"""Pruebas del código común. Desde la raíz: python -m unittest discover -s notebooks/common -t notebooks"""

import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from common import env as env_module
from common import a3c, dqn, markets, protocol, tuning
from common.agents import (BUY, HOLD, SELL, BuyHoldAgent, NoTradeAgent, RandomAgent, ReversionRuleAgent,
                           TrendRuleAgent, baseline_agents)
from common.dqn import DQNAgent, DQNConfig, EnsembleAgent, entrena_y_evalua, train_dqn
from common.env import SimpleTradingEnv
from common.markets import load_sp500, make_prices

ROOT = Path(__file__).resolve().parents[2]
ENUNCIADO = ROOT / "docs" / "raw" / "taller" / "taller_trader2.ipynb"


def definitions(source):
    """{nombre: AST} de las clases y funciones de primer nivel, sin posiciones ni comentarios."""
    return {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}


def dummy_fn(cfg):
    return {"fila": {"doble": cfg["x"] * 2}, "curva": np.arange(cfg["x"])}


class TestEnunciado(unittest.TestCase):
    """El entorno y los generadores son copia literal del enunciado (D-0.1)."""

    @classmethod
    def setUpClass(cls):
        if not ENUNCIADO.exists():
            raise unittest.SkipTest("docs/raw no está en este equipo")
        nb = json.loads(ENUNCIADO.read_text(encoding="utf-8"))
        cls.taller = {}
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                if source.lstrip().startswith(("class ", "def ")):
                    cls.taller.update(definitions(source))

    def test_entorno(self):
        ours = definitions(Path(env_module.__file__).read_text(encoding="utf-8"))
        self.assertEqual(ours["SimpleTradingEnv"], self.taller["SimpleTradingEnv"])

    def test_generadores(self):
        ours = definitions(Path(markets.__file__).read_text(encoding="utf-8"))
        for name in ["generar_precios", "generar_precios_browniano", "generar_precios_reversion_media",
                     "generar_precios_regimenes"]:
            self.assertEqual(ours[name], self.taller[name], name)


class TestProtocolo(unittest.TestCase):
    def test_series_reproducibles(self):
        np.testing.assert_array_equal(make_prices("regimenes", 100, 7), make_prices("regimenes", 100, 7))

    def test_semillas_no_se_cruzan(self):
        self.assertLess(protocol.train_series_seed(4, 9_999), protocol.VAL_SEED_OFFSET)
        with self.assertRaises(AssertionError):
            protocol.train_series_seed(5, 0)
        self.assertLess(protocol.VAL_SEED_OFFSET + protocol.N_VAL, protocol.TEST_SEED_OFFSET)

    def test_buy_hold_coincide_con_el_calculo_del_taller(self):
        prices = make_prices("browniano", protocol.N_STEPS, 3)
        _, balances, _, _ = protocol.run_episode(SimpleTradingEnv(prices), BuyHoldAgent())
        self.assertAlmostEqual(balances[-1], 1000.0 * 0.999 / prices[5] * prices[-1], places=6)

    def test_no_operar_conserva_el_capital(self):
        df = protocol.evaluate(NoTradeAgent(), "regimenes", n_test=3)
        self.assertTrue((df["retorno"] == 0).all() and (df["n_trades"] == 0).all())

    def test_baselines_reproducen_el_taller_1(self):
        """Mismas series y métricas que el Taller 1: si esto falla, las tablas dejan de ser comparables."""
        ref = pd.read_csv(ROOT / "results" / "taller1_tabular.csv")
        for market in markets.MARKETS:
            for name, agent in baseline_agents().items():
                row = ref[(ref.agente == name) & (ref.mercado == market)]
                if row.empty:
                    continue
                got = protocol.evaluate(agent, market)
                self.assertAlmostEqual(got["retorno"].median(), row.retorno_mediana.item(), places=8, msg=f"{name} {market}")

    def test_reglas_simples(self):
        up, flat, down = np.array([0.02, 0.0]), np.array([0.0, 0.0]), np.array([-0.02, 0.0])
        trend, reversion = TrendRuleAgent(0.01), ReversionRuleAgent(0.01)
        self.assertEqual([trend.act(o) for o in (up, flat, down)], [BUY, HOLD, SELL])
        self.assertEqual([reversion.act(o) for o in (up, flat, down)], [SELL, HOLD, BUY])

    def test_escalas_de_train(self):
        sigma_tendencia, sigma_retorno = protocol.train_scales("reversion_media")
        self.assertAlmostEqual(sigma_tendencia, 0.024, places=3)
        self.assertAlmostEqual(sigma_retorno, 0.020, places=3)

    def test_aleatorio_con_semilla(self):
        a, b = RandomAgent(seed=1), RandomAgent(seed=1)
        self.assertEqual([a.act(None) for _ in range(20)], [b.act(None) for _ in range(20)])

    def test_convergencia_curva_sintetica(self):
        rewards = 1 - np.exp(-np.arange(1000) / 150)
        self.assertTrue(300 < protocol.convergence_episode(rewards) < 400)


class TestConvergencia(unittest.TestCase):
    def history(self, returns, trades=10.0):
        n = len(returns)
        return {"val_episodios": np.arange(n) * 25, "val_retorno": np.array(returns), "val_tiempo": np.arange(n) * 2.0,
                "val_trades": np.full(n, trades)}

    def test_primer_punto_cerca_del_final(self):
        episode, steps, seconds = protocol.greedy_convergence(self.history([0, 0, 0.1, 0.3, 0.3, 0.3, 0.3, 0.3]), 494)
        self.assertEqual((episode, steps, seconds), (75, 75 * 494, 6.0))

    def test_sin_operar_no_hay_convergencia(self):
        history = self.history([0.0] * 5, trades=0.0)
        self.assertTrue(np.isnan(protocol.greedy_convergence(history, 494)[0]))
        self.assertEqual(protocol.convergence_status(history), "no opera")

    def test_si_no_mejora_no_hay_convergencia(self):
        """Antes devolvía el episodio 0, que parecía una convergencia instantánea."""
        history = self.history([0.1, -0.05, -0.08, -0.1, -0.1])
        self.assertTrue(np.isnan(protocol.greedy_convergence(history, 494)[0]))
        self.assertEqual(protocol.convergence_status(history), "no mejora")

    def test_nunca_converge_en_el_punto_inicial(self):
        episode, _, _ = protocol.greedy_convergence(self.history([0.0, 1.0, 1.0, 1.0, 1.0]), 494)
        self.assertEqual(episode, 25)

    def test_recalcula_la_convergencia_desde_las_curvas(self):
        curves = {"a": self.history([0.1, -0.05, -0.08, -0.1, -0.1]), "b": self.history([0.0, 1.0, 1.0, 1.0, 1.0])}
        df = protocol.recompute_convergence(pd.DataFrame({"id": ["a", "b"], "episodios_convergencia": [0, 3]}), curves)
        self.assertTrue(np.isnan(df.episodios_convergencia[0]))
        self.assertEqual(df.episodios_convergencia[1], 25)
        self.assertEqual(df.estado_convergencia.tolist(), ["no mejora", "converge"])

    def test_agrega_semillas_en_una_fila(self):
        rows = [dict.fromkeys(protocol.RESULT_COLUMNS, 1.0) | {"agente": "dqn", "mercado": "browniano",
                                                                 "retorno_mediana": r} for r in (0.1, 0.2, 0.9)]
        df = protocol.aggregate_seeds(rows)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.retorno_mediana[0], 0.2)


class TestDQN(unittest.TestCase):
    def agent(self, **kwargs):
        return DQNAgent(DQNConfig(memory_size=50, hidden=8, **kwargs), seed=0, scales=(0.5, 0.02))

    def test_objetivo_terminal_es_la_recompensa(self):
        y = self.agent(gamma=0.9).targets(torch.tensor([2.0]), torch.zeros(1, 2), torch.tensor([1.0]))
        self.assertAlmostEqual(y.item(), 2.0)

    def test_objetivo_usa_la_red_objetivo(self):
        agent = self.agent(gamma=0.5)
        with torch.no_grad():
            for p in agent.q_target.parameters():
                p.zero_()
            agent.q_target[2].bias[1] = 4.0
        y = agent.targets(torch.tensor([1.0]), torch.zeros(1, 2), torch.tensor([0.0]))
        self.assertAlmostEqual(y.item(), 1.0 + 0.5 * 4.0)

    def test_copia_a_la_red_objetivo_en_el_sentido_correcto(self):
        agent = self.agent()
        with torch.no_grad():
            agent.q[2].bias[0] = 7.0
        self.assertNotEqual(agent.q_target[2].bias[0].item(), 7.0)
        agent.sync_target()
        self.assertEqual(agent.q_target[2].bias[0].item(), 7.0)
        self.assertEqual(agent.q[2].bias[0].item(), 7.0)

    def test_escala_en_memoria_y_valores_reales_fuera(self):
        agent = self.agent()
        agent.remember(np.array([1.0, 0.0]), 2, 0.04, np.array([-1.0, 1.0]), True)
        np.testing.assert_allclose(agent.s[0], [2.0, 0.0])
        np.testing.assert_allclose(agent.s2[0], [-2.0, 1.0])
        self.assertAlmostEqual(float(agent.r[0]), 2.0, places=5)
        self.assertEqual(agent.done[0], 1.0)

    def test_sin_normalizar_no_escala(self):
        agent = self.agent(normalize_obs=False, normalize_reward=False)
        agent.remember(np.array([1.0, 0.0]), 2, 0.04, np.array([-1.0, 1.0]), False)
        np.testing.assert_allclose(agent.s[0], [1.0, 0.0])
        self.assertAlmostEqual(float(agent.r[0]), 0.04, places=6)

    def test_memoria_circular_borra_lo_mas_antiguo(self):
        agent = self.agent()
        for i in range(55):
            agent.remember(np.zeros(2), i % 3, float(i), np.zeros(2), False)
        self.assertEqual(agent.size, 50)
        self.assertAlmostEqual(float(agent.r[0]) * 0.02, 50.0, places=3)
        self.assertAlmostEqual(float(agent.r[4]) * 0.02, 54.0, places=3)

    def test_greedy_no_usa_epsilon_ni_el_generador(self):
        agent = self.agent(epsilon=1.0)
        obs = np.array([0.1, 0.0], dtype=np.float32)
        state = agent.rng.bit_generator.state
        self.assertEqual(agent.act(obs, greedy=True), int(agent.q_values(obs).argmax()))
        self.assertEqual(agent.rng.bit_generator.state, state)

    def test_aprende_un_objetivo_simple(self):
        """Con γ = 0 y recompensa fija por acción, la Q de cada acción debe acercarse a su recompensa."""
        agent = DQNAgent(DQNConfig(gamma=0.0, lr=0.01, memory_size=300, batch_size=64, hidden=16), seed=0, scales=(1, 1))
        for i in range(300):
            agent.remember(np.zeros(2), i % 3, float(i % 3), np.zeros(2), False)
        for _ in range(400):
            agent.learn()
        np.testing.assert_allclose(agent.q_values(np.zeros(2)), [0.0, 1.0, 2.0], atol=0.05)

    def test_guardar_y_cargar_conserva_la_politica(self):
        agent = self.agent()
        obs = np.array([0.3, 1.0], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp, mock.patch("common.dqn.RESULTS_DIR", Path(tmp)):
            agent.save("prueba")
            loaded = DQNAgent.load("prueba")
        np.testing.assert_allclose(loaded.q_values(obs), agent.q_values(obs))
        self.assertEqual(loaded.scales, agent.scales)

    def test_entrenamiento_corto_y_reproducible(self):
        params = {"mercado": "browniano", "seed": 1, "n_episodes": 3, "min_memory": 50, "eval_every": 2, "n_eval": 2,
                  "hidden": 8}
        with mock.patch("common.dqn.evaluate", wraps=lambda *a, **k: protocol.evaluate(*a, **{**k, "n_test": 2})):
            first = entrena_y_evalua(params)
            second = entrena_y_evalua(params)
        self.assertEqual(len(first["curva"]["retorno"]), 3)
        np.testing.assert_array_equal(first["curva"]["val_episodios"], [0, 2, 3])
        np.testing.assert_array_equal(first["curva"]["retorno"], second["curva"]["retorno"])


class TestMejorasDQN(unittest.TestCase):
    def agent(self, **kwargs):
        return DQNAgent(DQNConfig(memory_size=50, hidden=8, **kwargs), seed=0, scales=(1.0, 1.0))

    def fijar(self, net, valores):
        with torch.no_grad():
            for p in net.parameters():
                p.zero_()
            net[2].bias[:] = torch.tensor(valores)

    def test_sin_mejoras_el_objetivo_es_el_maximo(self):
        agent = self.agent(gamma=1.0)
        self.fijar(agent.q_target, [5.0, 1.0, 2.0])
        y = agent.targets(torch.tensor([0.0]), torch.tensor([[0.0, 0.0]]), torch.tensor([0.0]))
        self.assertAlmostEqual(y.item(), 5.0)

    def test_doble_dqn_elige_con_una_red_y_valora_con_la_otra(self):
        agent = self.agent(gamma=1.0, double_canonical=True)
        self.fijar(agent.q, [0.0, 0.0, 9.0])  # la red que aprende elige comprar...
        self.fijar(agent.q_target, [5.0, 1.0, 2.0])  # ...y la objetivo le da 2, no su máximo de 5
        y = agent.targets(torch.tensor([0.0]), torch.tensor([[0.0, 0.0]]), torch.tensor([0.0]))
        self.assertAlmostEqual(y.item(), 2.0)

    def test_la_accion_que_no_hace_nada_cuenta_como_mantener(self):
        agent = self.agent(gamma=1.0, double_canonical=True)
        self.fijar(agent.q, [5.0, 1.0, 2.0])
        self.fijar(agent.q_target, [5.0, 1.0, 2.0])
        y = agent.targets(torch.zeros(2), torch.tensor([[0.0, 0.0], [0.0, 1.0]]), torch.zeros(2))
        self.assertEqual(y.tolist(), [2.0, 5.0])  # sin activo no cuenta vender; con activo no cuenta comprar
        self.assertEqual(agent.act(np.array([0.0, 0.0], dtype=np.float32), greedy=True), BUY)
        agent.remember(np.array([0.0, 0.0]), SELL, 0.0, np.zeros(2), False)
        self.assertEqual(agent.a[0], HOLD)
        agent.remember(np.array([0.0, 1.0]), SELL, 0.0, np.zeros(2), False)
        self.assertEqual(agent.a[1], SELL)

    def test_recompensa_log_es_el_cambio_del_capital(self):
        """Con el activo en cartera y sin operar, lo aprendido es el log del retorno del precio, no el retorno simple."""
        cfg = DQNConfig(log_reward=True, n_episodes=1, min_memory=10_000, memory_size=600, n_eval=1, eval_every=1, hidden=8)
        agent, _, _ = train_dqn(cfg, "browniano", seed=0, n_steps=60)
        prices = make_prices("browniano", 60, protocol.train_series_seed(0, 0))
        self.assertEqual(agent.size, 54)
        held = [i for i in range(agent.size) if agent.s[i, 1] == 1 and agent.s2[i, 1] == 1]
        self.assertGreater(len(held), 0)
        for i in held:
            self.assertAlmostEqual(float(agent.r[i]) * agent.scale_reward, np.log(prices[6 + i] / prices[5 + i]), places=5)

    def test_elige_el_mejor_punto_y_no_el_ultimo(self):
        agent = self.agent()
        estados = []
        for v in (0.0, 1.0, 2.0):
            self.fijar(agent.q, [v, 0.0, 0.0])
            estados.append(copy.deepcopy(agent.q.state_dict()))
        marcas = iter([0.1, 0.5, 0.2])

        def falso(*a, **k):
            return pd.DataFrame({"retorno": [next(marcas)]})

        with mock.patch("common.dqn.evaluate", falso):
            episodio = dqn.select_checkpoint(agent, "browniano", [(0.3, 0, estados[0]), (0.9, 25, estados[1]), (0.0, 50, estados[2])], 2)
        self.assertEqual(episodio, 25)
        self.assertEqual(agent.q[2].bias[0].item(), 1.0)

    def test_conjunto_promedia_la_q(self):
        a, b = self.agent(), self.agent()
        self.fijar(a.q, [3.0, 0.0, 0.0])
        self.fijar(b.q, [0.0, 0.0, 2.0])
        self.assertEqual(EnsembleAgent([a, b]).act(np.zeros(2, dtype=np.float32)), SELL)
        self.fijar(b.q, [0.0, 0.0, 4.0])
        self.assertEqual(EnsembleAgent([a, b]).act(np.zeros(2, dtype=np.float32)), BUY)

    def test_ventaja_emparejada(self):
        base = [np.array([0.0, 0.0, 0.0])] * 3
        mejor = [np.exp([0.1, 0.1, 0.1]) - 1, np.exp([0.2, 0.2, 0.2]) - 1, np.exp([-0.1, -0.1, -0.1]) - 1]
        ventaja, ganan = tuning.ventaja_emparejada(mejor, base)
        self.assertAlmostEqual(ventaja, 0.1)
        self.assertEqual(ganan, 2)


class TestA3C(unittest.TestCase):
    def agent(self, **kwargs):
        return a3c.A3CAgent(a3c.A3CConfig(hidden=8, **kwargs), seed=0, scales=(0.5, 0.02))

    def test_retorno_de_n_pasos_con_bootstrap(self):
        np.testing.assert_allclose(a3c.n_step_returns([1.0, 2.0], 0.5, 8.0), [1 + 0.5 * 2 + 0.25 * 8, 2 + 0.5 * 8])

    def test_retorno_terminal_parte_de_cero(self):
        np.testing.assert_allclose(a3c.n_step_returns([1.0, 2.0], 0.5, 0.0), [2.0, 2.0])

    def test_la_ventaja_no_entrena_al_critico_por_el_actor(self):
        agent = self.agent(value_coef=0.0, entropy_coef=0.0)
        logits, values = agent.net(torch.tensor([[0.1, 0.0], [-0.2, 1.0]]))
        loss, _ = a3c.a3c_loss(logits, values, torch.tensor([2, 0]), torch.tensor([1.0, -1.0]), agent.cfg)
        loss.backward()
        self.assertEqual(agent.net.v.weight.grad.abs().sum().item(), 0)
        self.assertGreater(agent.net.pi.weight.grad.abs().sum().item(), 0)

    def test_la_mascara_quita_la_accion_que_no_hace_nada(self):
        agent = self.agent(mask_redundant=True)
        for trend in (-1.0, 0.0, 1.0):
            self.assertEqual(agent.probs([trend, 0.0])[SELL], 0)
            self.assertEqual(agent.probs([trend, 1.0])[BUY], 0)
            self.assertAlmostEqual(float(agent.probs([trend, 0.0]).sum()), 1, places=5)

    def test_la_entropia_con_mascara_no_da_nan(self):
        agent = self.agent(mask_redundant=True)
        x = torch.tensor([[0.1, 0.0], [-0.2, 1.0]])
        logits, values = agent.net(x)
        logits = a3c.mask_logits(logits, x[:, 1], True)
        loss, entropy = a3c.a3c_loss(logits, values, torch.tensor([2, 0]), torch.tensor([1.0, -1.0]), agent.cfg)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertLessEqual(entropy.item(), float(np.log(2)) + 1e-6)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in agent.net.parameters()))

    def test_escala_la_observacion_y_no_el_entorno(self):
        agent = self.agent()
        np.testing.assert_allclose(agent._scale_obs([0.25, 1.0]), [0.5, 1.0])

    def test_greedy_es_determinista_y_no_usa_el_generador(self):
        agent = self.agent()
        state = agent.rng.get_state().clone()
        actions = {agent.act([0.1, 0.0], greedy=True) for _ in range(5)}
        self.assertEqual(len(actions), 1)
        self.assertTrue(torch.equal(agent.rng.get_state(), state))

    def test_optimizador_compartido_sin_mover_los_pesos(self):
        agent = self.agent()
        before = copy.deepcopy(agent.net.state_dict())
        agent.net.share_memory()
        optimizer = a3c.make_shared_optimizer(agent.net, 1e-3)
        for k, v in agent.net.state_dict().items():
            torch.testing.assert_close(v, before[k])
        self.assertTrue(all(v.is_shared() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)))

    def test_guardar_y_cargar_conserva_la_politica(self):
        agent = self.agent()
        with tempfile.TemporaryDirectory() as tmp, mock.patch("common.a3c.RESULTS_DIR", Path(tmp)):
            agent.save("prueba")
            loaded = a3c.A3CAgent.load("prueba")
        np.testing.assert_array_equal(agent.probs([0.1, 1.0]), loaded.probs([0.1, 1.0]))
        self.assertEqual(loaded.scales, agent.scales)

    def test_entrenamiento_real_con_procesos(self):
        cfg = a3c.A3CConfig(n_episodes=6, n_workers=2, eval_every=4, n_eval=2, hidden=8, t_max=5)
        start = 8  # se redondea hacia arriba a tramos completos entre evaluaciones
        agent, history, seconds = a3c.train_a3c(cfg, "browniano", seed=1, n_steps=60)
        self.assertEqual(len(history["retorno"]), start)
        self.assertFalse(np.isnan(history["retorno"]).any())  # todos los episodios de los dos procesos
        self.assertEqual(history["val_episodios"][0], 0)
        self.assertEqual(history["val_episodios"][-1], start)
        self.assertGreater(seconds, 0)
        self.assertEqual(protocol.evaluate(agent, "browniano", n_test=1, n_steps=60).shape[0], 1)

    def test_los_procesos_cambian_los_pesos_compartidos(self):
        protocol.set_seed(3)
        fresh = a3c.A3CAgent(a3c.A3CConfig(hidden=8), seed=3)
        cfg = a3c.A3CConfig(n_episodes=4, n_workers=2, eval_every=4, n_eval=1, hidden=8, t_max=5)
        trained, _, _ = a3c.train_a3c(cfg, "browniano", seed=3, n_steps=60)
        self.assertFalse(torch.equal(fresh.net.common.weight, trained.net.common.weight))

    def test_elige_el_mejor_punto_y_no_el_ultimo(self):
        cfg = a3c.A3CConfig(n_episodes=4, n_workers=2, eval_every=2, n_eval=1, hidden=8, t_max=5, best_checkpoint=True)
        agent = a3c.A3CAgent(cfg, seed=0, scales=protocol.train_scales("browniano"))
        states = [{k: v.clone() for k, v in agent.net.state_dict().items()} for _ in range(3)]
        with mock.patch("common.a3c.evaluate", side_effect=[pd.DataFrame({"retorno": [r]}) for r in (0.1, 0.5, 0.2)]):
            chosen = a3c.select_checkpoint(agent, "browniano", [(0.0, 0, states[0]), (0.1, 2, states[1]), (0.2, 4, states[2])])
        self.assertEqual(chosen, 2)


class TestResultados(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.object(protocol, "RESULTS_DIR", Path(tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_save_y_load_results(self):
        row = protocol.summarize("x", "browniano", pd.DataFrame(
            {"retorno": [0.1], "sharpe": [0.0], "max_drawdown": [0.0], "n_trades": [1]}))
        protocol.save_results([row], "01_prueba")
        pd.DataFrame({"mercado": ["otro"]}).to_csv(protocol.RESULTS_DIR / "01_grid_prueba.csv")
        df = protocol.load_results()
        self.assertEqual(df.agente.tolist(), ["x"])
        self.assertEqual(df.fuente.tolist(), ["01_prueba"])

    def test_run_grid_en_procesos_y_con_cache(self):
        configs = [{"x": 2}, {"x": 3}]
        df, curves = protocol.run_grid(configs, dummy_fn, "01_grid_prueba", n_jobs=2)
        self.assertEqual(df.doble.tolist(), [4, 6])
        self.assertEqual(len(curves[df.id[1]]), 3)
        df2, _ = protocol.run_grid(configs, lambda c: self.fail("no debía reentrenar"), "01_grid_prueba", n_jobs=1)
        self.assertEqual(df2.doble.tolist(), [4, 6])

    def test_un_corte_conserva_lo_terminado(self):
        def falla_en_el_segundo(cfg):
            if cfg["x"] == 3:
                raise RuntimeError("corte")
            return dummy_fn(cfg)

        configs = [{"x": 2}, {"x": 3}]
        with self.assertRaises(RuntimeError):
            protocol.run_grid(configs, falla_en_el_segundo, "01_grid_corte", n_jobs=1)
        df, _ = protocol.run_grid(configs, lambda c: dummy_fn(c) if c["x"] == 3 else self.fail("no debía reentrenar"),
                                  "01_grid_corte", n_jobs=1)
        self.assertEqual(df.doble.tolist(), [4, 6])

    def test_la_version_invalida_la_cache(self):
        self.assertNotEqual(protocol.config_id({"x": 1}, 1), protocol.config_id({"x": 1}, 2))


class TestAjuste(unittest.TestCase):
    def test_margen_de_ruido(self):
        self.assertTrue(tuning.mejora_de_verdad("tendencia_ciclo", 700.0, 500.0))
        self.assertFalse(tuning.mejora_de_verdad("tendencia_ciclo", 600.0, 500.0))
        self.assertTrue(tuning.mejora_de_verdad("reversion_media", 0.33, 0.30))
        self.assertFalse(tuning.mejora_de_verdad("browniano", 0.05, 0.0))
        self.assertAlmostEqual(tuning.margen_log("tendencia_ciclo"), np.log(1.25))


class TestDatos(unittest.TestCase):
    def test_sp500_con_huella_correcta(self):
        df = load_sp500()
        self.assertEqual(len(df), 2262)
        self.assertTrue(df.date.is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
