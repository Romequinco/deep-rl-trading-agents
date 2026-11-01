"""Pruebas del código común. Desde la raíz: python -m unittest discover -s notebooks/common -t notebooks"""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from common import env as env_module
from common import markets, protocol
from common.agents import (BUY, HOLD, SELL, BuyHoldAgent, NoTradeAgent, RandomAgent, ReversionRuleAgent,
                           TrendRuleAgent, baseline_agents)
from common.dqn import DQNAgent, DQNConfig, entrena_y_evalua
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
        self.assertTrue(np.isnan(protocol.greedy_convergence(self.history([0.0] * 5, trades=0.0), 494)[0]))

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


class TestDatos(unittest.TestCase):
    def test_sp500_con_huella_correcta(self):
        df = load_sp500()
        self.assertEqual(len(df), 2262)
        self.assertTrue(df.date.is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
