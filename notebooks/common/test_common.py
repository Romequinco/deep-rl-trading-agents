"""Pruebas del código común. Desde la raíz: python -m unittest discover -s notebooks/common -t notebooks"""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from common import env as env_module
from common import markets, protocol
from common.agents import (BUY, HOLD, SELL, BuyHoldAgent, NoTradeAgent, RandomAgent, ReversionRuleAgent,
                           TrendRuleAgent, baseline_agents)
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

    def test_la_version_invalida_la_cache(self):
        self.assertNotEqual(protocol.config_id({"x": 1}, 1), protocol.config_id({"x": 1}, 2))


class TestDatos(unittest.TestCase):
    def test_sp500_con_huella_correcta(self):
        df = load_sp500()
        self.assertEqual(len(df), 2262)
        self.assertTrue(df.date.is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
