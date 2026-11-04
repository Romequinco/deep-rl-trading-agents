"""Controles de validez experimental de 03; no ejecutan el barrido completo."""

import unittest
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import torch

from common import a3c, extras
from common.dqn import DQNConfig, train_dqn
from common.env import SimpleTradingEnv
from common.protocol import set_seed


class TestExtras(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_plan_cambia_un_factor(self):
        base = asdict(extras.ExtraConfig())
        for job in extras.make_plan():
            differences = [k for k, value in job["config"].items() if value != base[k]]
            self.assertLessEqual(len(differences), 1)
            if differences:
                self.assertEqual(differences, [job["factor"]])

    def test_entorno_base_identico(self):
        prices = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, .02, 60)))
        original, extra = SimpleTradingEnv(prices), extras.ExtraTradingEnv(prices)
        np.testing.assert_array_equal(original.reset()[0], extra.reset()[0])
        for action in [2, 1, 0, 0, 2] * 10:
            a, b = original.step(action), extra.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[1:], b[1:])
            self.assertEqual(original.balance, extra.balance)

    def test_comision_en_efectivo_y_recompensa(self):
        for rate in (0, .001, .005):
            env = extras.ExtraTradingEnv(np.full(20, 100.0), commission=rate)
            env.reset()
            _, reward, *_ = env.step(2)
            self.assertAlmostEqual(env.balance, 1000 * (1 - rate))
            self.assertAlmostEqual(reward, -rate)
            env.step(1)
            _, reward, *_ = env.step(0)
            self.assertAlmostEqual(env.balance, 1000 * (1 - rate)**2)
            self.assertAlmostEqual(reward, -rate)

    def test_estado_no_lee_futuro(self):
        prices = np.linspace(100, 140, 60)
        changed = prices.copy()
        changed[26:] *= 10
        for features in ("base", "ventana", "ma30_vol", "ciego"):
            left = extras.ExtraTradingEnv(prices, features)
            right = extras.ExtraTradingEnv(changed, features)
            left.reset()
            right.reset()
            for t in (5, 10, 25):
                left.current_step = right.current_step = t
                np.testing.assert_array_equal(left._get_obs(), right._get_obs())

    def test_doble_selecciona_online_sin_canonizar_acciones(self):
        cfg = extras.ExtraConfig(double=True, hidden=2)
        agent = extras.create_agent(cfg, np.ones(2), 1., 0)
        for net in (agent.q, agent.q_target):
            for parameter in net.parameters():
                parameter.data.zero_()
        agent.q[-1].bias.data[:] = torch.tensor([10., 0., 0.])
        agent.q_target[-1].bias.data[:] = torch.tensor([1., 20., 0.])
        # SELL sin posición sigue estando disponible: no se mezcla con máscara.
        result = agent.targets(torch.tensor([0., 3.]), torch.zeros(2, 2), torch.tensor([0., 1.]))
        np.testing.assert_allclose(result.numpy(), [.9, 3.])

    def test_duelo_invariante_a_constante_en_ventajas(self):
        net = extras.DuelingQ(2, 4)
        x = torch.randn(5, 2)
        expected = net(x).detach().numpy()
        net.advantage.bias.data.add_(7.)
        np.testing.assert_allclose(net(x).detach().numpy(), expected, atol=1e-6)

    def test_clip_despues_de_normalizar(self):
        agent = extras.create_agent(extras.ExtraConfig(reward_clip=True), np.ones(2), .02, 0)
        agent.remember([0, 0], 2, .2, [0, 1], False)
        self.assertAlmostEqual(agent.r[0], 1.)

    def test_base_reproduce_entrenamiento_original(self):
        cfg = DQNConfig(n_episodes=2, n_eval=2, min_memory=10)
        extra_cfg = extras.ExtraConfig(n_episodes=2, n_eval=2, min_memory=10)
        original, _, _ = train_dqn(cfg, "reversion_media", seed=1, n_steps=500)
        extra, _, _ = extras.train_extra(extra_cfg, "reversion_media", seed=1)
        for a, b in zip(original.q.parameters(), extra.q.parameters()):
            np.testing.assert_array_equal(a.detach().numpy(), b.detach().numpy())

    def test_real_temporal_y_escalas_solo_train(self):
        data = extras.real_splits()
        self.assertEqual([len(data[s]) for s in ("train", "val", "nuevas")], [1259, 501, 502])
        cfg = extras.ExtraConfig(train_source="real")
        samples = extras.training_prices(cfg, "sp500", 1, 2)
        self.assertEqual(len(samples), 500)
        self.assertTrue(all(x in data["train"] for x in samples))

    def test_a2c_presupuesto_y_gradientes(self):
        cfg = a3c.A3CConfig(n_episodes=3, eval_every=3, n_eval=1)
        set_seed(0)
        initial = a3c.A3CAgent(cfg).net
        agent, history, _ = extras.train_a2c(cfg, "browniano", 0)
        self.assertEqual(len(history["retorno"]), 3)
        self.assertEqual(history["val_episodios"][-1], 3)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(initial.parameters(), agent.net.parameters())))

    def test_rechaza_test_oficial(self):
        with self.assertRaises(ValueError):
            extras.evaluate_extra(None, "browniano", extras.ExtraConfig(), "test")

    def test_evaluacion_lote_equivale_episodios_individuales(self):
        from common.protocol import episode_metrics, run_episode
        from common.markets import make_prices
        for features in ("base", "ventana", "ma30_vol", "ciego"):
            for commission in (0., .001, .005):
                cfg = extras.ExtraConfig(features=features, commission=commission)
                dim = {"base": 2, "ciego": 2, "ventana": 22, "ma30_vol": 4}[features]
                agent = extras.create_agent(cfg, np.ones(dim), 1., 2)
                prices = [make_prices("reversion_media", 70, s) for s in range(3)]
                batch = extras.batched_metrics(agent, prices, cfg)
                for i, series in enumerate(prices):
                    _, balances, actions, positions = run_episode(extras.ExtraTradingEnv(series, features, commission), agent)
                    expected = episode_metrics(balances, actions, positions)
                    for key, value in expected.items():
                        self.assertAlmostEqual(batch.loc[i, key], value, places=10)

    def test_evaluacion_actor_en_lote_equivale_original(self):
        from common.protocol import episode_metrics, run_episode
        from common.markets import make_prices
        agent = a3c.A3CAgent(a3c.A3CConfig(), scales=(.1, .02), seed=0)
        prices = [make_prices("regimenes", 80, s) for s in range(5)]
        batch = extras.batched_metrics(agent, prices, extras.ExtraConfig())
        for i, series in enumerate(prices):
            _, balances, actions, positions = run_episode(SimpleTradingEnv(series), agent)
            expected = episode_metrics(balances, actions, positions)
            for key, value in expected.items():
                self.assertAlmostEqual(batch.loc[i, key], value, places=10)


if __name__ == "__main__":
    unittest.main()
