"""Experimentos de un factor del notebook 03 (D-3.1).

No modifica el entorno, los generadores ni los agentes de 00–02. Los cambios
se implementan con una subclase de entorno y una subclase de DQN. Las métricas
siguen siendo las de protocol.py, medidas sobre el capital real.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from torch import nn

from . import a3c
from .agents import baseline_agents
from .dqn import DQNAgent, DQNConfig
from .env import SimpleTradingEnv
from .markets import EXTRA_MARKETS, load_sp500, make_prices, register_extra_markets
from .protocol import (N_FRESH, N_STEPS, RESULTS_DIR, aggregate_seeds, episode_metrics,
                       greedy_convergence, run_episode, run_grid, save_results,
                       set_seed, summarize, train_scales, train_series_seed)

CORE_MARKETS = ("tendencia_ciclo", "browniano", "reversion_media", "regimenes")
VERSION = 1


@dataclass
class ExtraConfig(DQNConfig):
    double: bool = False
    dueling: bool = False
    reward_clip: bool = False
    features: str = "base"       # base, ventana, ma30_vol, ciego
    commission: float = 0.001
    train_source: str = "individual"  # individual, mezcla, real


class ExtraTradingEnv(SimpleTradingEnv):
    """Extras causales; la lógica original se conserva, incluidos pasos y terminal.

    Las ventanas incompletas al principio usan solo historia disponible. No se
    retrasa el inicio del episodio para evitar cambiar también su duración.
    """

    def __init__(self, prices, features="base", commission=0.001):
        super().__init__(np.asarray(prices))
        if not 0 <= commission < 1:
            raise ValueError("la comisión debe estar en [0, 1)")
        self.features, self.commission = features, commission
        size = {"base": 2, "ciego": 2, "ventana": 22, "ma30_vol": 4}[features]
        self.observation_space = spaces.Box(-np.inf, np.inf, (size,), dtype=np.float32)
        self.log_returns = np.r_[0.0, np.diff(np.log(self.prices))]
        self.ma30 = pd.Series(self.prices).rolling(30, min_periods=1).mean().to_numpy()
        self.vol20 = pd.Series(self.log_returns).rolling(20, min_periods=2).std(ddof=0).fillna(0).to_numpy()
        # Precalcular indicadores causales evita recalcular la misma media en
        # cada interacción. No depende de acciones ni de precios posteriores.
        self.observations = np.zeros((len(prices), size), dtype=np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(self.prices, 6)
        means = windows.mean(axis=-1)
        self.observations[5:, 0] = np.clip((windows[:, -1] - means) / means, -1, 1)
        if features == "ciego":
            self.observations[:, 0] = 0
        elif features == "ventana":
            self.observations[:, 2:] = np.lib.stride_tricks.sliding_window_view(np.pad(self.log_returns, (19, 0)), 20)
        elif features == "ma30_vol":
            self.observations[:, 2] = self.prices / self.ma30 - 1
            self.observations[:, 3] = self.vol20

    def _get_obs(self):
        obs = self.observations[self.current_step].copy()
        obs[1] = self.position
        return obs

    def step(self, action):
        traded = (action == 2 and self.position == 0) or (action == 0 and self.position == 1)
        obs, reward, terminated, truncated, info = super().step(action)
        if traded and self.commission != 0.001:
            # super() ha cobrado 0,001. Sustituir ese coste, tanto en el
            # patrimonio como en la recompensa, sin tocar la dinámica original.
            ratio = (1 - self.commission) / (1 - 0.001)
            self.shares *= ratio
            self.cash *= ratio
            self.balance *= ratio
            reward += 0.001 - self.commission
            info["balance"] = self.balance
        return obs, reward, terminated, truncated, info


class DuelingQ(nn.Module):
    def __init__(self, obs_dim, hidden, n_actions=3):
        super().__init__()
        self.common = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, n_actions)

    def forward(self, x):
        h = self.common(x)
        advantage = self.advantage(h)
        return self.value(h) + advantage - advantage.mean(dim=-1, keepdim=True)


class ExtraDQN(DQNAgent):
    def __init__(self, cfg, scales, reward_scale, seed=0):
        self.feature_scales = np.asarray(scales, dtype=np.float32)
        super().__init__(cfg, obs_dim=len(scales), seed=seed,
                         scales=(float(scales[0]), float(reward_scale)))

    def _scale_obs(self, obs):
        return np.asarray(obs, dtype=np.float32) / self.feature_scales

    def targets(self, r, s2, done):
        if not self.cfg.double:
            return super().targets(r, s2, done)
        with torch.no_grad():
            choices = self.q(s2).argmax(dim=1, keepdim=True)
            future = self.q_target(s2).gather(1, choices).squeeze(1)
            return r + self.cfg.gamma * (1 - done) * future

    def remember(self, obs, action, reward, next_obs, done):
        if self.cfg.reward_clip:
            reward = float(np.clip(reward / self.scale_reward, -1, 1)) * self.scale_reward
        super().remember(obs, action, reward, next_obs, done)


def create_agent(cfg, scales, reward_scale, seed):
    agent = ExtraDQN(cfg, scales, reward_scale, seed)
    if cfg.dueling:
        agent.q = DuelingQ(len(scales), cfg.hidden)
        agent.q_target = DuelingQ(len(scales), cfg.hidden)
        agent.sync_target()
        agent.optimizer = torch.optim.Adam(agent.q.parameters(), lr=cfg.lr)
    return agent


@lru_cache(maxsize=1)
def real_splits():
    data = load_sp500()
    return {"train": data.loc[data.date.dt.year <= 2021, "close"].to_numpy(),
            "val": data.loc[data.date.dt.year.between(2022, 2023), "close"].to_numpy(),
            "nuevas": data.loc[data.date.dt.year.between(2024, 2025), "close"].to_numpy()}


def training_prices(cfg, market, seed, episode):
    series_seed = train_series_seed(seed, episode)
    if cfg.train_source == "real":
        prices = real_splits()["train"]
        start = np.random.default_rng(series_seed).integers(len(prices) - N_STEPS + 1)
        return prices[start:start + N_STEPS].copy()
    source = CORE_MARKETS[episode % len(CORE_MARKETS)] if cfg.train_source == "mezcla" else market
    # Para el experimento real, la transferencia browniana es el control fijado
    # antes de evaluar. Solo cambia la fuente de entrenamiento, no el test.
    source = "browniano" if source == "sp500" else source
    return make_prices(source, N_STEPS, series_seed)


@lru_cache(maxsize=64)
def fitted_scales(market, features, train_source):
    cfg = ExtraConfig(features=features, train_source=train_source)
    observations, returns = [], []
    for episode in range(50):
        prices = training_prices(cfg, market, 0, episode)
        env = ExtraTradingEnv(prices, features)
        observations.extend([env.reset()[0]])
        for t in range(6, len(prices)):
            env.current_step = t
            observations.append(env._get_obs())
        returns.extend(np.diff(prices) / prices[:-1])
    observations = np.asarray(observations)
    scales = observations.std(axis=0)
    scales[1] = 1.0  # posición binaria sin centrar
    scales = np.maximum(scales, 1e-8)
    if features == "ciego":
        scales[0] = 1.0
    if features in ("base", "ciego") and train_source == "individual" and market != "sp500":
        # Mismas constantes y orden de operaciones que el agente de 01.
        trend, reward = train_scales(market)
        if features == "base":
            scales[0] = trend
        return scales, reward
    return scales, float(np.std(returns))


def batched_metrics(agent, prices, cfg):
    """Evaluación greedy vectorizada; mismas cuentas que SimpleTradingEnv.

    Solo se agrupan inferencias sin aprendizaje. Cada serie conserva capital y
    posición propios. Es especialmente importante al evaluar 400 series.
    """
    prices = np.asarray(prices)
    n, length = prices.shape
    dim = {"base": 2, "ciego": 2, "ventana": 22, "ma30_vol": 4}[cfg.features]
    observations = np.zeros((n, length, dim), dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(prices, 6, axis=1)
    means = windows.mean(axis=-1)
    observations[:, 5:, 0] = np.clip((windows[..., -1] - means) / means, -1, 1)
    returns = np.c_[np.zeros(n), np.diff(np.log(prices), axis=1)]
    if cfg.features == "ciego":
        observations[:, :, 0] = 0
    elif cfg.features == "ventana":
        observations[:, :, 2:] = np.lib.stride_tricks.sliding_window_view(np.pad(returns, ((0, 0), (19, 0))), 20, axis=1)
    elif cfg.features == "ma30_vol":
        ma = pd.DataFrame(prices.T).rolling(30, min_periods=1).mean().to_numpy().T
        vol = pd.DataFrame(returns.T).rolling(20, min_periods=2).std(ddof=0).fillna(0).to_numpy().T
        observations[:, :, 2] = prices / ma - 1
        observations[:, :, 3] = vol
    positions = np.zeros(n, dtype=bool)
    cash, shares = np.full(n, 1000.), np.zeros(n)
    balances = np.full((n, length - 5), 1000.)
    trades = np.zeros(n, dtype=int)
    for t in range(5, length - 1):
        obs = observations[:, t].copy()
        obs[:, 1] = positions
        with torch.inference_mode():
            if isinstance(agent, ExtraDQN):
                actions = agent.q(torch.from_numpy(agent._scale_obs(obs))).argmax(dim=1).numpy()
            elif isinstance(agent, a3c.A3CAgent):
                actions = agent.net(torch.from_numpy(agent._scale_obs(obs)))[0].argmax(dim=1).numpy()
            else:
                actions = agent.q(torch.from_numpy(agent._scale_obs(obs))).argmax(dim=1).numpy()
        buy = (actions == 2) & ~positions
        sell = (actions == 0) & positions
        # Mantener el orden de cálculo del entorno, incluidas las comisiones.
        shares[buy] = (cash[buy] - cash[buy] * cfg.commission) / prices[buy, t]
        cash[buy] = 0
        sale = shares[sell] * prices[sell, t]
        cash[sell] = sale - sale * cfg.commission
        shares[sell] = 0
        positions[buy], positions[sell] = True, False
        trades += buy | sell
        balances[:, t - 4] = cash + shares * prices[:, t + 1]
    step = np.diff(balances, axis=1) / balances[:, :-1]
    std = step.std(axis=1)
    sharpe = np.divide(step.mean(axis=1), std, out=np.zeros(n), where=std > 0)
    return pd.DataFrame({"retorno": balances[:, -1] / 1000 - 1, "sharpe": sharpe,
                         "max_drawdown": (1 - balances / np.maximum.accumulate(balances, axis=1)).max(axis=1),
                         "n_trades": trades})


def evaluate_extra(agent, market, cfg, split="nuevas", n_series=None):
    if split not in ("val", "nuevas"):
        raise ValueError("03 no reutiliza el test oficial")
    if market == "sp500":
        prices_list = [real_splits()[split]]
    else:
        offset, count = (50_000, 100) if split == "val" else (600_000, N_FRESH)
        prices_list = [make_prices(market, N_STEPS, offset + i) for i in range(n_series or count)]
    if hasattr(agent, "q") or isinstance(agent, a3c.A3CAgent):
        return batched_metrics(agent, prices_list, cfg)
    rows = []
    for prices in prices_list:
        env = ExtraTradingEnv(prices, cfg.features, cfg.commission)
        _, balances, actions, positions = run_episode(env, agent)
        rows.append(episode_metrics(balances, actions, positions))
    return pd.DataFrame(rows)


def train_extra(cfg, market, seed):
    scales, reward_scale = fitted_scales(market, cfg.features, cfg.train_source)
    set_seed(seed)
    agent = create_agent(cfg, scales, reward_scale, seed)
    history = {name: [] for name in ("recompensa", "retorno", "trades", "val_episodios", "val_retorno", "val_trades", "val_tiempo")}
    elapsed = 0.0

    def checkpoint(episode):
        df = evaluate_extra(agent, market, cfg, "val", cfg.n_eval)
        for key, value in (("val_episodios", episode), ("val_retorno", df.retorno.median()),
                           ("val_trades", df.n_trades.mean()), ("val_tiempo", elapsed)):
            history[key].append(value)

    for episode in range(cfg.n_episodes):
        if episode % cfg.eval_every == 0:
            checkpoint(episode)
        start = time.perf_counter()
        env = ExtraTradingEnv(training_prices(cfg, market, seed, episode), cfg.features, cfg.commission)
        obs, _ = env.reset()
        total, trades, done = 0.0, 0, False
        while not done:
            action = agent.act(obs)
            trades += (action == 2 and env.position == 0) or (action == 0 and env.position == 1)
            balance = env.balance
            next_obs, reward, done, _, info = env.step(action)
            learned = float(np.log(info["balance"] / balance)) if cfg.log_reward else reward
            agent.remember(obs, action, learned, next_obs, done)
            agent.step_done()
            obs = next_obs
            total += reward
        agent.end_episode()
        elapsed += time.perf_counter() - start
        history["recompensa"].append(total)
        history["retorno"].append(env.balance / env.initial_balance - 1)
        history["trades"].append(trades)
    checkpoint(cfg.n_episodes)
    return agent, {k: np.asarray(v) for k, v in history.items()}, elapsed


def experiment(params):
    """Una configuración cacheable; guarda resultados por serie para emparejar."""
    torch.set_num_threads(1)
    register_extra_markets()
    cfg = ExtraConfig(**params["config"])
    market, seed = params["mercado"], params["seed"]
    agent, history, elapsed = train_extra(cfg, market, seed)
    results = evaluate_extra(agent, market, cfg)
    row = summarize("03_" + params["variante"], market, results, train_time=elapsed)
    episode, steps, seconds = greedy_convergence(history, N_STEPS - 6)
    row.update(episodios_convergencia=episode, pasos_convergencia=steps, tiempo_convergencia_s=seconds)
    history["eval_retornos"] = results.retorno.to_numpy()
    history["eval_trades"] = results.n_trades.to_numpy()
    # Pesos y escalas también quedan en caché, sin saturar models/ con ensayos.
    history["model_state"] = {k: v.clone() for k, v in agent.q.state_dict().items()}
    history["feature_scales"] = agent.feature_scales
    history["reward_scale"] = agent.scale_reward
    return {"fila": row, "curva": history}


def train_a2c(cfg, market, seed):
    """A2C vectorizado: tres rollouts, un gradiente medio y una actualización.

    Conserva red, Adam (incluida inicialización), retorno n-step, series por
    trabajador, entropía y presupuesto redondeado de train_a3c.
    """
    set_seed(seed)
    scales = train_scales(market)
    agent = a3c.A3CAgent(cfg, seed=seed, scales=scales)
    optimizer = a3c.make_shared_optimizer(agent.net, cfg.lr)
    generators = [torch.Generator().manual_seed(seed * 100 + rank) for rank in range(cfg.n_workers)]
    per_chunk = max(1, cfg.eval_every // cfg.n_workers)
    n_chunks = math.ceil(cfg.n_episodes / (cfg.n_workers * per_chunk))
    total_episodes = n_chunks * per_chunk * cfg.n_workers
    history = {key: [] for key in ("recompensa", "retorno", "trades", "entropia", "val_episodios", "val_retorno", "val_trades", "val_tiempo")}
    elapsed = 0.0

    def checkpoint(episode):
        df = evaluate_extra(agent, market, ExtraConfig(), "val", cfg.n_eval)
        for key, value in (("val_episodios", episode), ("val_retorno", df.retorno.median()),
                           ("val_trades", df.n_trades.mean()), ("val_tiempo", elapsed)):
            history[key].append(value)

    for episode in range(total_episodes // cfg.n_workers):
        if episode % per_chunk == 0:
            checkpoint(episode * cfg.n_workers)
        start = time.perf_counter()
        envs = [SimpleTradingEnv(make_prices(market, N_STEPS, train_series_seed(seed, episode * cfg.n_workers + rank)))
                for rank in range(cfg.n_workers)]
        obs = np.stack([env.reset()[0] for env in envs])
        totals, trades, entropies = np.zeros(cfg.n_workers), np.zeros(cfg.n_workers), []
        done = False
        while not done:
            states, actions, rewards = [], [], []
            for _ in range(cfg.t_max):
                scaled = obs.copy()
                scaled[:, 0] /= scales[0]
                states.append(scaled)
                with torch.no_grad():
                    probs = torch.softmax(agent.net(torch.from_numpy(scaled))[0], dim=-1)
                    acts = [int(torch.multinomial(probs[rank], 1, generator=generators[rank])) for rank in range(cfg.n_workers)]
                next_states, rs = [], []
                for rank, (env, action) in enumerate(zip(envs, acts)):
                    trades[rank] += (action == 2 and env.position == 0) or (action == 0 and env.position == 1)
                    nxt, reward, done, _, _ = env.step(action)
                    next_states.append(nxt)
                    rs.append(reward)
                obs = np.stack(next_states)
                actions.append(acts)
                totals += rs
                rewards.append(np.asarray(rs) / scales[1])
                if done:
                    break
            with torch.no_grad():
                last = np.zeros(cfg.n_workers) if done else agent.value(obs)
            returns = np.stack([a3c.n_step_returns(np.asarray(rewards)[:, rank], cfg.gamma, last[rank])
                                for rank in range(cfg.n_workers)], axis=1)
            logits, values = agent.net(torch.from_numpy(np.stack(states)).flatten(0, 1))
            loss, entropy = a3c.a3c_loss(logits, values, torch.tensor(actions).flatten(),
                                       torch.from_numpy(returns).flatten(), cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.net.parameters(), cfg.max_grad_norm)
            optimizer.step()
            entropies.append(entropy.item())
        elapsed += time.perf_counter() - start
        history["recompensa"].extend(totals)
        history["retorno"].extend(env.balance / env.initial_balance - 1 for env in envs)
        history["trades"].extend(trades)
        history["entropia"].extend([np.mean(entropies)] * cfg.n_workers)
    checkpoint(total_episodes)
    return agent, {k: np.asarray(v) for k, v in history.items()}, elapsed


def actor_experiment(params):
    torch.set_num_threads(1)
    cfg = a3c.A3CConfig(**params["config"])
    if params["variante"] == "a3c":
        # Sustituir solo la evaluación greedy del proceso principal. Los
        # trabajadores siguen ejecutando el A3C original. Contexto serial y
        # restauración automática: no se modifica common/a3c.py.
        with patch.object(a3c, "evaluate", actor_validation):
            agent, history, elapsed = a3c.train_a3c(cfg, params["mercado"], params["seed"])
    else:
        agent, history, elapsed = train_a2c(cfg, params["mercado"], params["seed"])
    results = evaluate_extra(agent, params["mercado"], ExtraConfig())
    row = summarize("03_" + params["variante"], params["mercado"], results, train_time=elapsed)
    ep, steps, seconds = greedy_convergence(history, N_STEPS - 6)
    row.update(episodios_convergencia=ep, pasos_convergencia=steps, tiempo_convergencia_s=seconds)
    history["eval_retornos"] = results.retorno.to_numpy()
    return {"fila": row, "curva": history}


def actor_validation(agent, market, n_test=None, n_steps=N_STEPS, greedy=True, split="val"):
    """Adaptador de la evaluación en lote al contrato del A3C original."""
    if n_steps != N_STEPS or not greedy:
        raise ValueError("03 solo agrupa evaluación greedy de 500 precios")
    return evaluate_extra(agent, market, ExtraConfig(), split, n_test)


def make_plan(seeds=range(5), n_episodes=500):
    """Plan fijado antes de evaluar; cada variante difiere en una sola clave."""
    base = asdict(ExtraConfig(n_episodes=n_episodes))
    changes = {"base": {}, "doble": {"double": True}, "duelo": {"dueling": True},
               "sin_objetivo": {"use_target": False}, "clip": {"reward_clip": True},
               "ventana": {"features": "ventana"}, "ma30_vol": {"features": "ma30_vol"},
               "log": {"log_reward": True}, "ciego": {"features": "ciego"},
               "mezcla": {"train_source": "mezcla"},
               "comision_0": {"commission": 0.0}, "comision_05": {"commission": 0.005}}
    jobs = []
    for label, change in changes.items():
        assert sum(base[k] != {**base, **change}[k] for k in base) == bool(change)
        for market in CORE_MARKETS:
            for seed in seeds:
                jobs.append({"variante": label, "mercado": market, "seed": seed,
                             "referencia": "base", "factor": next(iter(change), "ninguno"),
                             "config": {**base, **change}})
    for market in EXTRA_MARKETS:
        for seed in seeds:
            jobs.append({"variante": "base", "mercado": market, "seed": seed,
                         "referencia": "base", "factor": "mercado", "config": base.copy()})
    for label, source in (("transferencia", "individual"), ("real", "real")):
        for seed in seeds:
            jobs.append({"variante": label, "mercado": "sp500", "seed": seed,
                         "referencia": "transferencia", "factor": "train_source",
                         "config": {**base, "train_source": source}})
    return jobs


def paired_comparisons(df, curves):
    rows = []
    for (market, variant), group in df.groupby(["mercado", "variante"], sort=False):
        if variant in ("base", "transferencia", "a3c"):
            continue
        reference = "a3c" if variant == "a2c" else ("transferencia" if market == "sp500" else "base")
        ref = df[(df.mercado == market) & (df.variante == reference)].set_index("seed")
        advantages = []
        for row in group.itertuples():
            left = curves[row.id]["eval_retornos"]
            right = curves[ref.loc[row.seed, "id"]]["eval_retornos"]
            assert len(left) == len(right)
            advantages.append(float(np.median(np.log1p(left) - np.log1p(right))))
        med = float(np.median(advantages))
        margin = (np.log(1.25) if market == "tendencia_ciclo" else
                  (0.02 if market == "reversion_media" else (0.10 if market in CORE_MARKETS else np.nan)))
        rows.append({"mercado": market, "variante": variant, "referencia": reference,
                     "ventaja_log": med, "semillas_mejoran": sum(x > 0 for x in advantages),
                     "n_semillas": len(advantages), "min_ventaja": min(advantages), "max_ventaja": max(advantages),
                     "margen": margin, "supera_margen": bool(med > margin) if np.isfinite(margin) else pd.NA})
    return pd.DataFrame(rows)


def baseline_experiment(params):
    register_extra_markets()
    cfg = ExtraConfig(commission=params["commission"])
    result = evaluate_extra(baseline_agents()[params["baseline"]], params["mercado"], cfg)
    return {"fila": summarize("03_" + params["baseline"], params["mercado"], result),
            "curva": {"eval_retornos": result.retorno.to_numpy()}}


def run_suite(n_jobs=6, groups=None):
    plan = make_plan()
    selected = groups or ["dqn", "actores", "baselines"]
    if "dqn" in selected:
        (RESULTS_DIR / "03_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        df, curves = run_grid(plan, experiment, "03_grid_dqn", version=VERSION, n_jobs=n_jobs)
        save_results(aggregate_seeds(df), "03_extras_dqn")
        paired_comparisons(df, curves).to_csv(RESULTS_DIR / "03_grid_pareadas.csv", index=False)
    if "actores" in selected:
        actors = []
        for market in CORE_MARKETS:
            cfg = asdict(a3c.A3CConfig(lr=0.003 if market == "tendencia_ciclo" else 0.001,
                                      entropy_coef=0.0 if market == "reversion_media" else 0.01))
            for variant in ("a3c", "a2c"):
                for seed in range(5):
                    actors.append({"mercado": market, "seed": seed, "variante": variant, "config": cfg})
        # A3C crea procesos internos; ejecutar cada configuración sin anidar pools.
        df, curves = run_grid(actors, actor_experiment, "03_grid_actores", version=VERSION, n_jobs=1)
        save_results(aggregate_seeds(df), "03_extras_actores")
        paired_comparisons(df, curves).to_csv(RESULTS_DIR / "03_grid_pareadas_actores.csv", index=False)
    if "baselines" in selected:
        configs = [{"mercado": m, "baseline": b, "commission": c}
                   for m in (*CORE_MARKETS, *EXTRA_MARKETS, "sp500")
                   for c in ((0.0, 0.001, 0.005) if m in CORE_MARKETS else (0.001,))
                   for b in baseline_agents()]
        df, _ = run_grid(configs, baseline_experiment, "03_grid_baselines", version=VERSION, n_jobs=n_jobs)
        # Comisión en el nombre: el 04 no debe confundir condiciones distintas.
        df["agente"] += df.commission.map(lambda x: f"_c{x:g}")
        save_results(df, "03_extras_baselines")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--group", choices=("dqn", "actores", "baselines"), action="append")
    args = parser.parse_args()
    run_suite(args.jobs, args.group)
