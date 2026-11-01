"""DQN con PyTorch (D-0.2, diseño en D-2.2).

Las opciones marcadas como mejora las usa el §7 del notebook 01 (D-2.5). Con sus valores por defecto, el agente es el base.

Se implementa aquí y no en el notebook para poder lanzar barridos con `run_grid`.
Observación y recompensa se escalan dentro del agente con `train_scales(mercado)` (D-1.1): `evaluate()` y el
entorno siguen trabajando con los valores reales. El final de la serie es terminal (D-1.2).
"""

import copy
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
from torch import nn

from .agents import BUY, HOLD, N_ACTIONS, OBS_DIM, SELL, policy_string
from .env import SimpleTradingEnv
from .markets import make_prices
from .protocol import (N_STEPS, RESULTS_DIR, convergence_episode, evaluate, greedy_convergence, set_seed, summarize,
                       train_scales, train_series_seed)


@dataclass
class DQNConfig:
    gamma: float = 0.9
    lr: float = 1e-3
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995  # por episodio
    memory_size: int = 20_000
    min_memory: int = 1_000  # transiciones antes de empezar a entrenar
    batch_size: int = 128
    hidden: int = 64
    train_every: int = 4  # pasos de entorno por paso de gradiente
    target_update: int = 500  # pasos de entorno entre copias a la red objetivo
    use_target: bool = True
    normalize_obs: bool = True
    normalize_reward: bool = True
    n_episodes: int = 500
    eval_every: int = 25  # episodios entre puntos de la curva sin exploración (D-0.3)
    n_eval: int = 20  # series de validación de cada punto
    # Mejoras (notebook 01, §7)
    double_canonical: bool = False  # Doble DQN, y una acción que no hace nada cuenta como mantener
    log_reward: bool = False  # aprende con log(balance siguiente / balance), que es lo que manda en el capital
    best_checkpoint: bool = False  # se queda con el mejor punto en validación, no con el último


class DQNAgent:
    def __init__(self, cfg, obs_dim=OBS_DIM, n_actions=N_ACTIONS, seed=None, scales=(1.0, 1.0)):
        self.cfg = cfg
        self.n_actions = n_actions
        self.scale_trend = scales[0] if cfg.normalize_obs else 1.0
        self.scale_reward = scales[1] if cfg.normalize_reward else 1.0
        self.scales = tuple(scales)
        # Generador propio: make_prices reinicia el de NumPy con cada serie
        self.rng = np.random.default_rng(seed)
        self.epsilon = cfg.epsilon
        self.q = nn.Sequential(nn.Linear(obs_dim, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, n_actions))
        self.q_target = nn.Sequential(nn.Linear(obs_dim, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, n_actions))
        self.sync_target()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)

        self.s = np.zeros((cfg.memory_size, obs_dim), dtype=np.float32)
        self.a = np.zeros(cfg.memory_size, dtype=np.int64)
        self.r = np.zeros(cfg.memory_size, dtype=np.float32)
        self.s2 = np.zeros((cfg.memory_size, obs_dim), dtype=np.float32)
        self.done = np.zeros(cfg.memory_size, dtype=np.float32)
        self.size = 0
        self.position = 0
        self.steps = 0

    def neutral_obs(self, position):
        """Observación real con tendencia 0 y `position`."""
        return [0.0, float(position)]

    def _scale_obs(self, obs):
        scaled = np.asarray(obs, dtype=np.float32).copy()
        scaled[0] /= self.scale_trend
        return scaled

    def q_values(self, obs):
        """Valor de cada acción para una observación real del entorno."""
        with torch.inference_mode():
            return self.q(torch.from_numpy(self._scale_obs(obs))).numpy()

    def act(self, obs, greedy=False):
        """Acción epsilon-greedy; con `greedy=True`, la de mayor Q."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.q_values(obs)
        if self.cfg.double_canonical:
            q[SELL if obs[1] < 0.5 else BUY] = -np.inf
        return int(q.argmax())

    def remember(self, obs, action, reward, next_obs, done):
        """Guarda la transición ya escalada, en una memoria circular que borra lo más antiguo."""
        i = self.position
        if self.cfg.double_canonical and action == (SELL if obs[1] < 0.5 else BUY):
            action = HOLD
        self.s[i], self.a[i], self.r[i] = self._scale_obs(obs), action, reward / self.scale_reward
        self.s2[i], self.done[i] = self._scale_obs(next_obs), float(done)
        self.position = (i + 1) % self.cfg.memory_size
        self.size = min(self.size + 1, self.cfg.memory_size)

    def sync_target(self):
        """Copia la red que aprende en la red objetivo (θ⁻ ← θ)."""
        self.q_target.load_state_dict(self.q.state_dict())

    def targets(self, r, s2, done):
        """Objetivo de Bellman, sin gradiente: y = r si el episodio termina (D-1.2)."""
        with torch.no_grad():
            net = self.q_target if self.cfg.use_target else self.q
            values = net(s2)
            if not self.cfg.double_canonical:
                return r + self.cfg.gamma * (1.0 - done) * values.max(dim=1).values
            # La red que aprende elige entre las dos acciones con efecto, y la objetivo la valora
            chooser = self.q(s2).clone()
            has_asset = s2[:, 1] >= 0.5
            chooser[~has_asset, SELL] = -torch.inf
            chooser[has_asset, BUY] = -torch.inf
            return r + self.cfg.gamma * (1.0 - done) * values.gather(1, chooser.argmax(dim=1, keepdim=True)).squeeze(1)

    def learn(self):
        """Un paso de entrenamiento con un minibatch de la memoria. Devuelve la pérdida."""
        idx = self.rng.choice(self.size, min(self.cfg.batch_size, self.size), replace=False)
        s, a = torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx])
        r, s2 = torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx])
        y = self.targets(r, s2, torch.from_numpy(self.done[idx]))
        predicted = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(predicted, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def step_done(self):
        """Cuenta un paso de entorno; entrena y sincroniza la red objetivo cuando toca."""
        self.steps += 1
        if self.size >= self.cfg.min_memory and self.steps % self.cfg.train_every == 0:
            self.learn()
        if self.cfg.use_target and self.steps % self.cfg.target_update == 0:
            self.sync_target()

    def end_episode(self):
        """Decaimiento de epsilon."""
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def save(self, name):
        """Guarda los pesos en `results/models/<name>.pt`."""
        path = RESULTS_DIR / "models" / f"{name}.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save({"state": self.q.state_dict(), "cfg": asdict(self.cfg), "scales": self.scales}, path)
        return path

    @classmethod
    def load(cls, name):
        """Agente listo para evaluar a partir de `results/models/<name>.pt`."""
        data = torch.load(RESULTS_DIR / "models" / f"{name}.pt", weights_only=True)
        agent = cls(DQNConfig(**{**data["cfg"], "memory_size": 1}), scales=tuple(data["scales"]))
        agent.q.load_state_dict(data["state"])
        agent.epsilon = 0.0
        return agent


def train_dqn(cfg, market, seed=0, n_steps=N_STEPS):
    """Entrena con una serie nueva por episodio, usando `train_series_seed(seed, episodio)`.

    Devuelve (agente, historia, tiempo de entrenamiento en segundos, sin contar las evaluaciones). La historia tiene
    las dos curvas de D-0.3:
    - con exploración: `recompensa` y `retorno` de cada episodio de entrenamiento;
    - sin exploración: `val_retorno`, `val_trades` y `val_tiempo` en los episodios `val_episodios`.
    """
    scales = train_scales(market)
    set_seed(seed)
    agent = DQNAgent(cfg, seed=seed, scales=scales)
    history = {"recompensa": [], "retorno": [], "trades": [], "val_episodios": [], "val_retorno": [], "val_trades": [],
               "val_tiempo": []}
    train_time = 0.0
    saved = []

    def checkpoint(episode):
        df = evaluate(agent, market, n_test=cfg.n_eval, n_steps=n_steps, split="val")
        if cfg.best_checkpoint:
            saved.append((float(df["retorno"].median()), episode, copy.deepcopy(agent.q.state_dict())))
        history["val_episodios"].append(episode)
        history["val_retorno"].append(float(df["retorno"].median()))
        history["val_trades"].append(float(df["n_trades"].mean()))
        history["val_tiempo"].append(train_time)

    for episode in range(cfg.n_episodes):
        if episode % cfg.eval_every == 0:
            checkpoint(episode)
        start = time.perf_counter()
        env = SimpleTradingEnv(make_prices(market, n_steps, train_series_seed(seed, episode)))
        obs, _ = env.reset()
        total, trades, terminated = 0.0, 0, False
        while not terminated:
            action = agent.act(obs)
            trades += (action == BUY and env.position == 0) or (action == SELL and env.position == 1)
            balance = env.balance
            next_obs, reward, terminated, _, info = env.step(action)
            # El entorno y las métricas no cambian: solo aquello con lo que aprende el agente
            learned = float(np.log(info["balance"] / balance)) if cfg.log_reward else reward
            agent.remember(obs, action, learned, next_obs, terminated)
            agent.step_done()
            obs = next_obs
            total += reward
        agent.end_episode()
        train_time += time.perf_counter() - start
        history["recompensa"].append(total)
        history["retorno"].append(env.balance / env.initial_balance - 1)
        history["trades"].append(trades)
    checkpoint(cfg.n_episodes)
    if cfg.best_checkpoint:
        history["punto_elegido"] = [select_checkpoint(agent, market, saved, n_steps=n_steps)]
    return agent, {k: np.array(v) for k, v in history.items()}, train_time


def select_checkpoint(agent, market, saved, n_best=3, n_steps=N_STEPS):
    """Carga el mejor punto guardado. Cada punto se midió con pocas series, así que los mejores, junto con el último, se
    vuelven a medir con todas las de validación antes de elegir. Devuelve el episodio elegido."""
    order = sorted(range(len(saved)), key=lambda i: saved[i][0], reverse=True)
    best_score, best_index = -np.inf, len(saved) - 1
    for i in sorted({*order[:n_best], len(saved) - 1}):
        agent.q.load_state_dict(saved[i][2])
        score = float(evaluate(agent, market, n_steps=n_steps, split="val")["retorno"].median())
        if score > best_score:
            best_score, best_index = score, i
    agent.q.load_state_dict(saved[best_index][2])
    return saved[best_index][1]


class EnsembleAgent:
    """Decide con la media de la Q de varios agentes entrenados. Solo sirve para evaluar."""

    def __init__(self, agents):
        self.agents = agents

    def act(self, obs, greedy=True):
        q = np.mean([a.q_values(obs) for a in self.agents], axis=0)
        if self.agents[0].cfg.double_canonical:
            q[SELL if obs[1] < 0.5 else BUY] = -np.inf
        return int(q.argmax())


def evalua_modelos(params):
    """Función para `run_grid`: evalúa modelos ya guardados, sin entrenar. `params`: `modelos` (nombres), `mercado`, `split`
    y `conjunto` (True: un solo agente con la media de sus Q; False: el primero de la lista)."""
    torch.set_num_threads(1)
    agents = [DQNAgent.load(name) for name in params["modelos"]]
    agent = EnsembleAgent(agents) if params.get("conjunto") else agents[0]
    df = evaluate(agent, params["mercado"], split=params["split"])
    return {"fila": summarize("dqn_mejorado", params["mercado"], df), "curva": df["retorno"].to_numpy()}


def entrena_y_evalua(params):
    """Función para `run_grid`. `params` es un dict: `mercado`, `seed`, `split` (por defecto "val"), `guardar`
    (nombre para guardar los pesos, opcional) y cualquier campo de `DQNConfig`."""
    torch.set_num_threads(1)
    known = {f.name for f in fields(DQNConfig)}
    cfg = DQNConfig(**{k: v for k, v in params.items() if k in known})
    market, seed = params["mercado"], params["seed"]
    agent, history, train_time = train_dqn(cfg, market, seed)
    df = evaluate(agent, market, split=params.get("split", "val"))
    row = summarize("dqn", market, df, train_time=train_time)
    episode, steps, seconds = greedy_convergence(history, N_STEPS - 6)
    row.update({"episodios_convergencia": episode, "pasos_convergencia": steps, "tiempo_convergencia_s": seconds,
                "episodios_convergencia_taller1": convergence_episode(history["recompensa"]),
                "retorno_medio_final_train": float(np.median(history["retorno"][-50:])),
                "politica_sin_activo": policy_string(agent, 0), "politica_con_activo": policy_string(agent, 1)})
    history["eval_retornos"] = df["retorno"].to_numpy()  # por serie, para comparar en las mismas series
    if params.get("guardar"):
        agent.save(params["guardar"])
    return {"fila": row, "curva": history}
