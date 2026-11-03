"""A3C con PyTorch (D-0.2, variante en D-2.1, diseño en D-2.6).

A3C real: `n_workers` procesos juegan a la vez, cada uno con sus series, y aplican sus gradientes sin cerrojos sobre unos
pesos y un optimizador compartidos. `worker` tiene que estar en este módulo: en Windows los procesos se crean con `spawn`
y no pueden importar funciones definidas en una celda de un notebook.

Observación y recompensa se escalan dentro del agente con `train_scales(mercado)` (D-1.1): `evaluate()` y el entorno siguen
trabajando con los valores reales. El final de la serie es terminal: el último tramo parte de R = 0 (D-1.2).

`n_episodes` es el total entre todos los procesos, para comparar con DQN con el mismo número de pasos de entorno.
"""

import math
import threading
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import nn

from .agents import BUY, N_ACTIONS, OBS_DIM, SELL, policy_string
from .env import SimpleTradingEnv
from .markets import make_prices
from .protocol import (N_STEPS, RESULTS_DIR, convergence_episode, evaluate, greedy_convergence, set_seed, summarize,
                       train_scales, train_series_seed)

LOG_COLUMNS = ("recompensa", "retorno", "trades", "entropia")
MASKED_LOGIT = -1e9  # probabilidad 0 sin los NaN que da -inf en el gradiente
BARRIER_TIMEOUT = 900  # segundos: si un proceso muere, el principal no se queda esperando para siempre


@dataclass
class A3CConfig:
    gamma: float = 0.9
    lr: float = 1e-3
    hidden: int = 64
    n_workers: int = 3
    t_max: int = 20
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 5.0
    n_episodes: int = 500  # total entre todos los procesos
    normalize_obs: bool = True
    normalize_reward: bool = True
    mask_redundant: bool = False  # sin activo no se puede vender y con activo no se puede comprar
    best_checkpoint: bool = False  # se queda con el mejor punto en validación, no con el último (notebook 02, §7)
    eval_every: int = 24  # episodios totales entre puntos de la curva sin exploración (D-0.3)
    n_eval: int = 20  # series de validación de cada punto


class ActorCritic(nn.Module):
    """Una capa común y dos salidas: logits de la política (actor) y valor del estado (crítico)."""

    def __init__(self, obs_dim, n_actions, hidden):
        super().__init__()
        self.common = nn.Linear(obs_dim, hidden)
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = torch.relu(self.common(x))
        return self.pi(h), self.v(h).squeeze(-1)


def mask_logits(logits, position, mask):
    """Con `mask`, descarta la acción que no hace nada: vender sin activo o comprar con activo.

    Mantener queda como única forma de no operar, así que la entropía deja de engañar (06, "Qué vigilar").
    """
    if not mask:
        return logits
    logits = logits.clone()
    no_asset = position < 0.5
    logits[..., SELL] = torch.where(no_asset, torch.full_like(position, MASKED_LOGIT), logits[..., SELL])
    logits[..., BUY] = torch.where(no_asset, logits[..., BUY], torch.full_like(position, MASKED_LOGIT))
    return logits


class A3CAgent:
    def __init__(self, cfg, obs_dim=OBS_DIM, n_actions=N_ACTIONS, seed=None, scales=(1.0, 1.0)):
        self.cfg = cfg
        self.scale_trend = scales[0] if cfg.normalize_obs else 1.0
        self.scale_reward = scales[1] if cfg.normalize_reward else 1.0
        self.scales = tuple(scales)
        self.net = ActorCritic(obs_dim, n_actions, cfg.hidden)
        # Generador propio: make_prices reinicia el de NumPy con cada serie
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def neutral_obs(self, position):
        """Observación real con tendencia 0 y `position`."""
        return [0.0, float(position)]

    def _scale_obs(self, obs):
        scaled = np.asarray(obs, dtype=np.float32).copy()
        scaled[..., 0] /= self.scale_trend
        return scaled

    def probs(self, obs):
        """Probabilidad de cada acción para una observación real del entorno."""
        x = torch.from_numpy(self._scale_obs(obs))
        with torch.inference_mode():
            logits = mask_logits(self.net(x)[0], x[..., 1], self.cfg.mask_redundant)
            return torch.softmax(logits, dim=-1).numpy()

    def value(self, obs):
        """Valor del crítico, en unidades de recompensa escalada."""
        with torch.inference_mode():
            return self.net(torch.from_numpy(self._scale_obs(obs)))[1].numpy()

    def act(self, obs, greedy=False):
        """Muestrea de la política; con `greedy=True`, la acción más probable."""
        x = torch.from_numpy(self._scale_obs(obs))
        with torch.inference_mode():
            logits = mask_logits(self.net(x)[0], x[1], self.cfg.mask_redundant)
            if greedy:
                return int(logits.argmax())
            return int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=self.rng))

    def save(self, name):
        """Guarda los pesos en `results/models/<name>.pt`."""
        path = RESULTS_DIR / "models" / f"{name}.pt"
        path.parent.mkdir(exist_ok=True)
        state = {k: v.clone() for k, v in self.net.state_dict().items()}
        torch.save({"state": state, "cfg": asdict(self.cfg), "scales": self.scales}, path)
        return path

    @classmethod
    def load(cls, name):
        """Agente listo para evaluar a partir de `results/models/<name>.pt`."""
        data = torch.load(RESULTS_DIR / "models" / f"{name}.pt", weights_only=True)
        agent = cls(A3CConfig(**data["cfg"]), scales=tuple(data["scales"]))
        agent.net.load_state_dict(data["state"])
        return agent


def make_shared_optimizer(model, lr):
    """Adam con su estado en memoria compartida. Si no, se crearía en el primer `step()` y sería propio de cada proceso."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()  # con gradiente nulo crea el estado sin mover los pesos
    optimizer.zero_grad(set_to_none=True)
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                value.share_memory_()
    return optimizer


def n_step_returns(rewards, gamma, last_value):
    """Retorno de hasta `t_max` pasos, hacia atrás: R ← r + γR, desde V(último estado) o desde 0 si es terminal."""
    returns = np.zeros(len(rewards), dtype=np.float32)
    R = last_value
    for i in reversed(range(len(rewards))):
        R = rewards[i] + gamma * R
        returns[i] = R
    return returns


def a3c_loss(logits, values, actions, returns, cfg):
    """Pérdida de un tramo. La ventaja no lleva gradiente hacia el actor; los tres términos se promedian por paso, para
    que α no cambie de sentido con `t_max`. Devuelve (pérdida, entropía media)."""
    dist = torch.distributions.Categorical(logits=logits)
    advantage = returns - values
    policy_loss = -(dist.log_prob(actions) * advantage.detach()).mean()
    value_loss = advantage.pow(2).mean()
    entropy = dist.entropy().mean()
    return policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy, entropy


def worker(rank, shared_model, optimizer, cfg, market, seed, n_steps, log, barrier, scales, episodes_per_chunk, n_chunks):
    """Bucle de un proceso: copia los pesos compartidos, juega hasta `t_max` pasos y aplica sus gradientes.

    `log` es un tensor compartido con una fila por episodio. Cada `episodes_per_chunk` episodios, los procesos esperan
    en `barrier` a que el principal mida la política: así la evaluación no cuenta como tiempo de entrenamiento.
    """
    torch.set_num_threads(1)
    set_seed(seed * 100 + rank)  # acciones distintas en cada proceso
    agent = A3CAgent(cfg, shared_model.common.in_features, shared_model.pi.out_features, seed * 100 + rank, scales)
    local = agent.net
    for chunk in range(n_chunks):
        barrier.wait(BARRIER_TIMEOUT)  # el principal mide aquí
        barrier.wait(BARRIER_TIMEOUT)
        for k in range(episodes_per_chunk):
            episode = chunk * episodes_per_chunk + k
            index = episode * cfg.n_workers + rank  # series disjuntas entre procesos
            env = SimpleTradingEnv(make_prices(market, n_steps, train_series_seed(seed, index)))
            obs, _ = env.reset()
            total, trades, entropies, terminated = 0.0, 0, [], False
            while not terminated:
                local.load_state_dict(shared_model.state_dict())  # θ' = θ
                states, actions, rewards = [], [], []
                for _ in range(cfg.t_max):
                    states.append(agent._scale_obs(obs))
                    action = agent.act(obs)
                    trades += (action == BUY and env.position == 0) or (action == SELL and env.position == 1)
                    obs, reward, terminated, _, _ = env.step(action)
                    actions.append(action)
                    rewards.append(reward / agent.scale_reward)
                    total += reward
                    if terminated:
                        break
                last_value = 0.0 if terminated else float(agent.value(obs))
                x = torch.from_numpy(np.stack(states))
                logits, values = local(x)
                logits = mask_logits(logits, x[:, 1], cfg.mask_redundant)
                returns = torch.from_numpy(n_step_returns(rewards, cfg.gamma, last_value))
                loss, entropy = a3c_loss(logits, values, torch.tensor(actions), returns, cfg)
                local.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(local.parameters(), cfg.max_grad_norm)
                for shared, own in zip(shared_model.parameters(), local.parameters()):
                    shared.grad = own.grad  # gradientes de θ' aplicados sobre θ
                optimizer.step()  # sin cerrojos, como en el artículo
                entropies.append(entropy.item())
            log[index] = torch.tensor([total, env.balance / env.initial_balance - 1, trades, float(np.mean(entropies))],
                                      dtype=torch.float64)
    barrier.wait(BARRIER_TIMEOUT)  # medida final


def train_a3c(cfg, market, seed=0, n_steps=N_STEPS):
    """Lanza `cfg.n_workers` procesos, cada uno con sus propias series de `train_series_seed`.

    Devuelve (agente, historia, tiempo de entrenamiento en segundos, sin contar arranque ni evaluaciones). La historia
    tiene las dos curvas de D-0.3, con los mismos nombres que en DQN. Los episodios de un mismo número de cada proceso
    transcurren a la vez: `val_episodios` cuenta el total.
    """
    scales = train_scales(market)
    set_seed(seed)
    agent = A3CAgent(cfg, seed=seed, scales=scales)
    shared = agent.net
    shared.share_memory()
    optimizer = make_shared_optimizer(shared, cfg.lr)

    per_chunk = max(1, cfg.eval_every // cfg.n_workers)
    n_chunks = math.ceil(cfg.n_episodes / (cfg.n_workers * per_chunk))
    total_episodes = n_chunks * per_chunk * cfg.n_workers
    log = torch.full((total_episodes, len(LOG_COLUMNS)), float("nan"), dtype=torch.float64).share_memory_()

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(cfg.n_workers + 1)
    processes = [ctx.Process(target=worker, args=(rank, shared, optimizer, cfg, market, seed, n_steps, log, barrier, scales,
                                                  per_chunk, n_chunks), daemon=True) for rank in range(cfg.n_workers)]
    start = time.perf_counter()
    for p in processes:
        p.start()

    history = {"val_episodios": [], "val_retorno": [], "val_trades": [], "val_tiempo": []}
    train_time = 0.0
    saved = []

    def checkpoint(episode):
        df = evaluate(agent, market, n_test=cfg.n_eval, n_steps=n_steps, split="val")
        if cfg.best_checkpoint:
            saved.append((float(df["retorno"].median()), episode, {k: v.clone() for k, v in shared.state_dict().items()}))
        history["val_episodios"].append(episode)
        history["val_retorno"].append(float(df["retorno"].median()))
        history["val_trades"].append(float(df["n_trades"].mean()))
        history["val_tiempo"].append(train_time)

    try:
        barrier.wait(BARRIER_TIMEOUT)  # todos han arrancado
        startup = time.perf_counter() - start
        for chunk in range(n_chunks):
            checkpoint(chunk * per_chunk * cfg.n_workers)
            barrier.wait(BARRIER_TIMEOUT)
            start_chunk = time.perf_counter()
            barrier.wait(BARRIER_TIMEOUT)
            train_time += time.perf_counter() - start_chunk
        checkpoint(total_episodes)
    except threading.BrokenBarrierError:
        for p in processes:
            p.terminate()
        raise RuntimeError("un proceso de A3C ha fallado o no responde") from None
    for p in processes:
        p.join()

    if cfg.best_checkpoint:
        history["punto_elegido"] = [select_checkpoint(agent, market, saved, n_steps=n_steps)]
    values = log.numpy().copy()
    history.update({name: values[:, i] for i, name in enumerate(LOG_COLUMNS)})
    history["arranque_s"] = [startup]
    return agent, {k: np.array(v) for k, v in history.items()}, train_time


def select_checkpoint(agent, market, saved, n_best=3, n_steps=N_STEPS):
    """Carga el mejor punto guardado, con el mismo criterio que en DQN: los mejores y el último se vuelven a medir con todas
    las series de validación antes de elegir. Devuelve el episodio elegido."""
    order = sorted(range(len(saved)), key=lambda i: saved[i][0], reverse=True)
    best_score, best_index = -np.inf, len(saved) - 1
    for i in sorted({*order[:n_best], len(saved) - 1}):
        agent.net.load_state_dict(saved[i][2])
        score = float(evaluate(agent, market, n_steps=n_steps, split="val")["retorno"].median())
        if score > best_score:
            best_score, best_index = score, i
    agent.net.load_state_dict(saved[best_index][2])
    return saved[best_index][1]


def evalua_modelo(params):
    """Función para `run_grid`: evalúa un modelo guardado, sin entrenar. `params`: `modelo`, `mercado` y `split`."""
    torch.set_num_threads(1)
    df = evaluate(A3CAgent.load(params["modelo"]), params["mercado"], split=params["split"])
    return {"fila": summarize("a3c", params["mercado"], df), "curva": df["retorno"].to_numpy()}


def entrena_y_evalua(params):
    """Función para `run_grid` (con `n_jobs=1`: A3C ya lanza sus procesos). `params` es un dict: `mercado`, `seed`, `split`
    (por defecto "val"), `guardar` (nombre para guardar los pesos, opcional) y cualquier campo de `A3CConfig`."""
    torch.set_num_threads(1)
    known = {f.name for f in fields(A3CConfig)}
    cfg = A3CConfig(**{k: v for k, v in params.items() if k in known})
    market, seed = params["mercado"], params["seed"]
    agent, history, train_time = train_a3c(cfg, market, seed)
    df = evaluate(agent, market, split=params.get("split", "val"))
    stochastic = evaluate(agent, market, n_test=50, greedy=False, split=params.get("split", "val"))
    row = summarize("a3c", market, df, train_time=train_time)
    episode, steps, seconds = greedy_convergence(history, N_STEPS - 6)
    row.update({"episodios_convergencia": episode, "pasos_convergencia": steps, "tiempo_convergencia_s": seconds,
                "episodios_convergencia_taller1": convergence_episode(history["recompensa"]),
                "retorno_medio_final_train": float(np.median(history["retorno"][-50:])),
                "retorno_mediana_estocastica": float(stochastic["retorno"].median()),
                "n_trades_estocastica": float(stochastic["n_trades"].mean()),
                "entropia_final": float(np.mean(history["entropia"][-50:])),
                "politica_sin_activo": policy_string(agent, 0), "politica_con_activo": policy_string(agent, 1)})
    history["eval_retornos"] = df["retorno"].to_numpy()  # por serie, para comparar en las mismas series
    if params.get("guardar"):
        agent.save(params["guardar"])
    return {"fila": row, "curva": history}
