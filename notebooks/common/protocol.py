"""Protocolo común de entrenamiento y evaluación (D-0.1), igual que en el Taller 1.

- Entrenamiento: cada episodio usa una serie nueva del mismo mercado (`train_series_seed`).
- Validación: `N_VAL` series desde `VAL_SEED_OFFSET`. Solo se usan para elegir hiperparámetros.
- Test: `N_TEST` series desde `TEST_SEED_OFFSET`. Solo para el resultado final.
- Resultados: `save_results()` escribe `results/NN_nombre.csv` con las columnas de `RESULT_COLUMNS`,
  para que el 04 solo tenga que concatenar.
"""

import hashlib
import json
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .env import SimpleTradingEnv
from .markets import make_prices

N_STEPS = 500
N_TEST = 200
N_VAL = 100
N_SEEDS_FINAL = 5
TEST_SEED_OFFSET = 100_000
VAL_SEED_OFFSET = 50_000
TRAIN_SEED_STRIDE = 10_000

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULT_COLUMNS = ["agente", "mercado", "retorno_mediana", "retorno_medio", "std", "sharpe", "max_drawdown",
                  "n_trades", "episodios_convergencia", "tiempo_s"]


def train_series_seed(seed, episode):
    """Semilla de la serie del episodio `episode` para la semilla de entrenamiento `seed`."""
    series_seed = seed * TRAIN_SEED_STRIDE + episode
    assert series_seed < VAL_SEED_OFFSET, "las series de train invadirían las de validación"
    return series_seed


def set_seed(seed):
    """Fija las semillas de random, NumPy y PyTorch. Llamar también dentro de cada proceso de A3C."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_episode(env, agent, greedy=True):
    """Juega un episodio sin aprender. Devuelve recompensa total, balances, acciones y posiciones."""
    obs, _ = env.reset()
    action = agent.act(obs, greedy)
    total_reward = 0.0
    balances, actions, positions = [env.balance], [], []
    terminated = False
    while not terminated:
        positions.append(env.position)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        action = agent.act(obs, greedy)
        total_reward += reward
        balances.append(env.balance)
    return total_reward, np.array(balances), np.array(actions), np.array(positions)


def episode_metrics(balances, actions, positions):
    """Métricas de un episodio a partir de la evolución del balance (Sharpe por paso, sin anualizar)."""
    step_returns = np.diff(balances) / balances[:-1]
    peak = np.maximum.accumulate(balances)
    # Operación efectiva: comprar sin posición o vender con posición
    trades = int(np.sum(((actions == 2) & (positions == 0)) | ((actions == 0) & (positions == 1))))
    return {
        "retorno": balances[-1] / balances[0] - 1,
        "sharpe": step_returns.mean() / step_returns.std() if step_returns.std() > 0 else 0.0,
        "max_drawdown": float(np.max(1 - balances / peak)),
        "n_trades": trades,
    }


def evaluate(agent, market, n_test=None, n_steps=N_STEPS, greedy=True, split="test"):
    """Evalúa sin aprender sobre series de `split` ("test" o "val"). Devuelve un DataFrame (una fila por serie)."""
    offset = {"test": TEST_SEED_OFFSET, "val": VAL_SEED_OFFSET}[split]
    n_test = n_test or {"test": N_TEST, "val": N_VAL}[split]
    rows = []
    for i in range(n_test):
        env = SimpleTradingEnv(make_prices(market, n_steps, offset + i))
        _, balances, actions, positions = run_episode(env, agent, greedy)
        rows.append(episode_metrics(balances, actions, positions))
    return pd.DataFrame(rows)


def convergence_episode(rewards, window=50, frac=0.9):
    """Primer episodio en que la media móvil recorre `frac` del camino entre su valor inicial y el final.

    Es la métrica del Taller 1. No significa nada si el agente acaba sin operar: la curva sube solo
    porque deja de pagar comisiones al bajar la exploración (D-0.3).
    """
    if len(rewards) < window:
        return len(rewards)
    ma = pd.Series(rewards).rolling(window).mean().to_numpy()[window - 1:]
    target = ma[0] + frac * (ma[-1] - ma[0])
    reached = ma >= target if ma[-1] >= ma[0] else ma <= target
    return int(np.argmax(reached) + window - 1)


def summarize(agent_name, market, test_df, rewards=None, train_time=0.0):
    """Fila resumen con el esquema común `RESULT_COLUMNS`."""
    return {
        "agente": agent_name,
        "mercado": market,
        "retorno_mediana": test_df["retorno"].median(),
        "retorno_medio": test_df["retorno"].mean(),
        "std": test_df["retorno"].std(),
        "sharpe": test_df["sharpe"].mean(),
        "max_drawdown": test_df["max_drawdown"].mean(),
        "n_trades": test_df["n_trades"].mean(),
        "episodios_convergencia": convergence_episode(rewards) if rewards is not None else np.nan,
        "tiempo_s": train_time,
    }


def save_results(rows, name):
    """Guarda las filas resumen en `results/<name>.csv` (p. ej. name='01_dqn')."""
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    df.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    return df


def load_results():
    """Concatena los `results/NN_*.csv` con el esquema común. Los barridos (`NN_grid_*`) no se incluyen."""
    paths = [p for p in sorted(RESULTS_DIR.glob("[0-9][0-9]_*.csv")) if "_grid_" not in p.stem]
    frames = [pd.read_csv(p).assign(fuente=p.stem) for p in paths]
    frames = [f for f in frames if set(RESULT_COLUMNS) <= set(f.columns)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RESULT_COLUMNS + ["fuente"])


# ---------------------------------------------------------------------------
# Barridos en paralelo, con caché
# ---------------------------------------------------------------------------
def config_id(cfg, version=1):
    """Identificador de una configuración. Depende de `cfg` y de `version`, no del código:
    si cambia el algoritmo de verdad se sube `version`, y un comentario no invalida la caché."""
    payload = json.dumps({"cfg": cfg, "version": version}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_grid(configs, fn, name, version=1, n_jobs=None, recompute=False):
    """Ejecuta `fn(cfg)` para cada `cfg` en procesos separados y cachea cada resultado.

    `fn` debe estar en un módulo de `common/` (en Windows no se puede lanzar desde una celda) y devolver
    un dict con `fila` (dict de métricas) y `curva` (recompensa por episodio). Resúmenes en
    `results/<name>.csv`, que se versiona; lo demás en `results/cache/<name>/`, que no.
    Devuelve (DataFrame, {id: curva}).

    Los procesos compiten por la CPU, así que `tiempo_s` solo es comparable con `n_jobs=1`. A3C ya
    lanza sus propios procesos: usar `n_jobs=1` para no anidarlos.
    """
    cache = RESULTS_DIR / "cache" / name
    cache.mkdir(parents=True, exist_ok=True)
    ids = [config_id(c, version) for c in configs]
    done = {}
    if not recompute:
        for cid in ids:
            path = cache / f"{cid}.pkl"
            if path.exists():
                done[cid] = pickle.loads(path.read_bytes())
    todo = [(cid, c) for cid, c in zip(ids, configs) if cid not in done]
    if todo:
        n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
        print(f"Entrenando {len(todo)} configuraciones en {n_jobs} procesos ({len(configs) - len(todo)} en caché)...")
        cfgs = [c for _, c in todo]
        if n_jobs == 1:
            outputs = [fn(c) for c in cfgs]
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                outputs = list(pool.map(fn, cfgs))
        for (cid, _), out in zip(todo, outputs):
            (cache / f"{cid}.pkl").write_bytes(pickle.dumps(out))
            done[cid] = out
    df = pd.DataFrame([{**c, **done[cid]["fila"], "id": cid} for cid, c in zip(ids, configs)])
    df.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    return df, {cid: done[cid]["curva"] for cid in ids}
