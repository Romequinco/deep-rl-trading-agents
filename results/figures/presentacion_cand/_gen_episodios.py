"""Episodio de test representativo por mercado, para la presentación. Ejecutar desde notebooks/.

A diferencia de 01_episodios / 02_episodios (primera serie de validación), aquí se elige, por mercado,
la serie de test (de las 200) cuyo retorno con la semilla mediana de DQN está más cerca de la
mediana de esas 200: así el ejemplo coincide con la cifra que se muestra. Se usa la misma serie para A3C.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.a3c import A3CAgent
from common.agents import baseline_agents
from common.dqn import DQNAgent
from common.env import SimpleTradingEnv
from common.markets import MARKETS, make_prices
from common.protocol import N_SEEDS_FINAL, N_STEPS, RESULTS_DIR, TEST_SEED_OFFSET, run_episode
from common.style import AGENT_COLORS, MARKET_LABELS, POSITION_COLORS, log_axis, setup

setup()
OUT = RESULTS_DIR / "figures" / "presentacion_cand"
N_TEST = 200


def semilla_mediana(csv, cls, prefijo, m):
    final = pd.read_csv(RESULTS_DIR / csv)
    orden = np.argsort(final[final.mercado == m].sort_values("seed").retorno_mediana.to_numpy())
    return cls.load(f"{prefijo}_{m}_s{orden[len(orden) // 2]}")


agentes = {m: {"dqn": semilla_mediana("01_grid_final.csv", DQNAgent, "dqn", m),
               "a3c": semilla_mediana("02_grid_final.csv", A3CAgent, "a3c", m)} for m in MARKETS}

series = {}
for m in MARKETS:
    rets = np.array([run_episode(SimpleTradingEnv(make_prices(m, N_STEPS, TEST_SEED_OFFSET + i)), agentes[m]["dqn"])[1][-1] / 1000 - 1
                     for i in range(N_TEST)])
    i = int(np.argmin(np.abs(rets - np.median(rets))))
    series[m] = TEST_SEED_OFFSET + i
    print(m, "serie", series[m], f"retorno {rets[i]:+.3f}", f"mediana {np.median(rets):+.3f}")

for alg, etiqueta in (("dqn", "DQN"), ("a3c", "A3C")):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), sharex=True)
    for j, m in enumerate(MARKETS):
        precios = make_prices(m, N_STEPS, series[m])
        _, balance, _, posicion = run_episode(SimpleTradingEnv(precios), agentes[m][alg])
        validos = precios[5:-1]
        ax = axes[0, j]
        ax.plot(validos, color="black", lw=0.9, alpha=0.5)
        log_axis(ax)
        ax.scatter(np.flatnonzero(posicion == 1), validos[posicion == 1], s=5, color=POSITION_COLORS[1], label="con activo")
        ax.scatter(np.flatnonzero(posicion == 0), validos[posicion == 0], s=5, color=POSITION_COLORS[0], label="sin activo")
        ax.set_title(MARKET_LABELS[m])
        ax = axes[1, j]
        ax.plot(balance, color=AGENT_COLORS[alg], lw=2, label=etiqueta)
        ax.plot(run_episode(SimpleTradingEnv(precios), baseline_agents()["buy_hold"])[1], color=AGENT_COLORS["buy_hold"], lw=2, ls="--", label="buy & hold")
        ax.axhline(1000, color="gray", ls=":", lw=1)
        log_axis(ax)
        ax.set_xlabel("paso")
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("precio (escala log)")
    axes[0, 0].legend(fontsize=8, markerscale=3)
    axes[1, 0].set_ylabel("balance ($, escala log)")
    fig.suptitle(f"Un episodio típico por mercado ({etiqueta}, semilla mediana, serie cercana a la mediana)", x=0.01, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / f"pres_episodios_{alg}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
