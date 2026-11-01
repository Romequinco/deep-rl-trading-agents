"""Estilo común de las figuras: misma paleta y formato en todos los notebooks y en el informe.

Uso: `from common.style import setup, AGENT_COLORS, MARKET_LABELS, savefig` y `setup()` al principio.
"""

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd

from .protocol import RESULTS_DIR

AGENT_COLORS = {
    "aleatorio": "#9AA0A6",
    "buy_hold": "#F29D38",
    "no_operar": "#5F6368",
    "regla_tendencia": "#00A3A3",
    "regla_reversion": "#B56AC9",
    "qlearning": "#2B6CB0",
    "sarsa": "#2F9E6E",
    "dqn": "#6C5CE7",
    "a3c": "#D63031",
}
MARKET_COLORS = {
    "tendencia_ciclo": "#6C5CE7",
    "browniano": "#0984E3",
    "reversion_media": "#00A884",
    "regimenes": "#D63031",
}
MARKET_LABELS = {
    "tendencia_ciclo": "1 · Tendencia + ciclo",
    "browniano": "2 · Browniano geométrico",
    "reversion_media": "3 · Reversión a la media",
    "regimenes": "4 · Regímenes",
}
POSITION_COLORS = {1: "#2F9E6E", 0: "#D63031"}  # con activo / sin activo, como en el taller


def formato(r):
    """Retorno legible: múltiplo del capital a partir de x2 y porcentaje por debajo."""
    return f"x{1 + r:.1f}" if r >= 1 else f"{r:+.1%}"


def log_axis(ax, axis="y"):
    """Escala logarítmica con marcas legibles, también cuando el rango es menor que una década."""
    getattr(ax, f"set_{axis}scale")("log")
    target = getattr(ax, f"{axis}axis")
    target.set_major_locator(ticker.LogLocator(base=10, subs=(1, 1.5, 2, 3, 5, 7), numticks=20))
    target.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    target.set_minor_formatter(ticker.NullFormatter())


def setup():
    """Aplica el estilo común de matplotlib."""
    plt.rcParams.update({
        "figure.figsize": (11, 4.5),
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
    })


def savefig(fig, name):
    """Guarda la figura en `results/figures/<name>.png` (lo que se use en el informe)."""
    path = RESULTS_DIR / "figures" / f"{name}.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path)
    return path


def plot_episode(prices, balances, positions, label="Agente"):
    """Figura de dos paneles del taller: precio con posición (verde/rojo) y balance frente a buy & hold."""
    valid_prices = prices[5:-1]
    on = np.where(positions == 1, valid_prices, np.nan)
    off = np.where(positions == 0, valid_prices, np.nan)
    bh = 1000.0 * 0.999 / prices[5] * prices[5:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.plot(valid_prices, color="black", alpha=0.7, linewidth=1.5, label="Precio")
    ax1.plot(on, ".", color=POSITION_COLORS[1], label="Con activo")
    ax1.plot(off, ".", color=POSITION_COLORS[0], label="Sin activo")
    ax1.set_ylabel("Precio ($)")
    ax1.legend(loc="upper left")

    ax2.plot(balances, color="blue", linewidth=2, label=label)
    ax2.plot(bh, color=AGENT_COLORS["buy_hold"], linestyle="--", linewidth=2, label="Buy & Hold")
    ax2.axhline(1000, color="gray", linestyle=":", label="Capital inicial")
    ax2.set_xlabel("Pasos de tiempo")
    ax2.set_ylabel("Balance ($)")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_learning_curve(rewards, window=50, label=None, ax=None):
    """Recompensa por episodio de entrenamiento y su media móvil."""
    ax = ax or plt.gca()
    ax.plot(rewards, alpha=0.25)
    ax.plot(pd.Series(rewards).rolling(window).mean(), label=label)
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa total")
    return ax
