"""Figuras candidatas para la presentación (16:9). Ejecutar desde notebooks/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from common.style import setup, MARKET_COLORS, MARKET_LABELS, AGENT_COLORS
from common.markets import MARKETS
setup()
R = Path("../results"); OUT = R / "figures/presentacion_cand"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 10.5, "savefig.dpi": 200, "savefig.bbox": "tight"})
dqn = pd.read_csv(R / "01_grid_final.csv"); a3c = pd.read_csv(R / "02_grid_final.csv"); base = pd.read_csv(R / "00_baselines.csv")
def lab(r): return f"x{1+r:.0f}" if r >= 1 else f"{r:+.1%}".replace(".", ",")
def y(m, r): return np.log10(1 + np.asarray(r)) if m == "tendencia_ciclo" else 100 * np.asarray(r)

# A: duelo DQN vs A3C por mercado, semillas + referencias
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, m in zip(axes, MARKETS):
    b = base[base.mercado == m].set_index("agente").retorno_mediana
    best = b.drop(["aleatorio"]).max()
    for k, (name, d) in enumerate((("DQN", dqn), ("A3C", a3c))):
        s = d[d.mercado == m].retorno_mediana.values
        c = AGENT_COLORS["dqn" if name == "DQN" else "a3c"]
        ax.bar(k, y(m, np.median(s)), color=c, alpha=.85, width=.6)
        ax.scatter(np.full(len(s), k) + np.linspace(-.15, .15, len(s)), y(m, s), color="black", s=14, zorder=3)
        ax.text(k + .33, y(m, np.median(s)), lab(np.median(s)), ha="left", va="center", fontsize=10.5, fontweight="bold", bbox=dict(fc="white", ec="none", pad=1))
    ax.axhline(y(m, best), color="black", ls="--", lw=1.2); ax.axhline(y(m, b["buy_hold"]), color=AGENT_COLORS.get("buy_hold", "orange"), ls=":", lw=1.5)
    ax.set_xticks([0, 1], ["DQN", "A3C"]); ax.set_xlim(-.5, 1.95); ax.margins(y=.12); ax.set_title(MARKET_LABELS[m], color=MARKET_COLORS[m], fontweight="bold")
    ax.set_ylabel("log10(capital)" if m == "tendencia_ciclo" else "retorno (%)"); lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + .14 * (hi - lo))
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color="black", ls="--", label="mejor referencia"), Line2D([], [], color=AGENT_COLORS.get("buy_hold", "orange"), ls=":", label="buy & hold"),
                    Line2D([], [], color="black", marker="o", lw=0, label="semillas")], loc="lower center", ncol=3, bbox_to_anchor=(.5, -.08), frameon=False)
fig.tight_layout(); fig.savefig(OUT / "pres_duelo_a.png"); plt.close(fig)

# B: KPI simple - barras horizontales de "distancia al listón" (% del listón alcanzado en log) para 1 y 3
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
for ax, m in zip(axes, ("tendencia_ciclo", "reversion_media")):
    b = base[base.mercado == m].set_index("agente").retorno_mediana
    tab = pd.read_csv(R / "taller1_tabular.csv"); ql = tab[(tab.mercado == m) & (tab.agente == "qlearning")].retorno_mediana.iloc[0]
    rows = [("Buy & hold", b["buy_hold"], AGENT_COLORS.get("buy_hold", "orange")), ("Q-Learning", ql, AGENT_COLORS.get("qlearning", "#2b6cb0")),
            ("DQN", dqn[dqn.mercado == m].retorno_mediana.median(), AGENT_COLORS["dqn"]), ("A3C", a3c[a3c.mercado == m].retorno_mediana.median(), AGENT_COLORS["a3c"])]
    ref = np.log1p(b.drop("aleatorio").max())
    for i, (n, r, c) in enumerate(rows):
        v = 100 * np.log1p(r) / ref
        ax.barh(i, v, color=c); ax.text(max(v, 0) + 1.5, i, f"{v:.0f} %  ({lab(r)})", va="center", fontsize=10.5, bbox=dict(fc="white", ec="none", pad=1), zorder=5)
    ax.axvline(100, color="black", ls="--"); ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlim(0, 135); ax.set_xlabel("% del listón (regla de un umbral, en log-capital)")
    ax.set_title(MARKET_LABELS[m], color=MARKET_COLORS[m], fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "pres_liston_b.png"); plt.close(fig)
print("ok")
