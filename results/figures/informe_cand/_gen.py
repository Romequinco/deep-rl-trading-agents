"""Figuras candidatas para el informe PDF (2 páginas A4).

No entrena: solo lee results/*.csv, regenera series de precios con common.markets y recorta figuras existentes con PIL.
Tamaños pensados a escala real: ancho completo ~19 cm (7,5 in), media columna ~12 cm (4,7 in), columna lateral ~6,3 cm.
Ejecutar desde notebooks/:  python ../results/figures/informe_cand/_gen.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() if Path.cwd().name == "notebooks" else Path.cwd() / "notebooks"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image

from common.markets import make_prices
from common.protocol import N_STEPS, RESULTS_DIR, load_results
from common.style import AGENT_COLORS, MARKET_COLORS, MARKET_LABELS, formato, setup
from common.tuning import MARGEN, margen_log, mejora_de_verdad

OUT = RESULTS_DIR / "figures" / "informe_cand"
OUT.mkdir(exist_ok=True)
FIG = RESULTS_DIR / "figures"
FULL, HALF = 7.5, 4.7  # pulgadas

setup()
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "savefig.dpi": 200, "figure.dpi": 100, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.titlepad": 3, "lines.linewidth": 1.3,
})

MERCADOS = list(MARKET_LABELS)
CORTO = {"tendencia_ciclo": "1 · Tendencia + ciclo", "browniano": "2 · Browniano", "reversion_media": "3 · Reversión",
         "regimenes": "4 · Regímenes"}
NOMBRES = {"aleatorio": "Aleatorio", "buy_hold": "Buy & hold", "no_operar": "No operar", "regla_tendencia": "Regla tendencia",
           "regla_reversion": "Regla reversión", "qlearning": "Q-Learning (T1)", "sarsa": "SARSA (T1)", "dqn": "DQN", "a3c": "A3C"}
MEJOR_REF = {"tendencia_ciclo": "regla_tendencia", "browniano": "no_operar", "reversion_media": "regla_reversion",
             "regimenes": "no_operar"}


def csv(nombre):
    return pd.read_csv(RESULTS_DIR / f"{nombre}.csv")


def save(fig, nombre):
    fig.savefig(OUT / f"{nombre}.png")
    plt.close(fig)
    print("ok", nombre)


def color_title(ax, m, texto=None):
    ax.set_title(texto or CORTO[m], color=MARKET_COLORS[m], fontweight="bold")


# ---------------------------------------------------------------- datos comunes
tabulares = csv("taller1_tabular").query("agente in ['qlearning', 'sarsa']")
oficial = load_results().query("fuente in ['00_baselines', '01_dqn', '02_a3c']")
RES = pd.concat([oficial, tabulares], ignore_index=True)
ORDEN = [a for a in AGENT_COLORS if a in set(RES.agente)]
TABLA = RES.pivot(index="agente", columns="mercado", values="retorno_mediana").loc[ORDEN, MERCADOS]
SEMILLAS = pd.concat([csv("01_grid_final"), csv("02_grid_final")], ignore_index=True)


# ================================================================ 1. MERCADOS
def series(m, n, seed0=0):
    return np.array([make_prices(m, N_STEPS, seed0 + s) for s in range(n)])


def mercados_a():
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.55))
    for ax, m in zip(axes, MERCADOS):
        p = series(m, 1, seed0=3)[0]
        ax.plot(p, color=MARKET_COLORS[m], lw=1.1)
        if m == "tendencia_ciclo":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.axhline(p[0], color="grey", lw=0.6, ls=":")
        color_title(ax, m)
        ax.set_xticks([0, 250, 500])
        ax.tick_params(pad=1)
    axes[0].set_ylabel("precio")
    fig.text(0.5, -0.04, "paso", ha="center", fontsize=8)
    fig.tight_layout(w_pad=0.6)
    save(fig, "cand_mercados_a")


def mercados_b():
    fig, axes = plt.subplots(2, 2, figsize=(HALF, 3.0), sharex=True)
    for ax, m in zip(axes.ravel(), MERCADOS):
        P = series(m, 200)
        y = P / P[:, :1]
        q = np.percentile(y, [5, 25, 50, 75, 95], axis=0)
        x = np.arange(P.shape[1])
        c = MARKET_COLORS[m]
        ax.fill_between(x, q[0], q[4], color=c, alpha=0.15, lw=0)
        ax.fill_between(x, q[1], q[3], color=c, alpha=0.3, lw=0)
        ax.plot(x, q[2], color=c, lw=1.4)
        ax.plot(x, y[7], color="black", lw=0.6, alpha=0.7)
        ax.axhline(1, color="grey", lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_yticks({"tendencia_ciclo": [0.1, 1, 10, 100, 1000], "browniano": [0.5, 0.7, 1, 1.4, 2],
                       "reversion_media": [0.85, 0.9, 1, 1.1, 1.15], "regimenes": [0.3, 0.5, 0.7, 1, 1.4]}[m])
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"x{v:g}"))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        color_title(ax, m)
        ax.tick_params(pad=1)
    for ax in axes[1]:
        ax.set_xlabel("paso")
    for ax in axes[:, 0]:
        ax.set_ylabel("precio / inicial")
    leg = [Patch(color="grey", alpha=0.3, label="p25–p75"), Patch(color="grey", alpha=0.15, label="p5–p95"),
           Line2D([], [], color="grey", lw=1.4, label="mediana"), Line2D([], [], color="black", lw=0.6, label="una serie")]
    fig.legend(handles=leg, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), handlelength=1.4, columnspacing=1)
    fig.tight_layout(h_pad=0.4, w_pad=0.6)
    save(fig, "cand_mercados_b")


def mercados_c():
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 0.85))
    for ax, m in zip(axes, MERCADOS):
        P = series(m, 12, seed0=20)
        for p in P:
            ax.plot(np.log(p / p[0]), color="#B8BEC4", lw=0.5)
        ax.plot(np.log(P[3] / P[3][0]), color=MARKET_COLORS[m], lw=1.2)
        ax.axis("off")
        ax.set_title(CORTO[m], loc="left", fontsize=7.5, fontweight="bold", color=MARKET_COLORS[m], pad=1)
    fig.tight_layout(w_pad=0.4)
    save(fig, "cand_mercados_c")


# ================================================================ 2. RENDIMIENTO
def tr_market(m):
    if m == "tendencia_ciclo":
        return lambda v: 1 + np.asarray(v)
    return lambda v: 100 * np.asarray(v)


def rendimiento_a():
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.45), sharey=True)
    y = np.arange(len(ORDEN))
    for ax, m in zip(axes, MERCADOS):
        r = TABLA[m].to_numpy()
        log = m == "tendencia_ciclo"
        tr = tr_market(m)
        origen = 1 if log else 0
        ax.barh(y, tr(r) - origen, left=origen, height=0.72, color=[AGENT_COLORS[a] for a in ORDEN],
                edgecolor=["black" if a in ("dqn", "a3c") else "none" for a in ORDEN], linewidth=1)
        ext = {}
        for a in ("dqn", "a3c"):
            s = tr(SEMILLAS[(SEMILLAS.agente == a) & (SEMILLAS.mercado == m)].retorno_mediana.to_numpy())
            ax.scatter(s, np.full(len(s), ORDEN.index(a)), color="black", s=5, zorder=3)
            ext[a] = (s.min(), s.max())
        ax.axvline(tr(TABLA.loc[MEJOR_REF[m], m]), color="black", ls="--", lw=0.8)
        ax.axvline(origen, color="grey", lw=0.6)
        if log:
            ax.set_xscale("log")
            ax.set_xticks([1, 100, 10000], ["x1", "x100", "x10⁴"])
            ax.minorticks_off()
            ax.set_xlim(0.04, 40000)
            ax.set_xlabel("capital final (log)")
        else:
            lim = {"browniano": 42, "reversion_media": 52, "regimenes": 42}[m]
            ax.set_xlim(-lim, lim)
            ax.set_xlabel("retorno mediano (%)")
        for yi, a, ri in zip(y, ORDEN, r):
            v = tr(ri)
            der = v >= origen or log  # en el mercado 1 (log) la etiqueta siempre va a la derecha del origen
            v = max(v, origen) if log else v
            if a in ext:
                v = max(v, ext[a][1]) if der else min(v, ext[a][0])
            pos = v * (1.3 if der else 1 / 1.3) if log else v + (1 if der else -1) * 1.5
            ax.text(pos, yi, formato(ri), va="center", ha="left" if der else "right", fontsize=6.6,
                    fontweight="bold" if a in ("dqn", "a3c", MEJOR_REF[m]) else "normal")
        color_title(ax, m)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [NOMBRES[a] for a in ORDEN])
    for lab in axes[0].get_yticklabels():
        if lab.get_text() in ("DQN", "A3C"):
            lab.set_fontweight("bold")
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=0.5)
    save(fig, "cand_rendimiento_a")


def rendimiento_b():
    rank = TABLA.rank(axis=0, method="average").to_numpy()
    fig, ax = plt.subplots(figsize=(HALF, 2.55))
    ax.imshow(rank, cmap="RdYlGn", vmin=0, vmax=len(ORDEN) + 1, aspect="auto")
    for i, a in enumerate(ORDEN):
        for j, m in enumerate(MERCADOS):
            v = TABLA.loc[a, m]
            best = v == TABLA[m].max()
            ax.text(j, i, formato(v), ha="center", va="center", fontsize=7.5, fontweight="bold" if best or a in ("dqn", "a3c") else "normal")
            if best:
                ax.add_patch(Rectangle((j - .48, i - .46), .96, .92, fill=False, ec="black", lw=1.3))
    ax.set_xticks(range(4), [CORTO[m].replace(" · ", "\n") for m in MERCADOS])
    for lab, m in zip(ax.get_xticklabels(), MERCADOS):
        lab.set_color(MARKET_COLORS[m]); lab.set_fontweight("bold")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(ORDEN)), [NOMBRES[a] for a in ORDEN])
    for lab in ax.get_yticklabels():
        if lab.get_text() in ("DQN", "A3C"):
            lab.set_fontweight("bold")
    for s in (2.5, 4.5, 6.5):
        ax.axhline(s, color="white", lw=2.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    save(fig, "cand_rendimiento_b")


def rendimiento_c():
    """Dot plot: semillas de DQN y A3C frente a las referencias como líneas verticales."""
    refs = ["buy_hold", "no_operar", MEJOR_REF, "qlearning"]
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.75))
    for ax, m in zip(axes, MERCADOS):
        tr = tr_market(m)
        log = m == "tendencia_ciclo"
        for k, a in enumerate(("dqn", "a3c")):
            s = tr(SEMILLAS[(SEMILLAS.agente == a) & (SEMILLAS.mercado == m)].sort_values("seed").retorno_mediana.to_numpy())
            yk = 1 - k
            ax.scatter(s, np.full(len(s), yk) + np.linspace(-0.12, 0.12, len(s)), color=AGENT_COLORS[a], s=14, alpha=0.75, lw=0)
            ax.plot([tr(TABLA.loc[a, m])] * 2, [yk - 0.3, yk + 0.3], color=AGENT_COLORS[a], lw=2.4)
        ylo, yhi = -0.6, 1.6
        for a, ls in (("buy_hold", "-"), ("qlearning", "-"), (MEJOR_REF[m], "--")):
            v = tr(TABLA.loc[a, m])
            ax.axvline(v, color="black" if ls == "--" else AGENT_COLORS[a], lw=1.1, ls=ls, zorder=0)
        ax.set_ylim(ylo, yhi)
        ax.set_yticks([1, 0], ["DQN", "A3C"] if m == MERCADOS[0] else ["", ""])
        if log:
            ax.set_xscale("log")
            ax.set_xticks([1, 10, 100, 1000], ["x1", "x10", "x100", "x1000"])
            ax.minorticks_off()
            ax.set_xlim(0.6, 2000)
        else:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
            lo = {"browniano": -10, "reversion_media": -3, "regimenes": -15}[m]
            hi = {"browniano": 3, "reversion_media": 33, "regimenes": 3}[m]
            ax.set_xlim(lo, hi)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        color_title(ax, m)
    handles = [Line2D([], [], color="grey", marker="o", lw=0, ms=4, label="semillas"),
               Line2D([], [], color="grey", lw=2.4, label="mediana"),
               Line2D([], [], color=AGENT_COLORS["buy_hold"], lw=1.1, label="buy & hold"),
               Line2D([], [], color=AGENT_COLORS["qlearning"], lw=1.1, label="Q-Learning (T1)"),
               Line2D([], [], color="black", lw=1.1, ls="--", label="mejor referencia (regla / no operar)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(w_pad=0.6)
    save(fig, "cand_rendimiento_c")


# ================================================================ 3. CONVERGENCIA
def tiempos():
    t_dqn = csv("02_grid_tiempo_dqn").tiempo_s.median()
    dqn_ = csv("01_grid_final")
    factor = t_dqn / dqn_.tiempo_s.median()
    sem = pd.concat([dqn_.assign(tiempo_s=dqn_.tiempo_s * factor, tiempo_convergencia_s=dqn_.tiempo_convergencia_s * factor),
                     csv("02_grid_final")], ignore_index=True)
    return sem, t_dqn


def convergencia_a():
    sem, t_dqn = tiempos()
    mc = ["tendencia_ciclo", "reversion_media", "regimenes"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.2), gridspec_kw={"width_ratios": [1.5, 1]})
    for k, ag in enumerate(["dqn", "a3c"]):
        d = sem[(sem.agente == ag) & (sem.estado_convergencia == "converge")]
        for j, m in enumerate(mc):
            e = d[d.mercado == m].episodios_convergencia.to_numpy()
            x = j + (k - 0.5) * 0.36
            a1.scatter(np.full(len(e), x) + np.linspace(-0.06, 0.06, len(e)), e, color=AGENT_COLORS[ag], s=16, alpha=0.8, lw=0,
                       label=NOMBRES[ag] if j == 0 else None)
            if len(e):
                a1.hlines(np.median(e), x - 0.14, x + 0.14, color=AGENT_COLORS[ag], lw=2.2)
            a1.text(x, 540, f"{len(e)}/5", ha="center", color=AGENT_COLORS[ag], fontweight="bold", fontsize=7.5)
    a1.set_xticks(range(3), [CORTO[m] for m in mc])
    for lab, m in zip(a1.get_xticklabels(), mc):
        lab.set_color(MARKET_COLORS[m]); lab.set_fontweight("bold")
    a1.set(ylim=(0, 580), xlim=(-0.55, 2.55), ylabel="episodios hasta converger")
    a1.grid(axis="x", visible=False)
    a1.legend(loc="center right", handletextpad=0.2)
    a1.text(1.0, -0.2, "n/5 = semillas que convergen (en el mercado 2, ninguna)", transform=a1.transAxes, fontsize=6.8,
            va="top", ha="right", color="dimgray")
    t_a3c = sem[sem.agente == "a3c"].tiempo_s.median()
    conv = sem[sem.estado_convergencia == "converge"].groupby(["agente", "mercado"]).tiempo_convergencia_s.median()
    cats = ["Entrenamiento\n(500 ep.)", "Hasta converger\n(mercado 1)", "Hasta converger\n(mercado 3)"]
    vals = {"dqn": [t_dqn, conv["dqn", "tendencia_ciclo"], conv["dqn", "reversion_media"]],
            "a3c": [t_a3c, conv["a3c", "tendencia_ciclo"], conv["a3c", "reversion_media"]]}
    yy = np.arange(3)
    for k, ag in enumerate(("dqn", "a3c")):
        b = a2.barh(yy + (k - 0.5) * 0.38, vals[ag], 0.38, color=AGENT_COLORS[ag], label=NOMBRES[ag])
        a2.bar_label(b, fmt="%.1f s", fontsize=7, padding=2)
    a2.set_yticks(yy, cats)
    a2.invert_yaxis()
    a2.set_xlim(0, 40)
    a2.set_xlabel("segundos (mismo equipo)")
    a2.grid(axis="y", visible=False)
    a2.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=1)
    save(fig, "cand_convergencia_a")


def convergencia_b():
    """Episodios frente a segundos hasta converger, por semilla: DQN necesita menos episodios, A3C cada uno es más barato."""
    sem, _ = tiempos()
    d = sem[(sem.estado_convergencia == "converge") & sem.mercado.isin(["tendencia_ciclo", "reversion_media"])]
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    mk = {"tendencia_ciclo": "o", "reversion_media": "s"}
    for ag in ("dqn", "a3c"):
        for m in mk:
            g = d[(d.agente == ag) & (d.mercado == m)]
            ax.scatter(g.episodios_convergencia, g.tiempo_convergencia_s, marker=mk[m], s=26, color=AGENT_COLORS[ag], alpha=0.8,
                       edgecolor="white", lw=0.5)
    ax.set_xlabel("episodios hasta converger")
    ax.set_ylabel("segundos hasta converger")
    ax.set_xscale("log"); ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xticks([20, 50, 100, 200, 500]); ax.set_yticks([1, 2, 5, 10, 20, 50]); ax.set_ylim(1, 60)
    handles = [Line2D([], [], color=AGENT_COLORS["dqn"], marker="o", lw=0, label="DQN"),
               Line2D([], [], color=AGENT_COLORS["a3c"], marker="o", lw=0, label="A3C"),
               Line2D([], [], color="grey", marker="o", lw=0, label="mercado 1"),
               Line2D([], [], color="grey", marker="s", lw=0, label="mercado 3")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.8, handletextpad=0.1, borderaxespad=0.2)
    fig.tight_layout()
    save(fig, "cand_convergencia_b")


def crop_panels(path, cols, edges_frac, top_frac=0.0):
    """Recorta las columnas `cols` (0-based) de una figura 1×4 o 2×4 del notebook, sin el título general.
    `edges_frac`: bordes de cada columna (fracción del ancho), medidos a ojo sobre la figura original."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    edges = (np.array(edges_frac) * w).astype(int)
    top = int(h * top_frac)
    return [im.crop((edges[c], top, edges[c + 1], h)) for c in cols]


def hstack(imgs, gap=10):
    h = max(i.size[1] for i in imgs)
    out = Image.new("RGB", (sum(i.size[0] for i in imgs) + gap * (len(imgs) - 1), h), "white")
    x = 0
    for i in imgs:
        out.paste(i, (x, 0)); x += i.size[0] + gap
    return out


def vstack(imgs, gap=10):
    w = max(i.size[0] for i in imgs)
    out = Image.new("RGB", (w, sum(i.size[1] for i in imgs) + gap * (len(imgs) - 1)), "white")
    y = 0
    for i in imgs:
        out.paste(i, (0, y)); y += i.size[1] + gap
    return out


def label_strip(text, width, color, height=60):
    fig = plt.figure(figsize=(width / 200, height / 200), dpi=200)
    fig.text(0.01, 0.5, text, fontsize=9, fontweight="bold", color=color, va="center")
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return img.resize((width, height))


def convergencia_c():
    """Curvas de aprendizaje de 01 y 02 (mercados 1 y 3) recortadas: no hay curvas de A3C en CSV."""
    top = 0.13  # quita el título general
    e = [0, 0.266, 0.507, 0.758, 1]
    d = crop_panels(FIG / "01_dos_curvas.png", [0, 2], e, top_frac=top)
    a = crop_panels(FIG / "02_dos_curvas.png", [0, 2], e, top_frac=top)
    fila_d, fila_a = hstack(d), hstack(a)
    out = vstack([label_strip("DQN", fila_d.size[0], AGENT_COLORS["dqn"]), fila_d,
                  label_strip("A3C", fila_a.size[0], AGENT_COLORS["a3c"]), fila_a], gap=4)
    out.save(OUT / "cand_convergencia_c.png", dpi=(200, 200))
    print("ok cand_convergencia_c")


# ================================================================ 4. POLÍTICA
X_POL = np.linspace(-2.5, 2.5, 11)


def politicas():
    """Matriz semilla × tendencia con 1 = opera (compra sin activo / vende con activo), 0 = mantiene."""
    out = {}
    for ag, f in (("dqn", "01_grid_final"), ("a3c", "02_grid_final")):
        d = csv(f)
        for m in MERCADOS:
            g = d[d.mercado == m].sort_values("seed")
            out[ag, m, 0] = np.array([[1 if c == "C" else 0 for c in s] for s in g.politica_sin_activo])
            out[ag, m, 1] = np.array([[1 if c == "V" else 0 for c in s] for s in g.politica_con_activo])
    return out


def politica_a():
    pol = politicas()
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.35), sharey=True)
    for ax, m in zip(axes, MERCADOS):
        # filas: DQN sin (5), A3C sin (5), DQN con (5), A3C con (5); valor 0 mantener, 1 comprar, 2 vender
        bloques = [pol["dqn", m, 0], pol["a3c", m, 0], 2 * pol["dqn", m, 1], 2 * pol["a3c", m, 1]]
        M = np.vstack(bloques)
        ax.imshow(M, cmap=ListedColormap(["#EEF0F2", "#2F9E6E", "#D63031"]), vmin=0, vmax=2, aspect="auto",
                  extent=(-2.75, 2.75, M.shape[0] - 0.5, -0.5), interpolation="nearest")
        for s in (4.5, 14.5):
            ax.axhline(s, color="white", lw=1)
        ax.axhline(9.5, color="black", lw=1.2)
        ax.axvline(0, color="black", lw=0.5, ls=":")
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_xlabel("tendencia / σ")
        ax.grid(False)
        color_title(ax, m)
    axes[0].set_yticks([2, 7, 12, 17], ["DQN", "A3C", "DQN", "A3C"])
    axes[0].tick_params(axis="y", length=0)
    for lab, ag in zip(axes[0].get_yticklabels(), ["dqn", "a3c"] * 2):
        lab.set_color(AGENT_COLORS[ag]); lab.set_fontweight("bold")
    fig.text(0.004, 0.66, "sin activo", rotation=90, va="center", fontsize=7.5)
    fig.text(0.004, 0.30, "con activo", rotation=90, va="center", fontsize=7.5)
    handles = [Patch(color="#2F9E6E", label="comprar"), Patch(color="#D63031", label="vender"),
               Patch(color="#EEF0F2", ec="#BBBBBB", label="mantener")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07), handlelength=1.2)
    fig.tight_layout(w_pad=0.5, rect=(0.015, 0, 1, 1))
    save(fig, "cand_politica_a")


def politica_b():
    """Fracción de semillas que operan en cada tendencia: compra sin activo (arriba) y venta con activo (abajo)."""
    pol = politicas()
    fig, axes = plt.subplots(2, 4, figsize=(FULL, 2.3), sharex=True, sharey=True)
    for j, m in enumerate(MERCADOS):
        for i, (pos, accion) in enumerate(((0, "P(comprar)\nsin activo"), (1, "P(vender)\ncon activo"))):
            ax = axes[i, j]
            for ag, ls in (("dqn", "-"), ("a3c", "--")):
                f = pol[ag, m, pos].mean(axis=0)
                ax.step(X_POL, f, where="mid", color=AGENT_COLORS[ag], lw=1.6, ls=ls, label=NOMBRES[ag])
            ax.axvline(0, color="grey", lw=0.5, ls=":")
            ax.set_ylim(-0.08, 1.08)
            ax.set_yticks([0, 0.5, 1], ["0", "½", "1"])
            if j == 0:
                ax.set_ylabel(accion, fontsize=7.2)
            if i == 0:
                color_title(ax, m)
            else:
                ax.set_xlabel("tendencia / σ")
    axes[0, 0].legend(loc="center left", fontsize=7, handlelength=1.6, borderaxespad=0.1)
    fig.text(0.5, -0.03, "fracción de las 5 semillas cuya acción greedy es operar en cada nivel de tendencia",
             ha="center", fontsize=7, color="dimgray")
    fig.tight_layout(h_pad=0.3, w_pad=0.5)
    save(fig, "cand_politica_b")


def politica_c():
    """Recorte de 01_politica.png y 02_politica.png (mercados 1 y 3, filas compra/venta)."""
    top = 0.065
    e = [0, 0.272, 0.512, 0.758, 1]
    d = crop_panels(FIG / "01_politica.png", [0, 2], e, top_frac=top)
    a = crop_panels(FIG / "02_politica.png", [0, 2], e, top_frac=top)
    fila = hstack([hstack(d, gap=6), hstack(a, gap=6)], gap=40)
    w = fila.size[0]
    half = hstack(d, gap=6).size[0]
    lab = Image.new("RGB", (w, 60), "white")
    lab.paste(label_strip("DQN · Q(operar) − Q(mantener)", half, AGENT_COLORS["dqn"]), (0, 0))
    lab.paste(label_strip("A3C · probabilidad de operar", w - half - 40, AGENT_COLORS["a3c"]), (half + 40, 0))
    vstack([lab, fila], gap=2).save(OUT / "cand_politica_c.png", dpi=(200, 200))
    print("ok cand_politica_c")


# ================================================================ 5. EXTRAS (03)
def extras_a():
    par = pd.concat([csv("03_grid_pareadas"), csv("03_grid_pareadas_actores")], ignore_index=True)
    filas = [("ventana", "+20 retornos"), ("ma30_vol", "+MA30 y volatilidad"), ("doble", "Double DQN"),
             ("log", "Recompensa log"), ("ciego", "Placebo (sin tendencia)"), ("mezcla", "Mezcla 4 mercados"),
             ("a2c", "A2C (vs A3C)")]
    ix = par.set_index(["variante", "mercado"])
    margen = par.groupby("mercado").margen.first()
    ratio = np.array([[ix.loc[(v, m), "ventaja_log"] / margen[m] for m in MERCADOS] for v, _ in filas])
    fig, ax = plt.subplots(figsize=(HALF, 2.45))
    lim = 4
    ax.imshow(np.clip(ratio, -lim, lim), cmap="RdBu", norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), aspect="auto")
    for i, (v, _) in enumerate(filas):
        for j, m in enumerate(MERCADOS):
            r = ix.loc[(v, m)]
            sup = bool(r.supera_margen)
            ax.text(j, i, f"{r.ventaja_log:+.2f}  {int(r.semillas_mejoran)}/5", ha="center", va="center", fontsize=7,
                    fontweight="bold" if sup else "normal", color="white" if abs(ratio[i, j]) > 0.7 * lim else "black")
            if sup:
                ax.add_patch(Rectangle((j - .47, i - .44), .94, .88, fill=False, ec="black", lw=1.4))
    ax.set_xticks(range(4), [CORTO[m].replace(" · ", "\n") for m in MERCADOS])
    for lab, m in zip(ax.get_xticklabels(), MERCADOS):
        lab.set_color(MARKET_COLORS[m]); lab.set_fontweight("bold")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(filas)), [n for _, n in filas])
    for s in (1.5, 3.5, 5.5):
        ax.axhline(s, color="white", lw=2.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    save(fig, "cand_extras_a")


def extras_b():
    res3 = csv("03_grid_resumen_dqn").set_index(["mercado", "variante"])
    bas3 = csv("03_extras_baselines")
    bh = bas3[bas3.agente == "03_buy_hold_c0.001"].set_index("mercado").retorno_mediana
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.9), gridspec_kw={"width_ratios": [1, 1, 1.1, 1.1]})
    est = [("base", "base"), ("ventana", "+20 ret."), ("ma30_vol", "+MA30\n+vol.")]
    cols = ["#9AA0A6", AGENT_COLORS["dqn"], "#00A884"]
    # (1) estado en tendencia, (2) estado en reversión
    for ax, m in zip(axes[:2], ["tendencia_ciclo", "reversion_media"]):
        v = np.array([res3.loc[(m, k), "mediana"] for k, _ in est])
        y = 1 + v if m == "tendencia_ciclo" else 100 * v
        b = ax.bar(range(3), y, color=cols, width=0.7)
        ax.bar_label(b, labels=[formato(x) for x in v], fontsize=7, padding=1)
        ax.set_xticks(range(3), [n for _, n in est], fontsize=7)
        ax.set_ylim(0, max(y) * 1.22)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(False)
        ax.set_title(f"Estado · {CORTO[m].split(' + ')[0]}", color=MARKET_COLORS[m], fontweight="bold", fontsize=8)
    # (3) comisiones en reversión: operaciones y retorno
    ax = axes[2]
    cv = [("comision_0", "0 %"), ("base", "0,1 %"), ("comision_05", "0,5 %")]
    ops = [res3.loc[("reversion_media", k), "operaciones"] for k, _ in cv]
    ret = [res3.loc[("reversion_media", k), "mediana"] for k, _ in cv]
    b = ax.bar(range(3), ops, color=MARKET_COLORS["reversion_media"], width=0.7, alpha=0.85)
    ax.bar_label(b, labels=[f"{o:.0f} ops\n{formato(r)}" for o, r in zip(ops, ret)], fontsize=6.8, padding=1)
    ax.set_xticks(range(3), [n for _, n in cv])
    ax.set_ylim(0, max(ops) * 1.45)
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.grid(False)
    ax.set_xlabel("comisión", labelpad=1)
    ax.set_title("Comisión · 3 · Reversión", color=MARKET_COLORS["reversion_media"], fontweight="bold", fontsize=8)
    # (4) S&P 500
    ax = axes[3]
    sp = [("DQN\ntrain real", res3.loc[("sp500", "real"), "mediana"], AGENT_COLORS["dqn"]),
          ("DQN\ntransfer.", res3.loc[("sp500", "transferencia"), "mediana"], "#A29BFE"),
          ("Buy &\nhold", bh["sp500"], AGENT_COLORS["buy_hold"])]
    b = ax.bar(range(3), [100 * v for _, v, _ in sp], color=[c for *_, c in sp], width=0.7)
    ax.bar_label(b, labels=[formato(v) for _, v, _ in sp], fontsize=7, padding=1)
    ax.set_xticks(range(3), [n for n, *_ in sp], fontsize=7)
    ax.set_ylim(0, 55)
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.grid(False)
    ax.set_title("S&P 500 · test 2024–25", fontweight="bold", fontsize=8)
    fig.tight_layout(w_pad=1.2)
    save(fig, "cand_extras_b")


# ================================================================ 6. HIPERPARÁMETROS
def en_margenes(m, nuevo, base):
    if m == "tendencia_ciclo":
        return np.log((1 + nuevo) / (1 + base)) / margen_log(m)
    return (nuevo - base) / MARGEN[m]


def sens_dqn():
    norm = csv("01_grid_normalizacion")
    base = norm[norm.normalize_obs & norm.normalize_reward].groupby("mercado").retorno_mediana.median()
    out = {}
    for clave, f, p, elegido in (("α", "01_grid_lr", "lr", 0.001), ("γ", "01_grid_gamma", "gamma", 0.9),
                                 ("expl", "01_grid_epsilon", "epsilon_decay", 0.995), ("episodios", "01_grid_episodios", "n_episodes", 500)):
        pts = csv(f).groupby(["mercado", p]).retorno_mediana.median()
        for m in MERCADOS:
            d = {elegido: base[m], **pts[m].to_dict()}
            out[clave, m] = (dict(sorted(d.items())), elegido, base[m])
    return out


def sens_a3c():
    """Ajuste por etapas de 02 (misma lógica que 04_comparativa §2.1), devolviendo la etapa de cada parámetro."""
    BASE = {"lr": 0.001, "gamma": 0.9, "entropy_coef": 0.01, "t_max": 20, "mask_redundant": False, "n_episodes": 500}
    norm = csv("02_grid_normalizacion")
    las_dos = norm[norm.normalize_obs & norm.normalize_reward].groupby("mercado").retorno_mediana.median()
    actual = {m: dict(BASE) for m in MERCADOS}
    punto = {m: las_dos[m] for m in MERCADOS}
    out = {}
    for nombre, param, clave in (("lr", "lr", "α"), ("gamma", "gamma", "γ"), ("entropia", "entropy_coef", "expl"),
                                 ("tmax", "t_max", None), ("mascara", "mask_redundant", None), ("episodios", "n_episodes", "episodios")):
        pts = csv(f"02_grid_{nombre}").groupby(["mercado", param]).retorno_mediana.median()
        for m in MERCADOS:
            fila = {actual[m][param]: punto[m], **pts[m].to_dict()}
            ref = punto[m]
            mejor = max(pts[m].index, key=lambda v: pts[m][v])
            if mejora_de_verdad(m, pts[m][mejor], punto[m]):
                actual[m][param], punto[m] = mejor, pts[m][mejor]
            if clave:
                out[clave, m] = (dict(sorted(fila.items())), actual[m][param], punto[m])
    return out


PARAMS = [("α", "α (tasa de aprendizaje)"), ("γ", "γ (descuento)"), ("expl", "exploración: ε (DQN) / β (A3C)"), ("episodios", "episodios")]


def fmt_val(clave, v, ag):
    if clave == "expl":
        return (f"ε×{v:g}" if ag == "dqn" else f"β={v:g}")
    if clave == "α":
        return f"{v:.0e}".replace("e-0", "e-")
    return f"{v:g}"


def hiperparametros_a():
    """Heatmap: filas = valor probado (agrupado por parámetro), columnas = mercado, bloques DQN | A3C."""
    S = {"dqn": sens_dqn(), "a3c": sens_a3c()}
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.3))
    lim = 4
    for ax, ag in zip(axes, ("dqn", "a3c")):
        filas, M, T, elegidos, seps = [], [], [], [], []
        for clave, _ in PARAMS:
            vals = list(S[ag][clave, MERCADOS[0]][0])
            for v in vals:
                filas.append(f"{ {'α': 'α', 'γ': 'γ', 'expl': '', 'episodios': 'ep.'}[clave]} {fmt_val(clave, v, ag)}".strip())
                fila_m, fila_t, fila_e = [], [], []
                for m in MERCADOS:
                    d, eleg, ref = S[ag][clave, m]
                    x = d.get(v, np.nan)
                    fila_m.append(en_margenes(m, x, ref) if not np.isnan(x) else np.nan)
                    fila_t.append(formato(x) if not np.isnan(x) else "")
                    fila_e.append(v == eleg)
                M.append(fila_m); T.append(fila_t); elegidos.append(fila_e)
            seps.append(len(filas))
        M = np.array(M, dtype=float)
        ax.imshow(np.clip(M, -lim, lim), cmap="RdBu", norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), aspect="auto")
        for i in range(M.shape[0]):
            for j in range(4):
                ax.text(j, i, T[i][j], ha="center", va="center", fontsize=6.5,
                        color="white" if abs(M[i, j]) > 0.7 * lim else "black")
                if elegidos[i][j]:
                    ax.add_patch(Rectangle((j - .48, i - .45), .96, .9, fill=False, ec="black", lw=1.2))
        for s in seps[:-1]:
            ax.axhline(s - 0.5, color="white", lw=2.5)
        ax.set_xticks(range(4), [m.split(" · ")[0] for m in (CORTO[x] for x in MERCADOS)])
        for lab, m in zip(ax.get_xticklabels(), MERCADOS):
            lab.set_color(MARKET_COLORS[m]); lab.set_fontweight("bold")
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(filas)), filas, fontsize=7)
        ax.tick_params(length=0)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(NOMBRES[ag], color=AGENT_COLORS[ag], fontweight="bold", fontsize=9, pad=14)
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))
    fig.subplots_adjust(left=0.07, right=0.90, wspace=0.28, bottom=0.04, top=0.88)
    cax = fig.add_axes([0.925, 0.2, 0.012, 0.6])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("frente al elegido, en márgenes de ruido", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.text(0.01, 0.0, "Validación, mediana de 3 semillas. Recuadro = valor elegido (A3C: por mercado).", fontsize=6.8, color="dimgray")
    save(fig, "cand_hiperparametros_a")


def hiperparametros_b():
    """Líneas: Δ frente al elegido (en márgenes de ruido) por parámetro, mercados 1 y 3, DQN y A3C."""
    S = {"dqn": sens_dqn(), "a3c": sens_a3c()}
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.95), sharey=True)
    mk = {"tendencia_ciclo": "o", "reversion_media": "s"}
    for ax, (clave, titulo) in zip(axes, PARAMS):
        if clave == "expl":
            pos = {"dqn": {0.98: 0, 0.99: 1, 0.995: 2}, "a3c": {0.0: 0, 0.001: 1, 0.01: 2, 0.05: 3}}
        else:
            todos = sorted(set(S["dqn"][clave, "tendencia_ciclo"][0]) | set(S["a3c"][clave, "tendencia_ciclo"][0]))
            pos = {ag: {v: i for i, v in enumerate(todos)} for ag in ("dqn", "a3c")}
        for ag, ls in (("dqn", "-"), ("a3c", "--")):
            for m in mk:
                d, eleg, ref = S[ag][clave, m]
                xs = [pos[ag][v] for v in d]
                ys = [np.clip(en_margenes(m, x, ref), -6, 4) for x in d.values()]
                ax.plot(xs, ys, ls=ls, marker=mk[m], ms=3.5, color=AGENT_COLORS[ag], lw=1.1,
                        mfc=MARKET_COLORS[m], mec=MARKET_COLORS[m])
        ax.axhspan(-1, 1, color="grey", alpha=0.12, lw=0)
        ax.axhline(0, color="grey", lw=0.6)
        if clave == "expl":
            ax.set_xticks(range(4), [".98\n0", ".99\n.001", ".995\n.01", "–\n.05"], fontsize=6.5)
            ax.text(-0.02, -0.075, "ε×\nβ", transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
        else:
            ax.set_xticks(range(len(todos)), [fmt_val(clave, v, "dqn") for v in todos], fontsize=6.8)
        ax.set_title(titulo, fontsize=7.8)
        ax.set_ylim(-6.5, 4.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Δ en márgenes de ruido", fontsize=7.3)
    handles = [Line2D([], [], color=AGENT_COLORS["dqn"], label="DQN"), Line2D([], [], color=AGENT_COLORS["a3c"], ls="--", label="A3C"),
               Line2D([], [], color=MARKET_COLORS["tendencia_ciclo"], marker="o", lw=0, ms=4, label="mercado 1"),
               Line2D([], [], color=MARKET_COLORS["reversion_media"], marker="s", lw=0, ms=4, label="mercado 3"),
               Patch(color="grey", alpha=0.25, label="±1 margen (ruido)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(w_pad=0.5)
    save(fig, "cand_hiperparametros_b")


if __name__ == "__main__":
    for f in (mercados_a, mercados_b, mercados_c, rendimiento_a, rendimiento_b, rendimiento_c,
              convergencia_a, convergencia_b, convergencia_c, politica_a, politica_b, politica_c,
              extras_a, extras_b, hiperparametros_a, hiperparametros_b):
        f()
