"""Lectura de cachés, tablas y figuras de 03; no entrena ningún agente."""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .extras import (CORE_MARKETS, ExtraConfig, ExtraTradingEnv, create_agent,
                     make_plan, real_splits)
from .protocol import RESULTS_DIR, config_id, run_episode
from .style import MARKET_LABELS, formato, savefig


def load_grid(name):
    df = pd.read_csv(RESULTS_DIR / f"{name}.csv")
    folder = RESULTS_DIR / "cache" / name
    curves = {cid: pickle.loads((folder / f"{cid}.pkl").read_bytes())["curva"] for cid in df.id}
    return df, curves


def performance_table(df):
    table = df.pivot_table(index="variante", columns="mercado", values="retorno_mediana", aggfunc="median", sort=False)
    order = (*CORE_MARKETS, "tendencia_persistente", "saltos", "sp500")
    table = table.reindex(columns=[market for market in order if market in table.columns])
    return table.rename(columns=MARKET_LABELS).map(formato)


def export_series(df, curves, name):
    """Datos por serie versionables: permiten auditar las parejas sin reentrenar."""
    frames = []
    for row in df.itertuples():
        returns = curves[row.id]["eval_retornos"]
        frames.append(pd.DataFrame({"id": row.id, "variante": row.variante, "mercado": row.mercado,
                                    "seed": row.seed, "serie": np.arange(len(returns)), "retorno": returns}))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    return out


def effect_figure(comparisons, variants, title, name):
    table = comparisons[comparisons.variante.isin(variants)].pivot(index="variante", columns="mercado", values="ventaja_log")
    table = table.reindex(index=variants, columns=CORE_MARKETS)
    fig, ax = plt.subplots(figsize=(11, max(2.8, len(variants) * .6 + 1.5)))
    bound = max(.1, np.abs(table.to_numpy()).max())
    im = ax.imshow(table, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(4), [MARKET_LABELS[m] for m in CORE_MARKETS], rotation=12, ha="right")
    ax.set_yticks(range(len(table)), table.index)
    for y, variant in enumerate(table.index):
        for x, market in enumerate(table.columns):
            value = table.loc[variant, market]
            item = comparisons[(comparisons.variante == variant) & (comparisons.mercado == market)].iloc[0]
            mark = " *" if item.supera_margen else ""
            ax.text(x, y, f"{value:+.3f}{mark}\n{item.semillas_mejoran}/{item.n_semillas} semillas",
                    ha="center", va="center", fontsize=9)
    ax.set_title(title + "\nVentaja emparejada en log(capital); * supera el margen predefinido")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Ventaja frente a la referencia")
    fig.tight_layout()
    savefig(fig, name)
    return fig


def commission_figure(df):
    data = df[df.variante.isin(("comision_0", "base", "comision_05"))]
    x = [0, .1, .5]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, market in zip(axes.flat, CORE_MARKETS):
        d = data[data.mercado == market].groupby("variante")
        order = ["comision_0", "base", "comision_05"]
        y = np.log1p(d.retorno_mediana.median().reindex(order))
        low = np.log1p(d.retorno_mediana.min().reindex(order))
        high = np.log1p(d.retorno_mediana.max().reindex(order))
        ax.plot(x, y, marker="o", color="#6C5CE7")
        ax.fill_between(x, low, high, color="#6C5CE7", alpha=.15, label="Rango entre 5 semillas")
        ax.axhline(0, color="gray", linewidth=.8)
        ax.set(title=MARKET_LABELS[market], xlabel="Comisión por operación (%)", ylabel="log(capital final / inicial)")
        ax.legend(fontsize=8)
    fig.suptitle("Sensibilidad a la comisión: cada agente se entrena con su coste")
    fig.tight_layout()
    savefig(fig, "03_comisiones")
    return fig


def real_figure(df, curves):
    plan = {config_id(job, 1): job for job in make_plan()}
    real = df[df.mercado == "sp500"]
    prices = real_splits()["nuevas"]
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, color in (("transferencia", "#6C5CE7"), ("real", "#00A884")):
        balances = []
        for row in real[real.variante == label].itertuples():
            h = curves[row.id]
            cfg = ExtraConfig(**plan[row.id]["config"])
            agent = create_agent(cfg, h["feature_scales"], h["reward_scale"], row.seed)
            agent.q.load_state_dict(h["model_state"])
            balances.append(run_episode(ExtraTradingEnv(prices), agent)[1])
        values = np.asarray(balances) / 1000
        ax.plot(np.median(values, axis=0), color=color, label=label)
        ax.fill_between(np.arange(values.shape[1]), values.min(axis=0), values.max(axis=0), color=color, alpha=.12)
    ax.plot(np.r_[1., .999 * prices[6:] / prices[5]], color="#F29D38", linestyle="--", label="Buy & hold")
    ax.axhline(1, color="gray", linewidth=.8, label="No operar")
    ax.set(title="S&P 500 · 2024–2025 · mediana y rango entre semillas",
           xlabel="Sesiones desde el inicio de evaluación", ylabel="Capital / capital inicial")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "03_sp500")
    return fig


def seed_ranges(df):
    return df.groupby(["mercado", "variante"]).agg(
        mediana=("retorno_mediana", "median"), peor=("retorno_mediana", "min"),
        mejor=("retorno_mediana", "max"), operaciones=("n_trades", "median"),
        tiempo_s=("tiempo_s", "median")).reset_index()
