"""Generadores de series de precios.

Las cuatro funciones son copia literal de `docs/raw/taller/taller_trader2.ipynb` (mínimo obligatorio).
Los generadores extra están al final y se registran con `register_extra_markets()`.
`make_prices()` fija la semilla para que train y test sean reproducibles.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# 1. Tendencia suave + componente cíclica
def generar_precios(n_steps=1000):
    # Combinar una tendencia suave (drift) + componente cíclico + ruido
    t = np.linspace(0, 50, n_steps)
    drift = 0.0005 * t
    ciclo = np.sin(t) * 0.05
    ruido = np.random.normal(0, 0.1, n_steps)

    # Precio base acumulado
    retornos = drift + ciclo + ruido
    precios = 100 * np.cumprod(1 + retornos)
    return precios


# 2. Movimiento browniano
def generar_precios_browniano(n_steps=1000, start_price=100.0, mu=0.0001, sigma=0.02):
    """
    Genera precios usando Movimiento Browniano Geométrico.
    - mu: Tendencia o deriva esperada (drift).
    - sigma: Volatilidad (desviación estándar de los retornos).
    """
    dt = 1.0
    # Retornos logarítmicos distribuidos normalmente
    retornos = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_steps)
    precios = start_price * np.cumprod(np.exp(retornos))
    return precios


# 3. Reversión a la media
def generar_precios_reversion_media(n_steps=1000, start_price=100.0, mean_price=100.0, kappa=0.05, volatility=0.02):
    """
    Genera precios que tienden a revertir hacia un valor medio.
    - mean_price: El precio objetivo o de equilibrio.
    - kappa: Velocidad de reversión (cuánto más alto, más rápido vuelve al valor medio).
    - volatility: Magnitud del ruido aleatorio añadido en cada paso.
    """
    precios = np.zeros(n_steps)
    precios[0] = start_price

    for t in range(1, n_steps):
        # Fuerza de atracción hacia la media + ruido aleatorio
        delta = kappa * (mean_price - precios[t-1]) + np.random.normal(0, volatility * mean_price)
        precios[t] = precios[t-1] + delta

        # Evitar precios negativos o cero por seguridad
        precios[t] = max(precios[t], 0.1)

    return precios


# 4. Alternancia entre regímenes alcistas y bajistas
def generar_precios_regimenes(n_steps=1000, start_price=100.0):
    """
    Genera precios alternando entre regímenes alcistas y bajistas de forma estocástica.
    """
    precios = np.zeros(n_steps)
    precios[0] = start_price

    # Estado inicial: 0 = Alcista, 1 = Bajista
    estado = 0

    retornos = []
    for _ in range(n_steps):
        # Probabilidad de cambiar de régimen en cada paso
        if np.random.rand() < 0.02:
            estado = 1 - estado

        if estado == 0:  # Régimen alcista
            ret = np.random.normal(0.001, 0.01)
        else:            # Régimen bajista (caídas con más ruido)
            ret = np.random.normal(-0.0015, 0.025)

        retornos.append(ret)

    retornos = np.array(retornos)
    precios = start_price * np.cumprod(1 + retornos)

    # Asegurar que no haya valores negativos
    precios = np.maximum(precios, 0.1)
    return precios


# ---------------------------------------------------------------------------
# Generadores extra (más allá de los mínimos): añadir aquí y registrar abajo
# ---------------------------------------------------------------------------


MARKETS = {
    "tendencia_ciclo": generar_precios,
    "browniano": generar_precios_browniano,
    "reversion_media": generar_precios_reversion_media,
    "regimenes": generar_precios_regimenes,
}


def make_prices(market, n_steps, seed):
    """Serie de precios reproducible del mercado `market` con la semilla `seed`."""
    np.random.seed(seed)
    return MARKETS[market](n_steps=n_steps)


# Extras del Taller 1 para el notebook 03. Los cuatro generadores del taller permanecen intactos.
def generar_precios_tendencia_persistente(n_steps=1000, start_price=100.0):
    """Tendencias largas observables con ruido; cambio de signo con p=0.01."""
    returns = np.empty(n_steps)
    direction = 1
    for t in range(n_steps):
        if np.random.random() < 0.01:
            direction *= -1
        returns[t] = direction * 0.004 + np.random.normal(0, 0.008)
    return start_price * np.exp(np.cumsum(returns))


def generar_precios_saltos(n_steps=1000, start_price=100.0):
    """Difusión con saltos independientes: prueba de estrés sin señal predictiva."""
    log_returns = np.random.normal(0.0002 - 0.5 * 0.012**2, 0.012, n_steps)
    jumps = np.random.random(n_steps) < 0.015
    log_returns += jumps * np.random.normal(-0.04, 0.06, n_steps)
    return start_price * np.exp(np.cumsum(log_returns))


EXTRA_MARKETS = {
    "tendencia_persistente": generar_precios_tendencia_persistente,
    "saltos": generar_precios_saltos,
}
# Registro explícito para que importar este módulo no amplíe los barridos de 00-02.
# El notebook 03 y sus procesos llaman a register_extra_markets().
def register_extra_markets():
    MARKETS.update(EXTRA_MARKETS)


# Serie real para el notebook 03: procedencia en results/data/sp500_2017_2025.metadata.json
SP500_FILE = Path(__file__).resolve().parents[2] / "results" / "data" / "sp500_2017_2025.csv"


def load_sp500():
    """Cierres diarios del S&P 500 (2017-2025), tras comprobar la huella SHA-256 del contenido.

    La huella se calcula con saltos de línea LF para que no dependa del sistema operativo ni de git.
    """
    meta = json.loads(SP500_FILE.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if hashlib.sha256(SP500_FILE.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != meta["sha256"]:
        raise ValueError("El CSV real no coincide con su huella de procedencia")
    return pd.read_csv(SP500_FILE, parse_dates=["date"])
