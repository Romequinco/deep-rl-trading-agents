"""Ajuste de hiperparámetros por etapas, igual para DQN y A3C (D-2.3).

Un parámetro cada vez y por mercado, en validación. Un valor nuevo solo sustituye al actual si mejora más que el ruido:
unos 2 errores estándar de la mediana con 100 series (notebook 00, §5).
"""

import numpy as np
import pandas as pd
from IPython.display import display

from .protocol import run_grid
from .style import MARKET_LABELS, formato

MARGEN = {"browniano": 0.10, "reversion_media": 0.02, "regimenes": 0.10}
MARGEN_MULTIPLO = {"tendencia_ciclo": 1.25}


def mejora_de_verdad(mercado, nuevo, actual):
    """¿La mediana `nuevo` supera a `actual` por más que el ruido de `mercado`?"""
    if mercado in MARGEN_MULTIPLO:
        return (1 + nuevo) > MARGEN_MULTIPLO[mercado] * (1 + actual)
    return nuevo - actual > MARGEN[mercado]


def margen_log(mercado):
    """El mismo margen, en diferencia de log(1 + retorno), para comparar serie a serie."""
    return float(np.log(MARGEN_MULTIPLO[mercado])) if mercado in MARGEN_MULTIPLO else MARGEN[mercado]


def ventaja_emparejada(retornos, referencia):
    """Ventaja en log(1 + retorno) frente a `referencia`, serie a serie y semilla a semilla. Ambos: una lista por semilla,
    con el retorno de cada serie. Devuelve la mediana entre semillas de la mediana por serie, y cuántas semillas ganan."""
    por_semilla = [float(np.median(np.log1p(a) - np.log1p(b))) for a, b in zip(retornos, referencia)]
    return float(np.median(por_semilla)), sum(v > 0 for v in por_semilla)


class Ajuste:
    """Estado del ajuste: `actual[mercado]` es la configuración vigente y `puntos[mercado]`, su mediana en validación."""

    def __init__(self, base, mercados, semillas, fn, prefijo, n_jobs):
        self.mercados, self.semillas, self.fn, self.prefijo, self.n_jobs = list(mercados), list(semillas), fn, prefijo, n_jobs
        self.actual = {m: dict(base) for m in self.mercados}
        self.puntos = {}

    def etapa(self, nombre, parametro, valores):
        """Prueba `valores` de `parametro` frente al valor actual de cada mercado y actualiza `actual`."""
        configs = [{**self.actual[m], "mercado": m, "seed": s, parametro: v} for m in self.mercados for v in valores
                   if v != self.actual[m][parametro] for s in self.semillas]
        df, _ = run_grid(configs, self.fn, f"{self.prefijo}_grid_{nombre}", n_jobs=self.n_jobs)
        puntos = df.groupby(["mercado", parametro]).retorno_mediana.median()
        filas = {}
        for m in self.mercados:
            filas[MARKET_LABELS[m]] = {self.actual[m][parametro]: self.puntos[m], **puntos[m].to_dict()}
            mejor = max(puntos[m].index, key=lambda v: puntos[m][v])
            if mejora_de_verdad(m, puntos[m][mejor], self.puntos[m]):
                self.actual[m][parametro] = mejor.item() if hasattr(mejor, "item") else mejor
                self.puntos[m] = puntos[m][mejor]
        tabla = pd.DataFrame(filas).T.reindex(columns=sorted(valores))  # con [] una lista de booleanos sería una máscara
        elegidos = {MARKET_LABELS[m]: self.actual[m][parametro] for m in self.mercados}

        def marcar(fila):
            return ["background-color: #E3F4EA; font-weight: bold" if c == elegidos[fila.name] else "" for c in fila.index]

        display(tabla.style.apply(marcar, axis=1).format(formato).set_caption(f"{parametro}: mediana en validación. En verde, el elegido."))
        return tabla
