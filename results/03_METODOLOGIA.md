# 03 · Experimentos adicionales de un factor

Plan fijado el 4 de noviembre de 2026, antes de observar los resultados de estos
experimentos. Implementación: `notebooks/common/extras.py`. Análisis y resultados:
`notebooks/03_extras.ipynb`.

## Protocolo

- Cinco semillas (0–4), 500 episodios de 500 precios, una serie distinta por episodio.
- Referencia DQN idéntica a 01: α=0,001, γ=0,9, ε inicial 1, decaimiento 0,995,
  red 2→64→3, memoria 20.000, lote 128, entrenamiento cada 4 pasos y red objetivo
  cada 500. Reentrenada con el mismo software que las variantes; no se compara una
  implementación nueva contra un modelo antiguo entrenado con otra versión de PyTorch.
- Validación: semillas 50.000–50.019 cada 25 episodios, sin selección de checkpoint
  ni búsqueda de hiperparámetros adicionales. Normalización ajustada solo con train.
- Evaluación sintética: **400 series nuevas desde la semilla 600.000**, distintas
  de las de test oficial (100.000) y de las mejoras de 01–02 (400.000). Cada pareja
  usa exactamente las mismas series y semillas de entrenamiento.
- Principal: mediana entre semillas de la mediana del retorno por serie. También
  media, dispersión, drawdown, Sharpe por paso y operaciones del protocolo común.
- Ventaja emparejada: mediana entre semillas de la mediana por serie de
  `log(1 + retorno_variante) - log(1 + retorno_referencia)`.
- Márgenes heredados, descriptivos y **no pruebas de significación**: log(1,25)
  en tendencia, 0,02 en reversión, 0,10 en browniano y regímenes. Se informa también
  de cuántas semillas mejoran. No se proclama un ganador general por una celda.
- Los resultados son posteriores al test ya observado y no sustituyen a
  `01_dqn.csv` ni a `02_a3c.csv`. No se vuelven a ajustar usando estas series nuevas.

## Factores

| Variante | Único cambio respecto a la referencia |
|---|---|
| doble | Online selecciona la acción siguiente y target la valora; sin máscara ni acciones canónicas |
| duelo | Cabezas V y A; Q=V+A−media(A). Mantiene tronco de 64; añade 65 parámetros como parte de la arquitectura |
| sin_objetivo | Se desactiva la red objetivo |
| clip | Se recorta a [−1,1] la recompensa ya normalizada |
| ventana | Se añaden los últimos 20 log-retornos al estado original |
| ma30_vol | Se añaden distancia a MA30 y volatilidad de 20 retornos |
| log | Recompensa log(balance siguiente / balance actual) |
| ciego | Se anula la tendencia en train y evaluación: placebo sin información de precios |
| mezcla | El entrenamiento alterna los 4 generadores con el mismo total de episodios |
| comision_0 / comision_05 | Coste por transacción de 0 % / 0,5 % en entrenamiento y evaluación |
| mercados extra | Mismo DQN e hiperparámetros, con tendencia persistente y saltos |
| real | Frente a transferencia browniana, solo cambia la fuente de entrenamiento al histórico real |
| A2C | Frente al A3C de 02, cambia la actualización asíncrona por un gradiente medio síncrono |

Al ampliar el estado también aumenta el número de pesos de entrada; es una
consecuencia del factor «representación». El placebo conserva las dimensiones.
Las ventanas se rellenan con ceros al inicio cuando falta historia, y todos los
indicadores usan exclusivamente precios disponibles hasta ese instante. Se
conservan los 494 pasos negociables. La comisión se sustituye tanto en la
recompensa como en el efectivo/acciones. El entorno original no se modifica.

En la mezcla se mantiene el presupuesto total, no 500 episodios por generador:
se compara generalidad frente a especialización. La normalización se ajusta en
la distribución de entrenamiento correspondiente. La transferencia entre dos
mercados distintos no permite calcular una ventaja emparejada causal; en los
mercados extra se compara con baselines sobre las mismas series.

## A2C y A3C

Mismos 3 entornos, red y optimizador Adam, inicialización, semillas de las series,
t_max=20, γ y coeficiente de entropía de 02. En tendencia se conserva α=0,003;
en reversión β=0. Los 500 episodios solicitados se redondean a **504 efectivos**
en ambos, como ya hace A3C al trabajar con bloques de 24. A2C agrupa los tres
rollouts y aplica una actualización con el gradiente medio; A3C aplica gradientes
locales asíncronos. Las actualizaciones por parámetro difieren como parte de ese
factor. A2C es reproducible con semillas fijas; A3C no lo es bit a bit.

Los tiempos DQN de un pool de procesos incluyen contención y no permiten una
comparación de velocidad entre arquitecturas. A2C/A3C se ejecutan uno a uno en
el mismo equipo. Sus tiempos excluyen evaluación; los de A3C también excluyen
arranque. No se mezclan con tiempos de otros equipos.

A2C agrupa los tres entornos en un proceso; A3C usa tres procesos. Se mantiene
el mismo presupuesto de interacciones y se estudia la variante síncrona, pero
los segundos comparan estas implementaciones, no el coste intrínseco de los
algoritmos con idéntico uso de CPU. No se atribuye una diferencia de tiempo solo
a la sincronización.

## Histórico S&P 500

Se usa la copia local cuya huella verifica `load_sp500()`, sin descargar ni cambiar
datos: entrenamiento 2017–2021 (1.259 sesiones), validación 2022–2023 (501),
evaluación 2024–2025 (502). Entrenamiento con ventanas aleatorias de 500 precios
íntegramente dentro de train. La fuente real se compara con entrenamiento
browniano, manteniendo DQN y los hiperparámetros. Ambos se miden sobre el mismo
periodo real y contra buy & hold, aleatorio y no operar.

La normalización real se estima solo con ventanas de train. No se actualizan
pesos en validación ni evaluación. Las cinco semillas son cinco entrenamientos,
**no cinco históricos independientes**: existe una sola trayectoria histórica.
El periodo ya estaba disponible en el Taller 1: el análisis es retrospectivo y
no equivale a una prueba prospectiva. No se extrapolan sus resultados a inversión
real. La simulación conserva las limitaciones del repo (sin dividendos,
deslizamiento, spread ni intereses).

## Reproducción y ficheros

Desde `notebooks/`, con el Python de `.venv`:

```powershell
..\.venv\Scripts\python.exe -m common.extras --jobs 6
```

O ejecutar el notebook 03. Cada configuración se cachea atómicamente y se puede
reanudar. `03_plan.json` registra los 260 entrenamientos DQN; hay otros 40 de
A2C/A3C. `03_grid_*.csv` contiene detalle por semilla o por serie; no entra en la
comparativa oficial. `03_extras_*.csv` usa el esquema común y nombres de agente
con prefijo `03_` para distinguir condiciones. La caché local conserva los pesos
DQN y las curvas y retornos de ambos algoritmos. Las figuras están en
`results/figures/03_*.png`. Esta ejecución utilizó hasta 10 procesos DQN en un
equipo con 10 núcleos y 32 GB; el notebook mantiene 6 como valor conservador.

Fuentes de los algoritmos:
[Double DQN](https://arxiv.org/abs/1509.06461),
[Dueling](https://proceedings.mlr.press/v48/wangf16.html),
[A3C](https://proceedings.mlr.press/v48/mniha16.html),
[A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html).
