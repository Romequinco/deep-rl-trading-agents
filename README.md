# deep-rl-trading-agents · Agentes de Deep RL para trading

Práctica en grupo del Máster MIAX (módulo de Aprendizaje por Refuerzo): **DQN y A3C** para invertir en un único
activo sobre mercados sintéticos (tendencia + ciclo, browniano geométrico, reversión a la media y regímenes
alcista/bajista), comparados con un agente aleatorio y con buy & hold.

Continúa el Taller 1, con Q-Learning y SARSA tabulares:
[rl-trading-agents](https://github.com/Romequinco/rl-trading-agents).

**Grupo:** Óscar Romero Quincoces, Fernando Dapena Tauste y Daniel García López.

## Enunciado (resumen)

- **Entorno:** `SimpleTradingEnv` del taller. El estado es `(tendencia, posición)`, las acciones son
  vender / mantener / comprar, la comisión es del 0,1 % y la recompensa es la variación del precio
  con posición abierta menos los costes.
- **Mínimos:**
  1. DQN y A3C en los 4 mercados, con una serie nueva en cada episodio y evaluación promediada sobre series de test.
  2. Comparación con el aleatorio y con buy & hold, razonando si la estrategia es la esperada.
  3. Ajuste de α, ε (con decaimiento), γ y número de episodios.
  4. Tabla comparativa de rendimiento y tiempo de convergencia.
- **Entrega:** PDF de **2 páginas como máximo** (pruebas, resultados y conclusiones), una por grupo.
  - Antes del **martes 3/11:** un representante manda un email al profesor para fijar el orden de presentación.
  - Antes de la clase del **jueves 5/11:** primera versión por email.
  - **5/11, en clase:** presentación y discusión de 15 minutos.
  - **5/11, 23:59:** entrega final.
- **Evaluación:** los mínimos dan el aprobado (50 %). El otro 50 % se reparte entre ir más allá de los
  mínimos (10 %), la calidad de presentación (20 %), la justificación del diseño (10 %) y el análisis
  crítico (10 %).

## Estructura

```
├── notebooks/                    # todo el trabajo
│   ├── 00_entorno_y_baselines.ipynb
│   ├── 01_dqn.ipynb
│   ├── 02_a3c.ipynb
│   ├── 03_extras.ipynb
│   ├── 04_comparativa.ipynb      # solo lee results/: tabla final y figuras
│   └── common/                   # código común: única fuente de verdad
│       ├── env.py                # SimpleTradingEnv (copia literal del enunciado)
│       ├── markets.py            # 4 generadores, extras del Taller 1 y S&P 500
│       ├── agents.py             # referencias: aleatorio, buy & hold, no operar y reglas simples
│       ├── protocol.py           # series, evaluate(), métricas, save_results(), run_grid()
│       ├── dqn.py                # DQN y sus opciones de mejora
│       ├── a3c.py                # A3C (interfaz por implementar)
│       ├── tuning.py             # ajuste por etapas y margen de ruido
│       ├── style.py              # paleta y figuras comunes
│       └── test_common.py
├── docs/
│   ├── DECISIONES.md             # registro de decisiones D-X.n
│   ├── teoria/                   # síntesis del material de clase
│   └── raw/                      # material de clase (local, no versionado)
└── results/
    ├── NN_*.csv                  # resultados de cada notebook (esquema común)
    ├── taller1_tabular.csv       # referencia: baselines, Q-Learning y SARSA del Taller 1
    ├── data/                     # S&P 500 con procedencia y huella
    ├── figures/  models/         # figuras y pesos finales
    └── informe.md                # entregable → build_informe.py → PDF de 2 páginas
```

## Puesta en marcha

En Windows (PowerShell), desde la carpeta del proyecto:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name deep-rl-trading --display-name "Python (Deep RL Trading)"
.\.venv\Scripts\python.exe -m jupyterlab notebooks/
```

En otros sistemas:

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

Basta con la CPU: las redes son pequeñas. El material de clase no se versiona; quien lo necesite lo copia en
`docs/raw/`.

Pruebas: `.\.venv\Scripts\python.exe -m unittest discover -s notebooks/common -t notebooks`.

## Cómo trabajar en paralelo

- **Líneas independientes:** 01 (DQN), 02 (A3C) y 03 (extras) no dependen entre sí. Cada una guarda sus resultados
  con su prefijo (`results/NN_<nombre>.csv`), y el 04 los concatena.
- **El código va en `common/`** y el notebook lo usa para experimentar y analizar. Es además obligatorio para A3C y
  `run_grid`: en Windows, los procesos no pueden importar funciones definidas en una celda.
- **Notebooks:** los edita una sola persona a la vez y se suben ejecutados. Si hay conflicto, git conserva tu versión
  sin romper el JSON (`.gitattributes`). Para probar libremente, usa `notebooks/scratch_*.ipynb`, que no se versiona.
- **Decisiones:** cada una que afecte a los resultados se añade a [`docs/DECISIONES.md`](docs/DECISIONES.md); dos
  personas pueden añadir a la vez sin conflicto.

**Contrato común:**

- `common/env.py` y los 4 generadores son copia literal del enunciado y no se modifican (lo comprueba un test).
- Todo agente implementa `act(obs, greedy=False)`, y con eso `evaluate()` lo mide igual que a los baselines.
- Las series no se eligen a mano: train con `train_series_seed(seed, episodio)`, y validación y test con
  `evaluate(agente, mercado, split=...)`. Los hiperparámetros se eligen en validación y el test se usa una sola vez. Lo que se elija después
  de ver el test se mide en `split="nuevas"` (D-2.4).
- `save_results()` escribe las columnas `agente, mercado, retorno_mediana, retorno_medio, std, sharpe, max_drawdown,
  n_trades, episodios_convergencia, tiempo_s`. La métrica principal es la mediana del retorno.
- `run_grid()` entrena en paralelo con caché. La caché depende de la configuración y de `version`: si cambia el
  algoritmo de verdad, se sube `version`.
- El protocolo es el del Taller 1 y reproduce sus baselines, así que DQN y A3C se comparan directamente con los
  tabulares de `results/taller1_tabular.csv`.

## Resultado de DQN (`01_dqn`)

Mediana del retorno en las 200 series de test, con 5 semillas. Hiperparámetros: α = 0,001, γ = 0,9, decaimiento de ε 0,995 y 500 episodios.

| Mercado | DQN | Rango entre semillas | Buy & hold | Q-Learning | Mejor referencia |
|---|---|---|---|---|---|
| 1 · Tendencia + ciclo | x299,8 | x228 a x558 | x33,6 | x502,3 | Regla de tendencia: x630,4 |
| 2 · Browniano | −3,9 % | −6,4 % a 0 % | −3,5 % | 0 % | No operar: 0 % |
| 3 · Reversión a la media | +20,5 % | +4,7 % a +26,3 % | −0,3 % | +27,0 % | Regla de reversión: +29,2 % |
| 4 · Regímenes | 0 % | −7,9 % a 0 % | −12,5 % | 0 % | No operar: 0 % |

Bate a buy & hold donde hay señal, pero no a las reglas ni a los tabulares. Ninguna de las cuatro mejoras probadas supera el margen de
ruido (D-2.5); la más prometedora es quedarse con el mejor punto en validación.

## Documentación

- [`docs/DECISIONES.md`](docs/DECISIONES.md): decisiones del proyecto, cerradas y abiertas.
- `docs/teoria/`: síntesis del material de clase, con la fuente citada en cada concepto:
  - [`notebook_partida.md`](docs/teoria/notebook_partida.md): enunciado, rúbrica, sutilezas del entorno, comentarios
    del profesor, trampas y checklist. **Empezad por aquí.**
  - [`00_fundamentos_mdp.md`](docs/teoria/00_fundamentos_mdp.md): RL, bandidos, MDP, retorno y γ, V y Q, Bellman.
  - [`01_dp_y_montecarlo.md`](docs/teoria/01_dp_y_montecarlo.md): programación dinámica y Monte Carlo.
  - [`02_td_sarsa_qlearning.md`](docs/teoria/02_td_sarsa_qlearning.md): TD(0), SARSA y Q-Learning.
  - [`03_dqn.md`](docs/teoria/03_dqn.md): DQN, memoria de experiencias, red objetivo y consejos prácticos.
    **Núcleo del taller.**
  - [`04_doble_dqn_y_duelo.md`](docs/teoria/04_doble_dqn_y_duelo.md): Doble DQN y DQN en duelo.
  - [`05_policy_gradient.md`](docs/teoria/05_policy_gradient.md): gradiente de la política y REINFORCE.
  - [`06_actor_critico_y_a3c.md`](docs/teoria/06_actor_critico_y_a3c.md): actor-crítico, A2C y A3C. **Núcleo del taller.**
  - [`07_panorama_drl.md`](docs/teoria/07_panorama_drl.md): taxonomía, hitos, aplicaciones, entornos y librerías.
  - [`08_trading_en_clase.md`](docs/teoria/08_trading_en_clase.md): el DQN de trading de clase y qué ideas trasladar.
  - [`COBERTURA.md`](docs/teoria/COBERTURA.md): a qué documento va cada página, celda y tramo del material de clase.

## Estado

- [x] Estructura del repositorio y código común
- [x] 00 Entorno y baselines
- [x] Documentación de teoría (`docs/teoria/`)
- [x] 01 DQN
- [ ] 02 A3C
- [ ] 03 Extras
- [ ] 04 Comparativa
- [ ] Informe PDF (`results/informe.pdf`, se regenera con `python results/build_informe.py`)
