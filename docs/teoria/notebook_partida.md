# Empezad por aquí: enunciado, entorno y lo que dijo el profesor

> **Fuentes:**
> - `taller_trader2` (`docs/raw/taller/taller_trader2.ipynb`): enunciado del Taller 2, 25 celdas, con sus figuras de salida. Es la fuente que manda.
> - `taller_trader` (`docs/raw/MIAX-14/taller_trader.ipynb`): enunciado del Taller 1, para la comparación.
> - Transcripciones automáticas (`docs/raw/transcripciones/`), muy ruidosas. **T23** (23/10) y **T24** (24/10) son del profesor del **Taller 1**. **T30** (30/10) es del profesor del **Taller 2**. Tramos principales: T23 ~04:22-04:45, T24 ~00:00-00:06 y ~02:58-03:01, T30 ~00:00-00:02 y ~04:39-04:46, más alguna cita suelta que se indica. **No existe transcripción de la clase del 31/10**, en la que el profesor anunció que explicaría el taller.
> - Repositorio: `README.md`, `docs/DECISIONES.md`, `notebooks/common/` y `results/`.
> - Taller 1: `docs/raw/rl-trading-agents-main/docs/teoria/notebook_partida.md` (punto de partida de este documento), `docs/raw/rl-trading-agents-main/results/informe.md` (citado como **T1-informe**) y `docs/raw/rl-trading-agents-main/docs/DECISIONES.md`.
>
> **Convenciones:** `(T30 ~04:39)` = hora aproximada en la transcripción; todo lo que viene de ellas es paráfrasis. `(taller_trader2, celda 22)` = celda del enunciado, contando desde 0. `D-x.y` = decisión de `docs/DECISIONES.md` de este repositorio; `T1:D-x.y` = decisión del Taller 1, **con otra numeración**. **Nota:** = contenido propio, no del enunciado ni de los profesores. Escrito el 31/10/2026.

---

## 1. Qué pide exactamente la entrega

**Idea.** La práctica es la del Taller 1 con DQN y A3C en lugar de Q-Learning y Sarsa. Se entrega un PDF de dos páginas como máximo, se presenta en 15 minutos y lo que más se valora, aparte de los mínimos, es explicar bien lo obtenido.

### 1.1 Requisitos

| Requisito | Fuente | Dónde se hace |
|---|---|---|
| Implementar un agente de RL que aprenda a invertir en un entorno simulado con **un único activo** | celda 1 | `common/dqn.py`, `common/a3c.py` |
| **Ej. 1:** implementar "los algoritmos vistos en clase", al menos **DQN y A3C**, con al menos los **4 generadores** | celda 22 | **01** (DQN), **02** (A3C) |
| **Ej. 1:** muchos episodios, **cada uno con una serie distinta** de la misma función | celda 22 | `train_series_seed` · D-0.1 |
| **Ej. 1:** evaluar con **series nuevas** de la misma función, **promediando** sobre varias | celda 22 | `evaluate()` · D-0.1 |
| **Ej. 2:** analizar la solución de cada mercado, compararla con el **aleatorio** y con **buy & hold**, y **razonar si la estrategia es la esperada** | celda 22 | **00** (baselines), **01**, **02**, informe |
| **Ej. 3:** ajustar los hiperparámetros, "en particular" **α**, **ε** (probablemente con decaimiento), **γ** y **número de episodios** | celda 22 | **01**, **02** (`run_grid`) · D-2.1 para ε en A3C |
| **Ej. 4:** **tabla** con algoritmos y baselines por mercado, con al menos **rendimiento** y **tiempo de convergencia** | celda 22 | **04** · D-0.3 |
| **Más allá de los mínimos** (cuenta en la rúbrica): otras funciones, serie real, otros algoritmos... | celdas 5, 22, 24 | **03** (§8) |
| **PDF de 2 páginas como máximo:** pruebas realizadas, resultados y conclusiones | celda 23 | `results/informe.md` → `results/build_informe.py` · D-0.5 |
| **Una entrega por grupo** de TFM | celda 23 | — |

**Nota:** los notebooks 00 a 04 son `notebooks/00_entorno_y_baselines`, `01_dqn`, `02_a3c`, `03_extras` y `04_comparativa`. El enunciado no pide entregar código; solo el PDF.

### 1.2 Fechas

| Cuándo | Qué | Fuente |
|---|---|---|
| **Antes del martes 3/11** | Un representante del grupo escribe al profesor. Ese mismo día el profesor comunica a los representantes el orden de presentación | celda 23 |
| **Antes de la clase del jueves 5/11** | Primera versión del documento, por email | celda 23 |
| **Jueves 5/11, en clase** | Presentación y discusión con el profesor, **15 minutos como máximo** por grupo. Los comentarios sirven para mejorar la versión final | celda 23 |
| **5/11, 23:59** | Entrega final | celda 23 |

La dirección de correo está en la celda 23 del enunciado (no es la del Taller 1).

**Nota:** el enunciado no dice por dónde se sube la versión final. En el Taller 1 el profesor anunció una tarea en la plataforma del curso (T24 ~00:04); para el Taller 2 no consta, así que conviene preguntarlo en el primer email.

### 1.3 Rúbrica

| Bloque | Peso |
|---|---|
| Ejercicios y pruebas propuestos (garantizan el aprobado) | 50 % |
| Pruebas que vayan **más allá** de los mínimos | 10 % |
| **Calidad de presentación** de los resultados | 20 % |
| **Justificación** de las decisiones de diseño | 10 % |
| **Análisis crítico** de las soluciones | 10 % |

**Nota (consecuencias prácticas):**
- La presentación es el bloque mayor fuera de los mínimos. En dos páginas, cada figura y la tabla tienen que entenderse solas.
- Justificación y análisis crítico (20 % entre los dos) se apoyan en `docs/DECISIONES.md`: cada decisión que salga en el informe debería tener su `D-x.y` con evidencia.
- El enunciado solo fija "rendimiento" y "tiempo de convergencia". Las demás métricas (Sharpe, drawdown, operaciones) son decisión nuestra (D-0.3).

*Fuente: taller_trader2, celdas 1, 5, 22, 23 y 24; T24 ~00:04.*

---

## 2. Qué cambió frente al Taller 1

**Idea.** Comparando los dos notebooks línea a línea solo cambian los algoritmos, las fechas y el destinatario. El código del entorno, los generadores, los baselines, los ejercicios 2 a 4 y la rúbrica son idénticos, y las figuras de salida también (mismas imágenes byte a byte).

| Punto | Taller 1 (`taller_trader`) | Taller 2 (`taller_trader2`) | Celda |
|---|---|---|---|
| Algoritmos mínimos | Q-Learning y Sarsa | **DQN y A3C** | 22 |
| Fecha tope | 29/10/2026, 23:59 | **5/11/2026, 23:59** | 23 |
| Primera versión | Antes de la clase del jueves 29/10 | **Antes de la clase del jueves 5/11** | 23 |
| Email del representante | Antes del martes 27/10 | **Antes del martes 3/11** | 23 |
| Profesor y dirección de correo | Los del Taller 1 | Los del Taller 2 | 23 |

**Nota: lo que no está escrito pero se deduce del cambio de algoritmo.**

| Tema | Taller 1 | Taller 2 |
|---|---|---|
| Estado | Había que discretizar (10 bins por cuantiles, 20 celdas; T1:D-1.1) | **La red recibe la tendencia continua:** no hay bins, pero sí un problema de escala (§4.3, D-1.1) |
| Política | Tabla Q, ε-greedy | DQN: red Q, ε-greedy. A3C: política estocástica, **sin ε** |
| Ej. 3 | α, ε, γ, episodios | El texto no se adaptó: en A3C el papel de ε lo hace el coeficiente de entropía y hay que justificarlo (D-2.1). Se añaden hiperparámetros de red, memoria, *target* y procesos |
| Coste | Unos 19 s por 1 000 episodios (T1-informe §3) | Órdenes de magnitud más (§9, trampa 15) |
| Framework | No aplicaba | PyTorch en CPU, aunque el código de clase es Keras (D-0.2) |

*Fuente: comparación de taller_trader y taller_trader2; T1-informe §3; D-0.2, D-1.1, D-2.1.*

---

## 3. Qué dijo cada profesor en cada clase

**Idea.** T23 y T24 hablan del **Taller 1**, y las da otro profesor. Solo T30 es del profesor del Taller 2, que lo anuncia como la misma práctica con redes neuronales y deja las instrucciones concretas para una clase de la que no hay transcripción. Lo que se dijo para el Taller 1 orienta, pero **manda el enunciado**.

### 3.1 T23, 23/10: presentación del Taller 1 (profesor del Taller 1)

La clase cubrió fundamentos, bandits, MDP y programación dinámica. El taller se presentó al final (~04:22-04:41), todavía con el enunciado sin cerrar.

| Tema | Qué dijo (paráfrasis) | Hora |
|---|---|---|
| Formato de la sesión | Lo habitual es dejar trabajo para la semana y comentarlo el jueves con los alumnos | ~00:05 |
| Estado del enunciado | No está acabado, los ejercicios "no están del todo" definidos y los refinará | ~04:22, ~04:37 |
| Crear entornos | `SimpleTradingEnv` extiende el entorno de Gym. Basta definir `reset`, `_get_obs` y `step` para construir el entorno que se quiera | ~04:23-04:24 |
| Naturaleza del problema | "Muy sencillo", con apariencia financiera: una serie de precios y decidir cuándo entrar y cuándo salir, nada más | ~04:24 |
| Observación | Tendencia: valor actual menos la media "de los 5 o 6 pasos", más la posición. Se recorta para que no se dispare | ~04:25-04:26 |
| Discretización (Taller 1) | Con un método tabular hay que discretizar: "partís en 10 cachitos", "o los que sean" | ~04:27 |
| Comisión | 0,1 % por cada compra o venta, para hacerlo un poco más realista | ~04:28 |
| Balance | Se devuelve en `info` en cada paso para ver si se aprende a tener cada vez más dinero | ~04:30, ~04:34 |
| Fin del episodio | Se acaba al llegar al final de la serie, cuya longitud marca el número de pasos | ~04:31 |
| Precios reales | Quien quiera puede descargar una serie y pasarla al constructor. Él no los usará | ~04:31 |
| Generadores | Cada función tiene características distintas: browniano completamente aleatorio, reversión a la media, regímenes alcistas y bajistas. El realismo no es importante aquí | ~04:32 |
| Aleatoriedad y señal | Si todo es aleatorio el agente no tiene mucho que hacer; lo que debe descubrir es si hay alguna tendencia | ~04:34 |
| Mercado 1 | Tendencia, oscilación periódica y ruido. **Se puede aprender:** él entrenó un agente mejor que el aleatorio y que buy & hold | ~04:35 |
| Aleatorio | Gana algo en el ejemplo porque la serie es claramente alcista | ~04:35 |
| Objetivo | **Mejorar a buy & hold** | ~04:36 |
| Lectura de la figura | Verde: con el activo; rojo: sin él. Lo ideal es tenerlo cuando sube y no tenerlo cuando baja | ~04:36 |
| Entrenamiento | No usar siempre la misma serie: una distinta por episodio de la misma función o, como **reto mayor**, de **funciones distintas**, para que funcione bien en todas | ~04:36-04:37 |
| Algoritmos | "Algoritmos vistos en clase" es ambiguo y lo concretará | ~04:37 |
| Análisis | Para cada generador, decir qué esperaríais que fuese lo mejor con la información disponible y ver si el agente lo hace. Si encuentra otra cosa, entenderla. Ejemplo: en FrozenLake no había previsto que la política óptima se moviera directamente hacia el agujero | ~04:38 |
| Uso de IA | Le preocupa menos que programéis los algoritmos: le parece bien usar IA generativa. Él genera una solución y la revisa | ~04:38-04:39 |
| Lo que valora | **Entender la solución y analizar la convergencia.** Preguntará qué algoritmo converge más rápido y qué está haciendo la solución. Hay que entender los algoritmos aunque no se programen a mano | ~04:39 |
| Primera versión | Un resumen de "una cara o 2 caras" con los resultados, para discutir. Mencionó Word; el enunciado pide PDF | ~04:40 |
| Sesión del jueves | Revisar la solución con cada grupo; por grupos para poder hablar con todos en 2 h y media | ~04:40-04:41 |

### 3.2 T24, 24/10: normas del Taller 1 (profesor del Taller 1)

| Tema | Qué dijo (paráfrasis) | Hora |
|---|---|---|
| Entrega | El jueves a las 12 de la noche, con algo antes de la clase | ~00:02 |
| Sentido de la clase | Hablar con cada grupo y dar *feedback*, con un margen pequeño para corregir una errata o algo mal interpretado. También la usa para hacerse una idea de cómo evaluar | ~00:03 |
| Mínimos | Los ejercicios son el **mínimo**, pero "es súper abierto" y se pueden probar mil cosas. Valora la **capacidad de innovación, exploración y creatividad** | ~00:03-00:04 |
| Rúbrica | 50 % lo propuesto, 10 % ir más allá, 20 % presentación, 10 % justificar decisiones, 10 % análisis crítico | ~00:04 |
| Plataforma | Además del email, se abriría una tarea en la plataforma para subir la solución | ~00:04 |
| Extensión | "Van a ser 2 hojas" con los resultados, enviadas por email antes de ir a clase | ~00:04 |
| Organización | Con lo recibido planifica las sesiones; una persona de cada grupo escribe para saber cuántos sois | ~00:05 |
| Convergencia | Analizarla significa **mirar cuánto tarda cada algoritmo y comprobarlo**: "una cosa es lo que dicen los libros" y otra verificarlo uno mismo. Lo dijo comparando dos métodos de programación dinámica | ~00:52 |
| Versión previa | La versión inicial sirve para echarle un vistazo antes; sesiones de 15 minutos como máximo | ~02:59-03:00 |
| Corrección | Los comentarios servirán sobre todo si algo se está interpretando mal; se corrige hasta el final de ese día | ~03:00-03:01 |
| Formato | 2 páginas como máximo: pueden ser 2 diapositivas o un documento PDF con el resumen de lo hecho | ~03:01 |
| Dudas | Se le puede escribir por email | ~04:36 |

### 3.3 T30, 30/10: anuncio del Taller 2 (profesor del Taller 2)

| Tema | Qué dijo (paráfrasis) | Hora |
|---|---|---|
| Anuncio | Es **una continuación** del taller anterior, en vez de Q-Learning lo que se ve en este módulo: "exactamente la misma práctica, pero metiendo redes neuronales" | ~00:00 |
| Instrucciones | Pasará el guion al día siguiente y **confirmará entonces las instrucciones concretas** | ~00:00-00:01 |
| Formato | Por ser una variante, el jueves "igual" hace como en el otro: cada grupo describe en clase lo que está haciendo. Sería en el taller del jueves siguiente, de 6 y media a 9 ("creo") | ~00:01 |
| Alcance | El *dataset* y todo es lo mismo: **solo hay que cambiar el modelo** | ~00:01 |
| Amplitud del tema | El RL da "para un máster": muchos algoritmos, variantes y casos | ~00:02 |
| Taller 1, repaso | Tres acciones y estado con la tendencia de los últimos 6 días y la posición. Con Q-Learning hay que trocear en franjas, pero sin pasarse: cada estado debe visitarse muchas veces | ~00:08-00:10 |
| Realismo | La práctica es más sencilla que un uso real, donde se parte de indicadores, ventanas móviles y tendencias antes de discretizar | ~00:11 |
| Episodios | En el taller un episodio es un periodo de inversión, y con 10 o 20 no se entrena: harán falta muchos más | ~02:40 |
| Recompensa | Se puede dar en cada paso o solo al final, y hay comisión. Dar poca información a lo largo del episodio dificulta aprender, y **que un proyecto funcione depende mucho de cómo se diseñe la recompensa** | ~03:28-03:30 |
| Cuándo aprender | En el código de clase el aprendizaje se hace fuera del bucle de pasos, al acabar la época, y dice que suele ser típico | ~03:33 |
| Planes | Al día siguiente, primera parte con material nuevo y segunda explicando su taller y empezando a "cacharrear" | ~04:39, ~04:43 |
| Qué hay que hacer | El mismo ejercicio que él hace en CartPole: primero Q-Learning con estados discretizados, luego DQN sin discretizar. "Quiero que me hagáis lo mismo" con el entorno del taller | ~04:39 |
| GPU | Con una red tan sencilla no iría mucho más rápido: la complejidad está en recalcular el *target* y en simular el entorno. Va a tirar parecido a una CPU | ~04:40 |
| Duración como señal | En CartPole parece que tarda más a medida que aprende porque el episodio dura más. **Nota:** aquí no pasa, los episodios duran siempre 494 pasos | ~04:40 |
| Comportamiento y aprendizaje | Con un ε casi de la mitad, el agente se comporta medio al azar, pero Q-Learning puede haber aprendido ya mucho más. **Una cosa es cómo se comporta y otra lo que ha aprendido**: hay que mirar con ε = 0 | ~04:41 |
| Ruido y prudencia | Con ruido, el agente tiende a una estrategia más conservadora | ~04:42 |
| Reinicio del modelo | Una prueba le salió mal porque reinició el modelo sin querer y perdió lo entrenado | ~04:43 |
| Valor de DQN | Entrenado suficiente tiempo, tiene en cuenta los infinitos estados **sin ninguna discretización** | ~04:43 |

### 3.4 Conceptos de T23 que afectan a las decisiones

Son del profesor del Taller 1 y siguen vigentes:
- **ε con decaimiento:** en la práctica, empezar con ε alto para explorar mucho e ir bajándolo (T23 ~01:20). Respalda el ej. 3 para DQN.
- **No llevar ε a 0 si el entorno cambia:** si las propiedades del mercado cambian, hay que seguir explorando (T23 ~01:21). Sirve para justificar un ε mínimo y conecta con el mercado de regímenes. Evaluamos en *greedy* (D-0.1), lo que conviene justificar.
- **Markov:** sobre si el mundo financiero es markoviano dijo "me temo que no", y que depende de cómo se defina el estado (T23 ~01:34-01:35). Encaja con el análisis crítico de `(tendencia, posición)`.
- **γ:** se descuenta porque las recompensas lejanas son inciertas y para que la suma converja (T23 ~01:40). **Nota:** aquí el episodio es finito, así que γ fija cuánto pesa el futuro frente a la comisión inmediata.
- **Recompensas intermedias:** en ajedrez pueden despistar del objetivo a largo plazo (T23 ~00:14). Conviene tenerlo presente si se modifica la recompensa.

### 3.5 Lo que no está en estos tramos

Valores concretos de hiperparámetros para el taller, criterio exacto de convergencia para DQN y A3C, número de series de test, métricas más allá de rendimiento y convergencia, y si A3C debe ser asíncrono de verdad. Son decisiones del grupo y hay que justificarlas (D-0.3, D-2.1). Las instrucciones del 31/10 no están transcritas: **quien asistiera debería contrastar este documento con lo que se dijo.**

*Fuente: T23 ~00:05, ~00:14, ~01:20-01:40, ~04:22-04:45; T24 ~00:00-00:06, ~00:52, ~02:58-03:01, ~04:36; T30 ~00:00-00:02, ~00:08-00:11, ~02:40, ~03:28-03:33, ~04:39-04:46.*

---

## 4. El entorno `SimpleTradingEnv`

**Idea.** Un activo, tres acciones y una observación de dos números. Es muy simple, pero tiene sutilezas que condicionan el diseño. El código está copiado literalmente del enunciado en `notebooks/common/env.py` y no se modifica (lo comprueba `test_common.py`).

### 4.1 Observación, acciones y dinámica

| Elemento | Código del taller | Comentario |
|---|---|---|
| Observación | `[clip(trend, -1, 1), position]`, `float32`, `Box([-1,0],[1,1])` | Tendencia **continua** + posición 0/1 (celdas 3 y 4) |
| Tendencia | `window = prices[t-5 : t+1]`; `trend = (p_t − mean(window)) / mean(window)` | **6 precios**, incluido el actual. El markdown dice "últimas 6 observaciones" y el comentario del código "últimos 5 pasos": vale el código. Es la desviación relativa respecto a una media corta, **no una pendiente** |
| Acciones | `Discrete(3)`: 0 = vender, 1 = mantener, 2 = comprar | Vender sin activo o comprar con él equivale a mantener |
| Inicio | `reset()`: `current_step = 5`, `position = 0`, `cash = balance = 1000`, `shares = 0` | Empieza siempre fuera del mercado |
| Ejecución | Compra o venta **al precio `prev_price` = $p_t$**, con todo el capital | $p_t$ ya está en la observación: no hay mirada al futuro |
| Comisión | `cost_rate = 0.001` sobre el efectivo al comprar y sobre el valor de venta al vender | 0,1 % por operación |
| Balance | `cash + shares · p_{t+1}` con activo; `cash` sin él. Va en `info["balance"]` | Valor de mercado: una posición abierta al final **no paga la comisión de venta** |
| Fin | `terminated = current_step >= len(prices) − 1`; `truncated = False` siempre | Con `n = 500`, **494 decisiones** por episodio |

En fórmulas, con $\bar p_t=\tfrac16\sum_{i=0}^{5}p_{t-i}$ y $x_{t+1}\in\{0,1\}$ la posición tras ejecutar la acción:

$$\text{tendencia}_t=\frac{p_t-\bar p_t}{\bar p_t},\qquad r_t=-0{,}001\cdot\mathbb 1[\text{opera}]+x_{t+1}\,\frac{p_{t+1}-p_t}{p_t}$$

El paso de compra ya cobra la variación del precio; el de venta, no.

### 4.2 Sutilezas del entorno

1. **Recompensa y balance no miden lo mismo.** La comisión pesa igual (−0,1 %), pero la recompensa **suma** retornos simples sin ponderar por el capital y el balance **compone**. Maximizar $\sum r$ no equivale a maximizar el balance final, y con $\gamma<1$ ni siquiera se maximiza el retorno del episodio. **Nota:** la diferencia es aproximadamente $\sigma^2T/2$. Medida con buy & hold en 100 series (medianas): mercado 1, $\sum r=6{,}06$ frente a $\log(B_T/B_0)=3{,}24$; mercados 2-4, diferencia de ≈0,1.
2. **Acciones redundantes.** En cada estado, una de las tres acciones es igual que "mantener". **Nota:** tiene tres efectos con redes. En DQN, dos salidas estiman el mismo valor y el `max` elige entre ruido, un caso de libro de sobreestimación ([04_doble_dqn_y_duelo.md](04_doble_dqn_y_duelo.md)). En A3C, la probabilidad de "no hacer nada" se reparte entre dos salidas, así que la acción más probable puede no representar a la política: con $P=(0{,}3;\,0{,}3;\,0{,}4)$ sin activo, la *greedy* compra aunque el agente esperaría el 60 % de las veces. Y la exploración uniforme opera solo 1 de cada 3 pasos.
3. **El aleatorio paga mucho.** **Nota:** por lo anterior hace 494/3 ≈ 165 operaciones (164,8 medidas, `results/00_baselines.csv`), lo que por sí solo cuesta $1-0{,}999^{165}\approx15\,\%$. Por eso pierde 17-22 % en los mercados 2-4.
4. **La exploración cuesta recompensa.** **Nota:** con ε = 1 se pagan unas 0,16 unidades por episodio solo en comisiones. Al bajar ε, la curva de recompensa sube **aunque el agente no aprenda nada**: el Taller 1 lo observó en los mercados 2 y 4 (T1-informe, Figura 2).
5. **`reset()` no genera precios nuevos.** La serie se pasa al constructor (y `max_steps` se fija ahí), así que hay que crear un entorno por episodio.
6. **Los generadores usan `np.random` global sin semilla.** El repositorio usa `make_prices(market, n, seed)`, que llama a `np.random.seed` (trampa 5 de §9).
7. **El final es un límite de tiempo, no un estado terminal del MDP,** aunque se devuelva `terminated=True` (D-1.2).
8. **La observación no es (necesariamente) markoviana.** Resume 6 precios en un número y **no incluye el tiempo** ni la distancia a un nivel de referencia (T23 ~01:34).
9. **El recorte a ±1 no actúa.** **Nota:** en 200 series por mercado, la tendencia máxima en valor absoluto es 0,66 (mercado 1) y ≈0,13 (el resto). El rango teórico es $(-1, 5]$.
10. **Incompatibilidades con Gym.** **Nota:** `reset(self, seed=None)` no acepta `options`, así que `SyncVectorEnv` y `check_env` fallan con error (comprobado). Tampoco se puede dar un paso tras el final: `IndexError`. Como el entorno no se modifica, A3C debe usar bucles propios o un envoltorio.

### 4.3 Qué cambia con redes: la escala

**Nota:** medido en 200 series de 500 pasos por mercado (semillas 0-199), para completar D-1.1:

| Magnitud | Mercado 1 | Mercados 2-4 | Consecuencia |
|---|---|---|---|
| Tendencia (desviación típica) | 0,153 | ≈0,024 | Entrada de escala muy distinta entre mercados y frente a la posición (0/1) |
| Retorno por paso (desviación típica) | 0,106 | ≈0,02 | El ruido de la recompensa con posición abierta |
| Retorno medio por paso | +0,0126 | de −0,0002 a +0,0002 | La señal: en 2-4, unas 100 veces menor que el ruido |
| Comisión | 0,001 | 0,001 | Del orden de las diferencias entre acciones que la red debe resolver |
| $\lvert r\rvert$ máximo | 0,51 | ≈0,12 | **Recortar la recompensa a ±1 no hace nada** (trampa 8) |

- "Recompensa ≈1e-3 por paso" (D-1.1) es el orden de la **señal** y de la comisión. El **ruido** es de 2e-2 a 1e-1, y hay que tenerlo en cuenta al elegir la escala.
- Sin normalizar, la tendencia de los mercados 2-4 apenas mueve la entrada frente a la posición, y la red tenderá a ignorarla. No sirve normalizar con los límites de `observation_space` (±1), que no reflejan la escala real. Normalizar con estadísticos de train por mercado es lo que decide D-1.1, ya cerrada (`train_scales`).
- Las Q iniciales de una red sin entrenar pueden ser mayores que las diferencias entre acciones (≈1e-3). **La pérdida baja antes de que la política tenga sentido:** mirad la política, no solo la pérdida.
- **Lo que no cambia:** la red recibe la misma información que la tabla. Gana resolución (umbral continuo, sin bins) y generalización, **pero no ve nada nuevo.** Donde el Taller 1 falló por falta de información (mercado 4), no hay razón para esperar una mejora.

*Fuente: taller_trader2, celdas 3 y 4; `common/env.py`; `results/00_baselines.csv`; T23 ~01:34; T1-informe; D-1.1, D-1.2. Las medidas son propias.*

---

## 5. Los 4 generadores

**Idea.** Cada generador pide una estrategia distinta, y lo que se espera de vosotros es decir cuál antes de entrenar y contrastarla (T23 ~04:38). Con el mismo protocolo que el Taller 1, los resultados tabulares son una referencia directa (D-0.1).

### 5.1 Definición

| # | Función (`MARKETS`) | Parámetros y fórmula | Figura del enunciado |
|---|---|---|---|
| 1 | `generar_precios` (`tendencia_ciclo`) | $t$ = `linspace(0, 50, n)`; retorno $=0{,}0005\,t+0{,}05\sin t+\mathcal N(0;\,0{,}1)$; precio $=100\prod(1+\text{retorno})$ | Celda 8: plano hasta ≈300, picos de 8 000 y 20 000 y final en ≈5 800. Hay que verla en escala log |
| 2 | `generar_precios_browniano` (`browniano`) | `start_price=100, mu=1e-4, sigma=0.02`; log-retornos $\mathcal N(\mu-\sigma^2/2;\,\sigma)$ | Celda 11: 100 → 113 → 63, luego entre 60 y 88. Paseo sin patrón |
| 3 | `generar_precios_reversion_media` (`reversion_media`) | `mean_price=100, kappa=0.05, volatility=0.02`; $\Delta p=0{,}05(100-p)+\mathcal N(0;\,2)$; suelo 0,1 | Celda 14: oscila entre 88 y 117 en torno a 100 |
| 4 | `generar_precios_regimenes` (`regimenes`) | Cambio con $p=0{,}02$ por paso; alcista $\mathcal N(0{,}001;\,0{,}01)$, bajista $\mathcal N(-0{,}0015;\,0{,}025)$; empieza alcista; suelo 0,1 | Celda 17: baja a 60 hasta ≈150 y sube a 140 con ruido bajo |

**Nota, consecuencias de las fórmulas:**
- **Mercado 1:** el ciclo está en los **retornos**, con periodo $2\pi$ en $t$ (≈63 pasos con $n=500$), y la deriva crece de 0 a 2,5 % por paso. **La fase es la misma en todas las series,** porque $t$ es determinista. El agente no ve $t$. El periodo y la deriva por paso dependen de `n_steps`: siempre `N_STEPS = 500`.
- **Mercado 3:** el ruido son 2 unidades de precio, con vida media ≈14 pasos ($\ln 2/0{,}05$) y desviación estacionaria ≈6,4. El estado **no ve la distancia a 100,** solo la desviación frente a 6 precios.
- **Mercado 4:** regímenes de ≈50 pasos, con volatilidad 2,5 veces mayor en el bajista (T1:D-2.4).
- En los mercados 1, 2 y 4, `prices[0]` ya incluye un retorno; solo en el 3 vale exactamente 100.

### 5.2 ¿Hay señal en la observación?

**Nota:** retorno del siguiente paso según la tendencia observada (mismas 200 series que en §4.3). Es la pregunta previa al entrenamiento, y conviene incluirla en el 00 o en el informe.

| Mercado | corr(tendencia, retorno siguiente) | E[retorno] si tendencia en el 10 % inferior | E[retorno] si tendencia en el 10 % superior | Lectura |
|---|---|---|---|---|
| 1 | +0,197 | −2,20 % | **+4,54 %** | Seguir la tendencia, con margen enorme frente a la comisión |
| 2 | +0,005 | −0,01 % | +0,03 % | Sin señal |
| 3 | −0,045 | **+0,23 %** | −0,10 % | Comprar tras caídas: margen justo frente al 0,2 % de ida y vuelta, que el rebote acumulado de varios pasos hace rentable |
| 4 | +0,011 | −0,16 % | −0,11 % | Sin señal: los dos extremos son negativos |

### 5.3 Estrategia esperable, Taller 1 y redes

Mediana en 200 series de test; en mercado 1, múltiplo del capital (T1-informe, Tabla 1; `results/taller1_tabular.csv` y `results/00_baselines.csv`). "No operar" da 0 % por construcción.

| Mercado | Aleatorio | Buy & hold | Q-Learning | Sarsa | Ops./ep. | Conv. (ep.) Q-L · Sarsa | Política del Taller 1 |
|---|---|---|---|---|---|---|---|
| 1 | x4,3 | x33,6 | **x502** | **x472** | 48 · 57 | 189 · 203 | Sigue la tendencia: compra en bins 7-10, vende en 1-3 |
| 2 | −19,4 % | −3,5 % | **0 %** | **0 %** | 0 · 0 | — | No opera |
| 3 | −17,2 % | −0,3 % | **+27,0 %** | **+27,0 %** | 29 · 29 | 260 · 282 | Compra en bin 1, vende en bin 10 |
| 4 | −22,0 % | −12,5 % | **0 %** | **0 %** | 0 · 0 | — | No opera: el estado no ve el régimen |

Convergencia del duelo con la misma configuración (T1:D-2.5). Otros datos del Taller 1: α pequeña en todos los mercados (0,0005-0,001; con α = 0,1 caía a x14-x19), γ = 0 en el mercado 1 porque con un 10 % de ruido el futuro lejano se estima mal, y γ = 0,5-0,9 en reversión porque el rebote tarda varios pasos (T1-informe §3). Q-Learning y Sarsa acaban en políticas casi iguales. En regímenes, la mejor de 70 reglas fijas queda en −8,6 % en validación (T1:D-2.4). Con estado enriquecido (MA30 y volatilidad): x412, 0 %, +42,8 % y +7,8 %.

**Nota: propuesta, a contrastar con los resultados.**

| Mercado | Qué esperar | Qué cambia con redes | Cómo reconocer un fallo |
|---|---|---|---|
| 1 | Seguir la tendencia con muchas operaciones, y batir a buy & hold | Umbral continuo en lugar de bins. La escala de entrada es grande y el ruido del 10 % domina: α pequeña y γ bajo probablemente sigan valiendo. Dispersión enorme: usar mediana y escala log | Mediana por debajo de x33,6, política casi igual a buy & hold o resultado que oscila mucho entre semillas |
| 2 | No operar, o quedarse dentro: **los dos son defendibles** | Riesgo de sobreajustar ruido. La recompensa media de estar dentro es $494\cdot10^{-4}-0{,}001\approx+0{,}05$, aunque la mediana del balance es −3,5 %: **la recompensa y la métrica no coinciden** en este mercado | Muchas operaciones. Cualquier patrón con ganancia en validación que no se repite en test |
| 3 | Comprar tras caídas y vender tras subidas, con pocas operaciones | El mayor beneficio potencial del umbral continuo (T1 solo operaba en los extremos). Requiere normalizar: la señal está en ±0,04. γ debe cubrir el rebote | No opera (γ demasiado bajo o entrada sin normalizar) o sigue la tendencia |
| 4 | No operar | **No es esperable mejorar con el mismo estado:** falta información, no resolución. La vía es ampliar el estado (03), por ejemplo con volatilidad, que en el Taller 1 daba +8,5 % sola | Ganancia aparente que depende de la semilla, o estar siempre dentro (−12,5 %) |

**Ojo con "batir a buy & hold":** es el objetivo que dio el profesor (T23 ~04:36), pero en 2-4 lo bate también no operar. Por eso `no_operar` es baseline obligado (D-0.3): un agente con 0 % en esos mercados **no ha fallado** si esa es la mejor política para su información.

*Fuente: taller_trader2, celdas 5-17, con sus figuras; T23 ~04:32-04:38; T1-informe §2-§4; T1:D-2.4; `results/taller1_tabular.csv`; D-0.3. Medidas y expectativas propias.*

---

## 6. Baselines y figura del taller

**Idea.** El enunciado da el aleatorio y buy & hold, más una figura de dos paneles que sirve como formato para analizar cualquier agente.

- **Aleatorio:** `env.action_space.sample()` en cada paso (celda 19). En el repositorio, `RandomAgent(seed=0)`.
- **Buy & hold:** compra en `prices[5]` con comisión y valora a mercado hasta el final, sin vender (celda 21). `BuyHoldAgent` lo reproduce con la acción "comprar" y `test_common.py` lo comprueba.
- **No operar (nuestro):** `NoTradeAgent` (D-0.3).
- **Reglas simples (nuestro):** `TrendRuleAgent` y `ReversionRuleAgent`, con el umbral elegido en validación. En test dan x630 en el mercado 1 y +29,2 % en el 3, más que los tabulares del Taller 1: son el listón real (notebook 00, §6; D-0.3).
- **La figura (celda 21):** arriba, el precio con puntos verdes (con activo) y rojos (sin él); abajo, el balance del agente frente a buy & hold y al capital inicial. En el ejemplo, sobre el mercado 1, el precio pasa de ≈100 a un pico de ≈4 750 y acaba en ≈2 500; el aleatorio termina en ≈1 500 y buy & hold en ≈21 000, tras un pico de ≈43 000. Lo comenta el profesor en T23 ~04:35-04:36.
- **Errata de la celda 21. Nota:** `bh_balances` tiene un elemento por cada precio desde `prices[0]`, mientras que `balances` empieza en `prices[5]`: la curva de buy & hold queda **desplazada 5 pasos**. `plot_episode` de `common/style.py` ya lo alinea: usad esa función.
- `run_episode` y `evaluate` de `common/protocol.py` miden a todos con el mismo código. Basta con que el agente implemente `act(obs, greedy=False)`.

*Fuente: taller_trader2, celdas 18-21 y su figura; T23 ~04:32-04:36; `common/agents.py`, `common/style.py`, `common/protocol.py`.*

---

## 7. Mapa de los documentos y orden de lectura

**Idea.** Hay dos caminos: uno rápido para empezar a programar y uno completo para defender la solución en la discusión. Todos terminan con un apartado `## Aplicación al Taller 2`.

| Documento | Contenido | Material de origen | Para qué lo necesitáis |
|---|---|---|---|
| [00_fundamentos_mdp.md](00_fundamentos_mdp.md) | RL, MDP, retorno, Bellman | `slides1`, T23 | Vocabulario, γ, Markov |
| [01_dp_y_montecarlo.md](01_dp_y_montecarlo.md) | Programación dinámica y Monte Carlo | `slides2`, T24 | Base de TD; retornos MC |
| [02_td_sarsa_qlearning.md](02_td_sarsa_qlearning.md) | TD(0), Sarsa, Q-Learning | `slides3` | El objetivo de DQN sale de aquí |
| [03_dqn.md](03_dqn.md) | DQN: *replay*, *target*, código en CartPole | `DRL1`, `2-cart_pole_DQN` | **Imprescindible para 01** |
| [04_doble_dqn_y_duelo.md](04_doble_dqn_y_duelo.md) | Doble DQN y DQN en duelo | `DRL5` | Sobreestimación; extras |
| [05_policy_gradient.md](05_policy_gradient.md) | Teorema del gradiente, REINFORCE | `DRL6` | Base de A3C |
| [06_actor_critico_y_a3c.md](06_actor_critico_y_a3c.md) | Actor-crítico y A3C | `DRL7`, `3-cart_pole_actor_critic` | **Imprescindible para 02** |
| [07_panorama_drl.md](07_panorama_drl.md) | Taxonomía, hitos, entornos, librerías, consejos | `DRL2`-`DRL4`, T30 | Contexto y consejos de estabilidad |
| [08_trading_en_clase.md](08_trading_en_clase.md) | DQN para trading de clase y qué se traslada | `01-AR_trading_prueba`, `02-AR_trading_multivariable`, `PC5` | **Código de partida y sus errores** |

**Orden rápido (para programar ya):** este documento → [03_dqn.md](03_dqn.md) → [08_trading_en_clase.md](08_trading_en_clase.md) → notebook 01. Para A3C: [05_policy_gradient.md](05_policy_gradient.md) → [06_actor_critico_y_a3c.md](06_actor_critico_y_a3c.md) → notebook 02.

**Orden completo (antes del 5/11):** 00 → 01 → 02 → 03 → 04 → 05 → 06 → 08 → 07. Cada miembro debería poder explicar DQN y A3C aunque no los haya programado (T23 ~04:39).

*Fuente: `docs/teoria/`, a 31/10/2026; T23 ~04:39. Algunos documentos se estaban escribiendo a la vez que este, así que la descripción puede variar.*

---

## 8. Ideas para ir más allá (10 %)

**Idea.** Se valora la creatividad y la exploración (T24 ~00:03), pero el 10 % es un bloque pequeño: una o dos pruebas bien medidas y explicadas valen más que muchas sin concluir.

**Del enunciado o de los profesores:**
- **Más generadores o una serie real** (celda 5; T23 ~04:31). Ya están en `common/markets.py`: `tendencia_persistente`, `saltos` y `load_sp500()`.
- **Entrenar con generadores distintos** y evaluar en todos: un único agente robusto, "un reto mayor" (T23 ~04:37). Con redes es más natural que con tablas.
- **Otros algoritmos** (el enunciado dice "al menos"): Doble DQN y DQN en duelo ([04_doble_dqn_y_duelo.md](04_doble_dqn_y_duelo.md)), REINFORCE ([05_policy_gradient.md](05_policy_gradient.md)), A2C síncrono (D-2.1).
- **Comparar la velocidad de convergencia** y explicarla (T23 ~04:39; T24 ~00:52).
- **Diseño de la recompensa** (T30 ~03:28-03:30).

**Nota: propuestas propias.**
- **Comparar redes y tablas:** DQN frente a Q-Learning por mercado, con las mismas series. Responde a la pregunta natural del taller y `results/taller1_tabular.csv` ya está disponible.
- **Ampliar el estado** en un entorno derivado (no en `env.py`): ventana de retornos, volatilidad, media larga. Es donde las redes deberían notarse, porque una tabla escala mal: con 180 celdas el Taller 1 ya bajaba a 800-1 760 visitas por celda. Elegid los indicadores en validación y decid que se conocían los generadores (limitación señalada en T1-informe §5).
- **Añadir el tiempo al estado en el mercado 1:** como la fase es fija (§5.1), debería mejorar; es una forma clara de mostrar el límite de la observación, y hay que explicar que no se trasladaría a un mercado real.
- **Generalización fuera de distribución:** entrenar en un mercado y evaluar en otro (matriz 4×4).
- **Recompensa logarítmica** (sutileza 1 de §4.2); en el Taller 1 movió poco el resultado (x412 a x431).
- **Sensibilidad a la comisión:** 0 %, 0,1 % y 0,5 %, en un entorno derivado. En el Taller 1 fue lo que más pesó (T1-informe §4).
- **Política visualizada:** $Q$ o $\pi$ en función de la tendencia para cada posición, sobre el rango real. Con redes es una curva continua, muy buena figura para la presentación y para contrastar con §5.3.
- **Efecto de la normalización:** con y sin ella, como evidencia de D-1.1.
- **Asíncrono frente a síncrono:** A3C y A2C con el mismo presupuesto de pasos (D-2.1).

*Fuente: taller_trader2, celdas 5, 22 y 24; T23 ~04:31-04:39; T24 ~00:03, ~00:52; T30 ~03:28; T1-informe §4-§5; D-2.1.*

---

## 9. Trampas conocidas

**Idea.** Lista de fallos probables, para comprobarlos antes de dar un resultado por bueno. **Nota:** salvo donde se cita otra fuente, son deducciones propias del código o de los resultados del Taller 1, y las marcadas "comprobado" se han ejecutado.

**Sobre las fuentes**

1. **Mezclar talleres.** T23 y T24 hablan del Taller 1 y de otro profesor (§3). Las fechas de T24 ("martes 27", "jueves") **no son las vuestras**. El enunciado del Taller 1 está también en `docs/raw/MIAX-14/taller_trader.ipynb`: el vigente es `docs/raw/taller/taller_trader2.ipynb`.
2. **Mezclar decisiones.** `D-x.y` de este repositorio y `T1:D-x.y` tienen numeraciones distintas: D-1.1 aquí es la normalización y en el Taller 1 la discretización; D-2.1 aquí es la variante de A3C y allí el orden de actualización.
3. **Leer mal `results/taller1_tabular.csv`.** `retorno_mediana` es un retorno, no un múltiplo: 501,3 es x502,3 (el "x502" del informe) y 32,6 es x33,6. Además, su `episodios_convergencia` (132, 350...) **no coincide** con la tabla del informe (189, 260...), que viene del duelo con igual configuración (T1:D-2.5). Citad cuál usáis.

**Sobre el protocolo**

4. **Entrenar siempre con la misma serie**, o reutilizar el entorno creyendo que `reset()` da otra (comprobado que no): §4.2, sutileza 5. Usad `train_series_seed`.
5. **El RNG global se reinicia con cada serie (comprobado).** `make_prices` hace `np.random.seed(seed)`, así que todo lo que use `np.random` después depende de la serie: ε-greedy o el muestreo del *replay* quedarían correlados con los precios. Usad un `np.random.default_rng` propio en el agente, como `RandomAgent`, o `torch`.
6. **Semillas en A3C.** Si todos los procesos llaman a `train_series_seed(seed, episodio)` con los mismos argumentos, ven **las mismas series** y se pierde la decorrelación. Pero tampoco vale `seed + rank`: `train_series_seed` falla por encima de 49 999 (`seed` ≥ 5), y dos semillas se solapan si hay más de 10 000 episodios. Lo seguro es un contador de episodio único para todos los procesos, o `episodio·n_workers + rank`.
7. **Elegir hiperparámetros mirando el test.** Se eligen en validación (`split="val"`) y el test se usa una vez (D-0.1). En el Taller 1 la validación salió optimista: +34,0 % frente a +27,0 % (T1-informe §1).

**Sobre el entrenamiento**

8. **Copiar recetas sin mirar la escala.** Recortar la recompensa a ±1 no hace nada aquí, porque $\lvert r\rvert$ nunca llega a 1 (§4.3). Lo mismo con Huber con δ = 1: se reduce a MSE. El *reward clipping* a signo, en cambio, destruye la magnitud.
9. **Olvidar normalizar**, o normalizar con estadísticos de test o de todos los mercados a la vez. Si se normaliza, debe aplicarse igual en train, validación y test, y guardarse con el modelo (D-1.1).
10. **Fiarse de la curva de recompensa.** Sube sola al bajar ε (§4.2, sutileza 4). Y con ε alto, el comportamiento no refleja lo aprendido (T30 ~04:41): **medid la política *greedy* en validación** (D-0.3).
11. **`convergence_episode` engaña.** No significa nada si el agente acaba sin operar (`common/protocol.py`; D-0.3). Con A3C, los episodios además corren en paralelo: comparad pasos de entorno y tiempo.
12. **Un `terminated` mal tratado.** El último paso no es terminal de verdad, pero se trata como tal (D-1.2). Afecta a 1 de 494 pasos en DQN, y en A3C también al arranque del retorno de $n$ pasos: la regla es la misma en los dos.
13. **Evaluar A3C solo en *greedy*.** Por las acciones redundantes (§4.2, sutileza 2), la política determinista puede diferir mucho de la estocástica aprendida. Comparad las dos antes de sacar conclusiones. En la tabla va la *greedy* (D-0.1), y conviene decir si la estocástica difiere.
14. **Volver a ejecutar la celda que crea el modelo** y perder lo entrenado (T30 ~04:43). Guardad con `save` y usad la caché de `run_grid`.
15. **Quedarse sin tiempo.** Con lo medido en el equipo actual (notebook 00, §7), aprender en cada paso cuesta ≈0,34 s por episodio: 500 episodios son ≈3 min y 4 mercados × 5 semillas son ≈1 h **en secuencia, para un solo algoritmo y sin barridos**. Evaluar 200 series añade ≈10 s por agente y mercado. En otro equipo puede costar varias veces más. Conviene aprender cada varios pasos o al final de la época (T30 ~03:33), aunque un único paso de gradiente por episodio es muy poco ([03_dqn.md](03_dqn.md) §13); lanzar barridos con `run_grid` y reservar el resultado final con tiempo. Y `tiempo_s` solo es comparable con `n_jobs=1`.
16. **Funciones en celdas con A3C.** En Windows, `spawn` no puede importar funciones de una celda; `worker` va en `common/a3c.py`, con `set_num_threads(1)` y `set_seed` por proceso (README; D-0.2). No se anidan procesos: `run_grid(..., n_jobs=1)`.
17. **Usar GPU esperando ganar tiempo.** Con estas redes no compensa (T30 ~04:40; D-0.2).
18. **Modificar `env.py` o los generadores.** Lo detecta `test_common.py` y rompe la comparación. Las variantes van en un entorno derivado.

**Sobre el análisis**

19. **Dar la media en el mercado 1.** Buy & hold tiene media de x200 con desviación de 657 veces el capital (T1-informe §1). Mediana y escala logarítmica.
20. **Concluir de una semilla.** En el Taller 1, la diferencia x502/x472 reflejaba cuántas semillas caían en cada política, no un algoritmo mejor (T1-informe §3). Y la afirmación de que Q-Learning convergía antes era débil con 3 semillas. Con redes la varianza es mayor: 5 semillas (D-0.1) y mostrad la dispersión.
21. **Comparar con cifras del anexo.** El Q-L enriquecido del Taller 1 usa otra configuración y no se compara uno a uno (T1-informe, Tabla 1).
22. **Tratar "no operar" como fallo o "batir a buy & hold" como éxito.** §5.3.
23. **Valorar una ganancia sin contar la salida.** La posición abierta al final no paga la comisión de venta. Es coherente con buy & hold, pero conviene saberlo si el agente acaba siempre dentro.

*Fuente: `common/protocol.py`, `common/markets.py`, `common/agents.py`; `test_common.py`; D-0.1 a D-2.1; T1-informe; T30 ~03:33, ~04:40-04:43.*

---

## 10. Checklist final, ligada a la rúbrica

**Idea.** Comprobaciones antes de cada hito.

| Rúbrica | Comprobación | Dónde |
|---|---|---|
| Logística | ✅ Email del representante enviado **antes del 3/11** (2/11), preguntando también dónde se sube la versión final | Dirección en la celda 23 |
| Logística | Primera versión enviada **antes de la clase del 5/11** | `results/informe.md` → PDF |
| Logística | PDF final de **2 páginas como máximo**, una entrega por grupo, **5/11, 23:59** | `build_informe.py` · D-0.5 |
| Mínimos (50 %) | DQN y A3C entrenados en los 4 mercados, con una serie nueva por episodio | 01, 02 · D-0.1 |
| Mínimos | Test con series nuevas, resultados promediados: mediana, media ± std | `evaluate()` · D-0.1 |
| Mínimos | Comparación con aleatorio y buy & hold en cada mercado | 00, 04 |
| Mínimos | Para cada mercado: estrategia esperada frente a la encontrada, razonada | 01, 02, informe · §5.3 |
| Mínimos | Ajuste documentado de α, ε (decaimiento) o entropía en A3C, γ y número de episodios | 01, 02 · D-2.1 |
| Mínimos | Tabla algoritmos + baselines × mercado, con rendimiento y convergencia | 04 · D-0.3 |
| Más allá (10 %) | Una o dos pruebas de §8, con resultado y conclusión | 03 |
| Presentación (20 %) | Tabla legible, figura de precio/posición y balance, curvas de aprendizaje; todo cabe en 2 páginas | 04, `results/figures/` |
| Presentación | Guion de 15 minutos: entorno → diseño → resultados → análisis, y reparto entre los tres | — |
| Justificación (10 %) | Normalización (D-1.1), fin de episodio (D-1.2), convergencia (D-0.3) y variante de A3C (D-2.1) **cerradas** y citadas; PyTorch justificado (D-0.2) | `docs/DECISIONES.md` |
| Análisis crítico (10 %) | Límites del estado, recompensa frente a balance, dispersión del mercado 1, qué algoritmo converge antes y por qué, y qué aportan las redes frente a las tablas | Informe |
| Discusión | Cada miembro explica DQN (*replay*, *target*) y A3C (ventaja, entropía, asincronía), la convergencia observada y la solución de cada mercado (T23 ~04:39) | [03_dqn.md](03_dqn.md), [06_actor_critico_y_a3c.md](06_actor_critico_y_a3c.md) |

*Fuente: taller_trader2, celdas 22-24; T23 ~04:39; README; `docs/DECISIONES.md`.*

---

## Aplicación al Taller 2

**Nota: propuesta de plan.** Todo lo anterior se reduce a estos pasos:

1. **Hecho (2/11):** email del representante enviado; quien asistiera el 31/10 contrasta §3.5 con lo que se dijo.
2. **D-1.1 y D-1.2 ya están cerradas.** Empezad con la observación normalizada por mercado y validad el código en el mercado 1: tiene la señal más fuerte (§5.2), y el resultado del Taller 1 (x502) dice qué esperar.
3. **DQN y A3C primero en mercados 1 y 3,** que son los que tienen señal. En 2 y 4 el resultado esperable es no operar o quedarse dentro: si el agente opera mucho, sospechad antes que celebrar.
4. **Medir siempre la política *greedy* en validación,** con 494 decisiones por episodio, y comparar con `no_operar`, buy & hold y `results/taller1_tabular.csv`.
5. **Enviar una versión temprana del informe** para probar que cabe en 2 páginas y dedicar el resto del tiempo al análisis: es lo que preguntarán en la discusión.
