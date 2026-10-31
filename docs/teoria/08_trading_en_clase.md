# El DQN para trading visto en clase y qué se puede trasladar al Taller 2

> **Fuentes:** notebooks de clase `01-AR_trading_prueba` (DQN denso sobre el S&P 500, 2001-2010 entrenamiento y 2011 test), `02-AR_trading_multivariable` (mismo agente con 3 series semanales de acciones) y `PC5` (Proyecto de Consolidación 5: mismo código que el primero, sin salidas, con tres extensiones por programar); los datos `docs/raw/DRL/03-trading/GSPC_2001-10.csv`, `GSPC_2011.csv` y `financial_series.csv`; y los fragmentos de la transcripción `T30` (clase del 30 de octubre) en los que el profesor habla de bolsa, comisiones, diseño de la recompensa o de su experiencia aplicando RL a inversión. Para la comparación se usa `SimpleTradingEnv` del enunciado del Taller 2 (`taller_trader2`, celda 4).
>
> **Convenciones:** "(01-AR_trading_prueba, celda 9)" = celda 9 de ese notebook (numeración de 0). "(T30 ~01:12)" = hora aproximada en la clase. "PC5" es el notebook del Proyecto de Consolidación 5. **Nota** = contenido propio que no está en el material de clase (correcciones, cálculos propios, propuestas). Los errores del código de clase se señalan con su corrección razonada, nunca en silencio (ver §9). Para la teoría de DQN (red objetivo, replay, pérdida) ver [03_dqn.md](03_dqn.md); aquí solo se describe cómo lo concreta el código de trading.

---

## 1. Qué se vio en clase y para qué sirve
**Idea.** El profesor presentó el trading como la "misma práctica" del taller de Q-learning, pero sustituyendo la tabla Q por una red neuronal: el dataset y el entorno son los mismos, solo cambia el modelo (T30 ~00:00:30-00:01:15). Los notebooks de `03-trading` son esa versión con red: un agente DQN que, día a día, decide comprar o vender un paquete de acciones del activo.

- **Notebook 1** (`01-AR_trading_prueba`): un activo (S&P 500), estado = ventana de log-retornos, 2 acciones (comprar/vender un paquete), recompensa = beneficio realizado al vender, DQN denso 20→8→2.
- **Notebook 2** (`02-AR_trading_multivariable`): el mismo código con 3 series (IBM, Intel, Microsoft, semanales); solo se opera Microsoft, las otras dos son información extra del estado.
- **PC5**: copia del notebook 1 + enunciado con tres mejoras a implementar (§8). No trae solución ni salidas guardadas.
- El profesor insistió en que, en un proyecto real de inversión, lo que más trabajo da es el entorno (simular la inversión, comisiones, índices) y que el algoritmo es "estándar" (T30 ~01:12:19-01:12:57). En clase el entorno es deliberadamente sencillo.
- También avisó de que aplicar Deep RL a inversión es más potente pero más inestable que Q-learning tabular, exige cuidar los parámetros y sobreajusta si hay muchas variables (T30 ~00:16:21-00:16:46).

*Fuente: 01-AR_trading_prueba celdas 0-1; 02-AR_trading_multivariable celdas 0-1; PC5 celdas 0-3; T30 ~00:00:17-00:01:15, ~00:16:00-00:16:50, ~01:12:19-01:13:00.*

---

## 2. Los datos
**Idea.** Hay dos conjuntos distintos: el S&P 500 diario (notebook 1 y PC5) y una tabla semanal de 53 acciones (notebook 2). Los CSV se descargaban de una carpeta compartida con `urlretrieve` (01, celda 2; 02, celda 2; PC5, celda 4); en el repositorio están en `docs/raw/DRL/03-trading/`.

| Archivo | Filas de datos | Fechas | Columnas | Uso |
|---|---|---|---|---|
| `GSPC_2001-10.csv` | 2515 | 2001-01-02 a 2010-12-31 | `Date, Open, High, Low, Close, Adj Close, Volume` | Entrenamiento (01, PC5) |
| `GSPC_2011.csv` | 252 | 2011-01-03 a 2011-12-30 | las mismas 7 | Test (01, PC5) |
| `financial_series.csv` | 277 | 2011-01-09 a 2016-06-05, una fila por semana (domingos) | `Date` + 53 series de precios: 29 con sufijo `DAX30` (Daimler ... BMW) y 24 con sufijo `NASDAQ` (Coca Cola ... JP Morgan Chase) | 02 |

Detalles comprobados en los archivos (**Nota:** verificación propia, sin valores nulos en ninguno):
- GSPC: `Close` pasa de 1283,27 (2001-01-02) a 1257,64 (2010-12-31) en entrenamiento, y de 1271,87 a 1257,60 en 2011. En las filas que muestra la celda 3, `Adj Close` coincide con `Close`; el código usa `Close` (columna 4).
- El código lee solo `Close`: `float(line.split(",")[4])/1000` (01, celda 5), es decir, el precio se divide por 1000 y queda en torno a 0,8-1,6.
- Volatilidad diaria de los log-retornos de `Close` (**Nota:** cálculo propio): desviación típica ≈ 0,0138 en 2001-2010 (incluye 2008) y ≈ 0,0147 en 2011. El S&P 500 de 2011 acaba prácticamente donde empieza (-14 puntos), un año sin tendencia donde cualquier estrategia tiene difícil ganar.
- `financial_series.csv`: las columnas con sufijo "NASDAQ" no son todas del NASDAQ (hay valores del Dow Jones como Boeing o Caterpillar); es la etiqueta del conjunto de datos, no conviene darle más peso (**Nota**).
- Notebook 2 selecciona 3 columnas (`IBM NASDAQ`, `Intel NASDAQ`, `Microsoft NASDAQ`; hay una cuarta, Cisco, comentada) (02, celda 6), y separa por fecha: entrenamiento `Date < 2014-01-01` (156 filas, hasta 2013-12-29) y test `Date >= 2014-01-01` (121 filas, desde 2014-01-05) (02, celdas 7-8, 22). La serie principal es Microsoft (`SERIE_PRINCIPAL`, celda 9), que sube de ≈28,6 a ≈52,5 con una tendencia clara (figura de la celda 10).
- Escala: el notebook 1 divide por 1000; el notebook 2 **no** escala los precios (22-190 dólares). Esto cambia el tamaño de recompensas y pérdidas (ver §9).

*Fuente: 01-AR_trading_prueba celdas 2-5, 7; 02-AR_trading_multivariable celdas 2-10, 22; PC5 celdas 4-7; archivos CSV.*

---

## 3. El estado: ventana de log-retornos
**Idea.** En lugar de dar el precio al agente, se le da la historia reciente como rendimientos logarítmicos, que son estacionarios (aproximadamente) y están en torno a 0, de modo que sirven de entrada a una red sin normalización adicional. El "estado" es la ventana de los últimos `window_size` retornos terminando en el día actual (01, celda 5: "Estado: ventana que contiene la evolución de la serie financiera los últimos días").

$$
r_i=\ln\frac{x_i}{x_{i-1}},\qquad S_t=(r_{t-w+1},\dots,r_t)\in\mathbb{R}^{w}
$$

`getState(data, t, window_size)` (01, celda 5):
1. Toma el bloque de precios `data[max(t-w,0) : t+1]` (hasta $w+1$ precios, incluido el de hoy).
2. Si hay menos de $w+1$ (primeros días), rellena por delante con `data[0]` (padding), de modo que los retornos del relleno son 0.
3. Calcula `np.log(block[1:] / block[:-1])` y lo devuelve como matriz fila `(1, w)`, listo para `model.predict`.

El comentario del código recuerda la propiedad que justifica los log-retornos: $\ln(x_1/x_0)+\ln(x_2/x_1)+\ln(x_3/x_2)=\ln(x_3/x_0)$, los retornos se suman en el tiempo.

Ejemplo de la celda 8 (`t=5, window_size=10`): `[0, 0, 0, 0, 0, 0.0489, -0.0106, -0.0266, -0.0019, 0.0038]`; los 5 primeros ceros son el relleno y 0,0489 = ln(1,3476/1,2833) es el primer retorno real.

**Versión multivariable** (02, celda 11). Se aplica lo mismo a un `DataFrame` con $N$ series: el bloque es una matriz $(w+1)\times N$, se hace el log-cociente fila a fila (padding repitiendo la primera fila) y se aplana con `reshape(1,-1)`. El estado tiene dimensión $w\cdot N$: con $w=5$ y $N=3$ da `(1, 15)` (02, celda 18). El orden de aplanado es por tiempo y, dentro de cada instante, por serie (IBM, Intel, Microsoft); la red densa no lo usa, pero importa si luego se quiere remodelar a $(w, N)$ para una LSTM (**Nota**).

**Qué falta en este estado.** No contiene la posición/inventario del agente: ver §9, incoherencia 5, y PC5 (§8), que lo añade.

*Fuente: 01-AR_trading_prueba celdas 5-8; 02-AR_trading_multivariable celdas 11, 18; PC5 celda 7.*

---

## 4. El entorno con inventario (`Env`)
**Idea.** Una partida de entrenamiento es recorrer toda la serie de entrenamiento día a día: "volver al 2 de enero de 2001 y hacer trading hasta diciembre de 2010" (01, comentario de la celda 9). El agente mantiene un inventario (lista de precios de compra) y solo cobra recompensa cuando vende.

**Acciones (2)** (01, celda 9): `0 = comprar un paquete`, `1 = vender un paquete`.

```python
def step(self, action):
    reward = 0
    if action == 0:                              # comprar
        self.inventory.append(self.data[self.t]) # guardo el precio de compra
    elif action == 1 and len(self.inventory) > 0:# vender (solo si hay algo)
        precio_compra = self.inventory.pop(0)    # el paquete MÁS ANTIGUO (FIFO)
        reward = self.data[self.t] - precio_compra
    self.t += 1
    done = (self.t == self.final_t)
    S_ = getState(self.data, self.t, self.window_size)
    return S_, reward, done
```

Reglas, tal como están (01, celda 9; 02, celda 12; PC5, celda 8):
- **Inventario:** lista de precios de compra; el más antiguo es `inventory[0]` y el más reciente `inventory[-1]` (comentario de PC5, celda 8). Se vende siempre el más antiguo (FIFO).
- **Sin límite de tamaño:** se puede comprar un paquete cada día sin tope.
- **Recompensa:** 0 al comprar; al vender, $r_t = p_t - p_{\text{compra}}$ (precio de hoy menos precio de compra del paquete vendido, en las unidades del precio, es decir, ya dividido por 1000 en el notebook 1). El beneficio solo se "realiza" al vender.
- **Vender con el inventario vacío:** no ocurre nada y la recompensa es 0 (no hay penalización; el PC5 pide añadirla).
- **Sin comisiones** ni costes de transacción.
- **Fin de episodio:** `done` cuando se llega al último precio. El notebook 1 calcula `done` tras incrementar `t` (`t == final_t`); PC5 y el notebook 2 lo calculan antes (`t == final_t-1`); el resultado es el mismo número de pasos: `final_t = len(data)-1` (2514 pasos en entrenamiento del S&P; 155 en el entrenamiento semanal del notebook 2).
- **Sin look-ahead:** la acción del día $t$ se ejecuta al precio `data[t]`, que es el último precio que ve el estado (el estado termina en `t`). El estado siguiente ya incluye `data[t+1]`.
- El inventario al final del episodio **no se liquida**: las compras no vendidas no cuentan nunca. Por eso "comprar siempre" y "vender siempre" dan ambos beneficio 0 (enunciado PC5, celda 1).

Un detalle de diseño: el entorno devuelve `(S_, reward, done)`, no la interfaz de `gymnasium` (`obs, reward, terminated, truncated, info`) del Taller 2. Es el mismo patrón `reset/step` que el profesor ya había resaltado en otros entornos ("siempre tiene que haber un método llamado step o algo así", T30 ~01:14:54-01:15:25).

*Fuente: 01-AR_trading_prueba celda 9; 02-AR_trading_multivariable celda 12; PC5 celdas 1, 8; T30 ~01:14:40-01:15:25.*

---

## 5. El agente DQN
**Idea.** Es el DQN con experience replay del notebook de `catch` ("remember del notebook catch", comentario de 01, celda 17), adaptado a un estado vectorial. La red recibe el estado y devuelve un valor $Q(s,a)$ por acción; se entrena por regresión hacia un objetivo de Bellman calculado con la propia red.

**Red (notebook 1)** (01, celdas 12, 14, 15): `Dense(8, relu)` con `input_dim=window_size=20` + `Dense(2)` sin activación (186 parámetros: $20\cdot8+8+8\cdot2+2$); pérdida MSE; optimizador **SGD con learning_rate 0.001** (la línea de Adam está comentada). La salida lineal es lo correcto porque se estiman $Q$ que pueden ser negativos: el profesor lo planteó como pregunta en clase y concluyó que no debe haber activación en la salida porque "es un problema de regresión" (T30 ~00:33:07-00:33:48).

**Red (notebook 2)**: `Dense(8, relu)` con `input_dim = WINDOW_SIZE*N_series = 15` + `Dense(2)`, Adam(`learning_rate=0.001`), 146 parámetros (02, celdas 15-16).

**Alternativa LSTM (comentada)** (01, celda 13): `Reshape((10,1), input_dim=window_size)` → `LSTM(units=5)` → `Dense(2)`, Adam. Es una celda de texto entrecomillada (no se ejecuta) y fija `window_size = 10`, a diferencia del 20 de la red densa (ver §9).

**Escala.** El profesor avisó de que hay que cuidar la escala de entradas **y de salidas**: normalizar las entradas pero olvidar que $Q$ estima la recompensa total futura (que puede ser de miles) "echa completamente a la red" (T30 ~00:34:09-00:34:56). En el notebook 1 el `/1000` del precio y el uso de log-retornos mantienen ambas escalas pequeñas (pérdidas del orden de 1e-3).

**Clase `Agent`** (01, celda 10):
- Hiperparámetros por defecto: $\gamma=0.95$, $\epsilon_{ini}=1$, $\epsilon_{min}=0.01$, `epsilon_decay=0.995` (multiplicativo por episodio), `memsize=1000` (el notebook 1 usa 10000 al instanciarlo).
- `memory = deque(maxlen=memsize)` guarda tuplas `(state, action, reward, next_state, done)`.
- `choose_action`: con probabilidad $\epsilon$ (y si no está en modo evaluación) acción aleatoria; si no, `argmax` de $Q$. En evaluación (`is_eval=True`) nunca explora.
- `state_size` y `action_size` se leen del modelo (`input_shape[-1]`, `output_shape[-1]`), de modo que la misma clase vale para 2 o 3 acciones y para cualquier tamaño de estado.

**Experience replay eficiente (`get_batch` + `expReplay`).** Se muestrean `batch_size` transiciones al azar de la memoria (con reemplazo, `np.random.randint`), se predice en bloque y se sustituye solo la columna de la acción realizada:

$$
y_i=\begin{cases} r_i & \text{si } done_i\\ r_i+\gamma\max_{a'}Q(s'_i,a';\theta) & \text{si no}\end{cases},\qquad
\text{target}_i[a_i]=y_i,\ \ \text{target}_i[a\neq a_i]=Q(s_i,a;\theta)
$$

Como para las acciones no tomadas el objetivo es la propia predicción, su error es 0 y no generan gradiente (comentario del código: "the actions not taken [have] the same value as the prediction to not affect them"). Después `train_on_batch(inputs, targets)` da un único paso de gradiente con MSE y devuelve la pérdida. El docstring dice que esta función es "la misma [que la anterior] pero mucho más eficiente".

**Versión lenta (`expReplay_no_optimo`)**: recorre el minilote muestra a muestra con `predict` + `fit(epochs=1)`. En los notebooks 1 y PC5 está **rota** y nunca se llama (ver §9); en el notebook 2 está corregida y renombrada `expReplay_` (02, celda 13) y tampoco se usa.

**Nota:** el objetivo se calcula con la misma red que se entrena (no hay red objetivo separada) y las transiciones se muestrean con reemplazo; es el DQN "básico" de clase, sin las estabilizaciones del artículo de Mnih et al. Esto se ve en la práctica en la inestabilidad que el profesor describió al comparar con Q-learning (§10).

*Fuente: 01-AR_trading_prueba celdas 10-16; 02-AR_trading_multivariable celdas 13-17; T30 ~00:33:07-00:34:56.*

---

## 6. Entrenamiento, evaluación y resultados del notebook 1
**Idea.** Se entrena durante pocos episodios largos (cada episodio = diez años de datos) con un replay espaciado cada 64 pasos, y se evalúa sobre el año 2011 con $\epsilon=0$.

**Configuración** (01, celdas 11, 16):

| Parámetro | Valor |
|---|---|
| Episodios | `episode_count = 20` (el bucle hace `range(episode_count+1)` = 21 episodios) |
| `window_size` | 20 |
| Red | 20→8→2, SGD lr 0.001, MSE |
| `memsize` | 10000 (el comentario del código dice "1000") |
| `batch_size` | 128 (alternativas comentadas: 32, 1024) |
| `training_ratio` | 64 |
| $\gamma$, $\epsilon_{ini}$, $\epsilon_{min}$, decay | 0,95; 1; 0,01; 0,995 por episodio |
| Pasos por episodio | 2514 |

**Bucle** (01, celda 17):

```
para e en 0..episode_count:
    state = env.reset(); done = False; counter = 0; total_profit = 0
    mientras no done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        total_profit += reward
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        counter += 1
        si len(memory) > batch_size  y  counter % training_ratio == 0:
            loss = agent.expReplay(batch_size)        # 1 paso de gradiente
    si epsilon > epsilon_min: epsilon *= epsilon_decay
    guardar modelo (cada episodio)
```

**Training ratio.** `training_ratio = 64` significa un paso de gradiente cada 64 pasos del entorno, no uno por paso. Con un lote de 128, cada transición nueva "se usa" unas 2 veces en promedio (128/64) (**Nota:** cálculo propio). Del orden de 39 actualizaciones por episodio de 2514 pasos (el primer replay del episodio 0 sale en `counter=192`, porque hasta `counter>128` la memoria no supera el lote, salida de la celda 17). El profesor comentó, con el ejemplo de `catch`, que entrenar en cada acción es costoso y que una opción es sacar el replay del bucle `while` y hacerlo al final de la partida, compensando con más ejemplos por actualización o con un lote mayor; en el experimento en directo "no es mucho más rápido" (T30 ~03:31:15-03:35:04). El contador `training_ratio` del código de trading es la versión intermedia entre "entrenar en cada paso" y "entrenar al final del episodio".

**Resultados del notebook 1.**
- Pérdida del lote: ≈ 0,0010 al empezar (episodio 0) y ≈ 0,0027 en el episodio 20 (salida de la celda 17). La pérdida no baja: sube y se estabiliza, lo esperable con un objetivo que se mueve (el objetivo depende de la propia red, el profesor lo llamó "un problema de regresión dinámico", T30 ~00:42:40-00:43:08).
- Último episodio: `Total Profit: $17.12`, $\epsilon=0{,}9000874$ (celda 17).
- **Observación importante** (**Nota:** cálculo propio): con `epsilon_decay=0.995` por episodio, tras 21 episodios $\epsilon=0{,}995^{21}\approx0{,}90$, de modo que **el agente entrenó eligiendo acciones aleatorias el ~90 % del tiempo**; y con SGD de lr 0,001 y unas 800 actualizaciones en total, la red apenas se entrena. El beneficio de $17,12 es, en buena parte, el de una política casi aleatoria.
- **Evaluación sobre 2011** (celdas 18-24): se carga el modelo guardado (`models/model_ep.keras`), `is_eval=True`, `Env("GSPC_2011", 20)`, se parte de `state = environment.reset()` (estado de ceros, celda 21) y se registra el precio y la acción de cada día. Resultado: `GSPC_2011 Total Profit: -$1.38` (celda 23).
- La figura de la celda 24 (serie de 2011 con puntos verdes = comprar y rojos = vender, ≈ 250 días, precio 1,10-1,36) muestra compras y ventas entremezcladas sin patrón visible, sin tendencia clara en el año: el agente no ha aprendido una política con sentido (**Nota:** lectura de la figura).
- El "$" de `formatPrice` es engañoso: son unidades de (índice/1000) por paquete, sumadas sobre todos los paquetes vendidos (**Nota**).
- **No hay línea base** en el notebook: ni agente aleatorio ni *buy & hold*. El Taller 2 sí las exige (enunciado, ejercicio 2).
- La figura de la celda 15 solo es el diagrama de la red (`plot_model`): 20 → 8 → 2.

**Qué dijo el profesor sobre el entrenamiento.** Importaron `SGD` además de `Adam` para probar ambos (T30 ~01:11:58-01:12:13). Sobre el nº de episodios, hablando del taller de Q-learning que los alumnos hacían con otro profesor: con 10 partidas no se entrena y con 20 ya "estaréis un montón"; allí una partida es un periodo de inversión (T30 ~02:40:41-02:40:57). Se comentó también que el guion habla de 500 precios por episodio, que el profesor interpreta como unos 500 días (T30 ~02:41:00-02:41:23).

*Fuente: 01-AR_trading_prueba celdas 11, 16-24; T30 ~00:42:40-00:43:08, ~01:11:58-01:12:13, ~02:40:41-02:41:23, ~03:31:15-03:35:04.*

---

## 7. El notebook multivariable (`02-AR_trading_multivariable`)
**Idea.** Comprobar que el mismo agente puede recibir más información que la propia serie operada: los log-retornos de varias series entran juntos en el estado, aunque solo se opera la principal. El profesor lo había anunciado en clase: con el Ibex serían 35 valores durante los últimos días, y se podrían añadir índices de otros mercados o indicadores (T30 ~00:32:36-00:33:03).

**Diferencias respecto al notebook 1** (02):
- Datos en `DataFrame`; separación temporal 2011-2013 / 2014-2016 (celdas 7-8). Es la separación correcta: el test es posterior al entrenamiento.
- Estado: ventana $w=5$ × 3 series = 15 entradas (celdas 11, 15, 18).
- El `step` solo usa `SERIE_PRINCIPAL` (Microsoft) para comprar, vender y calcular la recompensa (celda 12). Incluye una variante inactiva (`if True: ... else:`) que interpreta "vender" como vender **todo** el inventario sumando todas las diferencias; no se usa.
- Entrenamiento: `episode_count=1000`, memoria 1000, batch 128, `training_ratio` 64; con 155 pasos por episodio solo hay 2 actualizaciones por episodio (counter 64 y 128) (celdas 14, 19, 20). Guarda el modelo cada 10 episodios.
- **Entrenamiento interrumpido:** la ejecución se detuvo con `KeyboardInterrupt` hacia el episodio ~128 de 1000 (última salida, celda 20), así que el modelo evaluado es el guardado en el episodio 120 (con $\epsilon\approx0{,}995^{121}\approx0{,}55$ en ese momento, **Nota:** cálculo propio). La pérdida baja de ≈2,0 (episodio 1) a ≈0,91 (episodio 128): está en la escala del dólar, porque los precios no se dividen por 1000.
- Episodio 0 (aleatorio, $\epsilon\approx1$): beneficio $5,80 (celda 20).

**Resultados** (celdas 24-25):
- Train (modelo en entrenamiento, `Env(df_train)`): beneficio $15,20.
- Test (2014-01 a 2016-06, 121 semanas): beneficio $27,45.
- Las figuras (celdas 24 y 25) muestran que el agente **casi siempre elige vender** (rojo) y compra solo en unos pocos puntos (≈una docena en train y ≈20 en test, contados a ojo), normalmente tras caídas del precio. Como solo se cobra al vender, una política "comprar poco, vender siempre" realiza beneficio sobre los pocos lotes que compró: si el precio sube después de comprar, gana. Es una política coherente con una serie de tendencia alcista como Microsoft, no una prueba de que haya aprendido a predecir (**Nota:** interpretación propia).
- Sigue sin haber comparación con *buy & hold*. **Nota:** comprar una acción de Microsoft el 2014-01-05 (36,91) y mantenerla hasta el 2016-06-05 (52,48) da +15,57 por acción; el $27,45 del agente suma lotes comprados en distintos momentos, y no es comparable sin fijar el capital invertido.
- Figura de la celda 10: Microsoft semanal de 2011 a 2016, tendencia alcista con ruido (28→52). Figura de la celda 17: diagrama de la red 15→8→2.

*Fuente: 02-AR_trading_multivariable celdas 6-12, 14-25; T30 ~00:32:36-00:33:03.*

---

## 8. PC5: las tres extensiones del ejercicio
**Idea.** PC5 parte del código de clase y pide arreglar sus defectos de modelado. Estas son las tres extensiones del enunciado (PC5, celda 1):

1. **Liquidar al final.** "Cuando el agente llegue al final de la partida (`done=True`) se vendan todos los paquetes de acciones que tenga en su cartera al precio de mercado en ese momento." Motivo: con la recompensa solo al vender, "comprar siempre" y "vender siempre" valen lo mismo, 0.
2. **Límite de cartera y posición en el estado.** El inventario no puede pasar de **5 paquetes**. El estado S debe incluir además el número de acciones en cartera dividido entre 5 (valor en [0,1]). Hay que modificar `reset` y `step` para que extiendan el array de salida con ese valor.
3. **Tercera acción "Mantener"** (ni comprar ni vender) y **recompensa negativa si intenta vender con la cartera vacía**. Hay que comprobar que, una vez entrenado, el agente nunca intenta vender con cartera vacía.

El resto del notebook es el código del notebook 1 con estos cambios de configuración (PC5, celdas 3-22): `window_size=10`, red 10→8→2 con `Adam(lr=0.001)`, `memsize=1000`, `epsilon_decay=0.95`, `batch_size=128`, `training_ratio=64`, 20 episodios sobre `GSPC_2001-10` y evaluación sobre `GSPC_2011`. **No hay salidas guardadas** ni figuras: es una plantilla sin ejecutar.

**Diseño propuesto** (**Nota:** no está en el material; es la forma natural de cumplir el enunciado):

- *Liquidación* (extensión 1): en el paso en que `done` pasa a ser `True`, vender todo el inventario al precio de `data[self.t]` y sumar al `reward` $\sum_i (p_t - p_i)$; después `inventory = []`. Esa suma entra en la última transición que se guarda en memoria.
- *Cartera máxima* (extensión 2): `elif action==COMPRAR and len(inventory) < 5` (si está llena, comprar es un no-op, con o sin penalización pequeña; el enunciado no lo fija). El estado pasa de $w$ a $w+1$ componentes: `np.append(getState(...), len(inventory)/5)`. Hay que cambiar también el `input_dim` de la red (`window_size+1`), el `reset` (que usa la variable global `window_size`, ver §9) y las 3 salidas del `Dense`.
- *Mantener y penalización* (extensión 3): acción `2 = mantener`; en `step`, si la acción es vender y el inventario está vacío, `reward = -c` (por ejemplo del tamaño de la desviación típica de un paso de recompensa, para que sea una señal perceptible a esta escala; hay que ajustarlo). Con 3 salidas el modelo tiene $11\cdot8+8+8\cdot3+3=123$ parámetros.
- *Comprobación* de que no vende con cartera vacía: contar en la evaluación cuántas veces `action==vender and len(inventory)==0` (debería ser 0 o casi 0). Una forma más robusta es enmascarar las acciones inválidas en `choose_action`, pero entonces la comprobación pedida deja de tener sentido: lo que se prueba es que la red lo ha aprendido (**Nota**).
- La extensión 2 es la que arregla el problema de información del estado (el agente no sabía si vender tendría efecto); la 3 lo arregla en recompensa; la 1, en la medida de resultados.

*Fuente: PC5 celdas 0-23 (enunciado en la celda 1).*

---

## 9. Bugs e incoherencias del código de clase
Cada punto da lo que hace el código y la corrección razonada.

| # | Dónde | Qué ocurre | Corrección / comentario |
|---|---|---|---|
| 1 | `expReplay_no_optimo` (01 c10; PC5 c9) | `self.model.predict(state, state=0)` y `self.model.fit(..., wpochs=1, ...)`. Son erratas: `predict` no tiene el argumento `state` (debería ser `verbose=0`) y `fit` no tiene `wpochs` (es `epochs`). Daría `TypeError` si se llamara. | Es código muerto: nunca se llama (se usa `expReplay`). En 02 c13 está corregido (`verbose=0`, `epochs=1`) y renombrado `expReplay_`. Además, esta versión no muestrea al azar: toma las últimas `batch_size-1` transiciones, correlacionadas, y hace un `fit` por muestra. |
| 2 | `Adam(lr=0.001)` (PC5 c11) | `lr` es el nombre antiguo del argumento. En Keras recientes el argumento es `learning_rate` (01 c12 y 02 c15 ya usan `learning_rate`); según la versión, `lr` da aviso o error. | Usar `Adam(learning_rate=0.001)`. |
| 3 | `Agent.__init__` (01 c10; PC5 c9; 02 c13) | Recibe `lr` (o `learning_rate`) y no lo usa: el optimizador se fija al compilar el modelo fuera de la clase. También `self.inventory` en `Agent` es una variable muerta (el inventario real está en `Env`). | Cosmético; no cambiar `lr` en el agente no tiene ningún efecto. |
| 4 | `Env.reset(self, t=0)` (01 c9) | El parámetro `t` solo se asigna a `self.t`, pero el estado se calcula con `getState(..., t=0, ...)` fijo: reset en `t≠0` daría un estado de otro día. En PC5 c8, `reset` llama a `getState(self.data, 0, window_size)` con la **variable global** `window_size` en vez de `self.window_size`. | Usar `self.window_size` y `getState(self.data, t, ...)`. |
| 5 | **Estado sin posición** (01, 02, PC5) | El estado solo tiene retornos pasados; no incluye el inventario. La recompensa de vender depende del precio de compra de los paquetes (oculto) y de si hay inventario (si no, vender no hace nada). El entorno no es markoviano en $s$: la misma $S_t$ puede tener $Q$ muy distintos según la cartera. | PC5 extensión 2 (añadir `len(inventory)/5`). El `SimpleTradingEnv` ya incluye la posición. |
| 6 | Recompensa solo al vender | Comprar no da señal (0); el beneficio no realizado no cuenta; el inventario final se pierde; no hay penalización por vender sin inventario, que se confunde con "no hacer nada"; el inventario no tiene límite. | PC5 extensiones 1-3. Un diseño con recompensa por paso (variación de valor de la cartera) evita estos problemas (ver §11). |
| 7 | Fin de episodio como `done` real | Se pasa `done=True` también al llegar al último precio (límite de tiempo, no fin real del problema), lo que anula el término $\gamma\max Q(s')$ en el último paso. | Efecto pequeño con $\gamma=0{,}95$. **Nota:** en gymnasium, esto es `truncated`, no `terminated`; en el Taller, D-1.2 (cerrada) lo mantiene como terminal (ver *Aplicación al Taller 2*). |
| 8 | **02 celdas 24 y 25: evaluación con `state` antiguo** | No se hace `state = environment.reset()` antes del bucle (en 01 c21 sí). La primera acción de la evaluación usa el `state` que quedó de la última iteración del entrenamiento (en c24) o de la evaluación anterior (en c25), que no es el estado inicial del nuevo entorno (el entorno nuevo sí empieza en $t=0$, con ceros). La dimensión coincide `(1,15)`, así que no da error y pasa desapercibido. | Hacer `state = environment.reset()` antes de cada evaluación. Afecta solo a una decisión de cada serie, pero es un fallo de método. Además, c24 imprime `df_test.columns` aunque evalúa `df_train` (cosmético). |
| 9 | Entrenamiento del 02 interrumpido | `episode_count=1000`, pero se paró con `KeyboardInterrupt` en el episodio ~128 y se guarda cada 10 episodios; el modelo evaluado es el del episodio 120 y se desconoce el resto de la curva. | Para comparar, guardar el último modelo o el mejor en validación y registrar la curva de beneficios por episodio (`total_profits` se llena pero nunca se grafica). |
| 10 | Exploración casi nula del aprendizaje (01) | `epsilon_decay=0.995` por episodio y solo 21 episodios: $\epsilon\approx0{,}90$ al final. El notebook imprime "Episode 20/20" pero hace 21 episodios (`range(episode_count+1)`). | Con pocos episodios largos, decaer $\epsilon$ **por paso** o con un decay mucho más rápido (como en PC5: 0,95 → $\epsilon\approx0{,}34$ tras 21 episodios). |
| 11 | Comentarios y parámetros incoherentes | 01 c16: el comentario dice "de las 1000 que tiene la memoria" con `memsize=10000`. 01 c13 (LSTM comentada): `Reshape((10,1), input_dim=window_size)` con 10 fijo mientras la red densa usa `window_size=20`; además `input_dim` en `Reshape` está obsoleto en Keras recientes (se usa `Input(shape=...)`). | Mantener `window_size` como única fuente de verdad. |
| 12 | PC5 c15: `agent.model.save("models/model_ep")` y `load_model('models/model_ep')` (c18-19) | Sin extensión. En Keras 3 guardar un modelo exige `.keras` (o `.h5`); 01 y 02 ya usan `model_ep.keras`. | Usar `models/model_ep.keras`. (**Nota:** no hay salidas de PC5, no se sabe cómo falló.) |
| 13 | Escalas distintas | 01 divide el precio por 1000 (recompensas ≈ 0,01, pérdidas ≈ 1e-3); 02 no escala (recompensas de dólares, pérdida ≈ 2). El profesor insistió en cuidar la escala de la salida (T30 ~00:34:09). El "$" de `formatPrice` en 01 no son dólares. | Normalizar siempre recompensas y estados. |
| 14 | Muestreo con reemplazo y objetivo con la misma red | `np.random.randint` puede repetir transiciones en un lote; no hay red objetivo. | Es el DQN básico de clase; ver el documento de DQN. |

*Fuente: celdas citadas en la tabla; T30 ~00:34:09.*

---

## 10. Consejos del profesor sobre trading y RL aplicado a inversión
**Idea.** En T30 el profesor mezcló la teoría con su experiencia en un proyecto de ayuda a la inversión en fondos; estas son las ideas que se pueden trasladar (parafraseadas; la transcripción es ruidosa).

- **Q-learning primero, luego Deep RL.** En su proyecto empezaron directamente por Deep RL y el primer resultado "después de meses fue un desastre": la red no era estable, no convergía, y cuando convergía predecía mal. Volvieron atrás y simplificaron (T30 ~00:16:00-00:16:46, ~02:53:32-02:54:05).
- **No empezar con muchas variables.** Partieron de ≈100 variables (indicadores con ventanas de 20, 40, 50 días) y acabaron con 7 indicadores de mercado agrupados por *clustering* en 5 estados de mercado; con más estados la red tarda más en converger (T30 ~02:52:49-02:54:45). "Inicialmente reduce al máximo tu problema."
- **Simplificar la acción.** No dejaron que el agente diseñase la composición de la cartera (demasiadas acciones posibles): definieron dos composiciones (más y menos arriesgada) y el agente solo decide cuándo cambiar de una a otra: 2 acciones (T30 ~02:54:46-02:56:27). El estado incluía además cómo le iba a la cartera (si estaba en pérdidas): en total unos 20 estados y 2 acciones (T30 ~02:56:28-02:56:51). Es decir, **el estado incluía la situación de la cartera**, lo que el código de clase no tiene (§9, #5).
- **Episodio = periodo de uso.** Si el modelo se va a usar un año, el episodio debe durar un año (T30 ~01:14:40-01:14:56). En su proyecto simulaban con datos pasados periodos de un año, una y otra vez, y recomendaron **empezar cada episodio en fechas aleatorias** en vez de volver siempre al mismo inicio, para no sobreajustar a una época (T30 ~02:56:51-02:57:41).
- **Recompensa.** Cada vez que se opera hay recompensa en el tiempo, no solo al final; se puede dar la composición de cartera en cada instante. Un enfoque alternativo es solo recompensa final, pero con poca inversión se da poca información durante la partida (T30 ~03:28:10-03:29:15). Las **comisiones** se pueden meter como recompensa negativa al operar (se paga) y positiva en las liquidaciones; el profesor preguntó a los alumnos si las estaban teniendo en cuenta (T30 ~03:28:32-03:29:07). En problemas complejos, dar refuerzos intermedios (el ejemplo era un coche autónomo) es mucho más fácil de aprender que un premio único al final, y "que funcione o no tu proyecto depende mucho de cómo hayas diseñado tu refuerzo" (T30 ~03:29:36-03:30:27). En algunos casos el refuerzo lo genera el propio agente internamente (T30 ~00:19:11).
- **Interpretabilidad.** Con una tabla Q de pocos estados de mercado se puede explicar a un inversor qué recomienda el sistema en cada situación (T30 ~02:57:47-02:58:21): una ventaja de Q-learning que se pierde con redes.
- **Simplifica a saco al principio**: pocas acciones, pocos estados, y desarrolla desde ahí; RL "no es como aprendizaje supervisado, es mucho más sensible a los parámetros" (T30 ~02:58:21-02:59:00).
- **Aparte: algoritmos genéticos** para optimizar carteras (pesos de activos, tamaños de ventana). Combinan individuos (p. ej. 5 activos de un padre y 5 del otro, y se normaliza a suma 1), con mutación y una función *fitness* no necesariamente derivable; no se usan para entrenar redes por lentitud (T30 ~03:04:53-03:08:28). No entra en el Taller 2.
- **Entornos de bolsa ya hechos.** Al final de la clase el profesor mostró un entorno de trading de Gym "ya programado" (el nombre sale deformado en la transcripción, "Jim Eny Trading"; **Nota:** probablemente un paquete tipo gym-anytrading) para probar algoritmos propios (T30 ~04:03:45-04:04:46).

*Fuente: T30 ~00:10:32-00:12:28, ~00:15:58-00:19:11, ~02:52:09-02:59:00, ~03:04:53-03:09:22, ~03:28:10-03:30:27, ~04:03:45-04:04:46.*

---

## 11. Comparación con `SimpleTradingEnv` (Taller 2)
**Idea.** El entorno del Taller 2 (`taller_trader2`, celda 4) corrige varios defectos del de clase (posición en el estado, recompensa por paso, comisión explícita), pero simplifica otros: una sola posición todo-o-nada y un estado de dos números.

| Aspecto | Entorno de clase (`Env`) | `SimpleTradingEnv` (Taller 2) |
|---|---|---|
| **Interfaz** | `reset()` → estado; `step(a)` → `(S', r, done)` | `gymnasium`: `reset()` → `(obs, {})`; `step(a)` → `(obs, r, terminated, truncated, info)` |
| **Acciones** | 2 (01, 02) o 3 (PC5): comprar un paquete, vender un paquete, [mantener] | 3: 0 = vender, 1 = mantener, 2 = comprar (nota: orden distinto al de PC5, que pone comprar = 0) |
| **Inventario** | Lista de paquetes (FIFO), sin límite (PC5: máx. 5) | Posición binaria 0/1: o todo el capital en el activo o todo en efectivo (capital inicial 1000) |
| **Acción inválida** | Vender con inventario vacío: no pasa nada, recompensa 0 (PC5: penalización) | Comprar con posición 1 o vender con posición 0: no-op, sin penalización |
| **Estado** | Ventana de $w$ log-retornos (×$N$ series); sin posición (PC5: + inventario/5) | `[tendencia, posición]`, continuo: tendencia $=\frac{p_t-\bar p_{t-5..t}}{\bar p}$ recortada a [-1,1] sobre 6 precios; posición 0/1 |
| **Recompensa** | 0 salvo al vender: $p_t-p_{compra}$ (precio, no relativa) | Densa por paso: variación relativa del precio $\frac{p_{t+1}-p_t}{p_t}$ si se mantiene posición tras la acción, más $-0{,}001$ si se compra o se vende |
| **Comisión** | Ninguna (el profesor la mencionó como mejora, T30 ~03:28:32) | 0,1 % por transacción, descontada del efectivo y también restada de la recompensa |
| **Fin de episodio** | `done` al llegar al último precio; inventario sin liquidar (PC5: se liquida) | `terminated=True` en el último precio; `truncated` siempre `False`; la posición no se liquida pero la recompensa ya refleja el valor de mercado |
| **Duración del episodio** | 2514 pasos (S&P 2001-2010) o 155 (semanal de entrenamiento) | 494 pasos con 500 precios (empieza en `current_step=5`; termina en `len(prices)-1`) |
| **Datos** | Serie real, fija; mismo episodio cada vez | Series sintéticas de 4 generadores, una nueva por episodio (enunciado, ejercicio 1) |
| **Escala de recompensa** | ≈ 0,01 (S&P/1000) o dólares (02) | ≈ 1e-3 por paso |
| **Agente** | DQN denso (sin red objetivo) | A implementar: DQN y A3C |
| **Línea base** | Ninguna | Aleatorio y *buy & hold* (celdas 19-21) |

**Diferencias que importan.**
- La posición ya está en el estado, así que la recompensa por paso (cambio de valor de la cartera) permite aprender sin esperar a una venta: no existe el problema de PC5, ni la necesidad de liquidar.
- La recompensa de `SimpleTradingEnv` se parece al *mark-to-market* del que habló el profesor ("le estás dando en cada momento la composición de tu cartera si mejora o no", T30 ~03:28:21).
- En clase el agente puede comprar varios paquetes; en el Taller solo hay dos estados de cartera. El problema es más sencillo, pero la política óptima se reduce a "cuándo estar dentro".
- El estado de dos números del Taller **resume** la historia (una media de 6 precios); el de clase da la ventana completa de retornos (más información, más dimensiones).
- **Nota:** escala del estado en el Taller: la tendencia es del orden de 0,01-0,05 mientras la posición es 0/1; D-1.1 (cerrada) la divide por su σ de train, en línea con el aviso de T30 ~00:34:09 sobre escalas.

*Fuente: 01-AR_trading_prueba celda 9; PC5 celdas 1, 8; taller_trader2 celdas 3, 4, 19-22; T30 ~03:28:10-03:29:07. La tabla y las notas de escala son comparación propia.*

---

## Aplicación al Taller 2
Para el Taller (DQN y A3C, `SimpleTradingEnv` con observación `[tendencia, posición]`, 3 acciones, comisión 0,1 %, recompensa ≈1e-3 por paso, 494 decisiones por episodio, series sintéticas) lo del trading de clase aporta lo siguiente. Todo lo que sigue son propuestas de extras (**Nota**).

**Qué se reutiliza tal cual.**
- La estructura `Agent` con `memory`, `get_batch` y `train_on_batch`, la salida lineal y `choose_action` con `is_eval`; el argumento `training_ratio`. Se traduce a PyTorch (D-0.2), donde el *target* se calcula con `no_grad` y `gather` sin construir la matriz de salidas ([03_dqn.md](03_dqn.md), Aplicación). En el Taller conviene indicar qué proporción de actualizaciones por paso se usa y justificarlo (el enunciado pide justificar decisiones de diseño).
- Evaluar con $\epsilon=0$ en series nuevas de la misma función generadora y promediar sobre varias (enunciado, ejercicio 1) y **resetear siempre `state`** antes de evaluar (§9, #8).
- Registrar por episodio el beneficio/balance (`total_profits`) y graficarlo: sirve para el "tiempo de convergencia" que pide la tabla del ejercicio 4.
- Línea base de agente aleatorio y *buy & hold* (ya en el notebook del Taller; no estaban en clase).

**Qué conviene arreglar respecto al código de clase.**
- Decaimiento de $\epsilon$: por episodios de 494 pasos y con muchos episodios, evitar el caso del notebook 1 (ε≈0,9 al final).
- El fin de la serie (§9, #7) se trata como terminal, como en el Taller 1, y no como *truncation*: lo decide D-1.2, con la misma regla en DQN y A3C.
- Escalar las entradas y las recompensas (≈1e-3): la señal es pequeña y la red puede quedarse en $Q\approx0$. D-1.1 (cerrada) lo hace en el agente con `train_scales(mercado)`, sin tocar el entorno.
- No usar `lr=`: `learning_rate=`.

**Extras que se pueden ensayar** (cada uno es opcional y cuenta como "pruebas que van más allá de los mínimos", 10 % de la nota):
1. **Ventana de log-retornos como estado.** Sustituir (o ampliar) la tendencia de 6 precios por los últimos $w$ log-retornos de `getState`, más la posición: `[r_{t-w+1}, ..., r_t, posición]`. En el Taller esto da más información para detectar el régimen (generador 4) o la reversión a la media (generador 3); con $w$ de 5-20, pocos parámetros extra. Comparar contra el estado original en cada generador.
2. **Multivariable.** Con series sintéticas hay un solo activo, así que "multivariable" aquí significa varias *características* del mismo precio (retornos, volatilidad en ventana corta, distancia a la media móvil larga). El profesor recomendó empezar con pocas variables y subir de una en una (T30 ~02:53:32); la tabla de comparación del Taller permite medir cuánto aporta cada una.
3. **LSTM (comentada en clase).** Probar la variante `Reshape → LSTM → Dense(n_acciones)` (01, celda 13) como alternativa a la red densa en DQN, con la ventana de retornos como secuencia $(w,1)$. Es más lenta y más inestable (el profesor avisó de la inestabilidad de Deep RL, §10); sirve como experimento, no como línea principal. En A3C, la LSTM exigiría mantener el estado oculto durante el episodio.
4. **Inicio aleatorio del episodio / series nuevas por episodio**: ya está pedido por el enunciado; es la versión sintética del consejo "empezar en fechas aleatorias".
5. **Cartera de más de un paquete (PC5)**: cambia el entorno del Taller, que fija 1 posición binaria; solo tiene sentido si se modifica `SimpleTradingEnv` y se documenta. No es prioritario.

**Riesgos.** El Taller compara con *buy & hold* y con el agente aleatorio sobre series sintéticas: en el movimiento browniano de deriva casi nula (generador 2, $\mu=0{,}0001$) no hay estrategia que bata al azar de forma fiable salvo reducir comisiones (no operar), y el análisis crítico del ejercicio 2 (razonar si la estrategia hallada es la esperada) es donde se obtiene nota (enunciado, evaluación). Ver los documentos de DQN y A3C de `docs/teoria/` para los algoritmos y los hiperparámetros.

*Fuente: síntesis de §3-§11; taller_trader2 celdas 3, 4, 22-24; propuestas propias marcadas como Nota.*
