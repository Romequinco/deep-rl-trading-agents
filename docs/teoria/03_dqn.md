# Deep Q-Network (DQN): de la tabla Q a la red neuronal

> **Fuentes:** **DRL1** (`docs/raw/DRL/transparencias/1-DQN.pdf`, "Aprendizaje por Refuerzo. Métodos Aproximados: DRL", 116 páginas en 52 grupos de animación, todas leídas como imagen). **T30** (transcripción de la clase del 30 de octubre, `docs/raw/transcripciones/`; tramos sobre DQN: ~00:02-01:50, ~02:20-02:58, ~03:15-03:36 y ~04:05-04:39). Notebooks de clase: **`1-catch`** (juego Catch con DQN, en Keras), **`1-cart_pole_QLearning`** (CartPole con tabla Q discretizada), **`2-cart_pole_DQN`** (CartPole con DQN). Artículos citados por el mazo: Mnih et al. (2013) y Mnih et al. (2015), *Human-level control through deep reinforcement learning*, Nature (DRL1 p29-p30).
>
> **Convenciones:** `(DRL1 pN)` = página N del mazo. `(T30 ~hh:mm)` = hora aproximada del cue, en paráfrasis de una transcripción automática y ruidosa. `(1-catch, celda N)` = celda del notebook. **Nota:** = contenido que no viene del material de clase (cálculos de comprobación, conocimiento externo, propuestas) o una errata detectada. Si en DRL1 aparece $Q'$ o $\hat Q$, en este documento es la red objetivo. El mazo escribe $r_t$ para el refuerzo recibido tras $a_t$; [00_fundamentos_mdp.md](00_fundamentos_mdp.md) escribe $R_{t+1}$ para lo mismo. Las erratas de las fuentes están reunidas en §14.
>
> **Otros documentos:** MDP y Bellman en [00_fundamentos_mdp.md](00_fundamentos_mdp.md); Q-Learning tabular en [02_td_sarsa_qlearning.md](02_td_sarsa_qlearning.md); sobreestimación, Doble DQN y duelo en [04_doble_dqn_y_duelo.md](04_doble_dqn_y_duelo.md); actor-crítico en [06_actor_critico_y_a3c.md](06_actor_critico_y_a3c.md); mapa de algoritmos y AlphaGo en [07_panorama_drl.md](07_panorama_drl.md).

---

## 1. Los límites de Q-Learning tabular

**Idea.** Q-Learning construye una tabla con una fila por estado y una columna por acción. Eso exige estados finitos y visitar cada uno muchas veces; en problemas reales no se cumple ninguna de las dos cosas.

**Qué es la tabla.** $Q(s,a)$ valora "cómo de bueno" es realizar $a$ desde $s$. Ejemplo del mazo, con la mejor acción de cada fila en verde (DRL1 p5):

| | $a_0$ | $a_1$ | $a_2$ |
|---|---|---|---|
| $S_0$ | 0,1 | **1** | 0,5 |
| $S_1$ | **0,95** | -0,2 | 0,3 |
| $S_2$ | 1,5 | **2** | -1 |
| $S_3$ | -0,2 | 0,1 | **0,4** |
| $S_4$ | **1,3** | -1,1 | 0,2 |

Si se ha aprendido bien, la política óptima es elegir en cada $s$ la acción con mayor $Q$ (DRL1 p5). $Q$ estima lo que se ganará desde ese estado en adelante, con descuento, si a partir de ahí uno se comporta de forma óptima (T30 ~00:04, ~00:05). La regla de actualización, tras observar $(s_t,a_t,r_t,s_{t+1})$, es (DRL1 p8, con la errata corregida, ver §14):

$$\Delta Q(s_t,a_t)=\alpha\,[\,r_t+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\,]$$

**Recordatorios del profesor antes de pasar a redes:**

- **Descuento.** $\gamma\in(0,1)$. Cuanto más cerca de 1, más lejos planifica el agente, porque los pesos futuros bajan más despacio. Tiene que bajar: en juegos que no terminan o con bucles, la suma de refuerzos podría ser infinita (T30 ~00:04-00:05).
- **Construir la tabla y usarla son cosas distintas.** Al usarla, se elige la acción con valor máximo. Al construirla no hace falta comportarse de forma óptima: puede ser mejor hacer "burradas", porque cuantas más cosas se prueban, más información se recoge (T30 ~00:06).
- **Q-Learning converge aunque se explore al azar,** si se recoge información suficiente (T30 ~00:06-00:07).
- **Supervisado frente a refuerzo.** Un modelo supervisado imita lo que ve en los datos: un coche entrenado así copia al conductor profesional. El aprendizaje por refuerzo aprende la estrategia óptima, sea cual sea la conducta con la que se exploró, siempre que la información sea suficiente (T30 ~00:07).

**Por qué falla en problemas reales** (T30 ~00:08-00:10):

1. **Con variables continuas hay infinitas filas.** Si se usan los datos exactos, la tabla no es finita. En el taller anterior hubo que partir la tendencia en franjas; en un coche autónomo no se puede (T30 ~00:08).
2. **Memoria.** Haría falta una memoria infinita o, en la práctica, inabarcable (T30 ~00:09).
3. **Visitas.** Aunque hubiera memoria, cada celda se actualiza por separado, así que hay que pasar por cada estado muchas veces. Y la misma situación exacta (posición, velocidad, temperatura, coches alrededor) no se repite (T30 ~00:09-00:10). El mazo lo dice igual: cada estado se visitará "0 o 1 veces a lo sumo" (DRL1 p12).
4. **Tamaño.** Una cámara de 100×100 píxeles con 256 niveles de gris da $256^{10000}$ estados (DRL1 p18). DRL1 p12 escribe $256^{100000}$, que es una errata (§14).

**Discretizar es la alternativa, con límites.** O se discretiza, o Q-Learning no es viable; y tampoco se pueden crear demasiados estados, porque el agente debe pasar muchas veces por cada uno (T30 ~00:10).

**Experiencia del profesor en un proyecto de inversión en fondos** (T30 ~00:11-00:16):

- Partían de indicadores generales con ventanas móviles y tendencias, se quedaban con 7 series y discretizaban los estados de mercado.
- K-means y las mezclas de gaussianas no les funcionaron bien: no garantizan "inercia", y el clúster podía saltar de un día a otro sin sentido.
- Usaron **modelos ocultos de Márkov** (HMM), que están pensados para series temporales y dan continuidad al estado.
- Recomendación: en cualquier proyecto de inversión con refuerzo, **empezar por Q-Learning**. Los algoritmos profundos son más potentes pero más inestables, muy sensibles a los hiperparámetros y propensos al sobreajuste si hay muchas variables. Ellos empezaron por DRL, tuvieron que retroceder y obtuvieron buenos resultados con Q-Learning antes de volver a DRL (T30 ~00:16). §10 da más detalle.

*Fuente: DRL1 p1-p12; T30 ~00:03-00:17.*

---

## 2. Motivación y el hito de Atari

**Idea.** El aprendizaje profundo aprende características de extremo a extremo en espacios enormes. Permite aproximar y generalizar en lugar de rellenar una tabla. DQN demostró en Atari que eso funciona pasando de píxeles a joystick.

**Motivación** (DRL1 p18): la maldición de la dimensionalidad hace que nunca haya muestras suficientes para llenar el espacio de estados. La respuesta es **aproximar, generalizar y explotar la continuidad**.

**Contexto histórico:**

- En 2012 el aprendizaje profundo supervisado triunfa en ILSVRC, la competición sobre ImageNet, y se desarrolla en buena parte gracias a ImageNet (DRL1 p20; la imagen muestra clasificaciones de ImageNet con sus 5 mejores etiquetas).
- En ese momento el estado del arte de RL con redes es, aproximadamente, TD-Gammon (1992), con 40 neuronas ocultas. La pregunta es qué se puede hacer con CNN profundas en RL (DRL1 p22).
- El *Atari Learning Environment* (ALE) hace para el RL profundo un papel análogo al de ImageNet (DRL1 p23).
- Atari 2600: consola, cartucho ROM, 128 bytes de RAM; la entrada son píxeles y la salida, acciones del joystick (DRL1 p24).
- Mnih et al. (2013/2015): un ordenador aprende a jugar de extremo a extremo, de los píxeles al joystick, sobre un emulador (DRL1 p29). p29 muestra la portada de Nature, p30 reproduce el artículo y p31, capturas de varios juegos. Sobre las cifras y el nombre mal escrito, ver §14.

**Bucle agente-entorno aplicado a Atari** (DRL1 p33-p34): el agente recibe el estado $s_t$ (la pantalla) y el refuerzo $r_t$ (el marcador) y emite la acción $a_t$ por el joystick; la consola devuelve $s_{t+1}$ y $r_{t+1}$.

**El refuerzo no siempre "viene de fuera".** En el esquema clásico lo genera el entorno, pero a veces lo produce el propio agente: el dolor al caerse de la bicicleta nace dentro del cuerpo, y en un coche autónomo un componente de software da un refuerzo negativo si se acerca demasiado a otro coche. En bolsa sí viene de fuera: son las ganancias (T30 ~00:18-00:19).

**Vídeo de Breakout** (DRL1 p32, `youtube.com/watch?v=V1eYniJ0Rnk`; T30 ~00:27-00:29). El agente solo recibe la pantalla y la orden de maximizar la puntuación, sin saber qué es una pelota ni qué hacen los controles.

| Entrenamiento | Comportamiento |
|---|---|
| 10 minutos | Reactivo: solo reacciona cuando la pelota llega a su lado |
| 2 horas | Planifica: se coloca donde espera el rebote y ya no pierde la pelota |
| 4 horas | Descubre el truco de abrir un túnel lateral para romper los ladrillos de arriba, que da muchos más puntos |

**Resultados del artículo mostrados en el mazo:**

- **Evolución del aprendizaje** (DRL1 p38, cuatro gráficas frente a 200 épocas). Las de arriba muestran la puntuación media por episodio; las de abajo, el valor $Q$ medio. Según el pie de figura que se ve en p30, corresponden a Space Invaders (a, c) y Seaquest (b, d). La puntuación sube con mucho ruido (hasta unos 1.800 y 5.000, con caídas bruscas); el valor $Q$ medio sube de forma mucho más suave y se estabiliza en torno a 9-10. **Nota:** por eso $Q$ medio es un buen complemento para seguir el entrenamiento, además de la puntuación.
- **Valoraciones aprendidas** (DRL1 p39). Cuatro capturas de Breakout sobre una curva del valor $V$ a lo largo de 120 fotogramas: $V$ ronda 16-17 (punto 1), cae a unos 15 (punto 2) y sube hasta 21 y más de 23 (puntos 3 y 4) a medida que se abre el túnel y la pelota llega a la zona superior. El modelo ha aprendido que esa situación vale más.
- **Comparación por juego** (DRL1 p40). Barras de puntuación normalizada para 49 juegos, de DQN frente al mejor modelo lineal, con una línea que separa "a nivel humano o superior" de "por debajo". Arriba destacan Video Pinball, Boxing y Breakout; abajo, Private Eye y Montezuma's Revenge, con casi el 0 %. El mazo concluye que bate la mejor marca humana en muchos juegos. **Nota:** el final de la lista son juegos de recompensa muy diferida, que enlaza con el diseño del refuerzo de §10.
- **Enduro** (DRL1 p41): jugador experto 368 puntos, DQN 661.

**Otros hitos, solo mencionados aquí.** AlphaGo (2016) mezcla supervisado y refuerzo y gana 5-0 a Fan Hui; AlphaGo Zero (2017) aprende solo jugando contra sí mismo y gana 100-0 a la versión anterior (DRL1 p44). El profesor comenta también Deep Blue, los estilos de juego "elegantes" en Go y el póker como juego de información incompleta (T30 ~02:28-02:33). También muestra el mapa de algoritmos: con o sin modelo, optimización de política o Q-Learning (T30 ~01:45-01:47). Todo ello se desarrolla en [07_panorama_drl.md](07_panorama_drl.md).

*Fuente: DRL1 p13-p44; T30 ~00:17-00:19, ~00:27-00:29, ~01:45-01:47, ~02:28-02:33.*

---

## 3. La red: de la tabla a un aproximador de Q

**Idea.** La tabla se sustituye por una red con muchos menos parámetros: entra lo observado en el estado y sale un valor $Q$ por cada acción (DRL1 p68).

```
datos observados en el estado  →  red neuronal  →  [ Q(s,a1), Q(s,a2), ..., Q(s,am) ]
```

Diseño (DRL1 p77):

- La entrada son los datos observados en el estado actual.
- La capa de salida tiene **tantas neuronas como acciones posibles**.
- La activación de la última capa debe ser **la identidad ("lineal")**, porque el problema es de regresión.
- Cada término de error se asocia **solo a una neurona de salida**, la de la acción realizada.

**Por qué la salida es lineal.** ReLU no sirve porque $Q$ puede ser negativa, y sigmoide tampoco porque limitaría los valores a $[0,1]$. Hay que dejarla libre (T30 ~00:33). En Keras basta con no poner activación, o escribir `activation="linear"` (T30 ~01:27; 1-catch, celda 17; 2-cart_pole_DQN, celda 6).

**Escala: un error muy típico.** Se normalizan las entradas pero no las salidas. Una red solo funciona bien si tanto las entradas como las salidas tienen una escala razonable. $Q$ estima el premio total que queda hasta el final, que puede llegar a miles: pedir a la red que estime algo del orden de 1.000 la estropea y puede impedir que aprenda (T30 ~00:34). También afecta a la estabilidad que los valores sean demasiado bajos o muy distintos entre sí (T30 ~02:35).

**Arquitectura Atari** (DRL1 p36; T30 ~00:19-00:21):

| Capa | Entrada | Filtro | Stride | Filtros/neuronas | Activación | Salida |
|---|---|---|---|---|---|---|
| conv1 | 84×84×4 | 8×8 | 4 | 32 | ReLU | 20×20×32 |
| conv2 | 20×20×32 | 4×4 | 2 | 64 | ReLU | 9×9×64 |
| conv3 | 9×9×64 | 3×3 | 1 | 64 | ReLU | 7×7×64 |
| fc4 | 7×7×64 | | | 512 | ReLU | 512 |
| fc5 | 512 | | | 18 | Lineal | 18 |

- En lugar de *max pooling* se usa *stride* (T30 ~00:19).
- Son 18 salidas porque el joystick tiene 18 estados discretos: quieto, 4 direcciones y 4 diagonales, y esos 9 otra vez con disparo (T30 ~00:20).
- **Por qué 4 fotogramas.** Con una sola foto no se sabe hacia dónde va la pelota. Con 2 ya se sacaría la dirección; en Atari usan los 4 últimos, en gris (DRL1 p36, "movimiento"; T30 ~00:30-00:31). El algoritmo solo ve la pantalla en bruto y los puntos (T30 ~00:30).
- **Nota:** es la forma de recuperar la propiedad de Márkov (ver [00_fundamentos_mdp.md](00_fundamentos_mdp.md) §3). En Catch sin rebotes no hace falta porque la fruta siempre baja; si rebotara, sí (T30 ~01:28).

**Qué cambia frente a una tabla** (T30 ~00:21-00:26):

- **Generalización.** A una red se le puede preguntar por una pantalla que nunca ha visto, y dará algo razonable si se parece a otras. En una tabla, un estado no visitado tiene ceros en toda la fila (T30 ~00:22-00:23).
- **Una actualización cambia todos los pesos.** Eso es bueno porque generaliza, y malo porque **puede desaprender**. Un coche que pasa días en ciudad y luego días solo en carretera puede olvidar cómo conducir en ciudad (T30 ~00:23-00:25). En la tabla esto no pasa: solo cambia una celda (T30 ~00:26). De ahí nace la memoria de experiencias (§8).

**Inferencia.** Una vez entrenada, se le pasa el estado, devuelve la valoración de cada acción y se elige la mayor (T30 ~00:31). La entrada en bolsa serían los datos numéricos de los últimos días, sin discretizar; por ejemplo, una serie multivariada con los 35 valores del IBEX más índices e indicadores (T30 ~00:32-00:33).

*Fuente: DRL1 p35-p36, p67-p68, p74-p77; T30 ~00:19-00:26, ~00:30-00:34.*

---

## 4. Retos: cobertura, correlación y convergencia

**Idea.** El aprendizaje profundo supervisado asume datos estáticos e independientes; el RL no tiene ninguna de las dos cosas. La aproximación de la función de valor es, en teoría, inestable.

**Retos generales** (DRL1 p49): carga computacional (30 imágenes de vídeo por segundo); entrada de alta dimensión; Q-Learning tabular no funciona; se sabe desde hace tiempo que la aproximación de la función de valor es teóricamente inestable; y el problema es aprender a partir de la retroalimentación, que provoca inestabilidad en los *targets*.

**Analogía del mazo** (DRL1 p49, p57): altavoz, micrófono y amplificador. El sonido del altavoz entra por el micrófono, se amplifica y vuelve a salir; ese bucle de realimentación provoca pitidos.

**Los tres problemas** (DRL1 p52):

| Problema | Qué dice | Dónde se ataca |
|---|---|---|
| **Cobertura** | La convergencia de Value Iteration, Q-Learning y SARSA depende de que al final se cubra todo el espacio de estados. En alta dimensión no se está ni cerca. También pasa en humanos: entrenar contra el mismo *sparring* da un conjunto limitado de habilidades (DRL1 p55) | Generalización de la red (§3) y exploración |
| **Correlación** | En supervisado los ejemplos no están correlacionados y el aprendizaje es estable. En RL las acciones determinan el siguiente estado del que se aprenderá (DRL1 p57) | Muestreo aleatorio de la memoria (§8) |
| **Convergencia** | Listada en p52 sin página propia en el mazo. **Nota:** su contenido está en p49 y p97: los *targets* cambian porque dependen de la misma red | *Target* dinámico (§5) y red objetivo (§9) |

**Nota:** la columna "Dónde se ataca" es una síntesis propia; el mazo no relaciona así problemas y técnicas.

*Fuente: DRL1 p45-p57.*

---

## 5. La función de coste

**Idea.** Se reformula Q-Learning como una regresión: la salida de la neurona de la acción realizada debe acercarse al *target* de Bellman. La rareza es que ese *target* lo calcula la propia red y por tanto cambia. El profesor avisa de que este es el salto conceptual más difícil y que hay que repetirlo hasta verlo (T30 ~00:32, ~00:35, ~00:44).

**Paso 1: la condición de consistencia** (DRL1 p66; T30 ~00:37-00:39). El retorno con descuento es

$$G_t=\sum_{k=0}^{\infty}\gamma^k\,r_{t+k}$$

(con la errata de índices de DRL1 p66 corregida, ver §14). Cuanto más lejos está un refuerzo, menos pesa: con $\gamma=0{,}9$, el inmediato pesa 1, el siguiente 0,9, luego 0,81... (T30 ~00:37).

$Q(s,a)$ es el retorno total óptimo si se realiza $a$ en $s$ y después se actúa de forma óptima (DRL1 p66). Como lo que se gana ahora es el refuerzo inmediato más $\gamma$ por lo que se gane después, y después se elegirá lo óptimo, cuando las $Q$ convergen debe cumplirse (Bellman, DRL1 p66):

$$Q(s_t,a)=r+\gamma\max_{a'}Q(s_{t+1},a')$$

Un teorema garantiza que es posible estimar $Q$ de modo que se cumpla (T30 ~00:39).

**Paso 2: deducir Q-Learning.** Si la derecha es mayor que la izquierda, $Q$ debe subir; si es menor, bajar. Lo "más cutre" que se puede hacer es sumar la diferencia multiplicada por $\alpha$: eso **es** Q-Learning (DRL1 p66; T30 ~00:39-00:40). Valores típicos de $\alpha$ tabular: 0,1 o 0,01 (T30 ~00:36).

**Paso 3: con una red.** $Q$ es ahora la salida de una neurona. Izquierda y derecha deben cuadrar, y si no cuadran hay que cambiar la red. La izquierda es la **predicción** y la derecha, el ***target*** (DRL1 p73; T30 ~00:41-00:42). Coste por transición (DRL1 p73, p77):

$$\text{loss}=\Big(\underbrace{r_t+\gamma\max_{a'}\hat Q(s_{t+1},a')}_{\text{Target}}-\underbrace{Q(s_t,a_t)}_{\text{Predicción}}\Big)^2$$

Sobre la figura: "Reward" señala a $r_t$ y "Decay Rate" a $\gamma$. **Nota:** $\hat Q$ con sombrero aparece ya desde p37, antes de presentar la red objetivo en p115. En la versión básica $\hat Q$ y $Q$ son la misma red (DRL1 p97).

**Sobre un minibatch** (T30 ~00:47, ~00:49). El alumno preguntó si el coste es una media. La fórmula de la transparencia es la contribución de una sola tupla; la función de coste real es la media en una muestra de $N$ transiciones:

$$\mathcal{L}(\theta)=\frac{1}{N}\sum_{j=1}^{N}\big(y_j-Q(s_j,a_j;\theta)\big)^2$$

**Estado terminal.** Si el episodio ha acabado no hay estado siguiente y no tiene sentido preguntar qué viene después. Se elimina el término del máximo (T30 ~04:35-04:37; 1-cart_pole_QLearning, celda 1; p116):

$$y_j=\begin{cases} r_j & \text{si el episodio termina en } j+1\\ r_j+\gamma\max_{a'}\hat Q(s_{j+1},a') & \text{en otro caso}\end{cases}$$

En tabla, $\Delta Q(s,a)=\alpha[r-Q(s,a)]$ (1-cart_pole_QLearning, celdas 1 y 31). El profesor lo omitió al principio para no liar, pero en el código hace falta el `if` y por eso la transición guarda un quinto dato, `done` o `game_over` (T30 ~01:31, ~04:35). **Nota:** olvidarlo es un error serio, porque hace que el valor del final de episodio se "propague" como si la partida continuara.

**El *target* dinámico: la diferencia con la regresión de siempre** (DRL1 p97; T30 ~00:42-00:46):

- En una regresión típica (el precio de una casa), el *target* es una columna estática de una tabla.
- Aquí el *target* depende de la salida de la propia red. Si cambia la red, cambian a la vez la predicción y el *target*. Hay que recalcularlo cada vez que se entrena (T30 ~00:43), y eso se hace dentro del bucle, no una vez fuera (T30 ~01:36).
- Por eso la red no se entrena de una vez sino a poquitos (T30 ~00:43), y estos modelos **pueden no converger** si no se tiene cuidado.
- Q-Learning, en cambio, acaba convergiendo "con muchas burradas" si explora lo suficiente. La red no: un *learning rate* demasiado alto, o algo poco estable, puede impedirlo (T30 ~00:46).
- **Analogía con las GAN** (T30 ~00:52-00:53): allí el objetivo también es dinámico, porque generador y discriminador dependen uno del otro. Si no se tiene cuidado, oscilan o desaprenden. Son mucho más sensibles a los hiperparámetros que Q-Learning.

**El coste no llega a 0.** En Q-Learning podría, jugando con las celdas. Una red tiene muchos menos parámetros y hace el papel de una tabla infinita, así que solo la aproxima: el coste no baja de cierto límite (T30 ~00:51). **Nota:** esa afirmación vale con un entorno determinista; si es estocástico, el error de cada muestra no se anula ni en una tabla. Por eso una curva de *loss* no indica por sí sola que el agente haya aprendido.

**Funciones de coste** (T30 ~01:02-01:06):

- **MSE:** es la habitual (DRL1 p97). Penaliza mucho los extremos, y al principio del entrenamiento los errores serán grandes (T30 ~01:02).
- **MAE:** disponible en Keras; se puede probar (T30 ~01:02).
- **Huber:** cuadrática para errores pequeños y lineal para grandes, lo que propuso un alumno (T30 ~01:03). Depende de un hiperparámetro $\delta$:

$$L_\delta(x)=\begin{cases}\tfrac12 x^2 & |x|\le\delta\\[2pt] \delta|x|-\tfrac12\delta^2 & |x|>\delta\end{cases}$$

El término raro $-\tfrac12\delta^2$ está para que las dos ramas coincidan en $|x|=\delta$ (ambas valen $\tfrac12\delta^2$) y la función sea continua (T30 ~01:04-01:05). La transcripción de la segunda rama es confusa; **Nota:** se ha reconstruido con la condición de continuidad que explica el profesor y coincide con la definición habitual. También se pueden programar funciones de coste propias (T30 ~01:03).

**Nota (semigradiente):** el *target* se trata como una constante, sin propagar gradiente por él. En el código de clase ocurre sin decirlo, porque sale de `predict` y es un array fijo. En PyTorch hay que garantizarlo con `torch.no_grad()` o `detach()`.

**Optimización.** Con esta función de coste se aplica el optimizador habitual, Adam o SGD, pero recalculando los *targets* una y otra vez (T30 ~00:47). "El dolor de cabeza con estos modelos" es que converjan, que no sobreajusten y que no olviden lo visto al principio (T30 ~00:47-00:48).

*Fuente: DRL1 p58-p77, p97; T30 ~00:35-00:54, ~01:02-01:06, ~04:35-04:37; 1-cart_pole_QLearning, celdas 1, 30, 31.*

---

## 6. Ejemplo numérico

**Idea.** El ejemplo de la transparencia muestra cómo se calcula el *target* de una sola transición, y por qué solo cambia una neurona.

**Planteamiento** (DRL1 p78, p86): red con 3 acciones y, por tanto, 3 neuronas de salida. El agente pasa de $S$ a $S'$ con la acción 0, recibe $r=2$, y $\gamma=0{,}9$. Salidas actuales de la red:

| Entrada | $Q(\cdot,a_0)$ | $Q(\cdot,a_1)$ | $Q(\cdot,a_2)$ |
|---|---|---|---|
| $S$ | **0,3** (acción realizada) | 0,1 | -1 |
| $S'$ | 1,1 | **1,5** (máximo) | 0,8 |

**Cálculo** (DRL1 p89):

$$\Delta Q(S,a_0)=\alpha\,[\,2+0{,}9\cdot 1{,}5-0{,}3\,]=\alpha\,[\,\underbrace{3{,}35}_{\text{target}}-\underbrace{0{,}3}_{\text{predicción}}\,]$$

**Salida deseada para $S$** (DRL1 p91, p93):

| | $a_0$ | $a_1$ | $a_2$ |
|---|---|---|---|
| Salida real | 0,3 | 0,1 | -1 |
| Salida deseada | **3,35** | 0,1 | -1 |

El resto de valoraciones "están ok de momento": no se sabe si están bien, pero se dejan como están (DRL1 p91; T30 ~00:58). En una tabla se subiría esa celda y ya; aquí se cambian los pesos de la red para que esa neurona dé un valor más alto (T30 ~00:58). Y el cálculo depende de la salida de la red para $S'$, por lo que todo va cambiando con el tiempo (T30 ~00:59). En la práctica esto se hace para cada una de, por ejemplo, 100 transiciones sacadas al azar (T30 ~01:00).

**Nota (comprobación):** el error es $3{,}05$ y su cuadrado, $9{,}30$. Las otras dos neuronas tienen error 0 y no aportan gradiente. Keras con `"mse"` promedia también entre salidas, de modo que aquí daría $9{,}30/3=3{,}10$: ese factor $1/m$ se absorbe en la tasa de aprendizaje efectiva. En una tabla con $\alpha=0{,}1$, la celda pasaría a $0{,}3+0{,}1\cdot 3{,}05=0{,}605$.

*Fuente: DRL1 p78-p93; T30 ~00:54-01:00.*

---

## 7. Q-Learning frente a DQN

**Idea.** La información pasa de celdas a pesos, y cada actualización pasa de tocar un valor a tocar muchos estados.

| | Q-Learning | DQN |
|---|---|---|
| Estructura | Tabla | Red neuronal |
| Información | En celdas | En pesos |
| Estados | Conjunto finito | Infinitos |
| Se actualiza | Una celda $Q(s,a)$ | Los pesos de la red |
| La actualización afecta en cada iteración a | Una celda | Muchos estados |

(DRL1 p95; T30 ~01:00-01:01.)

**Conclusiones sobre la función de coste** (DRL1 p101; T30 ~01:01-01:02):

- El problema de optimización es **una especie de regresión**,
- pero los *targets* son **dinámicos** y **dependen de la misma red**.
- Normalmente se usa **MSE** como función de coste.
- Para cada transición $S,a,r,S'$, el *target* deseado es la salida que dio la red para $S$, **salvo en la neurona de la acción realizada**, que hay que actualizar.

**Interpretabilidad.** Con pocos estados y acciones, Q-Learning es muy interpretable: se ve qué valora en cada estado. DQN es más una caja negra y haría falta añadir herramientas de explicabilidad (T30 ~00:54). En el proyecto de inversión del profesor, la tabla Q permitía decirle a un inversor qué recomienda el sistema cuando hay mucha volatilidad, por ejemplo (T30 ~02:57-02:58).

*Fuente: DRL1 p94-p101; T30 ~00:54, ~01:00-01:02, ~02:57-02:58.*

---

## 8. Memoria de experiencias y algoritmo básico

**Idea.** Se separa actuar de aprender. Cada transición se guarda en un *buffer* limitado y la red se entrena poco a poco con muestras al azar de todo lo vivido.

**Algoritmo básico** (DRL1 p106):

1. Inicializar los pesos aleatoriamente.
2. Inicializar un episodio e iterar hasta su fin:
   - Generar una experiencia $S,a,r,S'$ y meterla en una memoria limitada (*buffer*).
   - Tomar del *buffer* un minibatch de experiencias y entrenar la red con él.
3. Volver a 2 hasta el fin, por número de episodios o por convergencia.

**Por qué funciona** (DRL1 p109):

- Entrenar con minibatches elegidos al azar actúa contra **la correlación** de las experiencias.
- Si la red se reentrena solo con lo último, **"olvida" lo inicial**.
- *Experience replay*: las redes necesitan que les "recuerden" las experiencias pasadas.

En Q-Learning se observa algo y se cambia la tabla. En DQN se observa, se almacena en un *dataset* y, de vez en cuando, se saca al azar una muestra que cubre todo lo visto y se entrena la red "un poquito" (T30 ~00:25-00:26). Como con las personas: lo que no se repasa se olvida (T30 ~00:25).

**Remember y Replay.** Son los dos métodos que aparecen en el mazo (DRL1 p37, p73). El profesor señala que el nombre habitual, `remember`, no le gusta porque en realidad solo guarda la transición en la base de datos; quien "recuerda" es el *replay* (T30 ~01:31). Una transición completa son 5 datos: $(S,a,r,S',\text{done})$ (T30 ~01:31, ~00:48).

**Tamaño de la memoria.** Es un hiperparámetro con el que hay que jugar, no "cuanto más, mejor" (T30 ~02:48):

- **Demasiado pequeña:** retiene solo lo último y se pierde lo antiguo. El agente entra en bucle: aprende ciudad, olvida autopista, vuelve a ciudad... (T30 ~01:32).
- **Demasiado grande:** se conservan muchas observaciones de los inicios, generadas por una política aún casi aleatoria (T30 ~02:48).
- Depende de la complejidad: en un juego sencillo bastan unas 1.000 transiciones; un coche autónomo necesita muchísimas más (T30 ~01:11-01:12, ~01:30).
- **Política de borrado:** lo típico es eliminar la más antigua (FIFO). También se podría borrar una al azar (T30 ~01:32).

**Tamaño del minibatch.** También hay que encontrar un punto intermedio (T30 ~02:49):

- **Muy grande:** en una sola época la red cambia demasiado (T30 ~02:49). La analogía: con un *dataset* enorme bastan pocas épocas para entrenar muchísimo.
- **Muy pequeño:** con 1 ejemplo es muy ruidoso, porque se ajustan todos los pesos con un solo dato. Tras subirlo, el agente de Catch quedó claramente más estable (T30 ~04:29). Ver §11.
- Valores de clase: 1 y 10 en Catch; 128 en CartPole; "por ejemplo, 100" en la explicación (T30 ~00:49).

**Una sola época por muestra.** Esta es una recomendación explícita. Si se hacen 100 épocas sobre las 100 transiciones sacadas, la red sobreajusta a esa muestra y olvida el resto. Lo típico es sacar una muestra de toda la experiencia, entrenar **una sola época** y repetir: poco a poco irá convergiendo (T30 ~00:50). En Keras, `train_on_batch` equivale a un `fit` de una época (T30 ~01:40-01:41; 1-catch, celda 23; 2-cart_pole_DQN, celda 6).

**Dónde entrenar: dentro o al final del episodio.** Las fuentes no coinciden, y conviene saberlo:

| Opción | Fuente | Efecto |
|---|---|---|
| En cada paso | DRL1 p106, p116; 1-catch, celda 23 | Más actualizaciones; aprende en cada partida, pero es lento (T30 ~02:23) |
| Al final de cada episodio | "Lo normal" según el profesor (T30 ~01:39, ~02:23, ~03:33); 2-cart_pole_DQN, celda 15 | Mucho más barato por episodio; entrena menos y aprende más lento por episodio (T30 ~02:23). Hay que compensar con un minibatch mayor (T30 ~03:33) |

**Nota:** aunque el profesor diga que lo normal es entrenar al final, tanto el algoritmo básico de la transparencia como el de Atari entrenan en cada paso. Lo que cuenta es la proporción entre gradientes y experiencias nuevas; ver §13 para ver qué pasa cuando es muy baja.

*Fuente: DRL1 p102-p109; T30 ~00:25-00:26, ~00:48-00:51, ~01:30-01:41, ~02:23, ~02:48-02:50, ~03:31-03:35, ~04:29.*

---

## 9. Mejoras: recorte de recompensas, red objetivo y algoritmo DQN Atari

**Idea.** Hay dos trucos principales para estabilizar: acotar los refuerzos, para que los errores no se disparen, y congelar durante un rato la red que calcula los *targets*.

**Recorte de recompensas.** "Cuidado con refuerzos muy diferentes (Atari): por ejemplo, hacer clip entre -1 y 1" (DRL1 p115). DeepMind usa este tipo de truco porque tener valores demasiado bajos o diferencias muy grandes afecta a la estabilidad: los que son demasiado altos los recortan (T30 ~02:35).

**Nota:** recortar cambia el problema: el agente deja de distinguir un premio de 1 de uno de 1.000. Ayuda porque iguala la escala entre juegos con una misma tasa de aprendizaje. Si los refuerzos ya son pequeños, el recorte no actúa; es **escalar** lo que importa entonces (ver *Aplicación al Taller 2*).

**Red objetivo** (DRL1 p115):

- A veces el aprendizaje es inestable, porque el *target* lo da la misma red.
- Para estabilizar se usa una red $Q$ para generar la política y otra, $Q'$, para calcular los *targets*.
- $Q$ es la que se va modificando paso a paso.
- **Cada $k$ iteraciones se copia $Q$ en $Q'$.**

**Errata importante:** DRL1 p115 dice literalmente "se copia la Q' en Q" (y en los fotogramas previos, p111-p114, "cada C iteraciones"). Es al revés: de otro modo no se conservaría lo aprendido. El algoritmo de p116 lo confirma: *"Every C steps reset $\hat Q=Q$"*. Se copia la red que aprende en la red objetivo.

**Intuición (Nota).** Vuelve a la analogía del micrófono (§4): durante $C$ pasos el *target* no depende de lo que se acaba de cambiar, así que se corta el bucle de realimentación y la regresión se parece, por un rato, a una regresión normal.

**Nota sobre la implementación:** el truco de la "salida deseada" de §6 se mantiene, pero con cuidado. Las neuronas que no se actualizan deben copiar la **red en línea** evaluada en $S$ (para que su error sea 0), mientras que el máximo sobre $S'$ debe venir de la **red objetivo**.

**En clase.** En el tramo asignado de T30, las p111-p116 no se desarrollan. Hay una pregunta de un alumno sobre una "segunda red" en DeepMind; el profesor responde que hay muchas variantes con dos redes, en actor-crítico y en DQN, todas como intentos de estabilizar el entrenamiento, y que el algoritmo que han visto debería poder resolver cualquier problema: el problema es su estabilidad y el tiempo que tarda (T30 ~02:34-02:35). Actor-crítico se desarrolla en [06_actor_critico_y_a3c.md](06_actor_critico_y_a3c.md), y Doble DQN, en [04_doble_dqn_y_duelo.md](04_doble_dqn_y_duelo.md). **Ninguno de los tres notebooks de este documento implementa red objetivo ni recorte.**

**Algoritmo 1 de Mnih et al. (2015)** (DRL1 p116, "Algoritmo DQN Atari"):

```
Inicializar la memoria de repetición D con capacidad N
Inicializar la función acción-valor Q con pesos aleatorios θ
Inicializar la función objetivo Q̂ con pesos θ⁻ = θ
Para episodio = 1, M:
    Inicializar la secuencia s₁ = {x₁} y la secuencia preprocesada φ₁ = φ(s₁)
    Para t = 1, T:
        Con probabilidad ε elegir una acción aleatoria aₜ
        En otro caso aₜ = argmaxₐ Q(φ(sₜ), a; θ)
        Ejecutar aₜ en el emulador y observar el refuerzo rₜ y la imagen xₜ₊₁
        Hacer sₜ₊₁ = sₜ, aₜ, xₜ₊₁ y preprocesar φₜ₊₁ = φ(sₜ₊₁)
        Guardar la transición (φₜ, aₜ, rₜ, φₜ₊₁) en D
        Muestrear de D un minibatch aleatorio de transiciones (φⱼ, aⱼ, rⱼ, φⱼ₊₁)
        Calcular yⱼ (fórmula de abajo)
        Dar un paso de descenso por gradiente sobre (yⱼ − Q(φⱼ, aⱼ; θ))² respecto a θ
        Cada C pasos, hacer Q̂ = Q
```

$$y_j=\begin{cases} r_j & \text{si el episodio termina en el paso } j+1\\ r_j+\gamma\max_{a'}\hat Q(\phi_{j+1},a';\theta^-) & \text{en otro caso}\end{cases}$$

Aquí $\phi$ es el preprocesado, que apila los últimos fotogramas (84×84×4, §3).

**Nota (versión para estado vectorial, como el del Taller 2).** Combina p116 con los detalles de los notebooks:

```
θ ← aleatorio;  θ⁻ ← θ;  D ← vacío (FIFO, capacidad N);  ε ← ε_ini;  pasos ← 0
Para episodio = 1..M:
    s ← entorno.reset();  done ← False
    Mientras no done:
        a ← aleatoria con prob. ε;  si no, argmaxₐ Q(s, a; θ)
        s', r, done ← entorno.step(a)
        D.guardar(s, a, r, s', done);  s ← s';  pasos ← pasos + 1
        Si |D| ≥ tamaño mínimo:
            (sⱼ, aⱼ, rⱼ, s'ⱼ, doneⱼ) ← muestra aleatoria de tamaño B
            yⱼ ← rⱼ                                   si doneⱼ
            yⱼ ← rⱼ + γ · maxₐ' Q(s'ⱼ, a'; θ⁻)         si no       (sin gradiente)
            θ ← paso de Adam/SGD sobre media[(yⱼ − Q(sⱼ, aⱼ; θ))²]   (una sola pasada)
        Cada C pasos: θ⁻ ← θ
    ε ← max(ε_min, ε · decaimiento)
Evaluación: ε = 0, sin guardar ni actualizar nada
```

*Fuente: DRL1 p110-p116; T30 ~02:34-02:36.*

---

## 10. Recomendaciones prácticas del profesor

**Idea.** No hay un criterio general. Según el profesor, ajustar estas redes es prueba y error sobre muchos hiperparámetros (T30 ~02:50). Lo que sigue recoge sus avisos concretos.

**Hiperparámetros** (T30 ~02:50-02:51):

| Hiperparámetro | Qué controla | Consejo | Valores de clase |
|---|---|---|---|
| $\gamma$ | Hasta dónde planifica | Elegirlo según la duración del episodio: $\gamma^{T}$ no debe acercarse a 0 si el premio importa hasta el final (T30 ~04:21) | 0,9 (Catch, CartPole DQN); 0,999 (CartPole tabular, $0{,}999^{500}\approx 0{,}6$) |
| Tasa de aprendizaje | Cuánto cambian los pesos | No muy alta: con una sola época se modificarían demasiado (T30 ~02:51). Si no converge, bajarla (T30 ~01:11) | $\alpha=0{,}1$ (tabla); $10^{-3}$ (SGD) |
| Tamaño de memoria | §8 | Ni muy pequeño ni muy grande | 1.000 (Catch); 20.000 (CartPole) |
| Tamaño de minibatch | §8 | Ni muy pequeño ni muy grande | 1 y 10 (Catch); 128 (CartPole) |
| $\varepsilon$ y su decaimiento | Exploración | Ver abajo | 0,1 fijo; 1→0,01 con ×0,995; 1 con ×0,999 |
| Arquitectura | Capacidad | La más sencilla posible para que generalice (T30 ~00:51). Jugar con capas y neuronas (T30 ~00:52) | Densas de 50 y 100 neuronas |
| Función de coste | §5 | Empezar por MSE; probar MAE o Huber | MSE |

**Adam o SGD** (T30 ~01:09-01:11):

- **Adam** es el optimizador por defecto. Adapta el *learning rate* y acelera, pero "a veces se pasa de listo" y se vuelve inestable.
- **SGD** va siempre al mismo ritmo, poco a poco pero de manera consistente.
- **Recomendación:** probar primero Adam y, si no es estable, pasar a SGD (T30 ~01:10). `1-catch` usa Adam (celda 17), mientras que `2-cart_pole_DQN` deja Adam comentado y usa SGD (celda 6).
- El profesor añade que en GAN suele usarse SGD (T30 ~01:10). **Nota:** esto no es una regla general; DCGAN, por ejemplo, usa Adam.

**Qué hacer si no converge** (T30 ~01:10-01:12). Pintar la curva de error u observar el juego. Si no aprende, comprobar por este orden:

1. Que no haya un error de implementación.
2. Que el factor de aprendizaje no sea demasiado alto: bajarlo, o pasar a SGD.
3. Que la memoria no sea demasiado pequeña: ampliarla, o ajustar lo relacionado con ella.

**Inestabilidad y desaprendizaje.** Con datos fijos y *target* fijo, el descenso por gradiente tiende a mejorar. En DQN cambian ambos: la memoria se renueva al llenarse y los *targets* son dinámicos. Por eso puede ocurrir que todo vaya muy bien y de repente desaprenda y empiece a perder, como en las GAN (T30 ~02:46-02:47). Los distintos algoritmos atacan esta inestabilidad de diferentes formas (T30 ~02:47).

**Exploración** (T30 ~01:29, ~01:48, ~03:15, ~04:21):

- Siempre hace falta un factor de exploración; lo típico es mezclar lo aprendido con acciones al azar.
- Una estrategia típica es empezar muy alto e irlo bajando en cada episodio.
- **Cuidado:** si baja muy rápido, se congela la exploración antes de haber aprendido. El factor de decaimiento no puede ser ni muy alto ni muy bajo (T30 ~03:15).
- Un $\varepsilon$ residual explica algunas derrotas de un agente que ya sabe jugar (T30 ~02:39), y mientras siga siendo alto, las duraciones no reflejan lo aprendido. Lo realmente informativo es cuando $\varepsilon$ es bajo (T30 ~04:21, ~04:23).

**Seguimiento** (T30 ~01:11, ~03:25, ~04:24):

- Pintar la duración o el retorno por episodio, **suavizados** con una ventana móvil, por ejemplo las últimas 20 partidas, porque son muy ruidosos mientras $\varepsilon>0$.
- En Catch, media móvil de victorias por partida (1-catch, celda 29). Al ver la curva de un alumno, lo primero que pregunta el profesor es si está suavizada con una ventana de las últimas partidas o es el acumulado desde el inicio (T30 ~03:25-03:26). **Nota:** un acumulado reacciona cada vez más despacio y puede ocultar un desaprendizaje.
- Guardar el modelo periódicamente, para poder reutilizarlo; una ejecución interrumpida no devuelve el `history` (T30 ~03:15-03:16; 2-cart_pole_DQN, celda 15).

**Test *greedy*** (T30 ~03:17; 1-catch, celda 33; 2-cart_pole_DQN, celda 19). Una vez entrenado, se prueba con $\varepsilon=0$, eligiendo la acción más valorada y sin actualizar nada: no hay exploración ni entrenamiento. La función de test juega 10 partidas por defecto (T30 ~03:16). Un solo episodio puede engañar, porque el estado inicial es aleatorio (T30 ~04:25-04:26).

**Comprobar que los $Q$ tienen sentido** (T30 ~03:18-03:23). En Catch imprime los valores paso a paso:

- Deben ser positivos y menores que 1, porque como mucho se gana 1 y con descuento.
- La mejor acción debe **subir de valor conforme la fruta baja**, al acercarse el premio.
- Valores ligeramente superiores a 1 indican que la red no ha terminado de converger.
- Acciones casi empatadas pueden ser correctas: sin viento, quedarse quieto y moverse pueden valer casi igual.

**El entorno es lo que más cuesta** (T30 ~01:12, ~01:17, ~04:09-04:11):

- El agente suele ser estándar, incluso copiar y pegar; el trabajo real es preparar bien el entorno. En inversión: la simulación, los costes de transacción, los índices.
- Probarlo a mano y de forma exhaustiva antes de entrenar: moverse, comprobar los bordes, el refuerzo y el fin de episodio; ver si los valores se mueven como dicta la física.
- En un coche autónomo se construye primero un simulador muy trabajado, validado por conductores profesionales, antes de pasar a refuerzo.
- Se programa como una clase con `__init__`, `reset` y `step`, que devuelve estado, refuerzo y fin. Hay que distinguir el **estado interno** del juego de **lo que observa el agente** (T30 ~01:19-01:21).
- **Un episodio** debería durar el periodo real de uso del modelo: si se va a invertir un año, un año (T30 ~01:14).
- **Variar el punto de partida.** Lanzar siempre al agente al 1 de enero de 2000 sobreajusta a esa época; es mejor enviarlo a fechas aleatorias (T30 ~02:57).

**Diseño del refuerzo** (T30 ~03:27-03:30): "dependiendo de cómo lo diseñas, va a funcionar o no tu algoritmo".

- **Refuerzo solo al final.** Si en un juego muy complejo solo se sabe al final si se gana, cuesta mucho aprender. Un coche no se puede entrenar con "premio si llega intacto": explorando casi nunca llegará.
- **Refuerzos inmediatos, paso a paso.** Negativo si se acerca demasiado a otro coche o se salta un semáforo.
- **Pistas intermedias.** En Catch, un refuerzo positivo si la fruta se va acercando a la raqueta. También se puede dar más premio si se atrapa en el centro.
- **En inversión.** Se puede dar en cada paso según la variación de la cartera, que es como en el taller; o solo en las operaciones: comisión negativa al invertir, liquidación positiva al vender. **Con esto último, si se opera poco, se da muy poca información al agente.**

**Simplificar a saco.** Es la lección que más repite (T30 ~02:52-02:58). En su proyecto, empezaron con unas 100 variables y DRL: tras meses, el resultado fue un desastre. La red no convergía y, cuando lo hacía, predecía mal. Lo que funcionó fue reducir el problema:

- 7 indicadores agrupados en 5 estados de mercado, combinados con el estado de la cartera: unos 20 estados.
- Solo **2 acciones:** mantener la cartera actual o cambiar a la otra. Las dos composiciones (más y menos arriesgada) las define un experto; el agente solo indica cuándo hacer el cambio.
- Episodios de un año en fechas aleatorias.

"Mira que lo digo yo en mis clases", y aun así no lo hizo: "la solución es simplificar".

**Coste de entrenamiento y herramientas.**

- **Representación.** Si se da información elaborada (posición de fruta y raqueta) en vez de píxeles, aprende mucho más rápido: 2 entradas en lugar de 100 (T30 ~03:23).
- **Cuánto entrenar.** Parece mucho tiempo, pero en número de partidas es poco: cientos o miles, y mucho más si el juego es complicado. No basta con 10 o 20 partidas (T30 ~01:43, ~02:40).
- **Hardware.** Para ejemplos sencillos basta CPU; para imágenes o más tamaño se recurre a GPU (T30 ~01:07, ~01:43).
- **Librerías.** Una librería de RL en Keras que el profesor apreciaba dejó de mantenerse y falla con versiones nuevas de TensorFlow/Keras. No hay una librería estándar en Keras (T30 ~02:36-02:37).

*Fuente: T30 ~00:51-00:52, ~01:07-01:21, ~01:29, ~01:43, ~01:48, ~02:36-02:37, ~02:39-02:40, ~02:46-02:58, ~03:15-03:30, ~04:21-04:26.*

---

## 11. Código de clase: Catch (`1-catch`)

**Idea.** Es el ejemplo más sencillo de DQN completo: una red densa aprende a atrapar una fruta viendo solo los píxeles. Parece una tontería, pero el sistema debe extraer la información útil en bruto (T30 ~01:44).

**Origen y contexto** (celdas 0-1, 12-13): adaptación de "Keras plays catch" (E. Santana y otros), con explicaciones basadas en *Demystifying Deep Reinforcement Learning* (T. Matiisen). El markdown compara con el ajedrez: las recompensas son dispersas y diferidas, la mayor parte del tiempo $r=0$, y aparece el **problema de asignación de méritos** (*credit assignment*): determinar qué acción, tal vez muy anterior, fue la responsable del premio. Y si se planteara como clasificación imitando a expertos, sería caro y lento (celda 1).

**Entorno** (celda 4; T30 ~01:13-01:24):

| Elemento | Definición |
|---|---|
| Estado interno | `[fila_fruta, columna_fruta, columna_cesta]` |
| Observación | Matriz `grid_size`×`grid_size` con 1 en fruta y cesta, 0 en el resto, aplanada (100 entradas) |
| Acciones | 0 izquierda, 1 quieto, 2 derecha |
| Dinámica | La fruta baja una fila por paso; la cesta se mueve sin salirse |
| Viento | Con probabilidad `viento`, la fruta se mueve una columna al azar, sin atravesar bordes |
| Refuerzo | +1 si se atrapa, -1 si se escapa, al llegar abajo; 0 en el resto |
| Fin | Cuando la fruta llega a la última fila (9 pasos con `grid_size=10`) |

Celdas de apoyo: 5-6 son pruebas de Python (`randint` excluye el extremo; tuplas), 8-9 son variables y visualización, y 11 es el modo de juego manual para probar el entorno.

**Red** (celdas 17, 19-20; T30 ~01:26-01:29):

```python
model = Sequential()
model.add(Dense(hidden_size, input_shape=(grid_size**2,), activation="relu"))
model.add(Dense(num_actions))    # salida lineal, una neurona por acción
model.compile(Adam(), "mse")
```

Con `hidden_size=50` son unos 5.000 parámetros: **Nota:** exactamente 5.203 ($100\cdot50+50+50\cdot3+3$). Sin convoluciones, para ver si basta con eso. El markdown de la celda 13 habla de una entrada formada por $s$ y $a$, pero el código usa el esquema del mazo: entrada $s$ y una salida por acción.

**Parámetros** (celdas 15, 19): $\varepsilon=0{,}1$ fijo, 3 acciones, memoria 1.000, 50 neuronas, `batch_size=1`, `grid_size=10`, cesta de 3, $\gamma=0{,}9$ por defecto (celda 15, no se cambia en la celda 21). La celda 14 calcula $0{,}99^{10}$ para hacerse una idea del descuento.

**Memoria y cálculo de targets** (celda 15; T30 ~01:30-01:37). Es el núcleo del código:

```python
def remember(self, S, a, r, S_, game_over):
    self.memory.append([S, a, r, S_, game_over])
    if len(self.memory) > self.max_memory:
        del self.memory[0]                       # FIFO

def get_batch(self, model, batch_size=10):
    N = min(len(self.memory), batch_size)
    X = np.zeros((N, GRID_SIZE**2)); Y = np.zeros((N, num_actions))
    for i, idx in enumerate(np.random.randint(0, len(self.memory), N)):
        S, a, r, S_, game_over = self.memory[idx]
        X[i,:] = S
        targets = model.predict(np.array([S]), verbose=0)[0]   # salida actual para S
        if game_over:
            targets[a] = r
        else:
            targets[a] = r + self.discount*max(model.predict(np.array([S_]), verbose=0)[0])
        Y[i,:] = targets
    return X, Y
```

$Y$ es una matriz porque, con 3 neuronas de salida, en realidad hay 3 regresiones (T30 ~01:35). Es la "salida deseada" de §6. **Nota:** `randint` muestrea con reemplazo, y entrena desde la primera transición, sin esperar a tener un mínimo.

**Bucle de entrenamiento** (celda 23; T30 ~01:37-01:41):

```python
for episodio in range(epochs):
    entorno.reset(); S = entorno.observe().flatten(); game_over = False
    while not game_over:
        if np.random.rand() < epsilon: a = np.random.randint(num_actions)
        else: a = np.argmax(model.predict(np.array([S]), verbose=0)[0])
        S_, r, game_over = entorno.step(a); S_ = S_.flatten()
        exp_replay.remember(S, a, r, S_, game_over)
        S = S_
        X, Y = exp_replay.get_batch(model, batch_size)
        model.train_on_batch(X, Y)                # fit de 1 sola época
```

"A partir de ahora todo va a ser prácticamente este flujo": un `for` de episodios y dentro un `while` hasta el fin (T30 ~01:41).

**Número de partidas.** La celda 26 pone 4.000 y comenta que son necesarias unas 4.000; en clase dice 2.000 (T30 ~01:37). El resto de celdas: 25 y 27 sirven para descargar o guardar el modelo, 29 es la media móvil de victorias, 31-34 son la carga y el test, y 35 está vacía. El notebook no conserva salidas de entrenamiento: los resultados que siguen son los comentados en clase.

**Experimentos en clase:**

| Experimento | Resultado | Fuente |
|---|---|---|
| Base (batch 1, entrenamiento en cada paso, sin viento) | Gana 1 de cada 3 al principio; cerca de 1 de cada 2 a las 131 partidas; más tarde gana casi siempre, pero "baila" la cesta | T30 ~01:43, ~01:48, ~02:37, ~03:18 |
| Grid 20×20 con cesta de 5 | El problema se simplifica en proporción; sigue por debajo del 50 % al 10 % del progreso | T30 ~02:24 |
| Red convolucional (ejercicio) | Propuesta: grid de 15, kernel 3×3, 2 etapas de convolución y *pooling*, *flatten* y densas | T30 ~02:24-02:26 |
| Convolucional de un alumno, con muchos filtros | Unos 19.000 parámetros frente a 5.000: "una barbaridad" para este problema. Hay que reducir filtros | T30 ~02:44-02:45 |
| Entrenar al final de la partida con batch 10 | Equivale a las 10 muestras que antes se sacaban paso a paso. Gana casi siempre desde unas 800 partidas de 1.000 y la cesta queda quieta y estable, lo que el profesor atribuye a no ajustar con un solo ejemplo | T30 ~03:33-03:35, ~04:27-04:29 |

**Antes de culpar al agente, mirar con qué entorno se ha entrenado** (T30 ~02:19-02:23). Al ver que su agente se confunde en directo, el profesor sospecha que dejó el viento puesto y revisa el código: la última vez creó el entorno sin pasar el parámetro `viento`, cuyo valor por defecto es 0, así que no era la causa; además, lleva muy pocas partidas. **Nota:** en el taller equivale a comprobar mercado, serie y escala antes de interpretar un mal resultado.

**Con viento cambia la política óptima** (T30 ~02:37-02:39). Sin viento da igual recoger con la punta o con el centro; un agente sin viento se la juega hasta el último momento (T30 ~03:26-03:27). Con viento, lo óptimo es colocar el centro de la cesta bajo la fruta, para tener margen.

**Nota (comprobación con $\gamma=0{,}9$):** en la primera observación quedan 9 pasos y el premio llega en el noveno, así que el valor exacto de la mejor acción es $\gamma^{8}\approx 0{,}43$. El profesor lee alrededor de 0,51 en la primera pantalla y valores crecientes después (T30 ~03:20), lo que es razonable como aproximación.

*Fuente: 1-catch, celdas 0-35; T30 ~01:06-01:44, ~02:19-02:27, ~02:37-02:40, ~02:44-02:46, ~03:15-03:35, ~04:27-04:29.*

---

## 12. Código de clase: CartPole con tabla (`1-cart_pole_QLearning`)

**Idea.** Fuerza Q-Learning sobre un problema continuo para ver los límites y tener una referencia antes de usar una red.

**El problema** (celdas 6-7, 9-14; T30 ~04:06-04:08):

- **Acciones:** empujar el carro a izquierda o derecha con igual intensidad; no hay "no empujar".
- **Observación:** 4 variables continuas: posición, velocidad, ángulo y velocidad angular.
- **Refuerzo:** +1 por cada paso sin caerse. La estrategia óptima es la que mantiene el palo más tiempo.
- **Fin:** cuando el ángulo pasa de cierto umbral; el palo no llega a verse caer. Y hay un límite de duración, porque una política perfecta nunca acabaría. El profesor duda entre 200 y 500; en `CartPole-v1` es 500 (T30 ~04:14, ~04:20; figura de la celda 39).

**Comprobar el entorno** (T30 ~04:09-04:11). Con velocidad negativa, la posición baja; con empuje a un lado, cambia el signo de la velocidad angular. **Nota:** en las celdas 10-11, $0{,}02360706+0{,}02\cdot0{,}02435963=0{,}02409425$: el paso de tiempo que menciona el profesor vale 0,02.

**Referencias** (celdas 15, 19-22):

| Política | Duración |
|---|---|
| Siempre la acción 1 | 8 pasos |
| Aleatoria, 1.000 episodios | Media 22,0; máximo unos 74; sin tendencia (figura de la celda 21) |

**Discretización** (celdas 24-28; T30 ~04:15-04:18). Tres valores por variable, con `np.linspace`, y la conversión al más cercano con `argmin(abs(...))`:

| Variable | Valores |
|---|---|
| Posición | -2,4; 0; 2,4 |
| Velocidad | -0,05; 0; 0,05 |
| Ángulo | -0,21; 0; 0,21 |
| Velocidad angular | -0,05; 0; 0,05 |

Se usa un número impar para incluir el 0. Son $3^4=81$ filas por 2 acciones: "he simplificado a saco y aún así tengo 81 filas". Ejemplo: `discretiza_estado([1.9,-0.01,0.03,-0.02])` da `[2.4, 0, 0, 0]`.

**Nota:** el profesor atribuye el rango de la velocidad a la documentación (T30 ~04:15), pero en las celdas 11-14 aparecen velocidades de 0,22 y -0,37 y velocidad angular de 0,60, muy por fuera. La documentación de Gym da esas dos velocidades como no acotadas. En la práctica, casi todos los valores caen en el extremo: el rango funciona como un "signo con zona muerta".

**Agente** (celda 31): tabla en un `DataFrame` a ceros; ε-greedy, y si toda la fila vale 0 elige al azar. Actualización con caso terminal:

```python
if done: cambio = alpha*(r - Q[s_ant, a])
else:    cambio = alpha*(r + gamma*Q[s_nuevo].max() - Q[s_ant, a])
```

**Hiperparámetros** (celdas 35-38): $\gamma=0{,}999$, $\alpha=0{,}1$, $\varepsilon$ de 1 con ×0,999 por episodio, 10.000 episodios previstos. "Gamma va en función de lo que dura tu partida": la figura de la celda 36 muestra el peso $\gamma^t$ bajando de 1 a unos 0,61 en 500 pasos (T30 ~04:20).

**Resultado** (celda 38, interrumpida en el episodio 2.725, $\varepsilon\approx 0{,}065$; figura de la celda 39). **Nota:** la tabla es un resumen propio del log, que tiene una línea por episodio.

| Episodios | $\varepsilon$ al inicio | Duración media | Máximo |
|---|---|---|---|
| 1-227 | 0,999 | 20 | 74 |
| 455-681 | 0,63 | 39 | 192 |
| 909-1.135 | 0,40 | 65 | 291 |
| 1.363-1.589 | 0,26 | 81 | 278 |
| 1.817-2.043 | 0,16 | 109 | 429 |
| 2.044-2.270 | 0,13 | 136 | 500 |
| 2.498-2.724 | 0,08 | 121 | 442 |

La figura muestra una tendencia creciente con muchísimo ruido, y se mantienen muchos episodios muy cortos hasta el final. **Test** con $\varepsilon=0$: 426 pasos en un episodio (celda 42). La tabla final tiene valores del orden de 3-38 (celda 40) y se guarda en CSV (celda 44).

En la ejecución en clase (T30 ~04:20-04:26) el comportamiento fue similar: mucho más rápido que una red, ruido debido a $\varepsilon$, y una curva que claramente sube. En los vídeos de prueba el palo acaba cayendo por un lado, según el profesor, quizá por un estado inicial desfavorable o una zona no aprendida. Afirma que con un modelo continuo y suficiente entrenamiento se logra la estrategia perfecta (T30 ~04:26-04:27). **Nota:** la ejecución guardada de `2-cart_pole_DQN` no llega a demostrarlo (§13).

Las celdas 0-5 son título, teoría, importaciones, función de vídeo y lista de entornos de Gym; las 8, 16-18, 20, 23, 25, 29-30, 32-34, 37, 41, 43 y 45 son apoyo, títulos o quedan vacías.

*Fuente: 1-cart_pole_QLearning, celdas 0-45; T30 ~04:05-04:27.*

---

## 13. Código de clase: CartPole con DQN (`2-cart_pole_DQN`)

**Idea.** Es el mismo problema sin discretizar. Solo cambian el modelo y el aprendizaje; el bucle y el entorno se mantienen, porque buena parte del código "es copiar y pegar" (T30 ~04:30-04:34).

**Teoría que resume el notebook** (celdas 2-3): entrada $s$, una salida por acción, condición de consistencia $Q(s,a)=r+\gamma\max Q(s',a')$, e "inconsistencia" $(r+\gamma\max Q(s',a')-Q(s,a))^2$. **Nota:** en la celda 2 falta un paréntesis de cierre en esa fórmula.

**Red** (celdas 6, 8-10): entrada de 4 variables, una capa densa de 100 ReLU y salida lineal de 2. Total de 702 parámetros (500 + 202). El profesor cree que 100 es excesivo y que con menos iría más rápido (T30 ~04:32). Una red recién creada da valores cercanos a 0, como $[-0{,}0016;\,-0{,}0134]$ (celda 13). `predict` espera un conjunto de vectores, de ahí `reshape(1,-1)` (celdas 6, 12).

**Hiperparámetros** (celdas 6, 8, 15):

| Parámetro | Valor |
|---|---|
| $\gamma$ | 0,9 |
| Optimizador | SGD, $10^{-3}$ (Adam queda comentado) |
| Coste | MSE |
| Memoria | `deque(maxlen=20000)`, que borra sola lo más antiguo |
| $\varepsilon$ | 1 → mínimo 0,01, ×0,995 en cada llamada de *replay* |
| Batch | 128, sin reemplazo (`random.sample`) |
| Episodios | 1.000 previstos |
| Red objetivo y recorte | No hay |

***Replay* vectorizado** (celda 6, esencia):

```python
minibatch = random.sample(self.memory, batch_size)
# ... separar en estados, acciones, rs, estados_siguientes, dones
targets = self.model.predict(estados, verbose=0).copy()
Qs_sig  = self.model.predict(estados_siguientes, verbose=0)
for i in range(N):
    if dones[i]: targets[i][acciones[i]] = rs[i]
    else:        targets[i][acciones[i]] = rs[i] + self.gamma*Qs_sig[i].max()
self.model.train_on_batch(estados, targets)      # 1 sola época
```

Hay dos versiones: `experienceReplay` y `experienceReplay_lento`. La lenta es más legible y la otra más eficiente (T30 ~04:34). **Nota:** no son equivalentes. La lenta da $N$ pasos de gradiente con batch 1, recalculando con la red ya modificada; la rápida da un único paso con batch $N$.

**Bucle** (celda 15):

```python
for episodio in range(nepisodios):
    estado = entorno.reset(); done = False; tiempo = 0
    while not done:
        accion = agente.choose_action(estado)
        estado_nuevo, r, done, _ = entorno.step(accion)
        agente.remember(estado, accion, r, estado_nuevo, done)
        estado = estado_nuevo; tiempo += 1
    if len(agente.memory) > batch_size:
        agente.experienceReplay(batch_size)        # al final del episodio
```

Hay una línea comentada, `#if done: r = -10`, para castigar la caída: es otra opción de diseño del refuerzo. Y cada 50 episodios guarda el modelo.

**Resultado** (celdas 15, 17, 19). Interrumpido en el episodio 465 ($\varepsilon\approx 0{,}10$). La figura de la celda 17 muestra que las duraciones suben poco, de unos 10-50 a unos 20-100, con un máximo cerca de 109 y mucho ruido. **Test:** 88 pasos en un episodio. En clase se lanzó al final y se terminó sin ver el resultado: solo se comentó que va más lento a medida que el agente mejora, porque los episodios duran más, y que $\varepsilon$ es muy alto al principio (T30 ~04:37-04:38).

**Nota (lectura de este resultado).** Este es el aviso práctico más útil para el taller, y es una interpretación propia; el material no la desarrolla:

- **Muy pocas actualizaciones.** Un solo paso de SGD por episodio supone 465 pasos de gradiente en total, con $\alpha=10^{-3}$, frente a miles de transiciones recogidas. La memoria recibió mucho más de lo que se entrenó con ella.
- **$\gamma=0{,}9$ no sigue la regla de §10.** Con $0{,}9^{500}\approx 10^{-23}$, el horizonte efectivo es de unos 10 pasos, con $Q\approx 10$; el notebook tabular usa 0,999 para el mismo problema.
- **Comparación con la tabla.** La tabla alcanza 426 pasos con 2.725 episodios, mientras que este DQN interrumpido da 88: no prueba que DQN sea peor, pero muestra que no es mejor por defecto y que hay que dedicar actualizaciones y ajustes.
- **Lo que falla no es la teoría, es la configuración.** Antes de añadir complejidad, conviene comprobar cuántos pasos de gradiente se dan por transición.

Las celdas 0-1, 4 son importaciones y vídeo; 5 está vacía; 7 crea el entorno; 11-13 son pruebas; 14 y 18 son títulos; 16 guarda; 20 muestra el vídeo; 21 está vacía.

*Fuente: 2-cart_pole_DQN, celdas 0-21; T30 ~04:30-04:39.*

---

## 14. Erratas y discrepancias de las fuentes

**Nota:** todas las correcciones de esta sección son razonamiento propio, salvo donde se indica otra fuente.

| Dónde | Qué dice | Corrección o comentario |
|---|---|---|
| DRL1 p115 (y p111-p114) | "se copia la Q' en Q" | **Al revés:** se copia $Q$ (la que se entrena) en $Q'$ (objetivo). Lo confirma p116, "reset $\hat Q=Q$" |
| DRL1 p111-p115 | Fotogramas con "cada C iteraciones" y final con "cada k" | Es el mismo parámetro; p116 usa $C$ |
| DRL1 p12 | $256^{100000}$ | $100\times100=10.000$ píxeles, así que $256^{10000}$, como dice p18 |
| DRL1 p8, p62 | Último término $Q(s,a)$ en una fórmula con $s_t,a_t$ | Debe ser $Q(s_t,a_t)$, como en p86 |
| DRL1 p66 | $G_t=R_t+\gamma R_{t+1}+\dots=\sum_{k}\gamma^k R_{t+k+1}$ | Los dos lados no coinciden: la suma empieza en $R_{t+1}$. Vale $\sum_k\gamma^k R_{t+k}$ con la notación de $r_t$ del mazo, o se empieza en $R_{t+1}$ con la de [00_fundamentos_mdp.md](00_fundamentos_mdp.md) |
| DRL1 p66 | Bellman sin esperanza, con $r$ | Vale en un entorno determinista; en general hay una esperanza, y es muestrear transiciones lo que la aproxima |
| DRL1 p29 | "Minh et al." | Mnih et al. |
| DRL1 p29 | "aprende a jugar a 6 consolas Atari" | Son **juegos** de una consola. **Nota:** Mnih et al. (2013) usaron 7 y el de 2015, 49 (como muestra p40) |
| DRL1 p37, p73 | "Decay Rate" para $\gamma$ | Es el factor de descuento |
| DRL1 p37-p93 | $\hat Q$ en el *target* antes de presentar la red objetivo | En la versión básica, $\hat Q=Q$ (p97) |
| DRL1 p52 | "Convergencia" anunciada como tercer problema | No tiene transparencia propia (ver §4) |
| Fuentes entre sí | Entrenar en cada paso (p106, p116, `1-catch`) o al final del episodio ("lo normal", T30 ~01:39, `2-cart_pole_DQN`) | Las dos se usan; ver §8 |
| T30 ~01:37 y `1-catch`, celda 26 | 2.000 partidas frente a 4.000 | Y en clase bastaron unas 1.000 con batch 10 (T30 ~04:28) |
| T30 ~04:07 | Ángulo "de -0,4 grados a +24 grados" | **Nota:** la transcripción es ruidosa. La documentación de Gym da $\pm 0{,}418$ rad ($\pm 24°$) como rango observable, y el episodio termina en $\pm 0{,}2095$ rad y posición $\pm 2{,}4$, que son los límites de la celda 24 |
| T30 ~00:37 | $\gamma$ no puede ser exactamente 1 "por problemas numéricos" | **Nota:** en tareas episódicas con final garantizado, $\gamma=1$ es válido (ver [00_fundamentos_mdp.md](00_fundamentos_mdp.md) §4); el problema aparece en tareas continuas o con bucles (T30 ~00:05) |
| `1-cart_pole_QLearning`, celda 24 | Velocidades en $\pm 0{,}05$ | Valores reales mucho mayores (§12) |
| `1-catch`, celda 13 | La entrada es "$s$ y $a$" | El código usa solo $s$ (§11) |
| `2-cart_pole_DQN`, celda 2 | Paréntesis sin cerrar en la fórmula | $(r+\gamma\max Q(s',a')-Q(s,a))^2$ |

*Fuente: DRL1, T30 y notebooks indicados.*

---

## Aplicación al Taller 2

`SimpleTradingEnv`: observación continua `[tendencia, posición]`, 3 acciones, comisión del 0,1 %, recompensa ≈ $10^{-3}$ por paso, 494 decisiones por episodio. Según el profesor, en el taller un episodio son unos 500 precios y hará falta entrenar muchos más que 10 o 20 episodios (T30 ~02:41). Lo que no cita fuente es **Nota:** propuesta nuestra.

**Directo del material:**

- **Red:** entrada 2, una o dos capas densas pequeñas con ReLU, salida lineal de 3 (§3). Es un problema con menos entradas que CartPole, y el profesor considera excesivas 100 neuronas allí. No hay fotogramas: la observación ya contiene la tendencia.
- **Transición:** guardar $(s,a,r,s',\text{done})$ y aplicar $y=r$ en estados terminales (§5). **Nota:** el entorno devuelve `terminated=True` al acabarse la serie, pero es un límite de tiempo. D-1.2 (cerrada) lo mantiene como terminal, con $y=r$ en el último paso. Afecta a 1 de 494 pasos, y la regla es la misma en DQN y A3C.
- **Coste y optimización:** MSE con una pasada por minibatch; Adam primero y SGD o un $\alpha$ menor si es inestable (§10). Probar Huber si hay errores grandes.
- **Test *greedy*:** $\varepsilon=0$, sin actualizaciones, y varias series (protocolo D-0.1 de [DECISIONES.md](../DECISIONES.md)). Seguir la curva suavizada, sin fiarse de un episodio.
- **Serie nueva por episodio:** lo que recomienda el profesor con las fechas aleatorias (T30 ~02:57), y ya está en D-0.1.
- **Simplicidad:** empezar por lo mínimo y ampliar solo si hace falta (§10). Q-Learning ya es la referencia a batir.

**Nota (propuestas nuestras, a validar):**

1. **Escala: es el riesgo principal.** $10^{-3}$ es el orden de la señal y de la comisión; con posición abierta, el ruido por paso es de $2\cdot10^{-2}$ a $10^{-1}$ ([notebook_partida.md](notebook_partida.md) §4.3). Las diferencias entre acciones que la red debe resolver son del orden de $10^{-3}$: su contribución al error cuadrático ronda $10^{-6}$ y sus gradientes con MSE casi desaparecen, en especial con SGD. El recorte a $[-1,1]$ **no hace nada aquí**: hay que **escalar** por una constante positiva, que no altera la política óptima. D-1.1 (cerrada) fija esa constante: $r/\sigma_{\text{retorno}}$, con `train_scales(mercado)`. Medirlo imprimiendo mínimo y máximo de los *targets*, como la línea comentada de la celda 6 de `2-cart_pole_DQN`.
2. **Entrada.** La `tendencia` es del orden de $10^{-2}$ al lado de `posición` en $\{0,1\}$, y la red apenas la vería. D-1.1 (cerrada) la divide por $\sigma_{\text{tendencia}}$ dentro del agente, sin tocar el entorno copiado.
3. **Frecuencia de actualización.** Entrenar en cada paso, o cada pocos, no una vez por episodio. El resultado de §13 es el aviso: 494 transiciones por episodio y un solo gradiente sería muy poco. D-0.2 mide unos 3 ms por paso de entrenamiento en PyTorch, lo que lo hace viable.
4. **Memoria.** 494 transiciones por episodio implica que 1.000 no llegan a 2 episodios, y 20.000 son unos 40 episodios. Partir de esos 20.000 y variar; con un mínimo de arranque antes de entrenar.
5. **Red objetivo.** Incluirla desde el principio, con la copia en el sentido correcto ($\theta^-\leftarrow\theta$) y $C$ como hiperparámetro, y comparar con y sin ella para justificarla.
6. **$\gamma$.** La regla de $\gamma^T$ daría $0{,}999^{494}\approx0{,}61$, mientras que $0{,}99^{494}\approx0{,}007$ (horizonte de unos 100 pasos). La observación solo mira 5 pasos atrás, así que un horizonte largo quizá no aporte y aumente la escala y la inestabilidad: es un hiperparámetro que hay que probar.
7. **Exploración.** Como $\varepsilon$ se reparte en 494 pasos por episodio, conviene ligar el decaimiento a los episodios y no bajar muy rápido. Una acción aleatoria aquí puede costar comisiones repetidas, y eso genera datos de una política que opera en exceso: seguir también el número de operaciones.
8. **Semigradiente en PyTorch.** El *target* va dentro de `torch.no_grad()` y la predicción con `gather`, lo que evita construir la matriz $Y$ de Keras:
   ```python
   q = red(s).gather(1, a.unsqueeze(1)).squeeze(1)
   with torch.no_grad():
       y = r + gamma * (1 - done) * red_obj(s2).max(1).values
   loss = F.mse_loss(q, y)
   ```
9. **Mirar los $Q$.** Como en Catch (§10), imprimir los valores de las 3 acciones en estados representativos. **Con `posición=0`, vender y mantener hacen lo mismo**, y con `posición=1`, comprar y mantener también, de modo que esas parejas deberían valer casi igual si la red ha aprendido; una diferencia grande es una señal de alarma sencilla de comprobar.
10. **A3C.** Reutiliza la salida lineal por acción, la normalización y el seguimiento, pero prescinde de memoria y de red objetivo: ver [06_actor_critico_y_a3c.md](06_actor_critico_y_a3c.md).
