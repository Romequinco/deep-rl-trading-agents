# Fundamentos del aprendizaje por refuerzo y MDP

> **Fuentes:**
> - `slides1` (`docs/raw/MIAX-14/slides1.pdf`, 48 páginas): "Fundamentos de RL y Procesos de Decisión de Markov". Es el guion de la sesión del 23 de octubre.
> - `T23` (`docs/raw/transcripciones/`): transcripción automática de esa sesión (líneas 1-7600, hasta el 5x5 de Sutton & Barto). Es ruidosa. Aporta explicaciones, ejemplos y preguntas de alumnos que no están en las diapositivas. Se cita solo con la hora del cue, p. ej. `(T23 ~01:36)`.
> - Notebooks: `gridworld` (ejercicio 5×5 de Sutton & Barto, solo enunciado) y `clase_20261023` (resolución en clase del GridWorld de 4 estados).
> - Libros y cursos de referencia: Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, caps. 1 (1.1-1.6), 2 (2.1-2.4) y 3; clases 1 y 2 de D. Silver (2015, UCL/DeepMind) (slides1 p3).
>
> **Convenciones:** `(slides1 pN)` = página N de las diapositivas. `(gridworld, celda N)` y `(clase_20261023, celda N)` = celda del notebook (numeración desde 0). **Nota:** = aportación propia, no procedente del material de clase. También marcamos como Nota las erratas detectadas y damos entonces lo que dice la fuente y la corrección razonada. La notación sigue las diapositivas: $V^\pi, Q^\pi$ para una política $\pi$ y $V_*, Q_*$ (en las diapositivas, $V^*, Q^*$) para las óptimas.

---

## 1. Qué es el aprendizaje por refuerzo

**Idea.** El RL es un paradigma de aprendizaje automático en el que un agente aprende, interactuando con su entorno, a asociar situaciones (estados) con acciones para maximizar una señal de recompensa numérica a largo plazo (slides1 p4).

- Nadie le dice qué acción tomar. La descubre por prueba y error, aprendiendo de la experiencia (slides1 p4).
- Las acciones no solo determinan la recompensa inmediata, también el siguiente estado y, con él, todas las recompensas futuras (slides1 p4).
- Se sitúa junto al aprendizaje supervisado (clasificación, regresión), el no supervisado y el semi-supervisado como un paradigma distinto (slides1 p5).
- Ejemplos citados: juegos (Atari/DQN, ajedrez y Go con AlphaGo/AlphaZero), robótica y control (locomoción, vehículos autónomos), modelos de lenguaje e IA generativa, y finanzas (gestión de carteras) (slides1 p6).

**Diferencias con el aprendizaje supervisado (T23 ~00:06-00:20):**
- En supervisado se aprende un mapeo entrada → salida con datos etiquetados. En RL no hay etiquetas, solo una recompensa que dice lo bien o mal que lo está haciendo el agente. Por eso el RL explica gran parte de los éxitos recientes: etiquetar datos suele ser difícil, y definir una función de recompensa, aunque no es fácil, sí se puede hacer.
- La recompensa puede llegar retrasada, después de muchos pasos.
- El aprendizaje supervisado supone datos independientes e idénticamente distribuidos. En RL esa suposición no vale: las observaciones forman una secuencia, cada acción lleva a un nuevo estado que condiciona las siguientes, y estado, acción y recompensa están muy correlacionados. Aprender con correlaciones es más difícil y algunos algoritmos usan "trucos" para acercarse al caso i.i.d. (T23 ~00:19-00:20).

**Ejemplo guía de la clase: el ajedrez (T23 ~00:13-00:16).** El estado es la posición de las piezas en el tablero. La acción es mover una pieza. El objetivo es ganar la partida, es decir, la recompensa a largo plazo, no comer una torre por comerla. El oponente forma parte del entorno, de modo que el siguiente estado depende también de lo que haga él. El entorno es no determinista y "la mayor parte de las veces lo son" (T23 ~00:16).

**Otros ejemplos de la clase (T23 ~00:23-00:31):**
- Un humanoide que aprende a correr. La recompensa es la distancia avanzada, una recompensa parcial que da retroalimentación desde el primer paso.
- Un agente de Atari (Breakout) cuya única entrada son los píxeles de la pantalla. Un alumno matizó que se usa una pila de fotogramas, porque con uno solo no se sabe si la pelota sube o baja (T23 ~00:31).
- Un dron que compite en un circuito.

**Reward shaping y diseño de la recompensa (T23 ~00:25-00:27, ~00:48-00:49).**
- Una recompensa que solo se da al llegar a la meta es difícil de aprender si el agente nunca llega a ella al principio. A veces conviene dar recompensas parciales.
- Definir una buena recompensa es "quizás el problema más grande" del RL. Una recompensa que parece razonable lleva al agente a hacer cosas que no se habían previsto, sobre todo si no se penaliza lo indeseado.
- En ajedrez, dar recompensa por comer piezas distraería del objetivo (ganar). Se mencionó como práctica guiar al principio (partidas con pocas piezas, aperturas clásicas, entrenar contra humanos flojos) y luego dejar que el agente juegue contra sí mismo.

**RL frente a algoritmos genéticos (T23 ~00:32-00:36).** Un alumno preguntó por la diferencia. El profesor explicó que el RL aprende como los seres vivos, interactuando con el entorno y mediante recompensas, y estima un valor para cada acción o estado. Un algoritmo genético mantiene una población de individuos (cada uno codifica una política), deja pasar a los mejores y añade mutaciones, de modo que es una búsqueda aleatoria con muchas pruebas y sin concepto de valor.

**Lecturas.** Lectura obligatoria (slides1 p3): Sutton & Barto (2018), cap. 1 (secciones 1.1 a 1.6), cap. 2 (secciones 2.1 a 2.4, bandidos multibrazo) y cap. 3 completo (MDP finitos). El profesor sigue la 2.ª edición "casi a pie de la letra" (T23 ~00:10-00:12). Complementarias: clases 1 (*Introduction to RL*) y 2 (*Markov Decision Processes*) de Silver (slides1 p3).

**Calendario del bloque (T23 ~00:05-00:06).** Las sesiones del 23 y 24 de octubre son teóricas, de unas 5 horas cada una. Después hay una clase no presencial (online) de 2 h 30 el jueves siguiente, planteada como taller: el profesor deja ejercicios al final de cada sesión, que se pueden resolver hasta el final de la sesión del día 24, y pide algo entregable antes de ese jueves para comentarlo. **Nota:** los criterios de evaluación y plazos de ese taller están en [02_td_sarsa_qlearning.md](02_td_sarsa_qlearning.md), §9.

*Fuente: slides1 p3-p6; T23 ~00:05-00:36.*

---

## 2. Elementos formales de un sistema de RL

**Idea.** Un sistema de RL se describe con agente, entorno, política, recompensa, función de valor y, opcionalmente, un modelo del entorno.

| Elemento | Definición resumida | Fuente |
|---|---|---|
| Agente | Quien aprende y decide. Maximiza la recompensa acumulada a lo largo del tiempo | slides1 p7 |
| Entorno | Todo lo externo al agente. Responde a las acciones con una recompensa y un cambio de estado | slides1 p7 |
| Política $\pi$ | Mapeo de estados a acciones. Determinista: $a=\pi(s)$. Estocástica: $\pi(a\mid s)=P(A_t=a\mid S_t=s)$ | slides1 p8 |
| Recompensa $R_t\in\mathbb{R}$ | Escalar que da el entorno en cada paso. Define el objetivo inmediato. Puede ser estocástica | slides1 p9 |
| Función de valor | Recompensa total esperada a largo plazo desde un estado o par estado-acción | slides1 p10 |
| Modelo del entorno (opcional) | Representación que imita al entorno y sirve para planificar | slides1 p11 |

**Matices de la clase.**
- En ajedrez el agente es quien decide el movimiento. El entorno es el tablero y el oponente, con todo lo que se necesita para jugar la partida. Al actuar, el entorno devuelve una recompensa y un nuevo estado (T23 ~00:37-00:38).
- La política es "un conjunto de reglas" o una tabla que dice qué hacer en cada estado. Puede ser determinista o estocástica. Casi todas las que se vean en esos dos días serán deterministas (T23 ~00:39-00:40).
- La recompensa es el objetivo **inmediato**. El objetivo del agente es la recompensa acumulada a largo plazo (T23 ~00:40-00:41). La recompensa también puede ser estocástica: desde el mismo estado y con la misma acción puede darse recompensas o saltos de estado distintos (T23 ~00:40).
- **Recompensa frente a valor** (slides1 p10; T23 ~00:49-00:51). La recompensa es primaria e inmediata, el valor es secundario y a futuro. El agente quiere maximizar el valor, no la recompensa. Una acción puede dar mala recompensa y buen valor, y al revés: una recompensa inmediata muy buena que coloque al agente en una posición que acabe haciéndole perder.
- El valor es lo que determina las acciones y, por tanto, la política. Si se ha calculado bien la función de valor, la política "ya está hecha": elegir la acción que lleve a un estado de más valor. Muchos algoritmos de RL consisten solo en estimar la función de valor (T23 ~00:50-00:51). El valor a veces se asigna a un estado y a veces a un par estado-acción (T23 ~00:50).

**Modelo del entorno: model-based frente a model-free** (slides1 p11; T23 ~00:51-00:55).
- *Conocer* el entorno significa saber, para cada estado y acción, el siguiente estado (o su probabilidad) y la recompensa. *Modelarlo* es aproximarlo a partir de la experiencia. Se puede aprender conociendo el entorno, modelándolo o sin modelo.
- Los algoritmos *model-based* usan un modelo para planificar (simular el futuro antes de actuar). Los *model-free* aprenden directamente por ensayo y error sin modelo explícito.
- La programación dinámica no aprende un modelo: **conoce** el entorno. Es lo más sencillo con lo que trabajar, y el material de estas primeras clases siempre será de estos tipos (T23 ~00:54-00:55).

**Taxonomía de algoritmos** (slides1 p12, figura de OpenAI Spinning Up). Es un árbol con la raíz "RL Algorithms":
- *Model-Free RL*: ramas "Policy Optimization" (Policy Gradient, A2C/A3C, PPO, TRPO; DDPG, TD3 y SAC se sitúan entre ambas ramas) y "Q-Learning" (DQN, C51, QR-DQN, HER).
- *Model-Based RL*: ramas "Learn the Model" (World Models, I2A, MBMF, MBVE) y "Given the Model" (AlphaZero).

Comentario del profesor (T23 ~00:55-00:56): dentro de los model-free hay algoritmos basados en estimar la función de valor y otros que aprenden la política directamente, sin pasar por el valor. El árbol solo recoge algoritmos modernos y deja fuera clásicos como la programación dinámica. A partir de la semana siguiente se verán los de política y los de valor con redes neuronales (DQN y su extensión).

### Dilema exploración-explotación (slides1 p13)

- **Explotación:** elegir la acción con mayor recompensa esperada según el conocimiento actual. Es necesaria para maximizar la recompensa a corto plazo.
- **Exploración:** elegir acciones subóptimas según la estimación actual para obtener información. Es necesaria para descubrir mejores acciones a largo plazo.
- Ninguna estrategia pura sirve. Explotar desde el principio atrapa al agente en óptimos locales. Explorar siempre impide rentabilizar lo aprendido.

Matiz del profesor (T23 ~00:56-00:59): una acción que "parece mala" puede serlo solo porque se ha probado poco. Si se prueban cinco acciones un poco y se explota la que ha salido mejor, quizá la primera era mejor y no se ha probado lo suficiente.

### Bandido multibrazo, la versión sin estados (slides1 p14-p21)

- $k$ acciones (jugar cada una de $k$ tragaperras). Cada una tiene una recompensa esperada desconocida $q_*(a)=\mathbb{E}[R_{t+1}\mid A_t=a]$. Las acciones no modifican el estado del entorno (slides1 p14). Aquí recompensa y valor coinciden, porque no hay estado: se juega y se vuelve al mismo estado (T23 ~01:01).
- Estimación por promedio muestral: $Q_t(a)=\dfrac{\sum_{i=1}^{t-1}R_i\,\mathbb{I}(A_i=a)}{N_t(a)}$ (slides1 p15). $q_*$ es el valor real y $Q_t$ el estimado a partir de la experiencia (T23 ~01:03).
- **Greedy:** $A_t=\arg\max_a Q_t(a)$, con empates al azar. Nunca explora (slides1 p16). Como el valor real es una media con ruido, puede ocurrir que la mejor máquina dé malos resultados las primeras veces y se descarte para siempre (T23 ~01:04-01:05).
- **$\epsilon$-greedy:** con probabilidad $1-\epsilon$ explota y con $\epsilon$ elige una acción al azar entre las $|A|$ (slides1 p17):
$$P(A_t=a)=\begin{cases}1-\epsilon+\frac{\epsilon}{|A|} & a=\arg\max_b Q_t(b)\\ \frac{\epsilon}{|A|} & \text{resto}\end{cases}$$
  En la práctica se sortea un número en $[0,1)$: si es menor que $\epsilon$ se elige una acción al azar, y si no, la de mayor valor estimado (T23 ~01:17). A largo plazo puede convenir un $\epsilon\neq0$ (slides1 p17).
- **10-armed testbed** (slides1 p18, figura 2.1 de Sutton & Barto): $q_*(a)\sim\mathcal N(0,1)$ y $R\sim\mathcal N(q_*(a),1)$. La figura muestra, para las 10 acciones, una campana de recompensas centrada en cada $q_*(a)$ (valores como $q_*(3)\approx1{,}5$ y $q_*(6)\approx-1{,}5$). Aunque la acción 3 tiene la media más alta, una tirada concreta de otra acción puede salir mejor.
- **Resultados** (slides1 p19-p20; T23 ~01:09-01:12). Se hicieron 1000 pasos, promediados sobre 2000 ejecuciones (T23 ~01:09), con $\epsilon=0$ (greedy), $0{,}01$ y $0{,}1$ constantes. Valores leídos en las gráficas:

| Estrategia | Recompensa media (≈ pasos 600-1000) | % de acción óptima (paso 1000) |
|---|---|---|
| $\epsilon=0$ (greedy) | ≈ 1,05, se estanca pronto | ≈ 35 % (el profesor dice "un tercio") |
| $\epsilon=0{,}01$ | ≈ 1,3, sigue subiendo | ≈ 60 %, sigue subiendo |
| $\epsilon=0{,}1$ | ≈ 1,38, la más alta | ≈ 80 % |

  El greedy sube más rápido al principio pero no pasa de ahí. Con $\epsilon=0{,}1$ el agente llega a elegir la acción óptima casi el 80 % de las veces. Todas parten del 10 % (azar entre 10 acciones).
- **Por qué el greedy no se queda en el 10 %** (T23 ~01:12-01:18): al principio todos los valores son 0 y el empate se rompe eligiendo la primera acción. Esa recompensa pasa a ser su estimación. Si es positiva, el agente se queda con ella. Si es negativa, prueba otra. Por azar, en bastantes ejecuciones acierta con la mejor y se queda ahí, y de ahí sale el ≈ 33 %.
- **Decaimiento de $\epsilon$** (T23 ~01:10, ~01:20): lo más lógico es empezar con $\epsilon$ alto y bajarlo, porque con más pasos las estimaciones son mejores. En el límite de infinitos pasos convendría $\epsilon\to0$, pero "más pequeño y distinto de 0" garantiza que todas las acciones se exploran infinitas veces. En los experimentos se usó $\epsilon$ constante y no se probó un $\epsilon$ decreciente.
- **No siempre debe ir a 0** (T23 ~01:20-01:22). Si el entorno cambia con el tiempo, p. ej. se interpretan las acciones como invertir en un activo y el valor como su retorno medio, y las propiedades del mercado cambian, entonces hay que seguir aprendiendo. Con $\epsilon=0$ el agente deja de explorar y no se entera de que otro activo es ahora mejor. En un entorno real, "quizá siempre viene bien seguir explorando".
- Lo que aprende el agente es una estimación de la **media** de cada acción, no su distribución (T23 ~01:22-01:23).
- **Actualización incremental** sin guardar el historial (slides1 p21; T23 ~01:24-01:28):
$$Q_{n+1}=\frac1n\sum_{i=1}^{n}R_i=Q_n+\tfrac1n\,(R_n-Q_n)=Q_n\left(1-\tfrac1n\right)+\tfrac1n R_n$$
  con $Q_n=\frac{1}{n-1}\sum_{i=1}^{n-1}R_i$. Es un promedio ponderado: la nueva recompensa pesa $1/n$ y la estimación previa $1-1/n$. Con $n=1$ la estimación previa tiene peso 0. Con $n=2$ cada una pesa $1/2$, y cada vez la última pesa menos. El profesor lo comparó con la media de la altura de 100 000 personas: añadir un jugador de baloncesto casi no la mueve (T23 ~01:26).

**Nota:** esta es la regla "estimación nueva = estimación vieja + paso × (objetivo − estimación vieja)" que reaparece en Monte Carlo, SARSA y Q-Learning (ver [01_dp_y_montecarlo.md](01_dp_y_montecarlo.md) y [02_td_sarsa_qlearning.md](02_td_sarsa_qlearning.md)).

*Fuente: slides1 p7-p21; T23 ~00:37-01:28.*

---

## 3. El bucle agente-entorno y la propiedad de Markov

**Idea.** El agente y el entorno interactúan en pasos de tiempo discretos y generan una trayectoria. Si el estado contiene toda la información relevante del pasado, el problema es un proceso de decisión de Markov (MDP).

**Bucle de interacción** (slides1 p22). El diagrama tiene dos cajas, Agente (política $\pi$) y Entorno (dinámica del mundo). Una flecha lleva la acción $A_t$ del agente al entorno, y otras dos devuelven el estado $S_{t+1}$ y la recompensa $R_{t+1}$ al agente. Se genera la trayectoria (o historial)
$$S_0,A_0,R_1,S_1,A_1,R_2,S_2,A_2,R_3,\dots$$
El tiempo es discreto y la trayectoria puede ser finita o infinita (T23 ~01:29).

**Nota:** la diapositiva dice "el retorno devuelve una recompensa numérica y cambia de estado". Por el diagrama debe ser "el *entorno*".

**Propiedad de Markov** (slides1 p23-p24). El estado $S_t$ recoge toda la información relevante del pasado. La probabilidad del siguiente estado y recompensa depende solo de $(S_t,A_t)$, no de cómo se llegó a ellos:
$$P(S_{t+1}=s',R_{t+1}=r\mid S_t,A_t,S_{t-1},A_{t-1},\dots,S_0,A_0)=P(S_{t+1}=s',R_{t+1}=r\mid S_t,A_t)$$
- Un entorno que la cumple es **markoviano**. Una tarea de RL en uno de ellos es un **MDP**. Si estados y acciones son finitos, un **MDP finito (fMDP)** (slides1 p24).
- **Ejemplos de la clase.**
  - El ajedrez es markoviano: la posición del tablero contiene todo lo relevante, "me da igual cómo ha llegado" (T23 ~01:30-01:31).
  - **Contraejemplo:** el fotograma de Breakout. Con una sola imagen no se sabe de dónde viene la pelota, y su evolución depende de ello. Esa pantalla no es un estado markoviano (T23 ~01:32-01:33). Por eso se apilan fotogramas.
- **Hipótesis de trabajo.** Casi todas las tareas del curso se tratarán como fMDP. "Aunque la aproximación no es del todo válida, es una buena hipótesis de trabajo y simplifica mucho las cosas" (T23 ~01:33-01:34).
- **Finanzas (T23 ~01:34-01:35).** Cuando un alumno preguntó por el mundo financiero, el profesor respondió "me temo que no" es markoviano, aunque matizó que depende del entorno y de cómo se defina el estado, y que quizá en algunos casos particulares sea razonable. No quiso dar una respuesta tajante.

**Dinámica del fMDP** (slides1 p25; T23 ~01:35-01:37). Toda la dinámica del entorno es una única distribución $p:\mathcal S\times\mathcal R\times\mathcal S\times\mathcal A\to[0,1]$,
$$p(s',r\mid s,a)=P(S_{t+1}=s',R_{t+1}=r\mid S_t=s,A_t=a).$$
De ella se derivan la probabilidad de transición y la recompensa esperada:
$$P(s'\mid s,a)=\sum_{r}p(s',r\mid s,a),\qquad R(s,a,s')=\sum_r r\,\frac{p(s',r\mid s,a)}{p(s'\mid s,a)}.$$
Si se conoce $p$, el entorno es completamente conocido y se puede hallar la política óptima. En el resto del curso el profesor usará siempre $p(s',r\mid s,a)$ para calcular esperanzas (T23 ~01:36-01:37).

*Fuente: slides1 p22-p25; T23 ~01:28-01:37.*

---

## 4. Retorno y descuento

**Idea.** El agente maximiza el **retorno descontado**, la suma ponderada de las recompensas futuras.
$$G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\dots=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1},\qquad \gamma\in[0,1)$$
(slides1 p26). $\gamma$ es el factor de descuento y determina el valor presente de las recompensas futuras. En tareas episódicas que terminan en $T$, $G_t=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1}$ (slides2 p31).

**Recompensa frente a retorno** (T23 ~00:57, ~01:38). La recompensa es el valor inmediato de la acción, el "dolor o placer" de ese paso. El retorno es la suma, ponderada, de todas las recompensas futuras. La primera pesa 1, la siguiente $\gamma$, la siguiente $\gamma^2$, etc.

**Para qué sirve $\gamma$** (T23 ~01:39-01:41, ~01:54-01:57). El profesor dio dos motivos:
1. **Dar menos peso a lo lejano**: "más vale pájaro en mano". Con $\gamma=0{,}9$ la recompensa inmediata cuenta al 100 %, la siguiente al 90 %, la otra al 81 %. Una promesa de recompensa futura alta no garantiza obtenerla, porque la política es estocástica y el entorno también. Cuanto más lejos, más caminos posibles hay, así que una recompensa a 20 pasos "tiene que valer poco".
2. **Convergencia de la suma.** Si $\gamma=1$ y el horizonte es infinito con recompensas positivas, todos los retornos son infinitos y "todos los estados son igualmente buenos", de modo que no se puede aprender. Con $\gamma<1$ y recompensas acotadas la suma converge. El profesor lo comparó con descontar flujos de caja (T23 ~01:56).

**Tareas episódicas frente a continuas** (T23 ~01:43; slides1 p34). Si el número de pasos de cada episodio es finito, se puede relajar la condición y tomar $\gamma=1$. En el ejemplo del GridWorld de 4 estados se usa $\gamma=1$ (slides1 p34). La condición de existencia y unicidad de la solución de las ecuaciones de Bellman es $\gamma<1$ o que toda trayectoria termine en un número finito de pasos (slides2 p11).

**Nota:** la diapositiva exige $\gamma\in[0,1)$ y luego usa $\gamma=1$ en el ejemplo episódico. No es una contradicción: el rango abierto es el caso general (horizonte infinito) y $\gamma=1$ solo vale con episodios finitos, como explicó el profesor.

*Fuente: slides1 p26, p34; slides2 p11, p31; T23 ~01:37-01:44, ~01:54-01:57.*

---

## 5. Funciones de valor

**Idea.** Miden cómo de bueno es estar en un estado, o ejecutar una acción en un estado, **bajo una política $\pi$**.

- **Valor de estado:** $V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$ (slides1 p27).
- **Valor de acción:** $Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]$: se parte de $s$, se ejecuta $a$ y luego se sigue $\pi$ (slides1 p28).
- Relaciones entre ambas (slides1 p29):
$$V^\pi(s)=\sum_{a}\pi(a\mid s)\,Q^\pi(s,a),\qquad Q^\pi(s,a)=\sum_{s',r}p(s',r\mid s,a)\,[\,r+\gamma V^\pi(s')\,]$$

**Matices (T23 ~01:44-01:49).**
- El valor de un estado va ligado a una política: cada política asigna un valor distinto al mismo estado (una política aleatoria da unos valores y otra con lógica da otros).
- $V^\pi(s)$ es la media de los $Q^\pi(s,a)$ ponderada por la probabilidad de elegir cada acción.
- $Q^\pi(s,a)$ es, para cada posible salto $(s',r)$, la recompensa $r$ más el valor del estado siguiente multiplicado por $\gamma$ (ya no es la "primera recompensa" sino la segunda en adelante). Se pondera con la probabilidad $p(s',r\mid s,a)$ porque el mismo par $(s,a)$ puede llevar a estados y recompensas distintos.
- El valor de un estado terminal es 0, porque no hay transiciones ni recompensas desde él (T23 ~01:59, ~02:42).

*Fuente: slides1 p27-p29; T23 ~01:44-01:49.*

---

## 6. Ecuaciones de Bellman de expectativa

**Idea.** Descomponen el valor en recompensa inmediata esperada más valor descontado del siguiente estado esperado (slides1 p30-p31).
$$V^\pi(s)=\sum_{a}\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\,[\,r+\gamma V^\pi(s')\,]$$
$$Q^\pi(s,a)=\sum_{s',r}p(s',r\mid s,a)\Big[r+\gamma\sum_{a'}\pi(a'\mid s')\,Q^\pi(s',a')\Big]$$

- Se obtienen sustituyendo $Q^\pi$ en la expresión de $V^\pi$ (y al revés, para la ecuación de $Q^\pi$). El profesor dijo que "la forma más fácil de ver la relación de Bellman" es sustituir una fórmula en la otra (T23 ~01:49-01:54).
- La ecuación de $V^\pi$ es un **sistema de $|\mathcal S|$ ecuaciones lineales con $|\mathcal S|$ incógnitas**: una por estado. Se resuelve analíticamente (espacios pequeños) o con métodos iterativos de programación dinámica. $V^\pi$ es su única solución (slides1 p30).
- Como el sistema es lineal se puede escribir de forma matricial, pero con muchos estados no es viable: "para un entorno con 1000 estados no se puede hacer" analíticamente (T23 ~01:50-01:51). Un alumno lo comparó con un árbol de decisión que se recorre hacia atrás sumando. El profesor dijo que sí, con la salvedad de que la expansión desde cada estado y acción se pondera con probabilidades (T23 ~01:51-01:52).

*Fuente: slides1 p30-p31; T23 ~01:49-01:54.*

---

## 7. Política y funciones de valor óptimas

**Idea.** Se ordenan las políticas por su valor. Existe una óptima, y sus funciones de valor cumplen unas ecuaciones con máximo en lugar de promedio.

- **Orden parcial de políticas:** $\pi\ge\pi'$ si y solo si $V^\pi(s)\ge V^{\pi'}(s)\ \forall s$. Siempre existe al menos una política óptima $\pi_*$ mejor o igual que todas (slides1 p36).
- Puede haber varias óptimas, pero todas comparten $V_*(s)=\max_\pi V^\pi(s)$ y $Q_*(s,a)=\max_\pi Q^\pi(s,a)$, con $V_*(s)=\max_aQ_*(s,a)$ y $Q_*(s,a)=\sum_{s',r}p(s',r\mid s,a)[r+\gamma V_*(s')]$ (slides1 p37).

**Ecuaciones de optimalidad de Bellman** (slides1 p38): la media ponderada por $\pi$ se convierte en un máximo.
$$V_*(s)=\max_a\sum_{s',r}p(s',r\mid s,a)\,[\,r+\gamma V_*(s')\,]$$
$$Q_*(s,a)=\sum_{s',r}p(s',r\mid s,a)\Big[r+\gamma\max_{a'}Q_*(s',a')\Big]$$

**Nota:** los rótulos de la diapositiva p38 dicen "ecuación de optimalidad para $V^\pi(s)$" y "para $Q^\pi(s,a)$". Por las fórmulas debe ser $V_*$ y $Q_*$ (la política óptima no depende ya de $\pi$).

- Es un sistema de $|\mathcal S|$ ecuaciones **no lineales** con solución única. Esa solución es la función de valor óptima, que determina la política óptima: toda la probabilidad a la acción que maximiza recompensa más valor descontado del siguiente estado (slides1 p39; T23 ~03:09-03:10).
- El $\max$ no es lineal, así que impide la solución directa por inversión matricial. Hacen falta métodos iterativos (programación dinámica, Q-Learning, value iteration). **Los distintos métodos de RL difieren en cómo aproximan la solución de estas ecuaciones** (slides1 p39; T23 ~03:11).
- La iteración funciona así: se parte de una estimación inicial (por ejemplo ceros), se calcula la siguiente estimación con el lado derecho de la ecuación y se repite (T23 ~03:09).
- **Una política óptima no hace lo mismo en todos los estados** (T23 ~03:23-03:25). Un alumno preguntó si la política óptima encontrada servía "para todos los estados". El profesor aclaró que sí: la política óptima es mejor o igual que cualquier otra en *todos* los estados, pero eso no significa que repita la misma acción en todos. La política es una tabla que dice qué hacer en cada estado. En el ejemplo de 4 estados sale "ir abajo" en todos por la forma del entorno, no por necesidad.
- **Una política no óptima es peor en al menos un estado** (T23 ~03:28).

*Fuente: slides1 p36-p39; T23 ~03:01-03:11, ~03:22-03:28.*

---

## 8. Ejemplo de clase: GridWorld de 4 estados

**Entorno** (slides1 p32-p34; T23 ~01:57-02:01). Es el ejemplo de Sutton & Barto. El tablero es de 2×2: arriba $s_0$ y $s_1$, abajo $s_2$ (meta, resaltada en amarillo) y $s_3$.
- Estados $\{s_0,s_1,s_2,s_3\}$ con $s_2$ meta (terminal). Acciones $\{D,A\}$ (derecha, abajo). Los episodios empiezan en $s_0$. Es una tarea episódica.
- Entorno determinista. Desde $s_0$: $D\to s_1$ y $A\to s_2$. Desde $s_1$: $D\to s_1$ (choca contra el muro) y $A\to s_3$. Desde $s_3$: $D\to s_3$ (muro) y $A\to s_2$. Desde $s_2$ no hay transiciones.
- Recompensa $0$ en las acciones que llevan al terminal $s_2$ y $-1$ en el resto. $\gamma=1$, porque el número de transiciones es siempre finito.

**Por qué recompensa $-1$ y no $+1$ en la meta** (T23 ~02:00-02:01). Cada paso sin llegar a la meta resta. Así el agente aprende "ve a la meta lo más rápido posible". Con un $+1$ al llegar y $0$ en el resto, daría igual cuánto tardara. La alternativa sería $0$ en lo intermedio más un descuento $\gamma<1$, pero aquí se penaliza la demora con $\gamma=1$.

**Evaluación de la política equiprobable** (slides1 p35; T23 ~02:38-02:48). La política elige $D$ o $A$ con probabilidad $0{,}5$ en cada estado. La ecuación de expectativa para $s_0$ con $\gamma=1$: la mitad de las veces $D$ (recompensa $-1$, va a $s_1$) y la otra mitad $A$ (recompensa 0, va a $s_2$ con valor 0).
$$V^\pi(s_0)=0{,}5V^\pi(s_1)+0{,}5V^\pi(s_2)-0{,}5$$
$$V^\pi(s_1)=0{,}5V^\pi(s_1)+0{,}5V^\pi(s_3)-1$$
$$V^\pi(s_2)=0,\qquad V^\pi(s_3)=0{,}5V^\pi(s_3)+0{,}5V^\pi(s_2)-0{,}5$$
Solución: $V^\pi=(-2,-3,0,-1)$.

**Interpretación** (T23 ~02:48-02:49). $s_3$ es el mejor estado no terminal ($-1$), porque haga lo que haga llega a la meta en uno o más pasos con probabilidad $\tfrac12$ por paso. $s_1$ es el peor ($-3$): desde él hay que pasar por $s_3$ y se puede perder tiempo chocando contra el muro. La política "óptima" en términos intuitivos es bajar siempre.

**Optimalidad** (slides1 p40-p41; T23 ~03:11-03:17): se sustituyen las medias por máximos.
$$V_*(s_0)=\max(-1+V_*(s_1),\,0),\quad V_*(s_1)=\max(-1+V_*(s_1),\,-1+V_*(s_3)),$$
$$V_*(s_2)=0,\quad V_*(s_3)=\max(-1+V_*(s_3),\,0)$$
Solución $V_*=(0,-1,0,0)$. Resolución "a ojo" (T23 ~03:14-03:15): las recompensas son $0$ o $-1$ y los valores son sumas de recompensas, luego siempre $\le0$. Por tanto en $V_*(s_3)$ el $\max$ es $0$ (la opción de bajar), en $V_*(s_0)$ también, y $V_*(s_1)=-1$. La **política óptima** se obtiene eligiendo en cada estado la acción que maximiza recompensa inmediata más valor del siguiente estado: $\pi_*(s_0)=\pi_*(s_1)=\pi_*(s_3)=A$ (slides1 p41). En $s_0$: $D$ da $-1+V_*(s_1)=-2$ y $A$ da $0$, luego $A$ (T23 ~03:15-03:16). Para $s_2$ no hay acción porque es terminal.

**Comparación de políticas** (T23 ~03:26-03:28). Se evaluaron dos políticas: la aleatoria y "bajar siempre". Una tercera, "derecha en $s_0$ y abajo en $s_1$ y $s_3$", da $V(s_0)=-2$, $V(s_1)=-1$ y $V(s_3)=0$ (el profesor lo calculó de cabeza al final). Es mejor o igual que la aleatoria en todos los estados ($-2\ge-2$, $-1\ge-3$, $0\ge-1$) y peor que la óptima en $s_0$ ($-2<0$): una política no óptima es peor en al menos un estado. El profesor planteó el cálculo como ejercicio.

### Solución matricial e iterativa en clase (`clase_20261023`)

La ecuación de expectativa se escribe como $V=AV+b$ (T23 ~02:44-02:56). $A$ es la matriz de transición bajo la política equiprobable ponderada por $\gamma=1$ y $b$ el vector de recompensas esperadas (celdas 1 y 2):
$$A=\begin{pmatrix}0&.5&0&0\\0&.5&0&.5\\0&0&0&0\\0&0&0&.5\end{pmatrix},\qquad b=(-0{,}5,\,-1,\,0,\,-0{,}5)^\top$$
- **Directa** (celdas 3-5). De $V=AV+b$ se sigue $IV=AV+b\Rightarrow(I-A)V=b$. Se resuelve con `np.linalg.solve(np.eye(4)-A, b)` y da `[-2, -3, 0, -1]`. Se la llama "resolver el sistema $Ax=b$" y el profesor insistió en que no se hace a mano.
- **Iterativa** (celdas 6-10). Se parte de `V = np.zeros((4,1))` y se repite `V = A @ V + b`. El profesor probó primero con 10 iteraciones, que no bastaban (resultado `[-1.988, -2.987, 0, -0.999]`, celda 8), y subió a **20 iteraciones**, que dan `[-1.99999997, -2.99999997, 0, -1.]` (celda 10; T23 ~02:58-03:00). Es la forma habitual de resolverlo: no se sabe en cuántas iteraciones converge, pero converge. Valores iniciales cualesquiera, con la única condición de que los estados terminales valgan 0 (T23 ~03:00).
- **Óptima** (celdas 11-14). De nuevo desde ceros, 20 iteraciones de las ecuaciones de optimalidad escritas a mano y actualizadas **in-place**: `V[0]=max(-1+V[1],0)`, `V[1]=max(-1+V[1],-1+V[3])`, `V[2]=0`, `V[3]=max(-1+V[3],0)`. Da `[0, -1, 0, 0]`.
- Sin criterio de parada: 20 iteraciones bastan en este problema pequeño. **Nota:** la cota habitual para parar es $\max_s|V_{k+1}-V_k|<\theta$ (slides2 p13).
- **Nota:** en un modo "in-place" (cada línea usa ya los valores recién actualizados) el orden de las líneas puede cambiar la velocidad de convergencia, pero no el punto fijo.

*Fuente: slides1 p32-p35, p40-p41; clase_20261023 celdas 0-14; T23 ~01:57-02:01, ~02:38-03:00, ~03:11-03:28.*

---

## 9. Ejercicio de GridWorld 5×5 (`gridworld`, Sutton & Barto)

**Entorno** (gridworld, celda 1; slides1 p42; T23 ~03:28-03:31).
- Tablero de $5\times5$ y acciones $\{N,S,E,W\}$ (arriba, abajo, derecha, izquierda).
- Una acción que sacaría al agente del tablero lo deja en su casilla con $R=-1$.
- En la casilla $A$ toda acción lleva a $A'$ con $R=10$. En $B$ toda acción lleva a $B'$ con $R=5$. El resto, $R=0$. $\gamma=0{,}9$ (el valor del libro, sin motivo especial: la tarea es episódica y se podría usar otro, T23 ~03:32).
- La figura de la diapositiva (figura 3.1 de Sutton & Barto) muestra el tablero con $A$ en la fila 0 y columna 1, $A'$ en la fila 4 y columna 1, $B$ en la fila 0 y columna 3, $B'$ en la fila 2 y columna 3.

**Cuatro ejercicios** (el notebook trae el enunciado y celdas `# To do...`, sin solución):
1. Plantear la ecuación de expectativa de forma matricial para $\pi(a\mid s)=1/4$, con 25 estados numerados de 0 a 24 por filas (arriba a la izquierda es $s_0$). Ejemplo dado para $s_0$ (celda 3): salirse por arriba o por la izquierda da $-1$ y deja en $s_0$; derecha lleva a $s_1$ y abajo a $s_5$, con recompensa 0:
$$V^\pi(s_0)=\tfrac12(-1+\gamma V^\pi(s_0))+\tfrac14\gamma V^\pi(s_1)+\tfrac14\gamma V^\pi(s_5)\iff(1-\tfrac12\gamma)V(s_0)-\tfrac14\gamma V(s_1)-\tfrac14\gamma V(s_5)=-\tfrac12.$$
2. Resolverla con `np.linalg.solve` (celda 9).
3. Resolverla iterativamente desde `v = np.zeros((5,5))` (celdas 13-14).
4. Resolver iterativamente la ecuación de optimalidad desde ceros para obtener $V_*$ y la política óptima (celdas 19-20).

**Cómo lo resolvió el profesor en clase** (T23 ~03:31-03:42; la solución no está en el notebook).
- Con $A$ de $25\times25$ y $b$, la ecuación $V=AV+b$ se resuelve con `np.linalg.solve`. Dice que los resultados coinciden con los del libro.
- Para la versión iterativa escribió una función `update_value(V, gamma)` que primero multiplica $V$ por $\gamma$ (así todos los $V$ que se suman ya van descontados), crea una matriz nueva $W$ del mismo tamaño y guarda en $W$ las medias de las cuatro acciones (recompensa inmediata más $\gamma V$ del estado siguiente). Devuelve $W$ y el bucle hace `V = update_value(V, gamma)`.
- Para la política óptima cambió la media (`np.mean`, que en la transcripción aparece como "min") por `np.max` en todas partes. La política que sale es la del libro.
- Para decidir la acción óptima en cada casilla se mira, por cada acción, recompensa inmediata más el valor del estado al que lleva, y se elige el mayor (T23 ~03:41-03:42).

**Resultados de referencia.**
- Para la política equiprobable ($\gamma=0{,}9$), la figura 3.1 de Sutton & Barto (slides1 p42) da, por filas:
$$\begin{matrix}3{,}3&8{,}8&4{,}4&5{,}3&1{,}5\\1{,}5&3{,}0&2{,}3&1{,}9&0{,}5\\0{,}1&0{,}7&0{,}7&0{,}4&-0{,}4\\-1{,}0&-0{,}4&-0{,}4&-0{,}6&-1{,}2\\-1{,}9&-1{,}3&-1{,}2&-1{,}4&-2{,}0\end{matrix}$$
  La casilla $A$ vale $8{,}8$: haga lo que haga recibe 10, aunque luego aparece en $A'$ (T23 ~03:31). La esquina inferior izquierda es la peor por los golpes contra el borde ($-1$).
- **Nota (verificado):** al resolver el sistema lineal de $25$ ecuaciones con $\gamma=0{,}9$ se reproducen esos valores redondeados a un decimal. Iterando $V\leftarrow\max_a[\,r+\gamma V(s')\,]$ desde ceros se obtiene, de nuevo por filas, $V_*$:
$$\begin{matrix}22{,}0&24{,}4&22{,}0&19{,}4&17{,}5\\19{,}8&22{,}0&19{,}8&17{,}8&16{,}0\\17{,}8&19{,}8&17{,}8&16{,}0&14{,}4\\16{,}0&17{,}8&16{,}0&14{,}4&13{,}0\\14{,}4&16{,}0&14{,}4&13{,}0&11{,}7\end{matrix}$$
  Es la solución óptima del libro (figura 3.5), no incluida en el material de clase.

**Nota (errata):** la diapositiva p42 llama al ejercicio "ejemplo 3.5" y el notebook "ejemplo 3.8" de Sutton & Barto. En el libro el GridWorld de las figuras se presenta como ejemplo 3.5, y el 3.8 es el que resuelve las ecuaciones de optimalidad en ese mismo mundo. Es el mismo problema.

*Fuente: slides1 p42; gridworld celdas 0-20; T23 ~03:28-03:42.*

---

## 10. Ejemplos con solución óptima (FrozenLake y Taxi)

Se muestran, sin desarrollo teórico nuevo, los $V_*$ y $\pi_*$ de dos entornos de Gymnasium. Se resuelven con programación dinámica en [01_dp_y_montecarlo.md](01_dp_y_montecarlo.md).

- **FrozenLake-v1** (slides1 p43-p45). Tablero de $4\times4$: $S$ (salida) arriba a la izquierda, $G$ (meta) abajo a la derecha y cuatro agujeros en las posiciones (1,1), (1,3), (2,3) y (3,0).
  - `is_slippery=False` (p44): los valores en el camino óptimo van de $0{,}95$ en la salida a $1{,}00$ junto a la meta, con la política óptima dirigiendo hacia ella. Mapa de $V_*$ (flechas de $\pi_*$ entre paréntesis), por filas: fila 0: $0{,}95\,(\downarrow)$, $0{,}96\,(\rightarrow)$, $0{,}97\,(\downarrow)$, $0{,}96\,(\leftarrow)$; fila 1: $0{,}96\,(\downarrow)$, $0{,}00$ (agujero), $0{,}98\,(\downarrow)$, $0{,}00$ (agujero); fila 2: $0{,}97\,(\rightarrow)$, $0{,}98\,(\downarrow)$, $0{,}99\,(\downarrow)$, $0{,}00$ (agujero); fila 3: $0{,}00$ (agujero), $0{,}99\,(\rightarrow)$, $1{,}00\,(\rightarrow)$, $0{,}00$ (meta).
  - `is_slippery=True` (p45): la dinámica es estocástica y los valores son bastante menores. $V_*(0,0)=0{,}54$, $0{,}50$ en (0,1), $0{,}74$ en (3,1) y $0{,}86$ en (3,2). La política óptima cambia: en la fila 0 prefiere ir hacia arriba o izquierda (pegarse a la pared) para no caer en un agujero. En la figura, los agujeros y la meta tienen valor $0{,}00$. Mapa completo de $V_*$ (flechas de $\pi_*$), por filas: fila 0: $0{,}54\,(\leftarrow)$, $0{,}50\,(\uparrow)$, $0{,}47\,(\uparrow)$, $0{,}46\,(\uparrow)$; fila 1: $0{,}56\,(\leftarrow)$, $0{,}00$, $0{,}36\,(\leftarrow)$, $0{,}00$; fila 2: $0{,}59\,(\uparrow)$, $0{,}64\,(\downarrow)$, $0{,}62\,(\leftarrow)$, $0{,}00$; fila 3: $0{,}00$, $0{,}74\,(\rightarrow)$, $0{,}86\,(\downarrow)$, $0{,}00$. **Nota:** los valores bajan mucho respecto al caso determinista (de $0{,}95$ a $0{,}54$ en la salida) porque en cada intento hay probabilidad de deslizarse en dirección perpendicular (explicación propia).
  - **Nota:** que los valores no-deslizantes decrezcan 0,01 por casilla sugiere $\gamma=0{,}99$ y recompensa 1 solo al llegar (es una deducción, la diapositiva no da $\gamma$).
- **Taxi-v4** (slides1 p46-p48). Mapa de $5\times5$ con paradas $R$, $G$, $Y$, $B$ y muros. Se muestran $V_*(s)$ y $\pi_*(s)$ en un mapa de calor para dos casos:
  - Pasajero dentro del taxi, destino $R$ (p48): $V_*=20{,}0$ en la casilla $R$ con acción de dejar al pasajero ($D$), y decrece unas $1{,}1$-$1{,}2$ unidades por casilla de distancia, hasta $10{,}7$ en las esquinas lejanas.
  - Pasajero en $B$, destino $R$ (p47): el valor más alto es $10{,}7$ en $B$ con la acción de recoger ($P$), y $3{,}2$ en la casilla más lejana.
  - **Nota:** que $18{,}8=-1+0{,}99\cdot20$ y $17{,}6=-1+0{,}99\cdot18{,}8$ sugiere $\gamma=0{,}99$ y $-1$ por paso (deducción, no dada).

*Fuente: slides1 p43-p48.*

---

## Aplicación al Taller 2

**Marco del Taller 2.** DQN y A3C operan un activo en un entorno sintético (`SimpleTradingEnv`) con los cuatro generadores de precios del enunciado. La observación es continua, $[\text{tendencia},\ \text{posición}]$. Hay 3 acciones (vender, mantener, comprar). La comisión es del 0,1 %. La recompensa es ≈ $10^{-3}$ por paso y cada episodio tiene 494 decisiones. En las secciones previas todo era tabular y con $p$ conocida. Esto es lo que sigue vigente al pasar a redes neuronales y lo que no.

**Qué sigue igual**

1. **Trayectoria y MDP.** El episodio es la trayectoria $S_0,A_0,R_1,\dots$ de 494 pasos (slides1 p22). Es una tarea episódica de horizonte finito, así que $\gamma=1$ es válido y $\gamma<1$ también. El agente interactúa con un entorno que no conoce: es **model-free** (slides1 p11) y no resuelve $V=AV+b$ ni itera sobre $V_*$ como en `clase_20261023`, porque no dispone de $p(s',r\mid s,a)$. Los precios los genera una serie nueva en cada episodio.
2. **Markov en finanzas.** El profesor avisó de que el mercado "me temo que no" es markoviano y que depende de cómo se defina el estado (T23 ~01:34-01:35). En el Taller 2 el estado es `[tendencia, posición]`:
   - La **posición** sí hace falta para la propiedad de Markov. La comisión y la recompensa de cada paso dependen de la posición actual y de la acción, y sin ella la misma `tendencia` tendría recompensas distintas.
   - La **tendencia** resume el pasado reciente del precio. No garantiza Markov (el precio puede tener memoria que ese resumen no recoja). Es una hipótesis de trabajo, "buena y que simplifica mucho" (T23 ~01:33), no un hecho. Con redes neuronales se podría ampliar el estado con más retardos o rendimientos, como hacía Atari con una pila de fotogramas (T23 ~00:31, ~01:32), pero el enunciado fija la observación de dos componentes.
3. **Retorno y $\gamma$.** El agente maximiza el retorno $G_t=\sum_k\gamma^kR_{t+k+1}$ (slides1 p26), no la recompensa inmediata. Recompensa y valor son cosas distintas: una operación puede costar comisión hoy (recompensa mala) y tener valor positivo si deja una buena posición (T23 ~00:50). $\gamma$ sigue siendo el hiperparámetro que decide cuánto pesa el futuro.
4. **$V$ y $Q$ siguen siendo los objetos que se aprenden.** La diferencia es que ya no son tablas: DQN aproxima $Q_\theta(s,a)$ con una red y A3C aproxima además $V_\phi(s)$ (el crítico) y una política $\pi_\theta(a\mid s)$ (el actor). Las relaciones $V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)$ y $Q^\pi=\mathbb E[r+\gamma V^\pi(s')]$ (slides1 p29) siguen siendo válidas y justifican, por ejemplo, la ventaja $A=Q-V$ de A3C. Al ser el estado continuo, las sumas sobre $s'$ pasan a esperanzas estimadas con muestras.
5. **Bellman es el objetivo de entrenamiento.** La ecuación de optimalidad de $Q_*$ (slides1 p38) se convierte en el *target* de DQN, $y=r+\gamma\max_{a'}Q(s',a')$, y la de expectativa en el *target* del crítico de A3C. Lo que cambia entre métodos es "cómo aproximan la solución" (slides1 p39). El detalle está en [02_td_sarsa_qlearning.md](02_td_sarsa_qlearning.md) y en los documentos de DQN y A3C de esta carpeta.
6. **Exploración.** El dilema de slides1 p13 sigue ahí. Con DQN se usa $\epsilon$-greedy, y el consejo de T23 es relevante para finanzas: $\epsilon$ alto al principio y decreciente, pero **no necesariamente a 0 en mercados no estacionarios** (T23 ~01:20-01:22). En A3C la exploración la da la propia política estocástica (por ejemplo con un bonus de entropía).

**Qué deja de valer**

- **No hay solución directa ni DP.** Con estado continuo no existe la matriz $A$, ni tiene sentido `np.linalg.solve`, ni iterar $V\leftarrow\max_a[\dots]$ sobre una tabla. El $\max$ de la ecuación de optimalidad sigue siendo no lineal y se aproxima por muestreo y descenso de gradiente (slides1 p39).
- **La media de la recompensa del bandido** ($Q_{n+1}=Q_n+\tfrac1n(R_n-Q_n)$, slides1 p21) pasa a ser un paso de gradiente con tasa de aprendizaje $\alpha$ sobre los pesos de la red, no sobre una tabla.
- **La garantía de convergencia de $\gamma<1$ y unicidad de la solución** (slides2 p11) es para el caso tabular. Con redes no hay garantía, por lo que hacen falta estabilizadores (estudiados en los documentos de DQN y A3C).

**Nota (propuestas propias, no del material de clase).**
- Con recompensa ≈ $10^{-3}$ por paso y 494 pasos, $\gamma=1$ da retornos del orden de $0{,}5$ en un episodio y $\gamma=0{,}99$ un horizonte efectivo de unos 100 pasos con retornos del orden de $0{,}1$. El retorno de un episodio completo es pequeño, lo que sugiere normalizar o escalar la recompensa (p. ej. multiplicarla por una constante) antes de entrenar la red, sin cambiar la política óptima (solo cambia la escala de $Q$).
- La comisión del 0,1 % es del mismo orden que la recompensa típica por paso. Cambiar de posición debe estar justificado: por eso la **posición** es imprescindible en el estado y un $\gamma$ cercano a 1 ayuda a ver que una operación solo compensa si la tendencia persiste varios pasos.
- Para la memoria del taller conviene justificar $\gamma$ y $\epsilon$ con las ideas de este documento: horizonte, varianza del retorno y no estacionariedad del mercado.

*Fuente: slides1 p11, p13, p21-p22, p26, p29, p38-p39; slides2 p11; T23 ~00:31, ~00:50, ~01:20-01:22, ~01:33-01:35.*
