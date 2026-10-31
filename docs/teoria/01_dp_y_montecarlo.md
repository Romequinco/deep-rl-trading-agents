# Programación dinámica, métodos tabulares y Monte Carlo

> **Fuentes:**
> - **slides2** (`docs/raw/MIAX-14/slides2.pdf`, mazo "Programación Dinámica y Métodos Tabulares", MIAX-14, p1-p54). Lectura obligatoria indicada en p3: Sutton & Barto (2018), cap. 4 (secciones 4.1-4.6) y cap. 5 (5.1-5.7); complementaria: lecciones 3, 4 y 5 del curso de D. Silver (2015).
> - **T23** (transcripción de la clase del 23 de octubre, tramo del final: gridworld, Gymnasium, FrozenLake y Taxi) y **T24** (transcripción de la clase del 24 de octubre, tramo de DP y Monte Carlo). Son automáticas y ruidosas; se parafrasea y se cita con la hora aproximada del cue, por ejemplo `(T24 ~01:26)`.
> - **Notebooks** (`docs/raw/MIAX-14/`): `frozen_lake.ipynb`, `taxi.ipynb`, `gambler.ipynb` y `blackjack.ipynb`.
>
> **Convenciones:** "(slides2 pN)" = página N del mazo. "(frozen_lake, celda N)" = celda N (empezando en 0) del notebook. "(T24 ~HH:MM)" = hora aproximada de la transcripción. **Nota:** = contenido propio, aclaración o errata que no está en las fuentes. Prerrequisitos (MDP, Bellman, $\epsilon$-greedy del bandido): [00_fundamentos_mdp.md](00_fundamentos_mdp.md). El paso natural siguiente es [02_td_sarsa_qlearning.md](02_td_sarsa_qlearning.md).

---

## 1. Métodos tabulares: las tres familias

**Idea.** Aprender por refuerzo en un fMDP (MDP finito) es resolver las ecuaciones de optimalidad de Bellman (slides2 p4):

$$V_*(s)=\max_{a}\sum_{s',r}p(s',r\mid s,a)\,[\,r+\gamma V_*(s')\,],\qquad Q_*(s,a)=\sum_{s',r}p(s',r\mid s,a)\,\Big[\,r+\gamma\max_{a'}Q_*(s',a')\Big].$$

**Nota (errata de slides2 p4):** los títulos de los recuadros dicen "$V^\pi(s)$" y "$Q^\pi(s,a)$", pero las ecuaciones que contienen son las de optimalidad ($V_*$, $Q_*$, con $\max$).

La $p(s',r\mid s,a)$ es la dinámica del entorno: es propia del entorno y **no depende del agente**; lo único que depende del agente es la política (T24 ~00:10). Si se conoce, se resuelven las ecuaciones; si no, hay que estimarla o evitarla (T24 ~00:11).

Tres obstáculos para resolver Bellman tal cual (T24 ~00:11): (1) el entorno puede no ser markoviano (se supondrá que sí); (2) aunque lo sea, puede no conocerse $p$; (3) aunque se conozca, puede haber tantos estados que la tabla sea inabordable. En T24 ~00:12 se fija el supuesto de partida: el número de estados es lo bastante pequeño como para guardar la función de valor en una **tabla**. De ahí "métodos tabulares" (slides2 p5): $V$ es un vector con una casilla por estado y $Q$ es una matriz de filas = estados y columnas = acciones, y cada casilla es la estimación actual (T24 ~00:12).

Las tres familias y su descripción (slides2 p6-p8):
- **Programación dinámica (DP):** conocimiento completo y perfecto de $p(s',r\mid s,a)$. Hace *bootstrapping*: actualiza la estimación de un estado a partir de las estimaciones de los siguientes. En lugar de invertir matrices, resuelve Bellman de forma **iterativa**: dada la estimación actual, calcula la siguiente, y repite (T24 ~00:13).
- **Monte Carlo (MC):** sin modelo; aprende de la experiencia (episodios, trayectorias de estado-acción-recompensa), **sin bootstrapping**: promedia retornos reales observados hasta el final del episodio. Solo vale para tareas episódicas. El profesor lo describe como "un poco de fuerza bruta" (T24 ~00:14-00:16): se juegan muchos episodios completos y se promedian los retornos observados tras cada estado o par estado-acción; si la tarea no acaba nunca, el retorno total no se puede conocer.
- **Diferencia temporal (TD):** híbrido. Sin modelo (como MC) y con bootstrapping (como DP): en vez de esperar al final del episodio, va paso a paso con la recompensa inmediata y el valor estimado del estado al que llega (T24 ~00:17). Es la familia de Q-Learning y, con redes neuronales, de DQN, el algoritmo que dio éxito en los juegos de Atari (T24 ~00:18). Ver [02_td_sarsa_qlearning.md](02_td_sarsa_qlearning.md).

| Propiedad | Programación dinámica (DP) | Monte Carlo (MC) | Diferencia temporal (TD) |
|---|---|---|---|
| Requiere modelo $p(s',r\mid s,a)$ | Sí (model-based) | No (model-free) | No (model-free) |
| Bootstrapping | Sí | No | Sí |
| Tipo de tarea | Episódica y continua | Exclusivamente episódica | Episódica y continua |

(slides2 p9; el profesor lo repasa en T24 ~00:19: MC "tiene que esperar hasta el final" y TD "solo mira un paso hacia delante", por eso se aplica a cualquier tarea). *Bootstrapping* = actualizar la estimación de un estado usando estimaciones de otros estados.

*Fuente: slides2 p4-p9; T24 ~00:07-00:20.*

---

## 2. Programación dinámica

**Idea.** Calcula funciones de valor óptimas **dado un modelo perfecto** del fMDP: $p(s',r\mid s,a)$ se supone conocida. Una vez obtenida $V_*$ o $Q_*$, la política óptima es un paso trivial (slides2 p10): en cada estado se elige la acción que maximiza la recompensa inmediata esperada más el valor esperado descontado del estado siguiente, y eso se puede calcular porque se conocen las probabilidades de transición (T24 ~00:20). Con otros métodos (sin modelo) esto no es posible a partir de $V$, solo a partir de $Q$ (T24 ~00:20). El esquema general es alternar **evaluar** una política y **mejorarla** (T24 ~00:21).

### 2.1 Evaluación de políticas (slides2 p11-p14)
Calcular $V^\pi$ para una $\pi$ arbitraria resolviendo la ecuación de expectativa de Bellman:
$$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\,[\,r+\gamma V^\pi(s')\,],\quad\forall s.$$
Con $p$ conocida es un sistema de $|\mathcal S|$ ecuaciones lineales con $|\mathcal S|$ incógnitas. Existencia y unicidad están garantizadas si $\gamma<1$ o si la tarea es episódica y todo episodio termina en un número finito de pasos (slides2 p11). Razón (T24 ~00:22): sin esa condición el valor podría irse a infinito; con episodios finitos se suman recompensas un número finito de veces, y con $\gamma<1$ en tareas continuas la suma converge.

**Iterative policy evaluation** (slides2 p12): sucesión $V_0,V_1,\dots$ con $V_0$ arbitraria (la única condición es $V_0=0$ en los terminales, T24 ~00:23) y la ecuación de expectativa como regla de actualización:
$$V_{k+1}(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\,[\,r+\gamma V_k(s')\,].$$
$V_k\to V^\pi$ cuando $k\to\infty$ bajo las mismas condiciones.
- **Dos implementaciones** (slides2 p13): dos arrays ($V_k$ y $V_{k+1}$) o un único array **in-place**, donde parte de las actualizaciones usa valores ya modificados en el mismo barrido. El profesor señala que el segundo suele ser algo más rápido porque aprovecha las actualizaciones en cuanto están disponibles (T24 ~00:25).
- **Criterio de parada:** $\max_s|V_{k+1}(s)-V_k(s)|<\theta$, con $\theta>0$ (slides2 p13). La convergencia exacta solo se da en el infinito; $\theta$ fija el grado de aproximación deseado (T24 ~00:26).

```
Iterative Policy Evaluation (slides2 p14)
Entrada: política π(a|s), dinámica p(s',r|s,a), γ ∈ [0,1), umbral θ > 0
Salida:  V ≈ V^π
1  Inicializar V(s) arbitrariamente ∀s  (V(s)=0 si s es terminal)
2  repetir
3      Δ ← 0
4      para cada s ∈ S:
5          v ← V(s)
6          V(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [ r + γ V(s') ]
7          Δ ← max(Δ, |v − V(s)|)
8  hasta que Δ < θ
9  devolver V
```
Es la versión in-place (se sobrescribe $V(s)$ dentro del barrido) (T24 ~00:29).

### 2.2 Mejora de políticas (slides2 p15-p18)
Dada una $\pi$ **determinista** con $V^\pi$ conocida, en un estado $s$ se prueba una acción $a\neq\pi(s)$ y luego se sigue $\pi$. Su valor es
$$Q^\pi(s,a)=\sum_{s',r}p(s',r\mid s,a)\,[\,r+\gamma V^\pi(s')\,]$$
(slides2 p15). Se restringe a políticas deterministas por sencillez; las conclusiones valen también para las estocásticas (T24 ~00:30). Si $Q^\pi(s,a)\ge V^\pi(s)$, tiene sentido elegir $a$ cada vez que se pase por $s$.

- **Teorema de mejora de políticas** (slides2 p16): si $\pi,\pi'$ son deterministas y $Q^\pi(s,\pi'(s))\ge V^\pi(s)\ \forall s$, entonces $V^{\pi'}(s)\ge V^\pi(s)\ \forall s$.
- **Política greedy** (slides2 p17): $\pi'(s)=\arg\max_aQ^\pi(s,a)=\arg\max_a\sum_{s',r}p\,[r+\gamma V^\pi(s')]$ cumple la condición del teorema por construcción, porque se elige justo la acción que maximiza $Q^\pi$ (T24 ~00:34).
- **Optimalidad** (slides2 p18): si $V^{\pi'}=V^\pi$ para todo $s$, entonces $V^{\pi'}(s)=\max_a\sum_{s',r}p\,[r+\gamma V^{\pi'}(s')]$, que es la ecuación de optimalidad de Bellman: $\pi$ y $\pi'$ son óptimas (T24 ~00:35). Una definición equivalente de política óptima es "la que es greedy respecto a su propia función de valor" (T24 ~00:38).

### 2.3 Iteración de políticas (slides2 p19-p22)
Partiendo de una $\pi_0$ arbitraria alterna evaluación y mejora:
$$\pi_0\xrightarrow{\text{Eval}}V^{\pi_0}\xrightarrow{\text{Mejora}}\pi_1\xrightarrow{\text{Eval}}V^{\pi_1}\to\dots\to\pi_*\xrightarrow{\text{Eval}}V_*$$
Garantiza la política óptima en un número finito de iteraciones (slides2 p19). A una pregunta de un alumno sobre si podría quedarse en un máximo local, el profesor responde que no: converge al óptimo siempre que se cumplan las condiciones de existencia ($\gamma<1$ o episodios finitos); no hay máximos locales (T24 ~00:41).

```
Policy Iteration (slides2 p20-p22)
Entrada: dinámica p, γ ∈ [0,1), θ > 0
Salida:  V ≈ V*, π ≈ π*
1  Inicializar V(s) ∈ R y π(s) ∈ A(s) arbitrariamente  (V(s)=0 si s terminal)
2  bucle:
3      # Evaluación de la política (π determinista)
4      repetir
5          Δ ← 0
6          para cada s:
7              v ← V(s)
8              V(s) ← Σ_{s',r} p(s',r | s, π(s)) [ r + γ V(s') ]
9              Δ ← max(Δ, |v − V(s)|)
10     hasta que Δ < θ
11     # Mejora de la política
12     policy_stable ← true
13     para cada s:
14         a_old ← π(s)
15         π(s) ← argmax_a Σ_{s',r} p(s',r | s,a) [ r + γ V(s') ]
16         si a_old ≠ π(s): policy_stable ← false
17     si policy_stable: salir del bucle
18 devolver V, π
```
- **Criterio de parada:** política estable (ninguna acción cambia en un barrido de mejora) (T24 ~00:44).
- La suma sobre acciones desaparece respecto a §2.1 porque $\pi$ es ahora determinista: solo hay una acción por estado (T24 ~00:41).
- Aclaración a un alumno: primero se evalúan **todos** los estados hasta converger; después se mejora en **todos** los estados; la política no se cambia estado a estado durante la evaluación (T24 ~00:42-00:43).
### 2.4 Iteración de valores (slides2 p23-p25)
Inconveniente de policy iteration: la fase de evaluación necesita varios barridos hasta converger a $V^\pi$, lo que es muy costoso; la fase de mejora solo requiere un barrido (T24 ~00:45). *Value iteration* hace **un único barrido** de evaluación por iteración y combina evaluación y mejora en una sola regla basada en la ecuación de optimalidad:
$$V_{k+1}(s)=\max_a\sum_{s',r}p(s',r\mid s,a)\,[\,r+\gamma V_k(s')\,]\quad\forall s$$
Converge a $V_*$ bajo las mismas condiciones de existencia (slides2 p24). La función de valor intermedia no es exacta, pero converge igual (T24 ~00:47). Ya no hay política explícita durante el bucle: se aplica directamente Bellman de optimalidad (T24 ~00:47). Es más compacto y fácil de programar, y es lo usado en los notebooks (T24 ~00:48).

```
Value Iteration (slides2 p25)
Entrada: dinámica p, γ ∈ [0,1), θ > 0
Salida:  V ≈ V*, π ≈ π*
1  Inicializar V(s) arbitrariamente  (V(s)=0 si s terminal)
2  repetir
3      Δ ← 0
4      para cada s:
5          v ← V(s)
6          V(s) ← max_a Σ_{s',r} p(s',r|s,a) [ r + γ V(s') ]
7          Δ ← max(Δ, |v − V(s)|)
8  hasta que Δ < θ
9  para cada s:  π(s) ← argmax_a Σ_{s',r} p(s',r|s,a) [ r + γ V(s') ]
10 devolver V, π
```
**Convergencia y velocidad (T24 ~00:51-00:53).** Ambos algoritmos convergen a la política óptima y la precisión la fija $\theta$; lo que difiere es cuánto se tarda en alcanzar ese umbral. El profesor espera que value iteration sea más rápido (cada evaluación completa de policy iteration cuesta muchos barridos) y pide **medirlo** en el ejercicio de la práctica: cuánto tarda cada uno (y cuántas iteraciones) en cumplir $\Delta<\theta$. "Una cosa es lo que dicen los libros y otra probarlo uno mismo" (T24 ~00:52).

**Cómo está implementado en los notebooks (T24 ~00:53-01:00).** Se parte de `V = np.zeros(n_states)` (cualquier inicialización arbitraria vale mientras los terminales valgan 0). Se extrae la dinámica de `env.unwrapped.P`, que para cada par (estado, acción) da una lista de tuplas `(probabilidad, siguiente_estado, recompensa, es_terminal)` cuyas probabilidades suman 1, que es la $p(s',r\mid s,a)$ en el formato de Gymnasium. Bucle `while True` que rompe cuando $\Delta<\theta$; contador de iteraciones para poder analizar la convergencia. Por cada estado: guardar el valor viejo, calcular un array `q_values` con el valor de cada acción (suma ponderada por probabilidad de `reward` más $\gamma V$ del siguiente estado, o 0 si el siguiente es terminal) y asignar `np.max(q_values)`. La política se extrae al final con `argmax` en lugar de `max`. Código y resultados: §5.1 y §5.2.

### 2.5 Limitaciones de DP (slides2 p29)
Tiene garantías matemáticas de convergencia, pero:
1. Exige que el entorno sea un fMDP con **dinámica completamente conocida**; en la práctica el entorno no suele ser markoviano o $p(s',r\mid s,a)$ no se conoce.
2. Cada iteración recorre **todos** los estados; con muchísimos estados una sola iteración puede ser demasiado lenta y almacenar y barrer la tabla es inviable.

Matiz del profesor (T24 ~01:14-01:15): la mayoría de los estados de un problema grande, por ejemplo el ajedrez, ni siquiera ocurren en la práctica, y DP los visita todos igualmente.

*Fuente: slides2 p10-p29; T24 ~00:20-00:53, ~01:14-01:15.*

---

## 3. Monte Carlo (MC)

**Idea.** Estiman funciones de valor y políticas óptimas a partir de **experiencia muestreada** (secuencias de estados, acciones y recompensas), sin conocer la dinámica. Solo valen para tareas episódicas: se necesita llegar al final del episodio para conocer todas las recompensas posteriores a un estado y calcular su retorno. Al final de cada episodio se actualizan la estimación y la política (primero evaluación, luego mejora) (slides2 p30; T24 ~01:16). El bucle general es el mismo que en DP, evaluar-mejorar, y solo cambia cómo se evalúa: con experiencia, no con el modelo (T24 ~01:16).

### 3.1 Evaluación MC (slides2 p31-p33)
Se estima $V^\pi(s)$ a partir de episodios generados con $\pi$: $S_0,A_0,R_1,S_1,A_1,R_2,\dots,S_{T-1},A_{T-1},R_T$ (la acción y el estado se asocian a $t$ y la recompensa llega en $t+1$, T24 ~01:17). El retorno es
$$G_t=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1}.$$
Como $V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$, el **promedio empírico** de los retornos observados tras visitar $s$ converge a $V^\pi(s)$ cuando el número de visitas tiende a infinito (slides2 p31). Hay que conseguir muchos episodios en los que cada estado se visite alguna vez (T24 ~01:25).

**Cálculo hacia atrás (T24 ~01:18-01:23).** $G_t=R_{t+1}+\gamma G_{t+1}$, así que una vez conocida la lista de recompensas se calcula el retorno del último paso ($G_{T-1}=R_T$, pues el valor del estado terminal es 0) y se retrocede multiplicando por $\gamma$ el $G$ actual y sumando la recompensa del paso anterior. Por eso el bucle de los algoritmos va de $t=T-1$ a $0$.

- **First-visit:** promedia solo el retorno tras la primera visita a $s$ en cada episodio. **Every-visit:** tras cada visita. Ambas son consistentes: convergen a $V^\pi(s)$ cuando el número de episodios tiende a infinito (slides2 p32). El profesor explica que se estudia más la de primera visita (con un número finito de episodios) y es la que muestra; su motivo: la primera visita acumula más experiencia posterior del estado (T24 ~01:26).

```
First-Visit MC Policy Evaluation (slides2 p33)
Entrada: política π, γ ∈ [0,1]
Salida:  V ≈ V^π
1  Inicializar V(s) arbitrariamente ∀s;  Returns(s) ← lista vacía ∀s
2  mientras verdadero:
3      Generar un episodio siguiendo π: S0, A0, R1, ..., S_{T-1}, A_{T-1}, R_T
4      G ← 0
5      para t = T-1, T-2, ..., 0:
6          G ← γ G + R_{t+1}
7          si S_t ∉ {S_0, ..., S_{t-1}}:      # primera visita
8              añadir G a Returns(S_t)
9              V(S_t) ← promedio(Returns(S_t))
```
**Nota (omitido en el pseudocódigo):** el bucle es infinito en el mazo; el profesor reconoce que habría que añadir una tolerancia de parada (T24 ~01:30). Y en vez de guardar las listas `Returns` se puede promediar de forma **incremental** (T24 ~01:27): $V\leftarrow V+\frac1n(G-V)$, que es lo que hace el notebook (§5.4). Como MC promedia retornos sin usar el valor anterior, la inicialización de $V$ no importa (T24 ~01:31, ~01:35).

### 3.2 MC frente a DP (slides2 p34-p35)
En MC las estimaciones de cada estado son **independientes** y salen de retornos reales, sin bootstrapping: $V(S_t)\leftarrow\text{promedio}(G_t)$. En DP la estimación de un estado se construye a partir de estimaciones de otros: $V(S_t)\leftarrow\mathbb{E}_\pi[R_{t+1}+\gamma V(S_{t+1})]$ con la dinámica conocida (slides2 p34). Ambas son aproximaciones, pero de cosas distintas (T24 ~01:32-01:34): en MC el valor esperado se **aproxima por un promedio de muestra finita** (el sumando es exacto: retornos reales); en DP el valor esperado es **exacto** (se conoce el modelo) pero se calcula **a partir de una estimación previa** $V_k$.

Cuatro diferencias (slides2 p35):
1. **Sesgo:** las estimaciones MC no tienen sesgo; las de DP están sesgadas en las fases iniciales por la elección de $V_0$. El profesor define sesgo como la capacidad del estimador de dar la solución exacta con infinitos datos; MC la tiene porque la media de infinitas observaciones es el valor esperado, y por eso a MC le da igual el valor inicial mientras que en DP puede ser muy relevante (T24 ~01:34-01:36).
2. **Varianza:** MC tiene alta varianza, porque $G_t$ acumula una secuencia larga de acciones y transiciones estocásticas: desde el mismo estado y acción el episodio evoluciona cada vez de forma distinta, el retorno cambia, y el promedio de una muestra finita queda disperso. DP/TD solo miran un paso, con mucho menos ruido que cien pasos (T24 ~01:36-01:37). Es el compromiso habitual: poco sesgo implica mucha varianza y al revés, igual que en aprendizaje automático (un modelo muy flexible tiene poco sesgo y mucha varianza, y se sobreajusta; uno rígido como un modelo lineal tiene mucho sesgo y poca varianza) (T24 ~01:38-01:39).
3. **Independencia de Markov:** MC no se ve afectado si los estados no cumplen estrictamente la propiedad de Markov, porque promedia retornos reales; DP depende de ella (T24 ~01:40).
4. **Coste:** en MC el coste de estimar un estado no depende del número de estados (slides2 p35). El profesor añade que se puede estimar el valor **solo de los estados que interesan**: por ejemplo, iniciar siempre las simulaciones en una posición concreta del tablero de ajedrez y evaluar solo esa. A cambio, hay que esperar al final del episodio (T24 ~01:40).

| | MC | DP (y TD, por el bootstrapping) |
|---|---|---|
| Sesgo | Nulo | Inicial, depende de $V_0$ |
| Varianza | Alta (suma de $T-t$ recompensas aleatorias) | Baja (un paso) |
| Depende de Markov | No | Sí |
| Coste por estado | Independiente de $|\mathcal S|$ | Barrido de todos los estados (DP) |

### 3.3 Estimar $Q$ y necesidad de exploración (slides2 p36-p38)
Sin modelo, conocer $V(s)$ no basta para elegir acciones: habría que conocer el estado siguiente y sus probabilidades. Por eso en los métodos model-free el objetivo primario es $Q(s,a)$, la tabla de valor de cada acción en cada estado (T24 ~01:41-01:42). El algoritmo es el mismo, pero la visita se refiere a pares $(s,a)$: se genera un episodio y se mide el retorno que sigue a la (primera) visita de cada par (slides2 p36).

Problema de las **acciones no visitadas** (slides2 p37): con una $\pi$ determinista solo se observa una acción por estado y el valor del resto de pares queda sin explorar; así no se descubre que otra acción es mejor (T24 ~01:44). Y es difícil de evitar porque el bucle de mejora hace la política greedy, y la política greedy es determinista (T24 ~01:44). Hay que **forzar la exploración** con dos estrategias típicas (slides2 p38; T24 ~01:44-01:45):
1. **Exploring starts (inicios exploratorios):** cada episodio comienza en un par $(S_0,A_0)$ elegido al azar entre todos los de $\mathcal S\times\mathcal A$, con probabilidad $>0$ para todos. Garantiza ver cada par al menos como primera visita del episodio.
2. **Políticas soft:** $\pi(a\mid s)>0\ \forall s,a$ (en la práctica, greedy con una pequeña probabilidad de acción al azar).

**Dos pegas (T24 ~01:45-01:46).** Exploring starts no es realista (hay problemas en los que siempre se empieza igual). Y una política soft impide que el algoritmo converja a la política óptima, que es determinista; convergerá a la mejor política entre las soft, "con suerte".

### 3.4 Control MC (slides2 p39-p41)
"Control" aquí es buscar la política óptima (evaluación era calcular el valor de una política) (T24 ~01:47). Igual que en DP, alterna evaluación de $Q^{\pi_k}$ y mejora greedy:
$$\pi_0\xrightarrow{\text{Eval}}Q^{\pi_0}\xrightarrow{\text{Mejora}}\pi_1\xrightarrow{\text{Eval}}Q^{\pi_1}\to\dots\to\pi_*\xrightarrow{\text{Eval}}Q_*,\qquad \pi_{k+1}(s)=\arg\max_aQ^{\pi_k}(s,a).$$
Primera aproximación (slides2 p39): exploring starts y pasos de evaluación y mejora completos (infinitos episodios), de modo que $Q^{\pi_k}$ es exacta. El teorema de mejora aplica (slides2 p40):
$$Q^{\pi_k}(s,\pi_{k+1}(s))=\max_aQ^{\pi_k}(s,a)\ \ge\ Q^{\pi_k}(s,\pi_k(s))=V^{\pi_k}(s),$$
por lo que $\pi_{k+1}\ge\pi_k$ y, si son iguales, son óptimas (T24 ~01:49).

**Consideraciones prácticas (slides2 p41; T24 ~01:50-01:51).** La evaluación completa es imposible (habría que fijar una tolerancia y $Q$ no será exacta), y evaluar con muchos episodios es muy costoso. Se alterna entre evaluación y mejora **episodio a episodio**: tras cada episodio se evalúa y se mejora la política solo en los estados visitados en ese episodio.

### 3.5 MC con exploring starts (MC ES) (slides2 p42, p47)
```
Monte Carlo Exploring Starts (slides2 p42)
Entrada: γ ∈ [0,1]
Salida:  π ≈ π*, Q ≈ Q*
1  Inicializar Q(s,a) arbitrariamente, π(s) arbitraria, Returns(s,a) ← lista vacía
2  mientras verdadero:
3      Elegir S0 ∈ S, A0 ∈ A(S0) al azar (todos los pares con prob > 0)
4      Generar un episodio con π desde (S0,A0): S0,A0,R1,...,S_{T-1},A_{T-1},R_T
5      G ← 0
6      para t = T-1, ..., 0:
7          G ← γ G + R_{t+1}
8          si (S_t,A_t) ∉ {(S_0,A_0),...,(S_{t-1},A_{t-1})}:
9              añadir G a Returns(S_t,A_t)
10             Q(S_t,A_t) ← promedio(Returns(S_t,A_t))
11             π(S_t) ← argmax_a Q(S_t,a)
```
Notas del mazo (slides2 p47): los retornos de un par se acumulan y promedian **independientemente de la política con la que se observaron**; el algoritmo converge a la política óptima, pero exploring starts es poco realista, porque no permite aprender de interacción real con el entorno; para evitarlo hacen falta políticas soft.

**Matices del profesor (T24 ~02:11-02:14).**
- Como la política cambia tras cada episodio, los retornos de un mismo par $(s,a)$ se obtuvieron con **políticas distintas** y se promedian todos. No es tan exacto como la evaluación con política fija, pero el algoritmo converge igualmente y hay demostraciones de convergencia. **Nota:** (de memoria, a verificar en el libro) Sutton & Barto (cap. 5.3) presentan la convergencia de MC ES como intuitivamente inevitable pero aún sin demostrar formalmente, y la señalan como cuestión teórica abierta; es un matiz frente a la afirmación de clase.
- Ejemplos de falta de realismo: no se puede aprender blackjack empezando con tres cartas en la mano ni ajedrez, donde toda partida empieza en la misma posición (T24 ~02:12-02:13).
- Si se quisieran incluir apuestas de distinta cuantía en blackjack, habría que ampliar las acciones y redefinir la recompensa según la cantidad apostada (T24 ~02:14).

### 3.6 MC on-policy (slides2 p48-p49)
Evalúa y mejora **la misma política** que usa para interactuar. Para que no se vuelva determinista y deje de explorar, la mejora se restringe a políticas $\epsilon$-greedy respecto a $Q$:
$$\pi(a\mid s)=\begin{cases}1-\epsilon+\dfrac{\epsilon}{|\mathcal A(s)|}&a=\arg\max_bQ(s,b)\\[2mm]\dfrac{\epsilon}{|\mathcal A(s)|}&\text{resto}\end{cases}$$
Con probabilidad $\epsilon$ se elige una acción al azar (que puede ser también la greedy) y con $1-\epsilon$ la greedy, de ahí el reparto (T24 ~02:16). Es el algoritmo de MC ES con la política siempre $\epsilon$-greedy y sin exploring starts; converge a la mejor política **entre las $\epsilon$-greedy**, no a la óptima (que es solo greedy) (slides2 p48). Si $\epsilon$ es muy pequeño, convergerá casi a la óptima, pero se necesitan cada vez más episodios para visitar todo; lo razonable es **empezar con $\epsilon$ alto y decrecerlo** (T24 ~02:17).

```
On-Policy First-Visit MC Control (slides2 p49)
Entrada: ε > 0, γ ∈ [0,1]
Salida:  π ε-greedy ≈ π*, Q ≈ Q*
1  Inicializar Q(s,a) y Returns(s,a) como en MC ES
2  Inicializar π como ε-greedy respecto a Q
3  mientras verdadero:
4      Generar un episodio siguiendo π: S0,A0,R1,...,S_{T-1},A_{T-1},R_T
5      G ← 0
6      para t = T-1, ..., 0:
7          G ← γ G + R_{t+1}
8          si (S_t,A_t) ∉ {(S_0,A_0),...,(S_{t-1},A_{t-1})}:
9              añadir G a Returns(S_t,A_t)
10             Q(S_t,A_t) ← promedio(Returns(S_t,A_t))
11             A* ← argmax_a Q(S_t,a)          (empates arbitrarios)
12             para todo a ∈ A(S_t):
13                 π(a|S_t) ← 1 − ε + ε/|A(S_t)|  si a = A*,  ε/|A(S_t)|  si a ≠ A*
```
(T24 ~02:17-02:18: la línea 11 calcula la acción greedy, pero la acción realmente ejecutada será $\epsilon$-greedy sobre ella.)

### 3.7 MC off-policy (slides2 p50-p52)
Resuelve el dilema exploración-explotación con **dos políticas** (slides2 p50): la **objetivo** $\pi(a\mid s)$, la que se optimiza, que acaba siendo determinista y óptima, y la de **comportamiento** $b(a\mid s)$, exploratoria y estocástica, que genera las trayectorias e interactúa con el entorno. El agente juega con $b$ pero aprende $\pi$ (T24 ~02:19). Requiere **cobertura**: $\pi(a\mid s)>0\Rightarrow b(a\mid s)>0$ (si $b$ nunca toma una acción, no se puede explotar). Para evaluar $\pi$ con datos de $b$ se ponderan los retornos (idea de T24 ~02:20-02:23): si una secuencia no podría haberla generado $\pi$, sus retornos no sirven para actualizar $\pi$; cuanto más probable sea con $\pi$ relativamente a $b$, más peso.

**Muestreo por importancia** (slides2 p51): la razón es el cociente de las probabilidades de la secuencia bajo $\pi$ y $b$. Las probabilidades de transición del entorno aparecen arriba y abajo y **se cancelan**, así que no hace falta el modelo (T24 ~02:22):
$$\rho_{t:T-1}=\frac{\prod_{k=t}^{T-1}\pi(A_k\mid S_k)\,P(S_{k+1}\mid S_k,A_k)}{\prod_{k=t}^{T-1}b(A_k\mid S_k)\,P(S_{k+1}\mid S_k,A_k)}=\prod_{k=t}^{T-1}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)},$$
y el retorno ponderado aplicable a la política objetivo es $\rho_{t:T-1}G_t$.

```
Off-Policy MC Control (slides2 p52)
Entrada: política soft b
Salida:  π ≈ π* (greedy), Q ≈ Q*
1  Inicializar Q(s,a) arbitrariamente;  C(s,a) ← 0 ∀(s,a);  π(s) ← argmax_a Q(s,a)
2  mientras verdadero:
3      Generar un episodio con b: S0,A0,R1,...,S_{T-1},A_{T-1},R_T
4      G ← 0;  W ← 1
5      para t = T-1, ..., 0:
6          G ← γ G + R_{t+1}
7          C(S_t,A_t) ← C(S_t,A_t) + W
8          Q(S_t,A_t) ← Q(S_t,A_t) + (W / C(S_t,A_t)) [ G − Q(S_t,A_t) ]
9          π(S_t) ← argmax_a Q(S_t,a)
10         si A_t ≠ π(S_t): romper
11         W ← W · 1 / b(A_t|S_t)
```
Cómo leerlo (T24 ~02:23-02:27):
- $W$ es el cociente $\rho$ acumulado hacia atrás. El numerador es $\pi(A_t\mid S_t)$, que vale **1** porque $\pi$ es determinista y la acción coincide con la suya (de ahí que se rompa si no coinciden: sería 0). Por eso aparece "1" sobre $b(A_t\mid S_t)$.
- **Pega:** a partir del primer paso (yendo hacia atrás) en el que las acciones no coinciden, el resto del episodio anterior se ignora (multiplicado por 0), así que solo se aprende de la última parte del episodio.
- $C$ acumula las $W$ de cada par y se usa para promediar de forma **incremental** y ponderada (en lugar de guardar la lista de retornos): $Q\leftarrow Q+\frac{W}{C}[G-Q]$.

### 3.8 Limitaciones de MC (slides2 p53)
1. **No aprende online:** hay que esperar al final del episodio para calcular $G_t$ y actualizar (una actualización por episodio). Inaplicable con episodios muy largos o tareas continuas (T24 ~02:28).
2. **Varianza y factor de descuento:** la varianza de $G_t$ crece con la longitud del episodio y exige muchos episodios para estimaciones estables (T24 ~02:28-02:29).
3. **Off-policy ineficiente con trayectorias largas:** si $\pi$ difiere de $b$ en algún momento, $\rho$ se anula y se descarta el resto. Si ambas difieren mucho, puede que ni la última acción coincida y no se aprenda nada del episodio; en el mejor caso se usan dos o tres transiciones por episodio (T24 ~02:29-02:30). La idea de combinar dos políticas es muy buena, pero esta implementación es ineficiente; el profesor anticipa que en TD hay un algoritmo off-policy (Q-Learning) que funciona muy bien con una estrategia distinta (T24 ~02:30).

**Ejercicio propuesto** (slides2 p54): programar los algoritmos de control MC on-policy y off-policy ("On-Policy MC y On-Policy MC", texto que repite "On-Policy": **Nota:** casi seguro una errata por "On-Policy y Off-Policy") y probarlos en FrozenLake-v1, Taxi-v4 y Blackjack-v1, comparando los resultados con DP (T24 ~02:30).

*Fuente: slides2 p30-p54; T24 ~01:16-02:30.*

---

## 4. Exploración: resumen

**Idea.** Si la política es determinista (greedy) solo se visita una acción por estado y $Q$ del resto no se aprende. Soluciones y su coste:

| Estrategia | Qué hace | Problema |
|---|---|---|
| Exploring starts | Episodio comienza en $(S_0,A_0)$ al azar | Poco realista (slides2 p47) |
| Política soft / $\epsilon$-greedy | $1-\epsilon$ greedy, $\epsilon$ uniforme; mejor acción: $1-\epsilon+\epsilon/|A|$ | Converge a la mejor política $\epsilon$-greedy, no a la óptima (slides2 p48) |
| Off-policy | $b$ explora y $\pi$ greedy aprende | $\rho$ se anula si difieren; ineficiente con trayectorias largas (slides2 p53) |

La regla $\epsilon$-greedy es la del bandido (slides1 p17; ver [00_fundamentos_mdp.md](00_fundamentos_mdp.md)); en MC se aplica sobre $Q$ (slides2 p48). Consejo práctico: empezar con $\epsilon$ alto y reducirlo (T24 ~02:17).

*Fuente: slides2 p37-p38, p47-p48, p50, p53; slides1 p17; T24 ~01:44-02:17.*

---

## 5. Qué ilustra cada notebook

### 5.1 `frozen_lake`: value iteration en un MDP pequeño
- **Qué es** (T23 ~03:49-03:50): un lago helado en el que hay que ir de la casilla de salida a la meta evitando agujeros. `FrozenLake-v1` 4×4: 16 estados (posición `fila*4+col`, de izquierda a derecha y de arriba abajo) y 4 acciones: 0 izquierda, 1 abajo, 2 derecha, 3 arriba. Recompensa +1 al llegar a la meta, 0 al caer en un agujero (terminal) y 0 en el hielo; agujeros y meta son terminales; episodio de máximo 100 iteraciones (celda 5). Mismo código de ejecución con `gym.make(ENV_NAME, render_mode="rgb_array", is_slippery=...)` (celda 3). El episodio se juega con descuento (T23 ~03:52).
- **Gymnasium** (T23 ~03:45-03:56): es una biblioteca de entornos simulados (naves, péndulos, juegos de Atari, caminantes de MuJoCo, juegos de texto...) con una interfaz común: el entorno devuelve un estado, el agente le envía una acción y el entorno devuelve la recompensa. `gym.make(nombre)` lo crea y `reset()` lo pone en la situación inicial. Los "estados" se llaman *observations* porque no tienen por qué ser el estado real, sino lo que percibe el agente; para el profesor es lo mismo. Se accede al número de estados y de acciones con `observation_space` y `action_space` (T23 ~03:56; celda 3: 16 y 4).
- **`is_slippery`:** con `False` (el del notebook, celda 3), el movimiento es determinista. Con `True` (slides1 p44-p45 comparan ambos), intentar ir en una dirección sale bien con probabilidad 1/3 y con 1/3 cada una de las dos direcciones perpendiculares (T23 ~03:53-03:54, ~04:02). "Muy relevante: añade complicación".
- **Modelo.** `P = env.unwrapped.P`, con `P[s][a] = [(prob, siguiente_estado, recompensa, terminal), ...]` (celda 7). Por ejemplo, `P[0]` (sin deslizar): las cuatro acciones llevan con prob. 1 a `0`, `4`, `1` y `0` con recompensa 0 (izquierda y arriba chocan con el borde y dejan en el mismo sitio). Con deslizamiento, desde el estado 0 la acción "izquierda" daría 1/3 de quedarse (izquierda), 1/3 de ir abajo al estado 4 y 1/3 de intentar arriba y quedarse (T23 ~04:02). Ver el modelo supone **conocer el entorno**, algo que no siempre ocurrirá (T23 ~03:58).
- **`value_iteration(env, gamma=0.99, theta=1e-8)`** (celda 9): `V = np.zeros(n_states)`; bucle con `delta`; por cada estado `q_values[a] += prob * (reward + (0 if done else gamma * V[next_state]))`; `V[s] = np.max(q_values)` (in-place) y `delta = max(delta, |v_old − V[s]|)`. Para cuando `delta < theta`. La política se extrae con `np.argmax(q_values)`. `(0 if done ...)` es el valor 0 del terminal: sin bootstrapping tras terminar.
- **Resultado** (celda 11): "¡Entrenamiento completado en **7 pasos**!" (sin deslizar). Figura: $V_*$ en un mapa de calor con flechas de $\pi_*$. Valores: 0,95-0,99 en el camino hacia la meta, 1,00 junto a ella y 0,00 en agujeros y meta (terminales); el valor decae con la distancia por el descuento. La política elige el camino evitando agujeros (T23 ~03:55).
- **Políticas óptimas empatadas** (T23 ~03:43-03:44, ejemplo del gridworld previo): puede haber varias óptimas a la vez si dos acciones dan el mismo valor; el valor de la política es el mismo. Con varios estados con empates hay muchas combinaciones (al menos $2\times2\times\dots$).
- **Con `is_slippery=True`** (T23 ~04:06-04:09): la política óptima es poco intuitiva. En una casilla junto a un agujero lo mejor es ir en la dirección que no deja caer en ningún caso (hacia una pared, por ejemplo), aunque solo 1/3 de los intentos avanzan; incluso en algún estado concreto lo mejor es intentar ir hacia el agujero, porque cualquier otra dirección deja 2/3 de probabilidad de caer (T23 ~04:09). Los valores son algo menores que sin deslizar, y la lección del profesor: "emergen políticas que no parecen intuitivas, pero tienen sentido". Se invita a ejecutar la versión resbaladiza para verlo (T23 ~04:06). **Nota:** el notebook solo muestra la ejecución sin deslizar (7 pasos); la versión resbaladiza no queda registrada en él.

### 5.2 `taxi`: el mismo algoritmo con 500 estados
- **Qué es** (T23 ~04:10-04:12; celda 5): un taxi en una rejilla 5×5 con paredes que debe recoger a un pasajero (en una de las 4 estaciones R, G, Y, B, o ya dentro del taxi) y llevarlo a un destino (otra de las 4). Estado = (25 posiciones del taxi) × (5 posiciones del pasajero, incluido dentro del taxi) × (4 destinos) = **500** (celda 3: 500 estados, 6 acciones). Acciones: 0 sur, 1 norte, 2 este, 3 oeste, 4 recoger, 5 dejar. Si hay una pared o se sale del tablero, el taxi se queda en el mismo sitio (T23 ~04:13).
- **Recompensas** (T23 ~04:15): −1 por paso (resolver rápido), +20 al entregar al pasajero en el destino, −10 por recoger sin estar en el pasajero o dejarlo donde no toca. **Nota:** el cue de T23 dice "10" sin signo por el ruido de la transcripción; la ficha de Gymnasium indica −10. Coherente con `P[0]` (celda 7): todas las acciones dan −1 menos `5` (dejar), que da −10 sin salir del estado 0.
- **Nota:** 500 es el tamaño teórico; algunos estados no se dan nunca (T23 ~04:12). Según la documentación de Gymnasium (no es de clase), solo 404 son alcanzables: 400 con el pasajero en una estación distinta del destino o dentro del taxi, más 4 al acabar con éxito.
- **Código:** la función `value_iteration` es **idéntica** a la de FrozenLake (celda 9) (T23 ~04:16); cambia solo el entorno (`Taxi-v4`, sin `is_slippery`) y el dibujo. Resultado (celda 11): `gamma=0.99`, "completado en **13 pasos**" (T23 ~04:17).
- **Visualización** (celda 10; T23 ~04:17-04:19): al haber 500 estados solo se muestran los valores para un pasajero y destino fijos (celda 11: pasajero azul, destino rojo), en la rejilla 5×5; el índice del estado es `((fila*5+col)*5 + pasajero)*4 + destino`. La figura muestra $V_*$ y $\pi_*$: el mayor valor (10,7, con la acción `P` de recoger) está en la casilla del pasajero; los valores decrecen al alejarse (9,6; 8,5; ...; 3,2 en las esquinas más lejanas, aproximadamente −1,1 por casilla) y las flechas siempre van hacia la casilla vecina de mayor valor, rodeando las paredes. Con el pasajero ya dentro del taxi, el estado de mayor valor pasa a ser el del destino (donde se suelta y se cobra +20) (T23 ~04:19-04:20).
- **Vídeo** (celdas 12-14): `RecordVideo` graba 10 episodios con `policy[state]` (en lugar de acciones aleatorias) a 5 fps y se reproduce uno incrustado.
- **Moraleja** (T23 ~04:20-04:21): con modelo conocido y Markov, DP resuelve el problema aplicando Bellman y llegando a la política óptima en un número finito de pasos; las dificultades aparecen si el entorno no es markoviano, no se conocen las probabilidades o hay demasiados estados (entonces se buscan soluciones aproximadas).

### 5.3 `gambler`: value iteration con política no trivial
- **Problema** (Sutton & Barto, ej. 4.3; slides2 p26; T24 ~01:01-01:03; celda 2): un apostador lanza una moneda; si sale cara gana lo apostado, si sale cruz lo pierde; el juego termina al llegar a $100 o al perderlo todo. Apuestas enteras. fMDP no descontado y episódico. Estado = capital $s\in\{1,\dots,99\}$ (0 y 100 son terminales); acción = apuesta $a\in\{1,\dots,\min(s,100-s)\}$ (no tiene sentido apostar más de lo que falta para 100: con 80 solo se puede apostar 1-20); recompensa 1 solo al llegar a 100 (0 en el resto). La probabilidad de cara es **`p = 0.4`** (celda 3), no 0,5. Con $p<0{,}5$ la probabilidad de alcanzar 100 empezando con 50 es como máximo 0,4 (T24 ~01:03).
- **Implementación:** `v = np.zeros(100)` (el índice 0 solo sirve para que cada índice coincida con su capital, celda 4). Dos versiones de la actualización: copia (`update_value`, celda 5) e **in-place** (`update_value_in_place`, celda 6, la usada). En cada estado: $v(s)=\max_a\big[(1-p)\,v(s-a)+p\,v(s+a)\big]$, sumando $p$ directamente (recompensa 1, sin valor posterior) si $s+a=100$.
- **Primera iteración** (celda 7): $v=0$ hasta 49 y luego escalones de 0,4 (50-74), 0,64 (75-87), 0,784 (88-93), 0,8704... Los escalones aparecen porque la actualización in-place usa en el mismo barrido valores ya actualizados de capitales menores: el 0,4 de $s=50$ es apostar 50 y ganar con $p=0{,}4$; el 0,64 de $s=75$ es $0{,}4+0{,}6\cdot0{,}4$ (apostar 25) (slides2 p27; comprobable con las cifras de la celda 7).
- **32 iteraciones fijas** (celda 8, sin umbral $\theta$) y se grafican las iteraciones 1, 2, 3 y 32 (slides2 p27): las tres primeras son escalonadas y la 32 es una curva creciente suave que va de 0 a ≈0,96 en 99 (ejemplo: $v(50)=0{,}4$, $v(25)=0{,}16$, $v(75)=0{,}64$). Si $p$ fuera 0,5 sería la recta del 0 al 1; con $p<0{,}5$ queda por debajo (T24 ~01:04). El profesor insiste en que lo que se ilustra es cómo la función de valor inicial, lejana, se acerca poco a poco al óptimo.
- **Política** (celdas 9-10; slides2 p28): `get_policy` recorre acciones de menor a mayor y usa una **tolerancia `1e-9`** en la comparación (`pv > maxv + 1e-9`). Resultado: dientes de sierra con periodo 25: $1,2,\dots,12,12,11,\dots,1$ para capital 1-24, **25** en $s=25$, otra sierra para 26-49, **50** en $s=50$, y de nuevo **25** en $s=75$ (en la figura de slides2 p28 los picos aislados miden 25, 50 y 25 en $s=25,50,75$), con otra sierra de máximo 12 en cada tramo intermedio. Coincide con el libro. Sin la tolerancia aparecían picos espurios por **acciones empatadas en valor**; el notebook elige la **menor** apuesta entre empatadas simplemente porque las acciones se recorren de menor a mayor (celda 11). En T24 ~01:06-01:09 el profesor cuenta que esta forma extraña le costó entender.
- **Comprobación recorriendo en orden inverso** (celdas 12-14): al elegir la **mayor** apuesta entre las empatadas, la política sale como un triángulo: apostar $\min(s,100-s)$ (todo lo que se pueda), alcanzando 50 en $s=50$. Ambas son óptimas. El profesor explica (T24 ~01:06-01:11): con 50 hay que apostar los 50; con 99 apostar todo lo que falta (1) es lo único que tiene sentido; en esta versión de "apuesta máxima", con poco capital se apuesta todo el capital. Deja como **ejercicio** comprobar si todas las apuestas por debajo del máximo son equivalentes ("a lo mejor da igual lo que hagas, porque es puro azar", T24 ~01:11).
- **Lección práctica:** al hacer `argmax` de valores casi iguales, el desempate afecta a la política resultante y hay que fijar una tolerancia.
- **Sobre la intuición** (T24 ~01:05-01:06): un alumno propone apostar todo con 25 o apostar la mitad; el profesor responde que la política óptima no es una apuesta "racional" cualquiera, y que puede haber varias óptimas con la misma función de valor.

### 5.4 `blackjack`: Monte Carlo sin modelo
- **Reglas** (Sutton & Barto, ej. 5.1; T24 ~01:54-02:01; celdas 1-9): el jugador juega contra la banca con una de las dos cartas de la banca visible. Objetivo: acercarse a 21 sin pasarse; el as vale 11 (*usable*) o 1; las figuras valen 10. Cuando el jugador se planta, la banca descubre su carta y sigue sacando con reglas fijas (según la ficha de Gymnasium, hasta 17; el profesor no lo recuerda con exactitud, T24 ~02:01). **No hay barajas múltiples ni ventaja por contar cartas**: el entorno repone la baraja cada partida (T24 ~01:59).
  - `gym.make("Blackjack-v1", sab=True)` (celda 4): reglas del libro.
  - Estado = (suma del jugador, carta visible del dealer 1-10, as usable 0/1); celda 6: `(12, 1, 0)`. Acciones: 0 *stick* (plantarse), 1 *hit* (pedir). Recompensa: +1 ganar, −1 perder, 0 empatar (slides2 p44). "As usable" = un as que puede contar 11 sin pasarse; con 10, 4 y as no sería usable (T24 ~02:00).
- **¿Por qué blackjack para MC?** (T24 ~02:07-02:10): es un ejemplo académico con episodios **cortos** (1-3 acciones), pocos estados (sumas 4-21 → 18, × 10 cartas × 2 = 360 estados; × 2 acciones = 720 pares) y pocas acciones. Con 500 000 episodios, cada par se evalúa unas 700 veces de media (el profesor, redondeando a 1 000 pares, dice unas 500), y más los que se alcanzan a mitad de episodio; es suficiente para promediar sin demasiado ruido, porque la varianza viene de los episodios largos. Con episodios largos haría falta mucho más.
- **Evaluación MC de una política fija** (celdas 11-14; slides2 p45): política "plantarse con 20 o 21, pedir en otro caso" (`state[0] < 20 → 1`). **500 000 episodios**, sin descuento (se usa la recompensa final como retorno de todos los estados visitados), valor por **media incremental** `V ← V*(n-1)/n + G/n` con contador `num_visits`. Es una variante every-visit en la práctica, ya que se recorre `state_list` completo. Resultado: dos superficies 3D (con y sin as usable) sobre (suma del jugador × carta del dealer); el valor sube al acercarse a 20-21 y es ≈ −0,6 a −0,8 con sumas bajas (plano), "similar al libro" (celda 20). En la figura de slides2 p45 (ejes: suma del jugador 12-21 y carta del dealer A-10) la superficie sin as usable es casi plana por debajo de 20 y salta a ≈ +0,4/+0,9 con 20-21; con as usable es más irregular (≈ −0,4 a −0,1 con sumas bajas) y también salta con 20-21 (hasta ≈ 1,0). El profesor advierte de que **esto no es la política óptima**, solo la evaluación de una política concreta (T24 ~02:02-02:03).
- **Monte Carlo ES** (celdas 22-23; slides2 p46): **5 000 000 episodios** (celda 23, ~14 min a ~6 000 it/s; la evaluación de 500 000 tardó 1 min 27 s, celda 14). Política inicial aleatoria (0 o 1 con prob. 0,5 por estado); la primera acción de cada episodio es aleatoria (exploring starts) y el resto sigue la política; $Q(s,a)$ por media incremental; tras cada episodio, la política se actualiza **solo en los estados visitados**: `0` si $Q(s,0)>Q(s,1)$, `1` en caso contrario (empates → pedir carta). El estado inicial lo da el `reset()` (reparto aleatorio), no se fuerza.
  - **Discrepancia de fuentes:** slides2 p46 y T24 ~02:03 hablan de **500 000** iteraciones del algoritmo MC ES; el notebook (celda 23) usa **5 000 000**. El mapa de política resultante (celdas 26-27) es el de la diapositiva. **Nota:** el cue de T24 ~02:03 dice también "500 partidas" por un error de la transcripción.
- **Política óptima obtenida** (celdas 26-27; slides2 p46): sin as usable: pedir con 12-16 contra 7-10 y A; contra 2-3 solo con 12; plantarse siempre con 17+; contra 4, 5 y 6 plantarse con cualquier suma (también con 12). Con as usable: pedir hasta 17 siempre (contra cualquier carta); con 18 solo contra A, 9 y 10; plantarse con 19+. Interpretación del profesor (T24 ~02:03-02:06): con 17 sin as usable lo óptimo es plantarse, con 17 y as usable es pedir (porque si me paso, el as vuelve a valer 1); la decisión depende algo de la carta del dealer; es muy parecida a la estrategia básica que siguen los jugadores expertos.
- Ilustra el control MC por episodios con exploring starts, visto en §3.5.

*Fuente: frozen_lake celdas 0-11, taxi celdas 0-14, gambler celdas 0-14, blackjack celdas 0-27; slides2 p26-p28, p43-p46; T23 ~03:42-04:21; T24 ~00:53-01:11, ~01:54-02:14.*

---

## Aplicación al Taller 2

Marco del Taller 2: `SimpleTradingEnv`, observación continua `[tendencia, posición]`, 3 acciones (vender, mantener, comprar), comisión 0,1 %, recompensa por paso ≈ 1e-3 (variación de precio con posición abierta menos comisión al operar), **494 decisiones por episodio** (500 pasos, se empieza en el paso 5) y episodio terminado por tiempo. DQN y A3C, con los cuatro generadores de precios del enunciado.

- **DP no es aplicable.** Policy y value iteration (slides2 p10, p20-p25) necesitan $p(s',r\mid s,a)$, que Gymnasium da en FrozenLake/Taxi (`env.unwrapped.P`) y que en el Gambler se conoce por construcción. En el trading la recompensa depende de una serie de precios aleatoria sin modelo analítico: es la limitación 1 de slides2 p29. Además la observación es **continua** (`tendencia ∈ [-1, 1]`), por lo que ni siquiera cabe una tabla: es la razón de pasar a redes neuronales (ver los documentos de DQN y A3C de esta carpeta).
- **Qué aporta MC al Taller 2.** MC es la base del objetivo de aprendizaje "retorno real". El retorno $G_t=\sum_k\gamma^kR_{t+k+1}$ calculado **hacia atrás** con `G ← γG + R` (slides2 p33; T24 ~01:23) se reutiliza tal cual:
  - En **A3C con retornos $n$-step**: **Nota:** (propuesta, no está en el material de clase). $G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^nV(S_{t+n})$. Se calcula con el mismo bucle hacia atrás, pero **inicializando $G$ con el valor del crítico** $V(S_{t+n})$ en lugar de 0 (bootstrapping al truncar). $n=1$ es TD(0) (ver [02_td_sarsa_qlearning.md](02_td_sarsa_qlearning.md)); $n\to T-t$ es exactamente MC.
  - **Compromiso sesgo-varianza (slides2 p35; T24 ~01:36).** El retorno completo MC no tiene sesgo, pero suma hasta 494 recompensas aleatorias de precios sintéticos: la varianza es alta y MC solo actualizaría tras cada episodio. TD/bootstrapping actualiza en cada paso, pero con sesgo inicial por el valor del crítico. $n$-step en A3C es un punto intermedio: menor varianza que MC y menor sesgo que TD(0). **Nota:** (propuesta) con un $n$ del orden de 5-20 pasos se actualiza cada pocas decisiones, lo que resuelve la limitación 1 de MC (no aprende online, slides2 p53); $n$ es un hiperparámetro a justificar con validación.
  - **Ojo con el fin de episodio.** La observación no incluye el tiempo restante y el entorno marca `terminated` al llegar a `max_steps` (en realidad es un corte por tiempo). Con `0 if done` (como en `frozen_lake`, celda 9) el valor al final se fija a 0, lo que es coherente con DP/MC (el episodio acaba) pero introduce una no-estacionariedad que la red no ve (**Nota:** propuesta a valorar; para DQN/A3C en el entorno del taller conviene decidir y documentar si se trata como terminal real).
  - **Evaluación = MC.** La métrica final del taller (mediana/media del retorno de la política greedy sobre las series de test) es precisamente una **evaluación Monte Carlo** de la política: se promedian los retornos observados en muchos episodios (slides2 p31). Hereda sus propiedades: sin sesgo, pero con la varianza propia del número de series; con una única serie el resultado es muy ruidoso, de ahí el uso de varias semillas y series de validación/test.
- **Exploración.** Con una política greedy sobre $Q$ algunos pares (posición, tendencia, acción) nunca se visitarían (slides2 p37). DQN usa $\epsilon$-greedy (política soft, slides2 p38, p48) y A3C usa la propia política estocástica. Exploring starts **no es aplicable**: no se puede forzar al mercado a empezar en un estado y acción arbitrarios (**Nota:** salvo que el entorno de simulación permita reiniciar en una posición concreta, que no es el caso). Consejo de T24 ~02:17: empezar con $\epsilon$ alto y decrecer; relevante porque con tres acciones y recompensas ≈1e-3, **Nota:** (hipótesis propia) un $Q$ casi plano puede quedarse en "mantener" por empates en el `argmax`; ver la lección de la tolerancia en gambler, §5.3.
- **On-policy y off-policy.** A3C es on-policy (los trabajadores aprenden de la política con la que actúan; se parece al control MC on-policy, §3.6). DQN es off-policy (la política de comportamiento $\epsilon$-greedy difiere de la greedy objetivo, con memoria de repetición). **Nota:** MC off-policy necesita el ratio $\rho$ y sufre cuando $\pi$ y $b$ difieren (slides2 p53); DQN lo evita porque su objetivo de un paso solo compara la acción realizada con el máximo del siguiente estado (sin producto de ratios a lo largo del episodio), es decir, es la idea de Q-Learning que anunciaba el profesor (T24 ~02:30).
- **Detalles reutilizables.** Inicializar valores en cero; actualización incremental `V ← V + (1/n)(G − V)` en lugar de guardar listas (blackjack, celda 14; T24 ~01:27); tolerancia al desempatar en `argmax` (gambler, celda 9: `1e-9`); y no hacer bootstrapping tras un estado realmente terminal (frozen_lake, celda 9). En términos de coste: 494 pasos por episodio implican que los episodios son largos comparados con blackjack (1-3 acciones, T24 ~02:08), así que MC puro necesitaría muchos más episodios para el mismo ruido (slides2 p53).
