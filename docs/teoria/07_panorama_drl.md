# Panorama del aprendizaje por refuerzo profundo: taxonomía, hitos, aplicaciones, entornos y librerías

> **Fuentes:** DRL2 (`docs/raw/DRL/transparencias/2-Mapa_conceptual_DRL.pdf`, "Mapa conceptual de DRL", 11 pp.); DRL3 ("Algunos hitos del DRL", 13 pp.); DRL4 ("Librerías con Entornos de RL", 20 pp.). Transcripciones de clase: T23 (23 oct, ~00:12-00:36, aplicaciones y algoritmos genéticos), T30 (30 oct, ~01:45-01:50 taxonomía; ~02:29-02:36 Deep Blue y juegos; ~02:36-03:15 librerías, consejos prácticos, algoritmos genéticos y de enjambre; ~03:35-04:05 aplicaciones, Gym y *gym-anytrading*). Enunciado: `docs/raw/taller/taller_trader2.ipynb`.
>
> **Convenciones:** "(DRL2 p10)" = página 10 del mazo; "(T30 ~03:41)" = hora aproximada del cue de la transcripción (automática y ruidosa; se parafrasea y no se cita literalmente). "**Nota:**" = contenido propio que no está en el material de clase, o una errata/corrección razonada. Las transparencias de DRL3 casi no llevan texto: son imágenes que el profesor comentó en clase, así que el contenido de cada hito sale de la imagen y de la transcripción. Los nombres de investigadores de artículos y hitos públicos sí se citan; los de profesores y alumnos no.
>
> **Documentos relacionados** (`docs/teoria/`): MDP y fundamentos, Q-Learning tabular, DQN y A3C/actor-crítico se explican en sus propios documentos; aquí solo se sitúan en el mapa general.

---

## 1. Mapa conceptual: taxonomía de algoritmos de RL
**Idea.** Los algoritmos de RL se ordenan en un árbol de decisiones de diseño. La primera es si el agente **aprende o conoce un modelo del entorno** (*model-based*) o **no** (*model-free*); la segunda, en el caso *model-free*, es si optimiza **directamente la política** o **una función de valor Q**. El mazo DRL2 reproduce este árbol, tomado de la documentación *Spinning Up* de OpenAI (DRL2 p2-p10, enlace en cada página: `spinningup.openai.com/.../rl_intro2.html#a-taxonomy-of-rl-algorithms`). Las páginas p2-p10 son fotogramas de una sola animación; la última (p10) contiene el diagrama completo.

**El diagrama (DRL2 p10)**, de arriba abajo:

```
RL Algorithms
├── Model-Free RL
│   ├── Policy Optimization:  Policy Gradient, A2C / A3C, PPO, TRPO
│   │        └─ (flechas compartidas con la rama Q-Learning): DDPG, TD3, SAC
│   └── Q-Learning:           DQN, C51, QR-DQN, HER
└── Model-Based RL
    ├── Learn the Model:      World Models, I2A, MBMF, MBVE
    └── Given the Model:      AlphaZero
```

- Las cajas **DDPG, TD3 y SAC** están dibujadas en el centro, con flechas hacia *Policy Optimization* y hacia *Q-Learning* a la vez: son algoritmos híbridos que aprenden una función Q y una política a la vez (DRL2 p10). **Nota:** en *Spinning Up* se describen como algoritmos que "interpolan" entre ambas familias, y por eso conviene no encasillarlos en una sola.
- **Q-Learning** (rama de la derecha de *Model-Free*) agrupa DQN y sus variantes: C51 y QR-DQN (versiones "distribucionales", que aprenden la distribución del retorno y no solo su media) y HER (*Hindsight Experience Replay*) (DRL2 p10). **Nota:** esta descripción de C51, QR-DQN y HER viene de la literatura, no de las transparencias.
- **Policy Optimization** agrupa los métodos que optimizan la política (gradiente de la política, A2C/A3C, PPO, TRPO) (DRL2 p10). **Nota:** A2C/A3C son métodos *actor-crítico*: una red propone acciones (actor) y otra estima el valor (crítico).
- **Model-Based** se divide en aprender el modelo (*World Models*, I2A, MBMF, MBVE) o recibirlo ya hecho (*AlphaZero*, al que se le dan las reglas del juego) (DRL2 p10).

**Qué significa "basado en modelo" (explicación del profesor).** No se refiere a la red neuronal: se refiere a **modelizar explícitamente el entorno**, es decir, a tener dentro del software algo parecido a un gemelo digital que intenta aprender la dinámica del entorno. El Q-Learning que se ha visto no lo hace: asigna un valor a cada estado (o par estado-acción) sin simular el entorno (T30 ~01:45). Dentro de los *model-free*, el profesor sitúa la optimización de políticas (policy gradient, y los actor-crítico, que se verán después) y los algoritmos de tipo Q, donde está **DQN**, "una especie de Q-Learning con redes neuronales", con muchas variantes que se complican cuando se sigue el mapa hacia abajo (T30 ~01:46). Dentro de los *model-based* hay dos tipos: aquellos a los que se da el modelo completo (por ejemplo, las ecuaciones físicas de un motor que se quiere controlar) y aquellos que lo aprenden. En el curso se centran en los *model-free* y, por ahora, en DQN (T30 ~01:47).

**Cuadro resumen de criterios** (**Nota:** síntesis propia a partir del diagrama):

| Familia | ¿Usa modelo del entorno? | Qué se aprende | Ejemplos (DRL2 p10) |
|---|---|---|---|
| Q-Learning (*value-based*) | No | $Q(s,a)$; la política sale de $\arg\max_a Q$ | DQN, C51, QR-DQN, HER |
| Policy Optimization | No | $\pi_\theta(a\mid s)$ (y un crítico $V$ en los actor-crítico) | Policy Gradient, A2C/A3C, PPO, TRPO |
| Híbridos | No | Q y política | DDPG, TD3, SAC |
| Model-Based, aprender el modelo | Sí (lo aprende) | Dinámica del entorno + política/valor | World Models, I2A, MBMF, MBVE |
| Model-Based, modelo dado | Sí (conocido) | Política/valor sobre el modelo | AlphaZero |

*Fuente: DRL2 p2-p10; T30 ~01:45-01:48.*

---

## 2. Dónde cae cada algoritmo del Taller en la taxonomía
**Idea.** El enunciado exige implementar **al menos DQN y A3C** y comparar con el agente aleatorio y con *buy & hold* (enunciado, celda 22). Con el mapa de la §1:

| Algoritmo / referencia | Rama en el mapa (DRL2 p10) | Notas |
|---|---|---|
| **Q-Learning tabular** (visto en clase antes) | *Model-Free* → *Q-Learning*; es el antecesor de DQN | Tabla $Q(s,a)$; exige estados discretos, y la observación del taller es continua (ver el documento de TD/Q-Learning de `docs/teoria/`). |
| **DQN** | *Model-Free* → *Q-Learning* → caja **DQN** | Aproxima $Q$ con una red neuronal; no usa modelo del entorno; el profesor lo presenta como "Q-Learning con redes" (T30 ~01:46). |
| **A3C** | *Model-Free* → *Policy Optimization* → caja **A2C / A3C** | Actor-crítico: dos salidas (política y valor). Los actor-crítico se presentan el día siguiente a DQN en el curso (T30 ~01:46). |
| Extensiones de DQN (Double DQN, *Prioritized Replay*, *Dueling*, *Distributional*) | Cuelgan de DQN (DRL2 p11) y, la parte distribucional, de C51/QR-DQN (DRL2 p10) | Opcionales; ver §3. |
| **Agente aleatorio** y **buy & hold** | No son RL | Son *baselines* de comparación que pide el enunciado (celda 22). |
| Algoritmos genéticos / de enjambre | **Fuera del mapa de RL** | Optimización por población; ver §9. |

**Nota:** ni DQN ni A3C son *model-based*. En el taller los precios los generan cuatro funciones conocidas, pero el agente no las usa: aprende solo de la experiencia $(s,a,r,s')$. Además, la acción del agente (comprar o vender) no modifica la serie de precios, así que el problema es más un problema de **decidir la posición** con coste de transacción que de "controlar" un entorno que reacciona; esto simplifica el diseño del entorno, pero no cambia el lugar de los algoritmos en el mapa.

*Fuente: DRL2 p10-p11; T30 ~01:45-01:48; enunciado (celda 22).*

---

## 3. El linaje de DQN
**Idea.** DRL2 p11 (título "DQN") muestra un diagrama de DeepMind con la evolución de DQN hasta **Agent57**. Es una línea vertical de cuatro hitos, y a cada hito entran "bloques" de ideas que se añaden al algoritmo (círculos grandes con círculos pequeños dentro). La imagen es pequeña; se transcribe lo legible.

| Hito (año en el diagrama) | Ideas que se incorporan (círculo que entra) |
|---|---|
| **DQN (2015)** | punto de partida |
| *DQN improvements* | **Double DQN**, **Prioritised Replay**, **Dueling Heads**, **Distributed** (se leen así en el diagrama; "Distributed/Distributional" es ambiguo por la resolución) |
| **R2D2 (2019)** | + *Short-term memory*: **LSTM**, **GRU** |
| **Never Give Up (2019)** | + *Episodic memory*: **Memory Networks**, **Neural Episodic Control**, **Transformers**; + *Exploration*: **Curiosity**, **Intrinsic Motivation**, **Density Models**, **Hashing**, **Random Network Distillation**, "Cells" (lectura incierta), **Reachability** |
| **Agent57 (2020)** | + *Meta-controller*: **PBT**, **Bandits**, **Meta-Gradients**, **Adaptive Bandits** |

Lectura del diagrama: de DQN a sus mejoras, se arreglan problemas de estabilidad y sesgo de la estimación; con *memoria a corto plazo* (redes recurrentes) se maneja la observación parcial; con *memoria episódica* y *exploración intrínseca* se atacan los juegos con recompensas escasas; con el *meta-controlador* se elige, mediante bandits o PBT, qué política/horizonte usar en cada episodio. **Nota:** esta lectura es mía, basada en la literatura de R2D2, Never Give Up y Agent57; la transparencia solo da los nombres.

**Nota:** el diagrama dice 2019 para *Never Give Up*; la publicación de ese algoritmo (Badia et al.) suele citarse como 2020 (ICLR). Es una discrepancia menor de fecha en la fuente.

El profesor dijo que "hay un montón de variantes" de DQN, y que "esto se empieza a complicar" cuando se sigue el mapa (T30 ~01:46); que se estudiarán con el mapa a mano.

*Fuente: DRL2 p11; T30 ~01:46.*

---

## 4. Qué es un problema de RL: ejemplos agente-entorno (DRL3 p2-p5)
**Idea.** DRL3 abre con el esquema del bucle agente-entorno (DRL3 p2): el **agente** toma una **acción**, el **entorno** devuelve un **estado** y una **recompensa**, y se repite. Al lado, la imagen de un rover marciano. Después se proyectan otros ejemplos de agentes del mundo real:

- **Robots de exploración** (rover de Marte) (DRL3 p2): el profesor lo usa para ilustrar el control robótico (T30 ~03:36). Un robot complejo tiene cientos de motores finos; programar a mano la dinámica de todos es lo que hacía la **teoría de control** (calcular el voltaje exacto para cada motor); con RL se da una señal de recompensa de alto nivel ("cuanto más cerca del punto B, más premio") y el robot aprende a coordinar sus motores según el terreno (roca, arena, pendiente) (T30 ~03:37-03:40). Además, los motores se desgastan e incluso se rompen, y el agente tiene que adaptarse (T30 ~03:39).
- **Robot aspirador** (DRL3 p3): los más avanzados usan RL para decidir cómo moverse y cuánto tiempo dedicar a cada estancia para limpiar la casa en el menor tiempo (T30 ~03:40).
- **Coche autónomo** (DRL3 p4, con enlace a un vídeo) (T30 ~03:41).
- **Medicina**: el esquema de DRL3 p5 representa al **paciente** como entorno, su **estado de salud** como estado, la **supervivencia** como recompensa, y al agente como un "médico robot" que propone un **conjunto de acciones** (fármacos) entre las que el equipo médico elige la **acción escogida**, considerando preferencias del paciente y coste/disponibilidad.

**Nota:** el profesor dijo que el campo del RL aplicado "está explotando" y que, mirándolo bien, lo raro es *no* plantear un problema como RL, porque es la forma natural en la que aprendemos los humanos (T30 ~03:48).

*Fuente: DRL3 p2-p5; T30 ~03:35-03:41, ~03:48.*

---

## 5. Aplicaciones concretas descritas en DRL3
### 5.1 Dosificación de eritropoyetina en hemodiálisis (DRL3 p6)
**Idea.** Se muestra la página de un artículo: Martín-Guerrero, Gómez, Soria-Olivas, Schmidhuber, Climente-Martí y Jiménez-Torres, "A reinforcement learning approach for individualizing erythropoietin dosages in hemodialysis patients", *Expert Systems with Applications* 36 (2009) 9737-9742 (DRL3 p6). Los pacientes en hemodiálisis necesitan medicación que tiene efectos secundarios y es cara; los médicos, ante la duda, tienden a dar de más (T30 ~03:42). El problema se plantea como RL:

- **Estado:** resultados de las últimas analíticas (marcadores en sangre, con la hemoglobina como protagonista) y si el paciente ha mejorado o empeorado (T30 ~03:42). **Nota:** el profesor lo explicó oralmente; la transparencia solo reproduce acciones y recompensa.
- **9 acciones** (DRL3 p6): 1 subir la dosis de EPO $\beta$; 2 bajarla; 3 subir la dosis de darbepoetina $\alpha$; 4 bajarla; 5 aumentar el número de administraciones de EPO $\beta$; 6 reducirlo; 7 aumentar el de darbepoetina $\alpha$; 8 reducirlo; 9 no cambiar nada.
- **Recompensa** (ecuaciones 6-8 del artículo, DRL3 p6): suma de un término de *nivel* y uno de *cambio*:

$$\text{reward}=(\text{reward})_{Hb}+(\text{reward})_{\text{change}}$$

$$(\text{reward})_{Hb}=\begin{cases}10 & Hb\in[11.5,12.5]\ \text{g/dl}\\ 0 & Hb\in[10.0,11.5)\ \text{o}\ Hb\in(12.5,13.0)\\ -5 & Hb\in[9.0,10.0)\ \text{o}\ Hb\in[13.0,14.0)\\ -10 & \text{en otro caso}\end{cases}$$

$$(\text{reward})_{\text{change}}=\begin{cases}0 & |Hb(t)-Hb(t-1)|<1\ \text{g/dl}\\ -5 & 1\le|Hb(t)-Hb(t-1)|<1.5\ \text{g/dl}\\ -10 & \text{en otro caso}\end{cases}$$

**Lo que explica el profesor:** el primer término premia estar en el rango normal (+10 dentro, -5 algo fuera, -10 muy fuera). El segundo es "lo interesante": **penaliza que el paciente no esté estable** entre dos analíticas, de modo que la mejor recompensa exige estar bien en la última analítica *y* en la anterior. Así se evita una política que reacciona a cada medida subiendo o bajando la dosis continuamente (T30 ~03:44-03:45). El resultado fue que el agente aprendía a dosificar de forma más eficiente que los médicos, con menos efectos secundarios (T30 ~03:45).

**Lección para el Taller:** una recompensa puede combinar un término de *objetivo* con un término de *regularidad/estabilidad* que castiga cambios bruscos de acción. En trading, el análogo sería penalizar el exceso de operaciones (la comisión del entorno ya lo hace implícitamente). **Nota:** analogía propia.

### 5.2 Descuentos en campañas de marketing (DRL3 p7)
**Idea.** Gómez-Pérez, Martín-Guerrero, Soria-Olivas, Balaguer-Ballester, Palomares y Casariego, "Assigning discounts in a marketing campaign by using reinforcement learning and neural networks", *Expert Systems with Applications* 36(4), mayo 2009, pp. 8022-8031 (DRL3 p7). La transparencia muestra un **modelo RL** y su procedimiento:

- **Estado** $s_t=BMU_i$: la *Best Matching Unit* de un **mapa autoorganizado (SOM)** para la $i$-ésima transacción del cliente.
- **Acción** $a_t=a_t(V_C)$: $V_C$ son los créditos virtuales asignados (el descuento).
- **Recompensa inmediata** $r_t=LTV$ (*lifetime value* del cliente).
- **Procedimiento** (6 pasos): (1) entrenar un SOM; (2) calcular el estado del cliente por su BMU en el instante $t$; (3) acción de la empresa; (4) con la recompensa, obtener el siguiente estado; (5) calcular $Q^\pi(s,a)$; (6) estimar la política óptima $\pi^*(s)$.
- El diagrama dibuja las variables del cliente $S=S(R,F,A,A\text{-}P,E)$ entrando en un mapa hexagonal (SOM), la BMU (por ejemplo $BMU=9$) y la **tabla $Q(s,a)$** cuya fila muestra valores como 8.2, 7.1, 11, 1.3, 9.2, 6.1, 4.3.

**Lección para el Taller:** es un ejemplo de **agregación de estados**: se comprime una descripción continua y de alta dimensión (el SOM) a un número pequeño de estados discretos para poder usar una tabla $Q$. **Nota:** esto enlaza con la discusión de §10 sobre simplificar el estado, y con el Q-Learning tabular que usaríamos si discretizáramos `[tendencia, posición]`.

*Fuente: DRL3 p6-p7; T30 ~03:41-03:45.*

---

## 6. Hitos de DeepMind y relacionados (DRL3 p8-p13)
**Idea.** Los hitos que el profesor recorre muestran que el RL profundo pasó de jugar videojuegos a descubrir algoritmos.

### 6.1 DQN en Atari (Breakout) (DRL3 p8)
La imagen es el juego *Breakout* con el logotipo de Google DeepMind. DQN (2015) consiguió jugar a videojuegos Atari mejorando el nivel humano usando **solo los píxeles de la pantalla** como entrada: no sabe qué es una pala ni una pelota; nadie le indica nada del juego (T23 ~00:21-00:22). Surgió cuando las redes convolucionales tuvieron sus primeros éxitos procesando imágenes: se coge la pantalla, se procesa con convoluciones y la salida se conecta a los mandos (derecha/izquierda) (T23 ~00:22). En el vídeo de entrenamiento mostrado en la clase del 23 de octubre se aprecia que al principio juega mal y, tras unos 20 min de entrenamiento, juega mejor que cualquier persona (T23 ~00:30). Un alumno comentó que se apilan varios fotogramas porque con uno solo no se sabe si la pelota sube o baja, y el profesor lo confirmó (T23 ~00:31).

### 6.2 AlphaGo contra Lee Sedol (DRL3 p9)
La imagen es el *Google DeepMind Challenge Match* (8-15 de marzo de 2016), AlphaGo contra el campeón Lee Sedol. El profesor comentó que "esas redes parecen tener intuición" y hacen movimientos elegantes: el algoritmo no imita al humano, aprende a optimizar el problema desde cero y encuentra estrategias que a los humanos no se nos habían ocurrido (T30 ~02:29).

### 6.3 AlphaGo Zero (DRL3 p10, "Aprendizaje reforzado")
Gráfica de **Elo** (eje vertical de -2000 a 5000) frente a un eje horizontal de 0 a 40, con tres series: *AlphaGo Zero 40 blocks* (línea continua), *AlphaGo Lee* (puntos verdes, ~3500) y *AlphaGo Master* (puntos azules, ~4700). En esta diapositiva solo se ven las dos líneas de referencia horizontales. **Nota:** en el artículo original el eje horizontal es el tiempo de entrenamiento (días) y la curva de AlphaGo Zero, que aprende solo por autojuego sin partidas humanas, supera primero a AlphaGo Lee y después a AlphaGo Master; la diapositiva, como animación, probablemente la dibujaba progresivamente. Es el punto de "aprende desde cero" que subrayó el profesor (T30 ~02:32).

### 6.4 AlphaTensor (DRL3 p11)
Entrada del blog de DeepMind, "Discovering novel algorithms with AlphaTensor", 5 de octubre de 2022, con el esquema de una multiplicación de matrices $A\,(4\times5)\times B\,(5\times5)=C\,(4\times5)$ y dos listas de fórmulas (algoritmo clásico y el encontrado). El algoritmo descubrió una forma más rápida de multiplicar matrices, no conocida hasta entonces; la recompensa es el número de operaciones: cuantas menos multiplicaciones haga, mejor, y si se equivoca en el resultado la recompensa es muy negativa (T30 ~03:46). **Nota:** en la formulación original, el agente juega a "descomponer un tensor" y cada paso cuesta una recompensa de $-1$, de modo que minimizar pasos equivale a minimizar el rango de la descomposición.

### 6.5 Stratego (DRL3 p12)
Entrada del blog de DeepMind "Mastering Stratego, the classic game of imperfect information" (1 de diciembre de 2022). Muestra el tablero y la codificación de la observación como capas (piezas del jugador, información pública del rival, jugadas pasadas, lagos) y una tabla de complejidad:

| | Ajedrez | Póker | Go 19×19 | Stratego |
|---|---|---|---|---|
| Turnos por partida | ~60 | ~15 | ~300 | ~1000 |
| Configuraciones iniciales | 1 | $10^6$ | 1 | $10^{66}$ |
| Complejidad del árbol de juego | $10^{123}$ | $10^{17}$ | $10^{360}$ | $10^{535}$ |

Pie de la imagen: "la escala de las diferencias entre ajedrez, póker, Go y Stratego". El profesor lo mencionó como ejemplo de **juegos de información imperfecta**: en ajedrez o Go se ve todo el tablero (información completa), pero en el póker o Stratego no se conocen las cartas/piezas del rival; el RL se puede seguir usando con información incompleta, y estos algoritmos "están explotando desde hace unos años" (T30 ~02:32-02:33, ~03:46).

### 6.6 AlphaDev: ordenación más rápida (DRL3 p13)
Artículo de Mankowitz, Michi, Zhernov, Gelmi, Selvi, Paduraru, Leurent, Iqbal, Lespiau, Ahern, Köppe, Millikin, Gaffney, Elster, Broshear, Gamble, Milan, Tung, Hwang, Cemgil, Barekatain, Li, Mandhane, Hubert, ..., Silver, "Faster sorting algorithms discovered using deep reinforcement learning", *Nature* 618, 257-263 (7 de junio de 2023) (enlace al blog de DeepMind). La figura compara el **código ensamblador original** con el de **AlphaDev**, con las instrucciones que cambian resaltadas (rojo en el original, verde en AlphaDev): el agente encontró una secuencia más corta de instrucciones para ordenar. El profesor lo presentó como un caso en el que RL aprendió a ordenar más rápido que los métodos conocidos, y subrayó que además los LLM se afinan con RL (T30 ~03:47).

**Nota:** estos tres últimos hitos (AlphaTensor, Stratego, AlphaDev) ilustran que el RL se usa para *descubrir* soluciones en espacios combinatorios enormes. No tienen relación directa con el taller, pero muestran que la recompensa se define sobre un objetivo medible (número de operaciones, instrucciones, victorias).

*Fuente: DRL3 p8-p13; T23 ~00:21-00:31; T30 ~02:29-02:33, ~03:45-03:47.*

---

## 7. RL, minimax y Deep Blue; RL con LLM
**Idea.** Un alumno preguntó si el *software* de ajedrez de hoy está basado en aprendizaje; el profesor explicó la diferencia con Deep Blue (T30 ~02:30-02:31):

- **Deep Blue** (IBM) jugó contra el campeón mundial de ajedrez, Kasparov, y perdió el primer torneo por poco. Cargaba una gran biblioteca de partidas pasadas y hacía una búsqueda **minimax**: qué movimientos puedo hacer, qué responde el adversario a cada uno, cómo contraataco... La búsqueda explota combinatoriamente, así que se limitaba a una profundidad de 4-5 movimientos. Con esa base de datos y esa combinatoria funcionaba muy bien, pero **la máquina no aprende nada**.
- Los algoritmos de RL de hoy "aprenden desde cero" y no dependen de bases de datos de otros jugadores (T30 ~02:32). En el ajedrez, T23 ~00:17-00:19: el profesor cree que los mejores motores de búsqueda siguen apoyándose en árboles de búsqueda y no solo en RL, y que probablemente se combinan; no lo afirmó con seguridad.
- El ajedrez es un entorno **no determinista desde el punto de vista del agente** porque el estado siguiente depende también de lo que haga el oponente (T23 ~00:16).
- **RL en los LLM:** los modelos de lenguaje tienen una última fase de afinado con RL cuya recompensa es la satisfacción del usuario (por ejemplo, valorar la respuesta con 5 o 2 estrellas); esto tiene pros y contras (el usuario queda más contento, pero el modelo puede acabar dándole siempre la razón) (T30 ~03:47; T23 ~00:23).

*Fuente: T30 ~02:29-02:33, ~03:47; T23 ~00:16-00:19, ~00:23.*

---

## 8. Más aplicaciones vistas en clase y el diseño de la recompensa
**Idea.** En la clase del 23 de octubre se enseñaron aplicaciones de RL (T23 ~00:21-00:30): videojuegos (cualquier algoritmo de estos supera a las personas desde DQN), robótica y control automático, vehículos autónomos, modelos generativos (afinado con feedback humano) y, cada vez más, finanzas. Vídeos de ejemplo: un simulador de persona que aprende a **correr** (estado = posición y fuerza de las articulaciones; acciones = mover cada articulación; recompensa = metros avanzados), aprender a **montar en bicicleta** (recompensa positiva si se mantiene, negativa si cae; nadie le explica la técnica), una **carrera de drones** entre una persona y un algoritmo, y *Breakout* (T23 ~00:24-00:30).

**Recompensa escasa vs. recompensa parcial.** Un alumno preguntó si una recompensa binaria (solo al llegar a la meta) dificulta aprender. El profesor dijo que depende del problema, pero que si el agente casi nunca llega a la meta al principio, apenas recibe señal; a veces conviene una **recompensa parcial** (como los metros avanzados), que da *feedback* positivo con cualquier avance (T23 ~00:25-00:26). Y que **definir una buena recompensa es "quizás el problema más grande" del RL**: una recompensa que parece razonable lleva al agente a hacer cosas que no se habían previsto, sobre todo en entornos reales (T23 ~00:26).

**Ejemplos de diseño de recompensas (T30 ~03:55-04:01):**
- *Levantarse del suelo (robot humanoide).* Si se da recompensa solo al final (cuando está de pie), al principio el agente no la encuentra explorando. Si se da recompensa pequeña por "cabeza más alta", el agente puede **quedarse a gatas** cobrando premio en cada paso, y el juego dura indefinidamente. La solución parcial es premiar la *mejora* de altura respecto a antes, o la altura relativa a los hombros, pero el problema se complica.
- *Robot que anda.* Un trabajo de fin de estudios usó un entorno de Gym con recompensa por ir de un punto a otro lo más rápido posible; el robot terminaba **a saltitos**. Para evitarlo se penalizó el tiempo en el aire; el robot terminó **arrastrándose**. Una penalización pequeña puede dominar si no se equilibra con el premio a la velocidad.
- *Coche autónomo.* Si se penaliza estar cerca de otro coche, pisar una línea continua, ir rápido... y no se calibra bien, el coche "ni arranca" (las penalizaciones totales superan lo que cobra por avanzar). Lo más sencillo es premiar solo al llegar y penalizar fuertemente el choque, pero entonces es muy difícil que la exploración aleatoria llegue al final sin chocar; hay que buscar soluciones intermedias.
- *Tráfico.* Controlar semáforos se puede plantear como RL; la recompensa sería el flujo total de coches medido por cámaras, con el tráfico conectado (los coches deben entrar y salir); primero hay que construir el simulador de calles (T30 ~04:01-04:03).

**Conclusión:** el diseño de la recompensa es crítico y no trivial; hay que buscar el equilibrio entre premios y castigos (T30 ~04:00).

**Nota (aplicable al Taller):** el entorno del Taller ya da una recompensa por paso (≈$10^{-3}$) que incluye la comisión, así que no hay recompensa escasa; el riesgo es el opuesto: que la comisión del 0,1 % haga que "no operar" sea el óptimo local (el agente aprende a quedarse quieto). Esto es una observación propia, relacionada con los ejemplos de arriba.

**Recordatorio RL vs. aprendizaje supervisado (T23 ~00:12-00:20).** En RL no hay etiquetas, solo una recompensa numérica a largo plazo y, a menudo, retrasada; el agente descubre la estrategia por prueba y error; y las acciones determinan no solo la recompensa inmediata sino el **estado siguiente**, que condiciona todas las recompensas futuras. Además, la suposición i.i.d. del aprendizaje supervisado **no se cumple**: las observaciones $(s,a,r)$ están muy correlacionadas, y algunos algoritmos necesitan "trucos" para acercarse a esa suposición (como el *experience replay* de DQN) (T23 ~00:19-00:20). Ver el documento de fundamentos (MDP) de `docs/teoria/` para el desarrollo formal.

*Fuente: T23 ~00:12-00:30; T30 ~03:55-04:03.*

---

## 9. Algoritmos genéticos y de enjambre: qué son y en qué se diferencian del RL
**Idea.** No son RL, pero el profesor los presentó como alternativa de optimización (y un alumno los mencionó a propósito de un vídeo de coches, T23 ~00:32). Son **optimización por población** inspirada en la evolución (T30 ~03:00-03:14).

**RL frente a genéticos (T23 ~00:32-00:36).** El RL aprende como los seres vivos interactúan con el entorno, mediante recompensas; el genético aprende como evoluciona una especie. Se define una población de individuos (cada uno con una política codificada en unos "genes"), se los pone a jugar, los que avanzan más se reproducen más, los hijos mezclan a los padres con **mutaciones** y se repite durante muchas generaciones. Dicho "un poco bruto": el algoritmo genético es esencialmente **azar guiado por selección**; las mutaciones no tienen dirección y de vez en cuando aparece una que funciona y se perpetúa. En RL, en cambio, hay un concepto de **valor** de cada acción que se va aprendiendo a lo largo de la interacción.

**Cómo funciona un genético (T30 ~03:00-03:06):**
1. Se define una **función de *fitness*** que puntúa lo bueno que es un individuo (no necesita ser analítica: puede ser un montón de `if`). Hay que decir qué parámetros se optimizan y si son continuos, discretos o binarios.
2. Se fija el tamaño de la población (por ejemplo, 100 individuos). Cada "época" o generación sustituye la población por los **hijos** de la anterior; los mejores se combinan más.
3. **Cruce** (*crossover*): cómo se combinan dos padres para generar el hijo. Puede ser muy simple: por ejemplo, en una cartera de 10 activos, tomar 5 pesos de un padre y 5 del otro y **normalizar** para que sumen 1.
4. **Mutación**: con una cierta probabilidad se cambia algo al azar en el hijo; es lo que mantiene la búsqueda explorando.

Siempre hay tres funciones: *fitness*, cruce y mutación (T30 ~03:06). Es "como una búsqueda aleatoria pero con una heurística que se enfoca en las zonas donde parece haber mejores individuos".

**Ventajas (T30 ~03:00-03:13):** admite funciones de coste no continuas ni derivables y parámetros no continuos (enteros, categóricos); en optimización de carteras sirve para fijar, por ejemplo, las ventanas de medias móviles (21, 14 o 31 días) o la ventana de estimación de la volatilidad; permite **restricciones** fácilmente (pesos no negativos, entre dos valores, presupuesto finito): a las soluciones que las violan se les asigna un *fitness* "horriblemente malo" ($-\infty$); busca óptimos **globales**, mientras que el descenso por gradiente tiende a óptimos locales; existen mezclas con descenso por gradiente (el genético acota la región, el gradiente afina). El profesor recomienda una librería de Python de computación evolutiva y programación genética (la citó con un nombre que la transcripción no recoge bien, por lo que no se puede identificar con seguridad).

**Límites:** son **muy lentos**; por eso no se usan para entrenar redes neuronales con miles de coeficientes (T30 ~03:06). Hay teoremas que garantizan que, con tiempo infinito, encuentran el mejor, pero encontrar una solución decente puede tardar mucho. El descenso por gradiente exige funciones continuas y derivables, y es muy rápido para llegar a óptimos locales (T30 ~03:08-03:13).

**Programación evolutiva y enjambres (T30 ~03:13-03:15).** Los genéticos son un caso particular de *algoritmos evolutivos*, que incluyen la evolución de programas (código). Otra familia son los **algoritmos de enjambre**, como los inspirados en las **hormigas**: al principio las hormigas exploran al azar; cuando una encuentra comida, las demás siguen su camino hasta que otra encuentra uno más corto; se comunican con un **rastro químico** (feromonas); acaban encontrando el camino más óptimo. Hay algoritmos de optimización global basados en esta dinámica.

**Nota:** el profesor no desarrolló más los métodos de enjambre; en el Taller no hacen falta. Conviene saber que existen porque pueden optimizar **hiperparámetros** o pesos de una estrategia sin necesidad de gradiente.

*Fuente: T23 ~00:32-00:36; T30 ~03:00-03:15.*

---

## 10. Consejos prácticos de clase sobre estabilidad, hiperparámetros y simplificación
**Idea.** Mientras entrenaba un DQN en directo sobre un juego (recoger manzanas con una cesta), el profesor dio consejos generales que se aplican al Taller (T30 ~02:34-03:00). Se resumen aquí porque son comentarios de contexto sobre el campo; el detalle de DQN está en su documento.

- **Inestabilidad.** A veces el agente va muy bien y de repente "desaprende". Pasa porque, a diferencia del aprendizaje supervisado (donde el *dataset* es fijo y el descenso por gradiente mejora siempre), aquí la **base de datos de experiencias cambia** (cuando se llena, se eliminan las más antiguas) y la política también cambia (T30 ~02:46-02:47). Por eso hay tantas variantes de algoritmos que atacan la estabilización de formas distintas (T30 ~02:47).
- **Variantes para estabilizar.** Un alumno preguntó por el uso de una segunda red. El profesor dijo que los *actor-crítico* usan dos redes (o dos capas de salida): una propone la acción y otra la critica y modula; y que también hay DQN con dos redes. Todo son intentos de que el entrenamiento no se desestabilice (T30 ~02:34). Otros trucos: **recortar (*clipping*) las recompensas** si son demasiado altas, porque unas puntuaciones demasiado bajas o con un rango muy amplio también afectan a la estabilidad (T30 ~02:35). **Nota:** una recompensa del Taller de ≈$10^{-3}$ por paso entra en el caso de "demasiado bajas" (señal diminuta), y el recorte no la arregla; conviene reescalarla (propuesta propia).
- **Tamaño de la memoria de repetición.** Es un hiperparámetro: no puede ser muy pequeño (solo retendría lo último) ni demasiado grande (al principio las experiencias vienen de una política mala, casi aleatoria); se busca un valor intermedio (T30 ~02:48-02:49). Lo mismo con el **tamaño del lote** (*batch*): si es muy grande la red cambia demasiado en cada actualización; se entrena "de a poquito" (T30 ~02:49-02:50). No hay un criterio general: es **prueba y error** (T30 ~02:50).
- **Hiperparámetros a vigilar:** $\gamma$ (cuánto planifica el algoritmo), la **velocidad de aprendizaje** (si es alta, los pesos cambian demasiado y el entrenamiento no es estable), el tamaño de la memoria, el tamaño del lote y la arquitectura de la red (T30 ~02:50-02:51). Y la tasa de exploración $\varepsilon$: conviene empezar alta e ir **decayendo**, pero con cuidado: si baja demasiado rápido se "congela" la política antes de que haya aprendido (T30 ~03:14-03:15; ya a ~01:48 el profesor había dicho que mantener $\varepsilon=0{,}1$ fijo es solo una opción). Con $\varepsilon$ constante (10 % de acciones aleatorias) un agente ya entrenado puede perder alguna partida por acciones aleatorias (T30 ~02:40).
- **Número de épocas.** Para conjuntos de datos enormes se entrenan pocas épocas (se citó que los grandes modelos de lenguaje se entrenan solo unas pocas) (T30 ~02:49). **Nota:** en el Taller cada episodio con una serie de precios nueva funciona como "dato nuevo", así que "muchos episodios" es lo adecuado (enunciado, celda 22).
- **Un episodio es una unidad de tiempo con sentido.** Un alumno observó que en el Taller un episodio serían 500 precios seguidos, "como 500 días" (T30 ~02:41). Coincide con el entorno (494 decisiones por episodio).
- **Simplificar el problema (experiencia del profesor en un proyecto real de recomendación de inversión) (T30 ~02:52-02:59).** El primer intento, con ~100 variables y mucha complejidad, fue "un desastre": la red no era estable, no convergía, y cuando convergía hacía malas predicciones. Lo que funcionó fue **simplificar al máximo**: (1) reducir las variables a 7 indicadores de mercado y agruparlas con *clustering* en **5 estados de mercado** (cuantos más estados, más cuesta converger); (2) añadir el estado de la cartera (si va en pérdidas o no), hasta ~20 estados; (3) no pedir al agente que diseñe toda la cartera, sino solo **2 acciones** (mantener la composición actual o cambiar entre una composición conservadora y otra arriesgada, ambas definidas por un experto); (4) entrenar con episodios de un año simulado con datos pasados, empezando en **fechas aleatorias** para no sobreajustar a una época. Con un Q-Learning de 20 estados × 2 acciones la tabla $Q$ se podía **interpretar** (qué recomienda el sistema en cada situación de mercado). Recomendación general: "simplificar a saco al principio" y desarrollar a partir de ahí, porque "como intentes resolver todo de un ataque, te la pegas": RL es mucho más sensible a los parámetros que el supervisado, donde la curva de pérdida siempre baja (T30 ~02:57-02:59).

*Fuente: T30 ~02:34-03:00.*

---

## 11. Entornos: OpenAI Gym / Gymnasium (DRL4 p1-p16)
**Idea.** **Gym** es una API estándar de RL y una colección diversa de entornos de referencia (cita de la página `gymlibrary.dev`, DRL4 p3). Su interfaz es "simple, *pythonic* y capaz de representar problemas generales de RL" (DRL4 p3). La barra lateral de la documentación muestra las categorías: **Atari, MuJoCo, Toy Text, Classic Control, Box2D, Third Party Environments**, además de un tutorial para crear un entorno propio ("Make your own custom environment") (DRL4 p3). La captura es del entorno *Lunar Lander* (DRL4 p3). Con Gym, el profesor distingue: Gym es el **entorno**, las librerías de algoritmos son el **agente** (esquema de DRL4 p17, §12).

| Categoría | Entornos mostrados | Descripción / uso |
|---|---|---|
| **Atari 2600 Arcade** (DRL4 p4-p8) | mosaico de juegos; *Tennis*, *Skiing*, *Demon Attack*, *Fishing Derby* | Observación = píxeles; son los de DQN (§6.1) |
| **MuJoCo** (*Multi-joint dynamics in contact*) (DRL4 p9) | *Ant*, *Humanoid*, *Half Cheetah* | Robots articulados con física realista (inercia, rozamiento, motores que no aceleran instantáneamente); una acción por articulación (T30 ~03:53-03:55) |
| **Toy Text** (DRL4 p10) | *Blackjack*, *Taxi*, *Frozen Lake* | Estados discretos; los típicos de Q-Learning tabular (T30 ~03:49) |
| **Classic Control** (DRL4 p11) | *Cart Pole*, *Mountain Car*, *Pendulum* | Cart Pole: un palo sobre un carrito que se mueve a izquierda/derecha para que no caiga; Mountain Car: empujar el carrito para alcanzar la bandera (T30 ~03:50) |
| **Box2D** (DRL4 p12) | *Bipedal Walker*, *Car Racing*, *Lunar Lander* | "Problemas que involucran control físico" |
| **Terceras partes** (DRL4 p13-p16) | *highway-env* (carretera, aparcamiento), ***anytrading*** | "Muchos más entornos" |

- **highway-env** (DRL4 p13-p15): simulador de conducción; las capturas muestran una carretera de cuatro carriles (coche verde = agente, coche azul = otro) y un aparcamiento (coche amarillo, plaza objetivo verde).
- **gym-anytrading** (DRL4 p16): entorno de trading ya programado. La captura muestra la salida `info: {'total_reward': -173.1000000000602, 'total_profit': 0.980652456904312, 'position': 0}` y una gráfica de precio (entre 1.040 y 1.060, ~60 pasos) con puntos verdes y rojos que marcan operaciones. **Nota:** `total_profit` < 1 indica pérdida respecto al capital inicial (=1) y `total_reward` suma las ganancias/pérdidas por operación; son medidas distintas y pueden ser ambas negativas o con signos diferentes.
- El profesor comentó que hay mucho material de Gym con videojuegos y robótica, pero **poco de finanzas** (T30 ~03:49); en tercera parte hay un entorno de bolsa, "algo como lo que habéis programado, pero ya programado": solo habría que optimizar con vuestro algoritmo (T30 ~04:04). Todos estos entornos son abiertos y sirven para experimentar con los algoritmos; también hay **competiciones** sobre cuál aprende antes a resolverlos (T30 ~03:52).
- **Gym → Gymnasium.** Gym lo mantienen ahora otros; el proyecto original se copió a otro paquete llamado **Gymnasium**, que es el que está actualizado (T30 ~03:51). **Nota:** `SimpleTradingEnv` del enunciado hereda de `gymnasium.Env` (`import gymnasium as gym`, `spaces`; primeras celdas del enunciado), así que tiene la forma de la API estándar (`reset`, `step`, `observation_space`, `action_space`) que usan las librerías de §12, con una excepción: su `reset(self, seed=None)` no acepta `options` (ver [notebook_partida.md](notebook_partida.md) §4.2).

*Fuente: DRL4 p1-p16; T30 ~03:49-04:05.*

---

## 12. Librerías de algoritmos: Baselines, Stable Baselines3 y Zoo (DRL4 p17-p20)
**Idea.** DRL4 p17 ("Gym y Baselines") dibuja el bucle agente-entorno (acción hacia el entorno; observación y recompensa de vuelta al agente) con dos flechas: **Baselines** señala al **agente**, **Gym** al **entorno**. Es decir: Gym pone los problemas y Baselines pone los algoritmos.

- **OpenAI Baselines** (DRL4 p18, `github.com/openai/baselines`): "un conjunto de implementaciones de alta calidad de algoritmos de RL. Estos algoritmos facilitan que la comunidad investigadora replique, refine e identifique nuevas ideas, y crearán buenas *baselines* para construir investigación sobre ellas. Nuestra implementación de DQN y sus variantes está aproximadamente a la par con las puntuaciones de los artículos publicados. Esperamos que se usen como base sobre la que añadir nuevas ideas y como herramienta para comparar un enfoque nuevo con los existentes."
- **Stable Baselines3** (DRL4 p19, `stable-baselines3.readthedocs.io/en/master/`): la página del mazo solo muestra el enlace. **Nota:** es la versión mantenida (PyTorch) de Stable Baselines, con implementaciones fiables de A2C, PPO, DQN, SAC, TD3, DDPG, entre otras; no incluye A3C asíncrono sino su versión síncrona A2C.
- **RL Baselines3 Zoo** (DRL4 p20, `github.com/DLR-RM/rl-baselines3-zoo`): "un *framework* de entrenamiento para agentes de Stable Baselines3". Proporciona *scripts* para entrenar, evaluar agentes, ajustar hiperparámetros, dibujar resultados y grabar vídeos; incluye una colección de **hiperparámetros ya afinados** para entornos y algoritmos habituales, y agentes entrenados con esos ajustes. Objetivos: (1) interfaz simple para entrenar y disfrutar los agentes; (2) comparar los distintos algoritmos de RL; (3) dar hiperparámetros afinados por entorno y algoritmo; (4) divertirse con los agentes entrenados. Es la versión SB3 del *rl-zoo* original de SB2.

**Lo que dijo el profesor sobre librerías (T30 ~02:36-02:37).** Hubo una librería de RL con "todo tipo de algoritmos", muy estable y bien documentada, que funcionaba con Keras, pero **dejó de mantenerse**: al cambiar las versiones de TensorFlow/Keras sin compatibilidad hacia atrás, hoy salen errores al usarla. "Es una pena: no hay una librería estándar con estos algoritmos", y hay que buscarlas en Python más que en Keras. **Nota:** por la descripción (Keras, abandonada), la librería a la que se refiere es probablemente `keras-rl`; la transcripción no lo deja claro. En ningún momento de las clases transcritas se menciona Stable Baselines por su nombre; solo aparece en DRL4.

*Fuente: DRL4 p17-p20; T30 ~02:36-02:37.*

---

## Aplicación al Taller 2
**Contexto del entorno:** `SimpleTradingEnv` (gymnasium), observación continua `[tendencia, posición]`, 3 acciones, comisión del 0,1 %, recompensa ≈$10^{-3}$ por paso, 494 decisiones por episodio.

1. **Lugar en la taxonomía (§2).** DQN = *Model-Free* → *Q-Learning* (caja DQN, DRL2 p10). A3C = *Model-Free* → *Policy Optimization* (caja A2C/A3C). Son las dos grandes ramas *model-free*, así que la comparación del Taller es en realidad **valor (DQN) frente a política + crítico (A3C)**. Los *baselines* (aleatorio, *buy & hold*) no son RL.
2. **¿Usar Stable Baselines3 en el Taller? Nota: propuesta propia, razonada.**
   - **El enunciado pide implementar** "los algoritmos vistos en clase", "al menos DQN y A3C" (celda 22). Usar `stable_baselines3.DQN` directamente no cumpliría el espíritu del ejercicio ni permitiría analizar hiperparámetros ($\alpha$, $\varepsilon$ con decaimiento, $\gamma$, episodios; punto 3 del enunciado), ni medir el tiempo de convergencia (punto 4) con control. Además, **SB3 no tiene A3C**, solo A2C (síncrono): no sería el mismo algoritmo.
   - **Sí conviene** usarlo como **contraste externo opcional**: entrenar un DQN o A2C/PPO de SB3 en el mismo entorno (acepta un `gymnasium.Env` con `Box` de dimensión 2 y `Discrete(3)`; pero `check_env` de Gymnasium falla porque `reset` no acepta `options`, comprobado, y con SB3 no se ha probado porque no está en `requirements.txt`: probablemente haga falta un envoltorio, sin tocar `env.py`) y comprobar que nuestra implementación llega a resultados comparables o razonables. Es el uso que se le da a un *baseline* de referencia (DRL4 p18). Si el resultado de SB3 supera claramente al nuestro, es una señal de *bug* o de hiperparámetros malos.
   - **Riesgos prácticos:** problemas de versiones (el profesor advirtió de librerías que dejan de funcionar al cambiar de versión, T30 ~02:37); habría que fijar versiones de `gymnasium` y `stable-baselines3`. La recompensa tan pequeña (≈$10^{-3}$) puede requerir normalización (por ejemplo, `VecNormalize`) tanto en SB3 como en nuestra versión.
   - **Zoo** (DRL4 p20) puede servir como **punto de partida de hiperparámetros** (tasa de aprendizaje, tamaño de memoria, ε-decay) para DQN/A2C, aunque estén afinados para otros entornos.
3. **Estabilidad y simplificación (§10).** Con observación de solo dos números y 3 acciones, el problema ya está "simplificado al máximo" como aconsejó el profesor; no hay que añadir variables. Atención a la inestabilidad de DQN (memoria de repetición, red objetivo, tasa de aprendizaje, $\varepsilon$ con decaimiento no demasiado rápido) y a que cada episodio use una serie de precios distinta, generada por la misma función (enunciado, celda 22).
4. **Recompensa (§8).** Revisar que "no operar" no sea un óptimo local trivial por la comisión de 0,1 %, y plantearse si conviene reescalar la recompensa de ≈$10^{-3}$ por paso. Proponer, como mucho, un término de regularidad al estilo de la recompensa de EPO (§5.1) solo si se observa *overtrading*; no hace falta para cumplir el enunciado.
5. **Evaluación.** Comparar DQN, A3C, aleatorio y *buy & hold* en las cuatro funciones de precios con series de test nuevas y promediadas, como pide el enunciado. *gym-anytrading* (§11) es un entorno distinto (no es el del Taller); no es necesario usarlo.
6. **No usar genéticos/enjambres como algoritmo principal (§9),** pero pueden servir, como **Nota** propia, para ajustar hiperparámetros sin gradiente si hubiera tiempo.
