# Aprendizaje por diferencia temporal: TD(0), Sarsa y Q-Learning

> **Fuentes:**
> - `slides3` (`docs/raw/MIAX-14/slides3.pdf`, "Métodos basados en Diferencia Temporal", pp. 1-43). Es el eje del documento. La p3 marca como lectura obligatoria Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, cap. 6, secciones 6.1-6.6, y como complementarias las clases 4 y 5 de Silver (2015).
> - `T24` (`docs/raw/transcripciones/`, clase del 24 de octubre, tramo de TD: horas 03:02 a 04:43 de la grabación). Explicaciones del profesor, preguntas de alumnos y una demostración en vivo de cliff walking. Es una transcripción automática y ruidosa; se cita con la hora del cue, p. ej. `(T24 ~03:12)`.
> - Para la última sección se usan los resultados del Taller 1 (`docs/raw/rl-trading-agents-main/README.md`) y el registro de decisiones del Taller 2 (`docs/DECISIONES.md`).
>
> **Convenciones:** "(slides3 pN)" = página N de las slides. "(T24 ~HH:MM)" = cue de la transcripción. **Nota:** = contenido propio que no está en el material de clase (ampliaciones, erratas detectadas, propuestas). Las slides no cubren Expected Sarsa, Double Q-Learning, n-step ni TD($\lambda$) (ver §10). Documentos relacionados: [`00_fundamentos_mdp.md`](00_fundamentos_mdp.md), [`01_dp_y_montecarlo.md`](01_dp_y_montecarlo.md) y [`03_dqn.md`](03_dqn.md).

---

## 1. Idea general de TD
**Idea.** TD combina lo mejor de DP y MC. Como MC, no necesita modelo del entorno (*model-free*) y aprende de experiencia muestreada: se pone a jugar, obtiene secuencias de estado-acción-recompensa y aprende de ellas. Como DP, actualiza a partir de estimaciones previas (*bootstrapping*), así que no espera al final del episodio y aprende con cada transición $(S,A,R,S')$ (slides3 p4; T24 ~03:04). Por eso aprende más rápido, sirve para aprendizaje *online* y en tiempo real, y vale para tareas episódicas y continuas (p4; T24 ~03:04).

**Taxonomía de los métodos tabulares** (diagrama de decisión, slides3 p5): primera pregunta, "¿requiere modelo?". Si sí, es Programación Dinámica (DP). Si no, segunda pregunta, "¿hace *bootstrapping*?". Si sí, es Diferencia Temporal (TD); si no, es Monte Carlo (MC).

Comparación de las tres familias (slides3 p6; T24 ~03:05-03:07):

| Propiedad | DP | MC | TD |
|---|---|---|---|
| Modelo $p$ | Sí (*model-based*) | No | No |
| *Bootstrapping* | Sí | No | Sí |
| Cuándo actualiza | Por barrido completo de los estados | Al final del episodio | Paso a paso ($t+1$) |
| Entorno | Episódico y continuo | Solo episódico | Episódico y continuo |
| Varianza de la muestra | Nula (calcula la esperanza) | Alta (suma de $T-t$ recompensas) | Baja (un paso estocástico) |
| Sesgo inicial | Alto (depende de $V_0$) | Nulo | Alto (depende de $V_0$) |

El profesor lo resume así: DP no tiene varianza pero sí sesgo alto por la estimación inicial; MC es lo contrario (varianza alta, sin sesgo); TD tiene el sesgo de DP pero algo menos de varianza que MC "porque no usa el episodio completo, hay menos ruido" (T24 ~03:06).

**Por qué importa aquí.** El profesor presenta Q-Learning como uno de los algoritmos más famosos e importantes del RL, y recuerda que su extensión a redes neuronales tuvo mucho éxito (T24 ~03:03). Su consejo práctico: si hubiera que usar un método tabular, iría directamente a Q-Learning antes que a los otros métodos comparados (T24 ~03:05). En la tarea del curso se pide como mínimo implementar solo Q-Learning y Sarsa, "los más famosos" (T24 ~03:05).

*Fuente: slides3 p4-p6; T24 ~03:02-03:07.*

---

## 2. Predicción TD(0)
**Idea.** El problema de predicción estima $V_\pi(s)$ para una política dada $\pi$ (slides3 p7). La regla de MC y la de TD(0) son la misma regla ("valor nuevo = valor viejo + $\alpha$ · (objetivo − valor viejo)"); lo único que cambia es el objetivo (T24 ~03:08-03:10).

**Repaso de MC** (slides3 p7-p8; T24 ~03:07-03:09). Si en $t$ se visita un estado no terminal $S_t$, MC espera al final del episodio para obtener $G_t$ y lo usa como objetivo:

$$V(S_t)\leftarrow V(S_t)+\alpha\,[G_t-V(S_t)]=(1-\alpha)V(S_t)+\alpha G_t .$$

- $\alpha\in(0,1]$ es la tasa de aprendizaje y $G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\dots$ actúa de objetivo (*target*). No se puede actualizar hasta llegar a $S_T$ (p8).
- Con $\alpha=1/k$ en la $k$-ésima visita al estado, la regla calcula el promedio incremental de los retornos observados, $V(S_t)\leftarrow\frac{k-1}{k}V(S_t)+\frac1kG_t$ (algoritmo MC estándar, p8). Con $\alpha$ constante se obtiene la variante *constant-$\alpha$ MC* (Sutton & Barto), que pesa todos los retornos por igual y es la que se usa de aquí en adelante para comparar con TD (p8; T24 ~03:08-03:09).

**TD(0).** No espera al final del episodio, solo un paso. Como no tiene $G_t$, lo sustituye por una estimación: la recompensa inmediata observada más el valor descontado de la estimación del estado siguiente (slides3 p9):

$$V(S_t)\leftarrow V(S_t)+\alpha\,[\,R_{t+1}+\gamma V(S_{t+1})-V(S_t)\,]$$

- **Objetivo TD** (*TD target*): $R_{t+1}+\gamma V(S_{t+1})$. Estima $G_t$ combinando una muestra real ($R_{t+1}$) con la estimación actual ($V(S_{t+1})$); ahí está el *bootstrapping* (p10; T24 ~03:10).
- **Error TD:** $\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$, la discrepancia entre la estimación actual y la mejorada. Es el error cometido en $t$ pero disponible en $t+1$ (p10; T24 ~03:11). Con él la regla es $V(S_t)\leftarrow V(S_t)+\alpha\delta_t$.
- El profesor subraya que **el error TD es lo que minimizan las redes neuronales entrenadas con Q-Learning** (DQN); lo anticipa y dice que lo ampliará otro profesor la semana siguiente (T24 ~03:11). Esta es la conexión directa con el Taller 2 (ver Aplicación).
- Se llama TD(0) porque mira un solo paso; se podría repetir con varios pasos (TD(2), TD(3)...). El profesor no describe esos algoritmos y remite a Sutton & Barto (T24 ~03:16).
- Ventaja de plantear MC y TD como una única regla de actualización con distinto objetivo: se pueden comparar bajo el mismo punto de vista (T24 ~03:12).

**Pseudocódigo TD(0)** (slides3 p16; T24 ~03:19-03:21). Entradas: política $\pi$ a evaluar, $\alpha\in(0,1]$, $\gamma\in[0,1]$. Salida: $V(s)$.

```
Inicializar V(s) arbitrariamente para todo s (con V(terminal) = 0)
para cada episodio:
    inicializar S
    mientras S no sea terminal:
        A <- accion dada por pi(.|S)
        tomar A, observar R, S'
        V(S) <- V(S) + alpha * [R + gamma * V(S') - V(S)]
        S <- S'
devolver V
```

Aquí $S'$ es el estado real al que se ha transitado, pero $V(S')$ viene de la estimación previa (T24 ~03:21). El profesor añade que el algoritmo converge y "no demasiado lento", es decir, funciona bastante bien (T24 ~03:21).

**Nota:** la p16 escribe "con $V(0)=0$ si $s$ es terminal"; se refiere a $V(\text{terminal})=0$, no al estado numerado 0.

*Fuente: slides3 p7-p10, p16; T24 ~03:07-03:12, ~03:16, ~03:19-03:21.*

---

## 3. DP, MC y TD: qué aproxima cada uno
**Idea.** La función de valor se puede escribir de dos maneras equivalentes (slides3 p11; T24 ~03:12-03:13): como esperanza del retorno, y como solución de la ecuación de expectativa de Bellman:

$$V_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]=\mathbb{E}_\pi[R_{t+1}+\gamma V_\pi(S_{t+1})\mid S_t=s].$$

Cada método parte de una de las dos y aproxima algo distinto:

- **DP** (p12; T24 ~03:13-03:14). Parte de Bellman. Calcula la esperanza **exacta** sobre acciones y sucesores con la dinámica conocida $p(s',r\mid s,a)$:
$$V(S_t)\leftarrow\sum_a\pi(a\mid S_t)\sum_{s',r}p(s',r\mid S_t,a)\,[\,r+\gamma V(s')\,].$$
Como no conoce $V_\pi$, aproxima $V_\pi(S_{t+1})$ por su estimación actual (*bootstrapping*). Solo aproxima $V$.
- **MC** (p13; T24 ~03:14-03:15). Parte de la definición de $V$ como esperanza del retorno. Sigue **una trayectoria completa** hasta $S_T$; los retornos son reales, no estimados (sin *bootstrapping*), pero la esperanza se aproxima por el promedio sobre una muestra finita de episodios.
- **TD(0)** (p14; T24 ~03:15-03:16). Parte de Bellman y hace las dos aproximaciones a la vez: sustituye la esperanza por **una muestra** de un paso y sustituye $V_\pi(S_{t+1})$ por su estimación actual. Cada método aproxima una cosa distinta, y TD tiene los dos tipos de error pero también las ventajas de ambos: no espera al final del episodio y aprende "súper rápido", con cada paso que da (T24 ~03:15-03:16).

**Diagramas de *backup*** (slides3 p15; T24 ~03:17-03:19). Círculos blancos = estados, puntos negros = acciones.
- DP: desde $S_t$ salen varias acciones y de cada una varios estados sucesores; se promedia con las probabilidades reales y se descuenta el valor estimado de cada sucesor (árbol completo de un nivel).
- MC: una sola cadena $S_t\to$ acción $\to$ estado $\to\dots$ hasta el cuadrado gris del estado terminal. El retorno es real pero corresponde a una trayectoria concreta; en otra visita habrá otra trayectoria y se promedia todo.
- TD(0): una sola arista $S_t\xrightarrow{A_t,\,R_{t+1}}S_{t+1}$. Mira un solo paso (no todo el árbol como DP, no todos los pasos como MC) y actualiza con la recompensa real más la estimación del estado siguiente.

**Sesgo y varianza.** TD tiene menos varianza que MC porque depende de una sola transición $(R_{t+1},S_{t+1})$ y no de toda la secuencia, que acumula ruido (p17, p6; T24 ~03:22). Su sesgo inicial depende de $V_0$ (p6), el de MC es nulo. **Nota:** las slides no desarrollan más este compromiso; es el mismo que reaparece en DQN (el objetivo depende de la propia red) y en los objetivos de $n$ pasos de A3C.

**Ventajas de TD(0)** (slides3 p17; T24 ~03:21-03:22): (1) no requiere modelo; aprende de la experiencia real; (2) es totalmente *online* y de actualización inmediata (solo espera al paso $t+1$), así que el aprendizaje es mucho más rápido; (3) es aplicable a tareas continuas, "incluso aunque no acabe nunca", no solo a episódicas como MC; (4) tiene menor varianza que MC.

*Fuente: slides3 p11-p15, p17; T24 ~03:12-03:19, ~03:21-03:22.*

---

## 4. Convergencia de TD(0) y comparación con MC
**Condiciones** (slides3 p18; T24 ~03:22-03:24). TD(0) converge a la función de valor verdadera $V_\pi$ cuando el número de pasos tiende a infinito si, para todo $s$,

$$\sum_{t=1}^{\infty}\alpha_t(s)=\infty\qquad\text{y}\qquad\sum_{t=1}^{\infty}\alpha_t^2(s)<\infty .$$

El profesor las describe como poco exigentes y dice que, como no se puede esperar al infinito, en la práctica se pone una tolerancia (T24 ~03:23).

**Con $\alpha$ constante** no hay garantía de converger al valor exacto: TD(0) acaba en un estado de **oscilación alrededor del valor verdadero** cuya amplitud depende de $\alpha$. Con $\alpha$ grande las oscilaciones son de mayor magnitud; con $\alpha$ pequeño, menores. Haciendo $\alpha\to0$ al final se garantiza converger a una solución pequeña (slides3 p18; T24 ~03:24). Nota del profesor: esto es solo evaluación de una política dada, aún no se mejora la política (T24 ~03:24).

**TD(0) vs constant-$\alpha$ MC** (slides3 p19-p20; T24 ~03:24-03:27). Ambos convergen asintóticamente a la verdadera función de valor con infinitos datos, pero TD(0) suele converger sustancialmente más rápido con datos en lotes finitos (*batch training*): toda la experiencia se limita a un conjunto de pasos o episodios y se reutiliza hasta converger. Con $\alpha$ constante y suficientemente pequeño, los dos convergen a **soluciones distintas**.

*Relevancia para finanzas* (T24 ~03:25-03:26): el profesor pone como ejemplo un problema de inversión en bolsa: no se pueden jugar infinitos episodios como en el ajedrez, solo se puede mirar el pasado y entrenar al agente con datos finitos. Es el mismo problema que el aprendizaje supervisado con una muestra finita: por bueno que sea el algoritmo, no está garantizado que aprenda la solución correcta (T24 ~03:27).

**¿A qué converge MC?** A la media muestral de los retornos observados para cada estado, $V(s)=\frac1N\sum_{i=1}^{N}G_i(s)$ (slides3 p20; T24 ~03:28). Con datos finitos eso es "la media de mis datos", no el valor esperado real: puede ser buena o mala estimación. La pregunta de la p20 es a qué converge TD(0).

**Ejemplo A-B** (Sutton & Barto; slides3 p21-p26; T24 ~03:29-03:42). MDP con dos estados no terminales A y B, $\gamma=1$, un lote de 8 episodios independientes; las acciones son irrelevantes (o el agente hace siempre lo mismo):

| Episodios | Secuencia (estado, recompensa, ...) |
|---|---|
| 1 | A, 0, B, 0 |
| 2-7 (seis veces) | B, 1 |
| 8 | B, 0 |

- **$V(B)$:** MC y TD(0) dan $0{,}75$ (p22). MC lo obtiene promediando los retornos tras B ($\frac{6\cdot1+2\cdot0}{8}=\frac68$). TD(0) observa que desde B siempre se va al terminal (valor 0) y promedia recompensa inmediata más valor del siguiente (1+0 seis veces, 0+0 dos veces). Llegan por caminos distintos al mismo resultado porque B siempre precede al terminal y el terminal vale 0 (T24 ~03:32-03:34).
- **$V(A)$ con MC:** A aparece solo en el episodio 1 y su retorno hasta el final es $0+0=0$; por tanto $V_{MC}(A)=0$ (p23). MC minimiza el error cuadrático medio respecto a los retornos observados en el conjunto de entrenamiento.
- **$V(A)$ con TD(0):** A se observa una vez, y el valor de A es la recompensa inmediata más el valor descontado del siguiente: $V(A)=0+V(B)=0{,}75$ (p24; T24 ~03:35-03:36).
- **Por qué difieren.** TD construye implícitamente el **modelo de máxima verosimilitud** del MDP subyacente (p25; T24 ~03:36-03:38). Diagrama de la p25: A $\to$ B con $R=0$ y $p=1$; B $\to$ Term con $R=1$ y probabilidad $6/8=0{,}75$, o con $R=0$ y probabilidad $2/8=0{,}25$. Si ese proceso inferido es correcto, TD(0) obtiene la función de valor **exacta en el sentido de DP** para él (usando el mismo objetivo "recompensa más valor descontado del siguiente" que DP), mientras MC solo promedia retornos históricos (p25-p26; T24 ~03:38-03:40).
- **Conclusión** (p26): MC minimiza el error cuadrático respecto a los retornos históricos; TD(0) minimiza el error de predicción del modelo de Markov implícito. Como la dinámica del problema es markoviana, la estimación de TD(0) es mejor predictor de los retornos futuros con datos no vistos que la respuesta "memorizada" por MC (T24 ~03:40-03:41). El profesor añade que, aunque no se conozca la dinámica, si el entorno es markoviano TD suele funcionar mejor que MC (T24 ~03:40).

**Pregunta de un alumno y respuesta** (T24 ~03:41-03:42): ¿por qué no dan lo mismo si MC también promedia el camino? Porque no hay infinitos datos. Con un modelo real e infinitos datos MC vería muchos episodios que salen "por arriba" y muchos "por abajo" desde A; con solo 8 episodios la única vez que se vio A fue un camino "por abajo", y MC concluye que $V(A)=0$. TD(0) reutiliza el valor aprendido de B ("si desde A siempre he ido a B, me lo creo") y asigna a A el valor descontado de B. Con infinitos datos ambos convergen a lo mismo; como en la práctica se entrena con una muestra finita, suelen funcionar mejor los métodos de tipo TD (T24 ~03:42).

**Nota:** la condición de las slides ($\sum\alpha_t=\infty$, $\sum\alpha_t^2<\infty$) es la de Robbins-Monro, y la convergencia asume además que todos los estados se visitan infinitas veces y que la política $\pi$ es fija. La afirmación "TD(0) converge más rápido" es una observación empírica (Sutton & Barto, ej. 6.2), no un teorema.

*Fuente: slides3 p18-p26; T24 ~03:22-03:42.*

---

## 5. Control TD: evaluación y mejora, on-policy vs off-policy
**Idea.** El problema de control busca la política óptima $\pi^*$. El planteamiento es el de siempre: alternar evaluación y mejora (evaluación, mejora, evaluación...), donde la evaluación es TD y la mejora es hacer la política *greedy* respecto a la función de valor medida (T24 ~03:43). Sin modelo del entorno hay que estimar $Q(s,a)$ en lugar de $V(s)$ (slides3 p27; T24 ~03:43). La regla general es:

$$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\,[\,\text{target}-Q(S_t,A_t)\,]$$

La naturaleza del *target* decide si el algoritmo es *on-policy* u *off-policy* (p27; T24 ~03:44). El profesor presenta dos objetivos:

- **On-policy (Sarsa):** evalúa y optimiza la misma política que usa para decidir e interactuar con el entorno (p28).
- **Off-policy (Q-Learning):** optimiza una política distinta de la que usa para seleccionar acciones (p32).

Precisión del profesor sobre el término: "política que evalúo" significa **la política que se utiliza para calcular el *target***. Si la que determina el *target* es la misma que selecciona las acciones, el algoritmo es *on-policy*; si son distintas, *off-policy* (T24 ~03:44).

*Fuente: slides3 p27, p28, p32; T24 ~03:43-03:45.*

---

## 6. Sarsa (on-policy)
**Idea.** Usa la transición $(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})$, donde $A_{t+1}$ se elige con la política exploratoria actual (p. ej. $\epsilon$-greedy) (slides3 p28; T24 ~03:45). El profesor comenta que el nombre viene de esa quíntupla (S, A, R, S', A') y no tiene más misterio (T24 ~03:46).

$$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\,[\,R_{t+1}+\gamma\,Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\,]$$

$$\text{Target}_{\text{Sarsa}}=R_{t+1}+\gamma\,Q(S_{t+1},A_{t+1})\quad(\text{p29})$$

- Es *on-policy* porque la misma política determina el comportamiento (elegir $A_{t+1}$) y define el objetivo; el objetivo usa la acción **real** que se va a ejecutar a continuación (slides3 p30; T24 ~03:46-03:47). Diagrama de la p30: $(S_t,A_t)\to R_{t+1}\to S_{t+1}\to$ "selección vía $\pi$ (p. ej. $\epsilon$-greedy)" $\to(S_{t+1},A_{t+1})$. Una vez elegida $A_{t+1}$ el *target* queda definido.
- La política debe ser $\epsilon$-greedy para garantizar la exploración de todas las opciones. Por eso no puede ser la política óptima (que es determinista), y converge a **la mejor política entre las $\epsilon$-greedy** (p30; T24 ~03:47).
- Si se hace tender $\epsilon\to0$, Sarsa también converge a la política óptima; Q-Learning no necesita esa condición porque la política que explora es distinta de la que optimiza (T24 ~03:51-03:52).

**Pseudocódigo** (slides3 p31; T24 ~03:48-03:51). Entradas: $\alpha\in(0,1]$, $\epsilon>0$, $\gamma\in[0,1]$. Salida: $Q(s,a)$.

```
Inicializar Q(s,a) arbitrariamente (con Q(terminal,.) = 0)
para cada episodio:
    inicializar S
    elegir A desde S con la politica derivada de Q (p. ej. epsilon-greedy)
    mientras S no sea terminal:
        tomar A, observar R, S'
        elegir A' desde S' con la politica derivada de Q (epsilon-greedy)
        Q(S,A) <- Q(S,A) + alpha * [R + gamma * Q(S',A') - Q(S,A)]
        S <- S';  A <- A'
devolver Q
```

La acción $A'$ se arrastra: la que sirve para actualizar es la misma que se ejecuta en el paso siguiente. "Estoy aprendiendo de las acciones reales; por eso lo llamamos *on-policy*" (T24 ~03:51).

*Fuente: slides3 p28-p31; T24 ~03:44-03:52.*

---

## 7. Q-Learning (off-policy)
**Idea.** $Q$ aproxima directamente la función de valor de acción óptima $Q^*$, y esto no depende de la política que se siga para explorar, siempre que todos los pares estado-acción se visiten eventualmente (slides3 p32).

$$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\,[\,R_{t+1}+\gamma\max_a Q(S_{t+1},a)-Q(S_t,A_t)\,]$$

$$\text{Target}_{\text{Q-learning}}=R_{t+1}+\gamma\max_a Q(S_{t+1},a)\quad(\text{p33})$$

**Dos políticas** (slides3 p34; T24 ~03:52-03:54):
- **Política objetivo** (*target policy*): puramente codiciosa respecto a la tabla $Q$ actual, $a^*=\arg\max_aQ(S_{t+1},a)$. Define el objetivo y es la que se está optimizando; al final del algoritmo es la política óptima.
- **Política de comportamiento** (*behavior policy*): suele ser estocástica y exploratoria, como $\epsilon$-greedy. Con probabilidad pequeña $\epsilon$ elige al azar y con $1-\epsilon$ la acción óptima. Ejemplo del profesor: si la mejor acción es la 3, se elige la 3 el 90 % de las veces y una acción aleatoria el 10 % (T24 ~03:56).

Intuición del diagrama de la p34: en Sarsa solo había la transición real $S_t\to S_{t+1}$; en Q-Learning, desde $S_{t+1}$ se consideran **todas** las acciones posibles (los puntos negros, con un arco que indica el máximo) y se queda uno con la mejor para actualizar. La acción que realmente se ejecuta se elige después, con la política de comportamiento (T24 ~03:54).

**Convergencia.** Converge a la política óptima si la política de comportamiento garantiza la visita de todos los pares estado-acción; $\epsilon$-greedy lo garantiza (slides3 p35; T24 ~03:55). $Q$ converge a la función de valor óptima, y una vez tenida la política óptima es elegir en cada estado la acción que maximiza $Q$; esa política final no es la que se usa para actuar durante el entrenamiento, porque a esa se le añade $\epsilon$ (T24 ~03:57-03:58).

**Pseudocódigo** (slides3 p36; T24 ~03:55-03:57). Parámetros iguales a Sarsa.

```
Inicializar Q(s,a) arbitrariamente (con Q(terminal,.) = 0)
para cada episodio:
    inicializar S
    mientras S no sea terminal:
        elegir A desde S con la politica derivada de Q (epsilon-greedy)
        tomar A, observar R, S'
        Q(S,A) <- Q(S,A) + alpha * [R + gamma * max_a Q(S',a) - Q(S,A)]
        S <- S'
devolver Q
```

Diferencia estructural con Sarsa: no se arrastra ningún $A'$; la acción se elige al inicio de cada paso y, tras observar $S'$, el objetivo usa el máximo en lugar de una acción muestreada (T24 ~03:55-03:56).

**Nota:** una consecuencia práctica del $\max$ es que, con aproximadores de funciones, el $\max$ de estimaciones ruidosas sobreestima el valor (*maximization bias*); Double Q-Learning y el objetivo de DQN con red objetivo lo mitigan. No está en las slides ni en la clase; se trata en [`03_dqn.md`](03_dqn.md).

*Fuente: slides3 p32-p36; T24 ~03:52-03:58.*

---

## 8. Sarsa vs Q-Learning en *cliff walking*
**Entorno** (Sutton & Barto, ej. 6.6; slides3 p37-p39; T24 ~03:58-04:00): tipo GridWorld de 4 filas por 12 columnas; el objetivo es ir de la salida S a la meta G.
- Inicio S=(3,0); meta G=(3,11); acantilado C en las casillas (3,1)-(3,10).
- Acciones arriba, abajo, izquierda y derecha, **deterministas**; si se sale del tablero se queda donde está.
- Recompensa $-1$ por cada transición entre casillas normales y $-100$ al entrar en el acantilado, que además devuelve al agente a S. El episodio **no termina** al caer; solo termina al llegar a G.
- Tarea episódica y sin descuento ($\gamma=1$).
- No hay recompensa positiva por alcanzar G: el estado G es terminal y se incentiva llegar rápido porque cada paso suma $-1$ (T24 ~04:00).

**Predicciones** (T24 ~04:01-04:05): Q-Learning debería converger a la política óptima, ir directo por el borde ("subir, venir hacia acá y bajar"), porque es lo que minimiza las penalizaciones. Para Sarsa, un alumno propone que podría llegar a la óptima si se ajusta $\epsilon$ a la baja. El profesor responde que con $\epsilon$ fijo no, "porque la política que busca es la misma que usa para explorar"; aprende la mejor política entre las $\epsilon$-greedy para ese $\epsilon$ concreto, y solo con $\epsilon\to0$ tendería a la óptima (T24 ~04:02-04:03).

**Resultados.**
- Q-Learning encuentra la **ruta óptima**, pegada al acantilado (en la figura, sube de S a la fila 2, la inmediatamente encima de las casillas C, la recorre entera y baja a G), porque la política que evalúa es determinista (slides3 p40). Es el camino más corto, sin tocar el acantilado.
- Sarsa encuentra una ruta **subóptima pero más segura** para una política $\epsilon$-greedy: sube hasta la fila superior (fila 0), cruza y baja a G (p41). Razonamiento del profesor: como explora con $\epsilon$-greedy, si va muy cerca del acantilado "alguna vez se va a caer", así que el objetivo incorpora ese riesgo y lo más seguro es alejarse (T24 ~04:04-04:05).
- **Rendimiento *online*:** el de Q-Learning es peor porque eventualmente cae al acantilado al explorar (p42; T24 ~04:05-04:06). Figura de la p42: recompensa por episodio frente a episodios (0-5000). La curva de Sarsa sube muy rápido desde cerca de $-100$ y se estabiliza en torno a $-20$ casi sin variación. La de Q-Learning oscila de forma amplia, aproximadamente entre $-30$ y $-70$ (la mayor parte del tiempo cerca de $-40$/$-50$), siempre por debajo de la de Sarsa. El profesor lo describe como aparentemente contradictorio: Q-Learning aprende la mejor política, pero la recompensa que obtiene episodio a episodio es algo menor.
- **Rendimiento asintótico:** el de Q-Learning es óptimo, "siempre encuentra la mejor política" en el límite de infinitos episodios; el de Sarsa es subóptimo si se fija $\epsilon>0$ y óptimo solo cuando $\epsilon\to0$ (p43; T24 ~04:09).

**Pregunta de un alumno sobre la gráfica** (T24 ~04:06-04:08): ¿es una media? El profesor primero cree que es la recompensa media en tandas, y luego, al ejecutar su propio código, confirma que cada punto es la recompensa real acumulada en un episodio, no un promedio de varios (T24 ~04:20).

**Nota:** la diferencia viene del *target*. Sarsa incorpora en $Q(S',A')$ el coste esperado de los pasos exploratorios cerca del precipicio; Q-Learning supone que después actuará de forma codiciosa. La curva de la figura es la del libro de Sutton & Barto; las slides no indican $\epsilon$, $\alpha$ ni la suavización (el ejemplo original usa $\epsilon=0{,}1$, $\alpha=0{,}5$ y promedia sobre varias ejecuciones; dato de fuera de clase).

*Fuente: slides3 p37-p43; T24 ~03:58-04:09, ~04:20.*

---

## 9. Demostración en clase y consejos para la tarea
**Código de la demo** (T24 ~04:09-04:21). El profesor muestra un entorno de cliff walking programado a mano sobre Gym (clase con tamaño de cuadrícula 4×12, posición inicial (3,0), meta (3,11), acantilado en la fila 3, 4 acciones, `reset`, `step`, `render`).
- Un agente aleatorio cae al acantilado enseguida (T24 ~04:14).
- Política $\epsilon$-greedy: con probabilidad $\epsilon$ una acción aleatoria entre 0 y 3; si no, el `argmax` de $Q$ (T24 ~04:16).
- $Q$ inicializada a ceros con forma (4 filas, 12 columnas, 4 acciones); el único requisito es 0 en terminales (T24 ~04:16). $\epsilon=0{,}1$ fijo; $\gamma=1$; $\alpha$ con decaimiento multiplicándolo por 0,99 (el valor inicial se entiende como 0,5 pero el cue es dudoso, y no queda claro si el decaimiento es por paso o por episodio; el profesor lo actualiza al final de cada episodio, ~04:21); hay un número máximo de pasos por episodio, tras el cual se corta el episodio (cifra poco clara).
- **Sarsa:** en cada paso toma la acción $A$ ($\epsilon$-greedy), observa $R,S'$, elige $A'$ con la misma política y actualiza $Q[S,A]$ con $R+\gamma Q[S',A']$; $A'$ es la acción que se ejecutará en el paso siguiente (T24 ~04:17-04:20).
- **Q-Learning:** idéntico salvo que el objetivo usa la acción *greedy* $\arg\max Q(S',\cdot)$, mientras la acción que realmente ejecuta el agente se obtiene con $\epsilon$-greedy (T24 ~04:29).
- Se entrena unos 5000 episodios. Al principio las recompensas por episodio están cerca de $-100$ y suben rápido; el agente aprende pronto a alejarse del acantilado (T24 ~04:21-04:22).

**Experimentos improvisados** (T24 ~04:22-04:33): cambiar la penalización del acantilado, agrandar el tablero (con más filas hay más estados, así que 5000 episodios no bastan y hay que explorar más; se multiplica el número de episodios) y añadir decaimiento de $\epsilon$ (×0,99 por episodio) en Sarsa para comprobar que converge hacia una política mejor. El profesor apunta que lo esperable es que al aumentar la penalización del acantilado el agente se aleje más. Los resultados mostrados fueron poco concluyentes y con algún fallo de la demo; el último, con $\epsilon$ decayendo, dio una política "un poco más abajo, muy parecida a la anterior" (T24 ~04:35). No hay una conclusión cuantitativa que recoger.

**Qué pide la tarea** (T24 ~04:33-04:35): hacer pruebas con el entorno de trading, ver cómo afectan los distintos hiperparámetros a los resultados y **entender si los resultados tienen sentido**: preguntarse qué se esperaría como política óptima. El profesor había dado la función generadora de precios; dice que no sabe si son completamente aleatorios y un alumno responde que esperaría "no hacer nada" (en un mercado aleatorio, cualquier política da 0 en promedio). Pide además probar distintos agentes y compararlos (por qué Sarsa hace una cosa y Q-Learning otra) (T24 ~04:34-04:35).

**Criterios de evaluación y entrega del taller** (T24 ~00:02-00:05; ~04:35-04:36). Los ejercicios de la descripción del taller son el **mínimo**: el taller es muy abierto ("se pueden probar 1 000 cosas") y el profesor valorará la innovación, la exploración y la creatividad. Reparto indicado: 50 % por hacer lo que se propone; hasta 10 % por ir más allá de esos mínimos; 20 % por la calidad y la presentación de los resultados; hasta 10 % por justificar las decisiones tomadas; y un análisis crítico de las soluciones (el peso de este último apartado no se oye, por el reparto resultante sería el 10 % restante). Plazo: el jueves siguiente a las 12 de la noche, mediante una tarea de la plataforma del curso; antes de la clase de ese día hay que enviar por email una primera versión de los resultados, de unas 2 hojas, y en la clase (3 h) el profesor habla con cada grupo y da feedback, con un pequeño margen para corregir. Dudas por email, sin garantía de respuesta rápida. Al acabar la sesión quedaba "una semana, un poquito menos" para hacerlo (T24 ~04:36). **Nota:** esto se dijo para el primer taller (tabular, Q-Learning y Sarsa); no sabemos si el reparto y los plazos son los mismos en el Taller 2. Conviene comprobarlo en su enunciado.

*Fuente: T24 ~00:02-00:05, ~04:09-04:36.*

---

## 10. Hiperparámetros, inicialización y lo que las slides no cubren
- **$\alpha\in(0,1]$** (slides3 p16, p31, p36): tasa de aprendizaje. Convergencia exacta si $\sum\alpha_t=\infty$ y $\sum\alpha_t^2<\infty$; con $\alpha$ constante hay oscilación cuya amplitud crece con $\alpha$ (p18; T24 ~03:24). Con $\alpha=1/k$ se obtiene el promedio incremental (p8).
- **$\epsilon>0$** (p31, p36): parámetro de exploración; $\epsilon$-greedy garantiza explorar todas las opciones (p30). Sarsa es óptimo solo si $\epsilon\to0$ (p43).
- **$\gamma\in[0,1]$** (p16, p31, p36): factor de descuento (cliff walking se plantea sin descuento, p39).
- **Inicialización de $Q$:** arbitraria para todo $s,a$, con $Q(\text{terminal},\cdot)=0$ (p31, p36); igual con $V$ en TD(0) (p16). El sesgo inicial de TD depende de $V_0$ (p6).
- **Lo que las slides NO dicen:** no mencionan GLIE ni calendarios concretos de decaimiento de $\epsilon$ o $\alpha$ (solo la condición de $\alpha$ en p18 y "$\epsilon\to0$" en p43), ni Expected Sarsa, Double Q-Learning, $n$-step o TD($\lambda$). Cualquier uso de ellos sería **Nota** propia.

**Familias de métodos** (T24 ~04:40-04:41). Todo lo visto son métodos **basados en valor**: se estima la función de valor y de ahí se deriva la política. Hay otros métodos que buscan la política directamente sin preocuparse por el valor (los verá otro profesor), y los métodos **actor-crítico** combinan ambos: el actor realiza acciones y un crítico valora si lo ha hecho bien o mal; suelen funcionar algo mejor. Es el puente hacia A3C (ver Aplicación).

**Sesgo de supervivencia** (T24 ~04:41): comentario del profesor sobre datos financieros. Con series de 20 años se suelen coger los activos que existían hace 20 años y siguen existiendo, que por definición "han ido más o menos bien", y eso sesga los resultados de muchos artículos académicos. Es una advertencia para interpretar resultados y no afecta a los generadores sintéticos del taller. El profesor lo cuenta a propósito de una tesis que dirige sobre RL en finanzas con métodos más avanzados (actor-crítico), en entornos más realistas, donde "hay muchos sesgos" en los datos y los artículos previos dan resultados demasiado buenos (T24 ~04:41).

**Ejemplo de aplicación sin sesgo: ejecución de órdenes grandes** (T24 ~04:42-04:43). Un trabajo de fin de máster de años anteriores usó RL para decidir cuándo comprar muchas acciones sin mover el precio: si una entidad grande compra de golpe, la demanda sube el precio, así que se reparte la compra en trozos. El profesor lo cita como caso en el que "no hay sesgo". Es un ejemplo de que la recompensa puede modelar el impacto en el precio; el entorno del taller no lo tiene (comisión fija del 0,1 %).

*Fuente: slides3 p6, p8, p16, p18, p30, p31, p36, p39, p43; T24 ~03:24, ~04:40-04:43.*

---

## Aplicación al Taller 2

**De la regla tabular al objetivo de la red.** El Taller 2 sustituye la tabla $Q$ (20 estados × 3 acciones en el Taller 1) por una red $Q(s,a;\theta)$ que recibe la observación continua `[tendencia, posición]` y devuelve 3 valores (vender, mantener, comprar). La regla de Q-Learning (§7) deja de actualizar una celda y pasa a ser una pérdida:

$$y=r+\gamma\,(1-\text{done})\max_{a'}Q(s',a';\theta^-)\qquad L(\theta)=\big(y-Q(s,a;\theta)\big)^2 ,$$

con $\delta=y-Q(s,a;\theta)$ el error TD de §2. Es lo que el profesor anticipó: lo que minimizan las redes entrenadas con Q-Learning es el error TD (T24 ~03:11). La tabla es el caso particular con características *one-hot*, donde el paso de gradiente $\theta\leftarrow\theta+\alpha\,\delta\,\nabla_\theta Q$ se reduce a $Q[s,a]\mathrel{+}=\alpha\,\delta$. **Nota:** el objetivo $y$ se trata como constante (gradiente "semi", sin derivar por $\theta^-$); la red objetivo $\theta^-$, la *replay memory* y los detalles de DQN están en [`03_dqn.md`](03_dqn.md).

```python
# Nota: esquema en PyTorch del objetivo de DQN para un lote (s, a, r, s2, done)
with torch.no_grad():
    y = r + gamma * (1 - done) * q_target(s2).max(dim=1).values   # Q-Learning (off-policy)
loss = F.mse_loss(q(s).gather(1, a[:, None]).squeeze(1), y)       # media de delta^2
```

**Qué se hereda de cada pieza de este documento:**

| Concepto | En el Taller 2 |
|---|---|
| Objetivo TD $r+\gamma\max Q$ y `done` ($Q(\text{terminal},\cdot)=0$, p31/p36) | Es el $y$ de DQN. D-1.2 (cerrada): el entorno devuelve `terminated=True` al acabar la serie aunque es un límite de tiempo (`truncated` siempre `False`). Se trata como terminal, igual que en el Taller 1, con efecto en 1 de 494 pasos. |
| Off-policy (§7) | DQN es off-policy: $\epsilon$-greedy actúa y el $\max$ define el objetivo, lo que permite reutilizar transiciones antiguas (*replay*). |
| Predicción TD(0), $\delta=r+\gamma V(s')-V(s)$ (§2) | El crítico de A3C aprende $V(s)$ con esta regla; **Nota:** el mismo $\delta$ (o su versión a $n$ pasos) es la ventaja que pondera la actualización del actor. Ver el documento de actor-crítico. |
| Política estocástica y $\epsilon$ (§6, §10) | A3C no usa $\epsilon$: explora porque la política es estocástica (D-2.1, abierta); el ajuste de $\epsilon$ del enunciado se traslada al coeficiente de entropía y hay que justificarlo en el informe. |
| Convergencia con $\alpha$ (§4) | La tasa de aprendizaje es ahora la de la red. Con $\alpha$ constante hay oscilación alrededor de la solución (p18): conviene bajar la tasa y evaluar con el protocolo común (D-0.1: política *greedy*, 200 series de test una sola vez). |
| Datos finitos (§4, T24 ~03:25) | Cada episodio usa una serie nueva (500 pasos, 494 decisiones), pero sigue siendo una muestra finita del generador. |

**Resultados del Taller 1 que condicionan el diseño** (README del Taller 1; mediana del retorno en 200 series de test, 5 semillas, hiperparámetros ajustados en validación):

| Mercado | Aleatorio | Buy & hold | Q-Learning | SARSA | Estrategia aprendida |
|---|---|---|---|---|---|
| 1 · Tendencia + ciclo | x4,3 | x33,6 | x502 | x472 | Sigue la tendencia |
| 2 · Browniano | −19,4 % | −3,5 % | 0 % | 0 % | No opera |
| 3 · Reversión a la media | −17,2 % | −0,3 % | +27,0 % | +27,0 % | Compra tras la caída más fuerte, vende tras la subida más fuerte |
| 4 · Regímenes | −22,0 % | −12,5 % | 0 % | 0 % | No opera (límite del estado, no del algoritmo) |

- **Q-Learning y SARSA llegan a lo mismo**; la diferencia es de velocidad: Q-Learning converge en ≈100-260 episodios y SARSA en ≈200-360 (189 frente a 203 en el mercado 1, 260 frente a 282 en reversión). Es coherente con §8: en *cliff walking* la diferencia es grande porque caer cuesta $-100$ y el entorno es determinista; en trading un error cuesta solo el 0,1 % de comisión y $\epsilon$ decae a 0,01, así que "la ruta segura" de SARSA apenas difiere. Con $\epsilon_{\min}=0{,}05$ SARSA opera bastante menos, lo que reproduce en pequeño el efecto de la p41. **Nota:** en DQN, que es off-policy, el análogo es que el rendimiento *online* durante el entrenamiento (recompensa con $\epsilon>0$) subestima el de la política *greedy*; por eso no debe usarse como único criterio de convergencia (D-0.3, cerrada: la convergencia se mide en la curva de validación de la política *greedy*, en episodios, pasos de entorno y tiempo).
- **$\alpha$ muy pequeña y $\gamma$ baja en tabular:** $\alpha=0{,}001$ en todos los mercados (con 0,1 el agente no aprende) y $\gamma=0$ en el mercado 1, $\gamma=0{,}5$ en el 3. Con recompensa casi inmediata y ruidosa, el *bootstrapping* a largo plazo aporta poco. **Nota (propuesta):** empezar por $\gamma$ bajo (0-0,5) y barrer hacia 0,99; una $\gamma$ alta amplifica el error de *bootstrapping* sin ganar señal.
- **Mercados 2 y 4: no operar es lo correcto.** El baseline `no_operar` bate a *buy & hold* en los mercados 2-4 (D-0.3). Un buen agente DQN/A3C no debe "inventar" operaciones; si la red opera en el mercado 2, es sobreajuste. Consecuente con lo que pedía la tarea (T24 ~04:34): comprobar si lo aprendido tiene sentido.
- **Estado más informativo:** en el Taller 1, añadir distancia a MA30 y volatilidad20 mejoró reversión (+18 % a +43-44 %) y regímenes (0 % a +7,5-7,8 %); la red del Taller 2 solo ve `[tendencia, posición]`, así que cabe esperar que el techo siga estando en el estado y no en el algoritmo (como en el mercado 4). **Nota:** ampliar la observación sería una extensión del enunciado.

**Cuestiones de escala propias de la red** (no existen en la tabla):
- La recompensa es ≈1e-3 por paso y la comisión del 0,1 % es del mismo orden; los $Q$ reales son pequeños (con $\gamma=0{,}99$ y 494 pasos, del orden de décimas como máximo) y las diferencias entre acciones son aún menores. Una red inicializada con salidas $O(1)$ tiene gradientes dominados por ese sesgo inicial. D-1.1 (cerrada) decide normalizar las dos cosas con constantes por mercado calculadas solo con train (`train_scales`): la tendencia, cuyos percentiles 5/95 son ≈ ±0,23 en el mercado 1 y ≈ ±0,04 en el resto, se divide por su σ, y la recompensa, por la σ del retorno.
- $\epsilon$ pasa de decaer por episodio en tabular a decaer por episodio o por paso en DQN; la regla del Taller 1 ($\epsilon\leftarrow\max(\epsilon_{\min},\epsilon\cdot\text{decay})$ por episodio) sigue siendo un punto de partida válido, con la recomendación de §10: $\epsilon\to0$ (o $\epsilon_{\min}$ pequeño) para que la política *greedy* final coincida con la aprendida.

**Cuándo mirar SARSA aún.** **Nota:** un DQN *on-policy* con objetivo SARSA ($r+\gamma Q(s',a')$, $a'$ real) es posible y daría políticas más conservadoras mientras $\epsilon>0$, pero no es lo que pide el enunciado y con las comisiones de este entorno el efecto es pequeño (Taller 1). Solo merece la pena como experimento adicional.

*Fuente: slides3 p6, p16-p18, p27-p36, p40-p43; T24 ~03:11, ~03:25, ~04:34; README del Taller 1; `docs/DECISIONES.md` (D-0.1, D-0.3, D-1.1, D-1.2, D-2.1).*
