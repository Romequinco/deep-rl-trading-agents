# Doble DQN y DQN en duelo

> **Fuentes:** DRL5 (`docs/raw/DRL/transparencias/5-Doble_DQN_y_DQN_en_duelo.pdf`, 22 páginas, todas leídas, incluidas las imágenes). Referencias que cita el mazo: Van Hasselt, Guez y Silver (2015), *Deep Reinforcement Learning with Double Q-Learning* (el mazo lo titula "Deep Learning with Double Q-Learning"), y Wang, Schaul, Hessel, van Hasselt, Lanctot y de Freitas (2016), *Dueling Network Architectures for Deep Reinforcement Learning* (ICML). De la transcripción T30 (clase del 30 de octubre) solo se usan los fragmentos que tratan de los *targets* que dependen de la propia red, de la idea de "dos redes" y del estado terminal; el resto de T30 no pertenece a este documento.
>
> **Convenciones:** "(DRL5 pN)" = página N del mazo. "(T30 ~hh:mm:ss)" = hora aproximada en la transcripción. "**Nota:**" = contenido propio que no está en las fuentes (correcciones, pseudocódigo, propuestas). $Q_\theta$ es la red que se entrena (*online*) y $Q_{\theta^-}$ la red retardada (*target*); el mazo las llama **Q** y **Q'**.
>
> Prerrequisitos: DQN básico ([03_dqn.md](03_dqn.md)) y Q-learning ([02_td_sarsa_qlearning.md](02_td_sarsa_qlearning.md)).

---

## 1. El problema: el operador max sobreestima

**Idea.** En DQN el objetivo de regresión usa un `max` sobre las acciones del estado siguiente. Como las estimaciones de $Q$ tienen error, tomar el máximo de varias estimaciones ruidosas tiende a dar un valor mayor que el real: el `max` sobreestima (DRL5 p6-p8).

La ecuación básica de DQN es (DRL5 p8):

$$\text{loss}=\Big(\underbrace{r_t+\gamma\max_{a'}Q(s_{t+1},a')}_{\text{Target}}-\underbrace{Q(s_t,a_t)}_{\text{Predicción}}\Big)^2$$

que el mazo reescribe como

$$Y_t=r_t+\gamma\max_{a'}Q(s_{t+1},a'),\qquad \text{loss}=\big(Y_t-Q(s_t,a_t)\big)^2$$

y concluye: "el operador max hace que se tienda a sobreestimar la Q" (DRL5 p8).

**Nota (por qué ocurre).** Si $Q(s',a')=q^*(s',a')+\varepsilon_{a'}$ con ruido $\varepsilon$ de media cero, entonces $\mathbb E[\max_{a'}(q^*+\varepsilon)]\ge\max_{a'}q^*$ (el máximo es una función convexa). La misma red *elige* la acción (la que más "parece" valer, a menudo porque su ruido es positivo) y *evalúa* su valor (con ese mismo ruido). Al hacer *bootstrapping*, cada actualización arrastra además el sesgo del estado siguiente. Es el mismo sesgo del Q-learning tabular, ahora con aproximación de funciones.

**Contexto de T30.** El profesor insiste en que lo nuevo de DQN frente a una regresión normal es que el *target* es dinámico: depende de la salida de la misma red, así que cada vez que se entrena un poco hay que recalcular los *targets* (T30 ~00:42:33-00:43:40). Dice que es una de las cosas más complicadas de este salto conceptual (T30 ~00:43:53). En las conclusiones lo repite: se ha convertido el problema en regresión, pero "los targets son dinámicos, van cambiando en el tiempo, dependen de la misma red" (T30 ~01:01:54-01:02:05). Esa inestabilidad es la que Doble DQN y la red retardada intentan paliar.

*Fuente: DRL5 p6-p8; T30 ~00:42:33-00:43:53, ~01:01:54-01:02:05.*

---

## 2. Doble DQN

**Idea.** Separar la **selección** de la acción de su **evaluación**, usando dos redes: una elige la mejor acción y la otra le pone el valor (DRL5 p9-p13). Idea de investigadores de Google DeepMind (Van Hasselt, Guez, Silver, 2015) (DRL5 p4-p5).

Según el mazo (DRL5 p13), "solución: se tienen dos redes $Q$ y $Q'$". En vez de calcular el *target* como $Y_t=r_t+\gamma\max_{a'}Q(s_{t+1},a')$, se calcula como:

$$\text{acción}=\arg\max_{a'}Q(s_{t+1},a')$$
$$Y_t=r_t+\gamma\,\max_{a'}Q'(s_{t+1},\text{acción})$$

- La red $Q$ se actualiza en cada paso.
- $Q'$ se actualiza más lentamente: cada $k$ pasos se copia $Q$ en $Q'$; entre medias, $Q'$ no se actualiza (DRL5 p13).
- Consecuencia que da el mazo: "el aprendizaje es más estable" (DRL5 p14).

### Erratas y aclaración de las fórmulas

**Nota (errata en DRL5 p9-p13).** La segunda línea del mazo, $Y_t=r_t+\gamma\max_{a'}Q'(s_{t+1},\text{acción})$, conserva un `max` que sobra: una vez fijada la `acción`, $Q'(s_{t+1},\text{acción})$ ya es un único número y no hay nada que maximizar. Si se leyera como "máximo de $Q'$ sobre $a'$" se volvería a DQN con red retardada, y volvería la sobreestimación. La forma correcta, tal como aparece en Van Hasselt et al. (2015), es:

$$\boxed{\,Y_t=r_t+\gamma\,Q_{\theta^-}\!\Big(s_{t+1},\ \arg\max_{a'}Q_\theta(s_{t+1},a')\Big)\,}$$

**Quién hace qué:**

| Paso | Red que lo hace | Parámetros | Se actualiza |
|---|---|---|---|
| **Selecciona** la acción $a^*=\arg\max_{a'}Q(s_{t+1},a')$ | **$Q$** (la *online*, la que se entrena) | $\theta$ | cada paso |
| **Evalúa** $Q'(s_{t+1},a^*)$ | **$Q'$** (la retardada, *target*) | $\theta^-$ | copia de $\theta$ cada $k$ pasos |
| Predicción $Q(s_t,a_t)$ en la pérdida | **$Q$** | $\theta$ | cada paso |

Lectura coherente con las diapositivas: la red que se entrena ($Q$) propone la acción y la lenta ($Q'$) la valora. Si $Q$ elige una acción por ruido favorable, es poco probable que $Q'$, que está retardada y tiene un ruido distinto, la sobrevalore igual.

**Nota (matices que el mazo no menciona).**
- **Estado terminal:** si $s_{t+1}$ es terminal no hay estimación futura y $Y_t=r_t$. El profesor lo explica para DQN: en estado final no se calcula esa parte, "no tiene sentido preguntarse qué viene después del estado final"; por eso el código tiene un `if` de fin de juego y el *target* es solo $r$ (T30 ~04:36:19-04:37:04). Con una máscara: $Y_t=r_t+\gamma(1-d_t)\,Q_{\theta^-}(s_{t+1},a^*)$. En el taller, el fin de serie cuenta como terminal (D-1.2).
- El mazo escribe una sola $Q$ en el *target* de DQN "básico" (p6-p8). El artículo de DQN de Mnih et al. (2015) ya usaba una red *target* congelada durante $k$ pasos; lo que aporta Doble DQN es **elegir con la $Q$ online** y usar la *target* solo para evaluar. Con una red *target* ya existente, Doble DQN no añade redes: cambia una línea del cálculo del *target*.
- Alternativa a la copia dura cada $k$ pasos: actualización suave (Polyak) $\theta^-\leftarrow\tau\theta+(1-\tau)\theta^-$ con $\tau$ pequeño (p. ej. 0,005). No está en las fuentes.
- La pérdida cuadrática (MSE) pesa mucho los extremos; en T30 el profesor adelanta que luego se verá una función de coste que no será exactamente el MSE (T30 ~01:02:11-01:02:23). Una opción habitual es Huber; DRL5 no la desarrolla.

**Contexto de T30 (pregunta de un alumno).** Un alumno pregunta por qué DeepMind utiliza una segunda red y por qué da problemas una sola. El profesor responde que existen muchas variantes: actor-crítico (dos redes o dos capas de salida, una propone las acciones y la otra las "critica" y modula) y también variantes de DQN con dos redes; todas son "intentos para estabilizar, para que no se inestabilice el entrenamiento" (T30 ~02:34:09-02:34:57). El algoritmo base debería poder resolver cualquier problema, pero el problema de estos algoritmos es la estabilidad y el tiempo que tardan en encontrar la solución (T30 ~02:35:05-02:35:18). También afecta a la estabilidad que las puntuaciones sean demasiado bajas o con diferencias muy grandes; por eso se aplican trucos como saturar (*clipping*) los valores demasiado altos (T30 ~02:35:22-02:35:47).

*Fuente: DRL5 p3-p5, p9-p14; T30 ~02:34:09-02:35:47, ~04:36:19-04:37:04.*

---

## 3. DQN en duelo (Dueling DQN)

**Idea.** En lugar de que la última capa estime directamente $Q(s,a)$ de cada acción, la red se **parte en dos ramas**: una estima cuánto vale el estado, $V(s)$, y otra cuánto mejor o peor es cada acción respecto a las demás, $A(s,a)$. Después se combinan para reconstruir $Q(s,a)$ (Wang et al., 2016; DRL5 p16-p17).

**Descripción de los diagramas (DRL5 p18-p21).** Todos usan la misma red convolucional de ejemplo (imagen de entrada, tres bloques convolucionales y una capa densa grande):
- *DQN básica* (p18): tras la capa densa, una única capa de salida con tantas neuronas como acciones; cada una estima $Q(s,a)$.
- *DQN en duelo* (p19-p21): tras el último bloque convolucional, la red se bifurca en dos flujos, cada uno con su propia capa densa (dos columnas rojas). El flujo superior acaba en **una sola neurona** que estima $V(s)$ (p19). El flujo inferior acaba en **tantas neuronas como acciones**, que estiman la ventaja $A(s,a)$ "de cada acción sobre las otras" (p20). Ambos flujos se unen (la "Y" verde) en una capa de salida con tantas neuronas como acciones que estima $Q(s,a)$ como función de $V(s)$ y $A(s,a)$ (p21).

**Combinación (DRL5 p22):**

$$Q(s,a)=V(s)+A(s,a)-\operatorname{media}_{a'}A(s,a')=V(s)+A(s,a)-\frac1{|\mathcal A|}\sum_{a'}A(s,a')$$

La idea de partida (p17): $Q$ "se calcula como la suma de dos términos: función de valor $V(s)$ y ventaja $A(s,a)$ de cada acción".

**Nota (por qué se resta la media).** Con $Q=V+A$ a secas no es identificable: se puede sumar una constante a $V$ y restarla a $A$ y $Q$ no cambia, así que la red no sabría repartir. Restar la media de las ventajas fija el reparto: $A$ tiene media cero sobre las acciones y $V$ queda como el nivel medio de $Q$ en ese estado. Wang et al. también proponen restar el máximo en vez de la media; la media da un entrenamiento más estable y es la que recoge el mazo. La ventaja es $A(s,a)=Q(s,a)-V(s)$, definición clásica de RL que DRL5 no desarrolla.

**Nota (por qué ayuda).** En muchos estados las acciones valen casi lo mismo y lo importante es lo bueno o malo que es el estado. En la arquitectura clásica solo se actualiza la salida de la acción tomada, $Q(s,a_t)$. En la de duelo, cada actualización también modifica $V(s)$, compartido por todas las acciones, de modo que se aprende el valor del estado sin haber probado todas las acciones en él. El mazo no desarrolla esta motivación; es la del artículo.

**Nota (compatibilidad).** El duelo solo cambia la **arquitectura**: la salida sigue siendo un vector con una $Q$ por acción, así que el resto del algoritmo (memoria de repetición, $\varepsilon$-greedy, pérdida, red *target*) no cambia, y se puede combinar con Doble DQN.

*Fuente: DRL5 p15-p22.*

---

## 4. Cómo se integran en un DQN (pseudocódigo)

**Nota:** el pseudocódigo es propio, no está en las fuentes; sigue las fórmulas de DRL5 p9-p13 y p22.

```
Hiperparámetros: γ, α, ε (con decaimiento), tamaño del buffer D, batch B, k (copia de la red target)
Inicializar red online Q_θ  (arquitectura DQN estándar o en duelo)
Inicializar red target Q_θ⁻ ← copia de Q_θ
Inicializar buffer D vacío; contador de pasos n ← 0

para cada episodio:
    s ← reset()
    mientras no termine el episodio:
        a ← ε-greedy sobre Q_θ(s, ·)
        s', r, done ← step(a)
        guardar (s, a, r, s', done) en D
        s ← s'; n ← n + 1

        si len(D) ≥ B:
            muestrear minibatch {(s_i, a_i, r_i, s'_i, d_i)} de D

            # --- cálculo del target ---
            DQN (con red target):  y_i = r_i + γ (1 - d_i) · max_a' Q_θ⁻(s'_i, a')
            Doble DQN:             a*_i = argmax_a' Q_θ(s'_i, a')               # selecciona la online
                                   y_i  = r_i + γ (1 - d_i) · Q_θ⁻(s'_i, a*_i)  # evalúa la target
            (y_i sin gradiente: se trata como constante)

            L(θ) = media_i ( y_i - Q_θ(s_i, a_i) )²     # o Huber
            actualizar θ con un paso de gradiente (Adam)

        si n mod k == 0:  θ⁻ ← θ        # copia dura (o θ⁻ ← τθ + (1-τ)θ⁻ en cada paso)
```

**Cabeza en duelo** (solo cambia la red):

```
tronco(s) → h                              # capas comunes (MLP en nuestro caso)
V = Linear(h → 1)                          # flujo de valor: 1 neurona
A = Linear(h → n_acciones)                 # flujo de ventaja: n_acciones neuronas
Q = V + A - A.mean(dim=1, keepdim=True)    # DRL5 p22
```

La línea de Doble DQN en PyTorch (orientativo):

```python
with torch.no_grad():
    a_star = q_online(s_next).argmax(dim=1, keepdim=True)        # selecciona: Q (online)
    q_next = q_target(s_next).gather(1, a_star).squeeze(1)       # evalúa: Q' (target)
    y = r + gamma * (1.0 - done) * q_next
q_sa = q_online(s).gather(1, a.unsqueeze(1)).squeeze(1)
loss = F.smooth_l1_loss(q_sa, y)
```

Para el DQN con red *target* estándar basta con `q_next = q_target(s_next).max(dim=1).values`. `smooth_l1_loss` es Huber con umbral 1 por defecto: con errores tan pequeños como los del taller equivale a MSE/2, así que el umbral hay que ajustarlo a la escala.

*Fuente: DRL5 p9-p13, p22 (fórmulas); pseudocódigo propio.*

---

## 5. Resumen comparado

| | DQN | Doble DQN | DQN en duelo |
|---|---|---|---|
| Qué cambia | (base) | El **cálculo del target** | La **arquitectura** de la red |
| Problema que ataca | | Sobreestimación del `max` e inestabilidad | Aprender el valor del estado sin probar todas las acciones |
| Redes | 1 (con *target* congelada en la versión de 2015) | 2: $Q$ elige, $Q'$ evalúa | 1 con dos flujos ($V$ y $A$) |
| Fórmula clave | $Y=r+\gamma\max_{a'}Q(s',a')$ | $Y=r+\gamma Q'(s',\arg\max_{a'}Q(s',a'))$ | $Q=V+A-\overline{A}$ |
| Coste extra | | Prácticamente nulo | Una cabeza más |
| Combinables | | Sí | Sí |

*Fuente: DRL5 p6-p14, p16-p22.*

---

## Aplicación al Taller 2

(**Nota:** todo este apartado es propuesta propia, salvo las fórmulas citadas.)

**Entorno:** `SimpleTradingEnv`, observación continua `[tendencia, posición]`, 3 acciones, comisión 0,1 %, recompensa del orden de $10^{-3}$ por paso, 494 decisiones por episodio, con los cuatro generadores de precios del enunciado.

Por qué tendrían sentido aquí (apartado de "ir más allá"):

1. **Ruido alto, señal baja.** Los precios sintéticos tienen una componente aleatoria importante, así que $Q$ es muy ruidosa y la diferencia entre acciones es pequeña. Es el escenario donde el `max` más sobreestima: la red "encuentra" una acción que parece ganar solo por ruido. Con Doble DQN, el valor de esa acción lo da otra red con ruido distinto. En trading, la sobreestimación se traduce en operar de más y pagar comisiones (0,1 % por operación) por ganancias que no existen.
2. **Pocas acciones (3).** La sobreestimación crece con el número de acciones candidatas, así que con 3 el efecto será menor que en juegos con muchas acciones; es esperable una mejora modesta de Doble DQN. Merece probarlo igualmente: cuesta una línea.
3. **Dueling.** En muchos pasos las tres acciones valen casi lo mismo (la posición ya está tomada y el precio no se mueve de forma explotable). Lo que la red debe aprender sobre todo es el valor del estado, $V(s)$ (p. ej. "estoy largo en una tendencia alcista"), y las ventajas son diferencias pequeñas. El duelo comparte ese aprendizaje de $V$ entre las tres acciones.
4. **Escala de recompensa.** Con recompensas de $\sim10^{-3}$, las diferencias entre acciones son del orden de la comisión y la red puede no distinguirlas. La recompensa se escala con la σ del retorno (D-1.1, cerrada); aun así, hay que vigilar que $Q$ no colapse a una constante.
5. **Estabilidad.** El *target* dinámico (T30 ~01:02:02) se mitiga con la red retardada: $k$ del orden de cientos de pasos o copia suave con $\tau\approx0{,}005$. El *clipping* es el truco que menciona el profesor (T30 ~02:35:22-02:35:47), pero a $\pm1$ no actúa aquí, porque $|r|$ no pasa de 0,51 ([notebook_partida.md](notebook_partida.md) §4.3), y Huber con $\delta=1$ se reduce a MSE: para limitar los extremos (mercado 1) hay que elegir el umbral según la escala.
6. **Experimento sugerido:** comparar DQN, Doble DQN, DQN en duelo y ambos a la vez, con varias semillas, en cada uno de los cuatro generadores. Medir el retorno y también la $Q$ media del estado inicial: si crece sin que crezca el retorno real, hay sobreestimación.
7. **Red pequeña:** la observación tiene solo 2 valores, así que basta un MLP de 1-2 capas ocultas; el duelo se aplica sobre la última capa oculta con las dos cabezas ($V$: 1 salida, $A$: 3 salidas).

Relación con otros documentos: DQN completo en [03_dqn.md](03_dqn.md); Q-learning y el origen del sesgo del `max` en [02_td_sarsa_qlearning.md](02_td_sarsa_qlearning.md).
