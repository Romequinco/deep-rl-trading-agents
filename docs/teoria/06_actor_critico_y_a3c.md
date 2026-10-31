# Métodos actor-crítico, A2C y A3C

> **Fuentes:** DRL7 (`docs/raw/DRL/transparencias/7-Métodos_actor_crítico.pdf`), pp. 1–16, leídas todas sus imágenes; final de DRL6 (`6-Policy_gradient.pdf`), pp. 28–39. Notebooks `3-cart_pole_actor_critic` y `Copia de 3-cart_pole_actor_critic` (`docs/raw/DRL/02-cart_pole/`). Transcripciones T24 (clase del 24 de octubre) y T30 (clase del 30 de octubre), con búsqueda de términos también en T23. Código del proyecto: `notebooks/common/a3c.py`, `env.py`, `markets.py`, `protocol.py`, y `docs/DECISIONES.md`.
>
> **Fuente externa:** Mnih et al. (2016), *Asynchronous Methods for Deep Reinforcement Learning*, ICML, [arXiv:1602.01783](https://arxiv.org/abs/1602.01783), leído entero en su versión v2, con el material suplementario. Se cita como "(Mnih §4)" o "(Mnih Alg. S3)". Para A2C: la entrada de OpenAI [*Baselines: ACKTR & A2C*](https://openai.com/index/openai-baselines-acktr-a2c) y la [documentación de A2C de Stable-Baselines](https://stable-baselines.readthedocs.io/en/master/modules/a2c.html). Para procesos: las [notas de multiprocesamiento de PyTorch](https://docs.pytorch.org/docs/2.15/notes/multiprocessing.html).
>
> **Convenciones:** "(DRL7 p15)" = página del mazo; "(T30 ~02:34)" = hora del cue. `**Nota:**` = lo que no viene del material de clase, o una errata detectada. **La clase en la que se iba a explicar este tema no está transcrita**: de las transparencias solo queda lo escrito, y la explicación de A2C y A3C (§5 y §6) es entera **Nota**.

---

## 1. Tres cosas que no hay que confundir

**Idea.** Bajo el nombre "actor-crítico" el material reúne tres algoritmos distintos. Las fórmulas de uno no sirven para programar otro.

| | Transparencias (§3) | Notebook de clase (§4) | A3C (§5, **Nota**) |
|---|---|---|---|
| Qué hay en el material | Pseudocódigo completo (DRL7 p15) | Código Keras ejecutado | Solo el nombre (DRL7 p16) |
| Señal que multiplica a $\nabla\ln\pi$ | Error TD de un paso $\delta$ | $G_t$ normalizado $-\,V$ | Retorno de hasta $t_{max}$ pasos $-\,V$ |
| Objetivo del crítico | $R+\gamma v(S')$, con *bootstrapping* | $G_t$ normalizado, sin *bootstrapping* | Retorno de $n$ pasos, con *bootstrapping* |
| Cuándo actualiza | En cada paso | Al final del episodio | Cada $t_{max}$ pasos o al terminar |
| Pérdida del crítico | Cuadrática (implícita) | Huber | Cuadrática |
| Factor $I=\gamma^t$ | Sí | No | No |
| Modelos | Dos, con parámetros propios | Una red con capa compartida | Comparten capas en la práctica |
| Exploración | Muestrear $\pi$ | Muestrear $\pi$ | Muestrear $\pi$ + bonus de entropía |
| Paralelismo | No | No | Varios *workers* asíncronos |

En el Taller 2 se pide el tercero. El segundo solo sirve de base para la red; su bucle de entrenamiento **no** se puede traducir tal cual (§4).

---

## 2. De REINFORCE al actor-crítico

**Idea.** El actor-crítico parte de REINFORCE: mantiene el gradiente de la política, pero sustituye el retorno observado por una estimación con ayuda de un segundo modelo que aprende el valor.

**REINFORCE**, tal como se recuerda en DRL7 p3 (repite DRL6 p38; desarrollo completo en [`05_policy_gradient.md`](05_policy_gradient.md), §6). Por cada episodio generado con $\pi(a|s,\theta)$, para $t=0,\dots,T-1$:

$$G_t=\sum_{k=t+1}^{T}\gamma^{k-t-1}R_k\qquad\qquad\Delta\theta=\alpha\,\gamma^t\,G_t\,\nabla\ln\pi(A_t|S_t,\theta)$$

con $\nabla\ln\pi(A_t|S_t,\theta)=\dfrac{1}{\pi(A_t|S_t,\theta)}\nabla\pi(A_t|S_t,\theta)$ (DRL6 p39). Se generan episodios completos con la política actual (DRL6 p30). Qué defectos tiene y por qué el crítico los mitiga está en `05`, §7.

**Nota (errata):** DRL7 p2–p3 dice "$\pi(a|s,\theta)$ debe ser derivable respecto a." y la frase queda cortada; DRL6 p38 da la versión correcta: "respecto a $\theta$".

**La idea del método** (DRL7 p4–p5): usar dos modelos, uno para la política $\pi$ (**actor**) y otro para el valor $G$ (**crítico**). Pueden ser de tipos distintos, por ejemplo una red neuronal y un modelo lineal, cada uno con sus propios parámetros.

**Nota:** "el valor $G$" hay que entenderlo como la esperanza de $G$: el crítico aprende $v(s)\approx\mathbb{E}[G_t|S_t=s]$, como confirma DRL7 p6 y p15.

**Lo que dijo el profesor sobre el tema**, siempre de pasada:

- **Tres familias.** Unos métodos estiman la función de valor y de ahí sacan una política; otros buscan directamente la política sin ocuparse del valor; y los actor-crítico combinan ambas cosas. El actor realiza acciones sin preocuparse del valor, y el crítico las supervisa diciendo si lo ha hecho bien o mal. Suelen funcionar un poco mejor, porque combinan elementos de ambos (T24 ~04:40–04:41).
- **Dónde están.** En el mapa de algoritmos, los actor-crítico (A2C y A3C) van en la rama de optimización de políticas; DQN, en la de Q-learning (T30 ~01:46). La transcripción dice "a 12 a 13", que se interpreta como A2C y A3C. Ver [`07_panorama_drl.md`](07_panorama_drl.md).
- **Dos redes o dos salidas.** A la pregunta de un alumno sobre el uso de una segunda red, se responde que hay una red que propone las acciones y otra que la critica o modula: "en principio son 2 redes o 2 capas de salida", con muchísimas variantes (T30 ~02:34). Las dos opciones son válidas; el notebook usa la segunda.
- **Para qué.** Tanto los actor-crítico como las dos redes de DQN son intentos de que el entrenamiento no se inestabilice. Estos algoritmos deberían poder resolver cualquier problema: la dificultad está en la estabilidad y en cuánto se tarda (T30 ~02:34–02:35).
- **La escala importa.** Unas recompensas demasiado bajas, o con diferencias muy grandes, también afectan a la estabilidad; DeepMind las recorta (*clip*). Hay muchísimos trucos de este tipo (T30 ~02:35). Ver la Aplicación, más abajo.
- **Sin librería de referencia.** No hay una librería estándar con estos algoritmos en Keras; la que le gustaba al profesor dejó de mantenerse (T30 ~02:36).
- **Uso en finanzas.** Son métodos más avanzados, y se usan en investigación sobre RL en finanzas, donde el profesor advierte de los sesgos de los datos reales, en especial el de supervivencia (T24 ~04:40–04:42). No afecta al taller, cuyos datos son sintéticos.

**Diagrama oculto.** DRL7 p4–p5 contiene una imagen de un micrófono, un altavoz y un amplificador, con el rótulo "El loop de feedback provoca pitidos". En el PDF está **tapada por un rectángulo blanco**, así que no se ve en la página, aunque su texto sí está en el archivo. **Nota:** el profesor no llega a comentarla en lo disponible. La lectura razonable es esta: actor y crítico forman un bucle de realimentación, porque cada uno aprende de lo que produce el otro (el crítico valora la política actual y el actor se mueve según esa valoración), y un bucle así puede amplificar errores en vez de corregirlos.

*Fuente: DRL6 p28–p39; DRL7 p2–p5; T24 ~04:40–04:42; T30 ~01:46, ~02:33–02:36.*

---

## 3. Lo que dan las transparencias: actor-crítico de un paso

**Idea.** El actor-crítico de clase actualiza en cada paso: en vez de esperar a $G_t$, usa el error de Bellman $\delta$ del crítico, tanto para corregir al propio crítico como para decirle al actor si la acción salió mejor o peor de lo esperado.

**Entrada:** política parametrizada derivable $\pi(a|s,\theta)$ y función de valoración derivable $v(s,\mathbf{w})$. **Hiperparámetros:** $\alpha_\theta$, $\alpha_{\mathbf{w}}$ (tamaños de paso) y factor de descuento $\gamma$ (DRL7 p6).

**Pseudocódigo** (DRL7 p15):

```text
Inicializar parámetros θ y w (por ejemplo, a 0)
Por cada episodio:
    Inicializar S (primer estado del episodio). Inicializar I ← 1
    Mientras que S no sea terminal:
        Generar A de acuerdo a π, realizar A y observar S', R
        Calcular error en v de acuerdo a Bellman:
            δ = R + γ·v(S',w) − v(S,w)         (v es 0 si S' es terminal)
        Actualizar w:  Δw = α_w · δ · ∇v(S,w)
        Actualizar θ:  Δθ = α_θ · I · δ · ∇ln π(A|S,θ)
        I ← γ·I,   S ← S'
```

Las tres fórmulas:

$$\delta=R+\gamma\,v(S',\mathbf{w})-v(S,\mathbf{w})\qquad\Delta\mathbf{w}=\alpha_{\mathbf{w}}\,\delta\,\nabla v(S,\mathbf{w})\qquad\Delta\theta=\alpha_\theta\,I\,\delta\,\nabla\ln\pi(A|S,\theta)$$

Las páginas 7–14 son fotogramas de la animación que construye la p15: sin contenido adicional.

**Variantes.** "Este esquema tiene muchas variantes, por ejemplo A2C (*Advantage Actor-Critic*) y A3C (*Asynchronous Advantage Actor-Critic*)" (DRL7 p16). Es todo lo que dice el mazo sobre ellos.

**Nota: cómo se lee cada pieza.** Este algoritmo es el *One-step Actor-Critic (episodic)* de Sutton & Barto (2018), §13.5.

- **$\delta$ como crítica.** Si $\delta>0$, lo ocurrido ha sido mejor de lo que el crítico esperaba en $S$: $\Delta\theta$ sube la probabilidad de $A$. Si $\delta<0$, la baja. Es el error TD de [`02_td_sarsa_qlearning.md`](02_td_sarsa_qlearning.md), §2.
- **$\delta$ como ventaja.** Con el valor exacto, $\mathbb{E}[\delta\,|\,S=s,A=a]=Q^\pi(s,a)-V^\pi(s)$. Por eso $\delta$ es una muestra de la **ventaja** de la acción, y este es el puente con A2C y A3C (§5).
- **Por qué sustituye a $G_t$.** $G_t-v(S_t)$ también estima la ventaja, pero acumula el azar de todo el episodio. $\delta$ solo depende de una transición: menos varianza, pero sesgo mientras $v$ sea imperfecto.
- **$\Delta\mathbf{w}$ es un semi-gradiente.** Como en TD, el objetivo $R+\gamma v(S')$ se trata como constante: solo se deriva $v(S,\mathbf{w})$. En un framework automático, hay que cortar el gradiente del objetivo.
- **$I$ no es decorativo.** Es $\gamma^t$ acumulado paso a paso, el mismo factor de $\Delta\theta$ en REINFORCE (DRL7 p3). Pondera menos las decisiones tardías del episodio. Ni el notebook (§4) ni A3C (§5) lo usan. Ver `05`, Aplicación.
- **"Inicializar a 0" no vale para redes.** Es solo un ejemplo de la transparencia: sirve para modelos lineales, pero una red con todos los pesos a cero no rompe la simetría entre neuronas, y con ReLU ni siquiera hay gradiente. En una red se usa la inicialización aleatoria habitual.

*Fuente: DRL7 p6–p16.*

---

## 4. Lo que hace el notebook: no es el algoritmo de la transparencia

**Idea.** `3-cart_pole_actor_critic` es un ejemplo de Keras sobre CartPole-v0 (celdas 0–1) que se llama actor-crítico, pero actualiza **una vez por episodio** con **retornos Monte Carlo normalizados**. Por su mecánica está mucho más cerca de REINFORCE con línea base que de DRL7 p15.

**Configuración y red** (celdas 3–6): `gamma=0.99`, `seed=42`, `max_steps_per_episode=10000`. Entrada de 4 valores, una capa densa común de 128 neuronas ReLU, y dos salidas sobre esa capa: `action` (softmax de 2) y `critic` (lineal de 1). En total, 1 027 parámetros (640 + 258 + 129). La celda 4 lo indica: "comparten la capa inicial". La celda 7 solo dibuja el modelo. La celda 2 es un título.

**Bucle de entrenamiento, reducido a lo esencial** (celda 9). Optimizador `Adam(0.01)`, pérdida `Huber()`:

```python
with tf.GradientTape() as tape:
    for timestep in range(1, max_steps_per_episode):        # 1) juega el episodio entero
        action_probs, critic_value = model(state)
        action = np.random.choice(num_actions, p=np.squeeze(action_probs))
        action_probs_history.append(ops.log(action_probs[0, action]))
        critic_value_history.append(critic_value[0, 0])
        state, reward, done, *_ = env.step(action)
        rewards_history.append(reward)
        if done: break
    returns = []; discounted_sum = 0                        # 2) G_t hacia atrás
    for r in rewards_history[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    returns = (returns - np.mean(returns)) / (np.std(returns) + eps)   # 3) normaliza
    for log_prob, value, ret in zip(...):                   # 4) pérdidas
        diff = ret - value
        actor_losses.append(-log_prob * diff)
        critic_losses.append(huber_loss(ops.expand_dims(value, 0), ops.expand_dims(ret, 0)))
    loss_value = sum(actor_losses) + sum(critic_losses)     # 5) una sola actualización
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

En fórmulas, con $\tilde G_t=(G_t-\bar G)/(\sigma_G+\varepsilon)$ calculado dentro del episodio:

$$L=\sum_t\Big[-\ln\pi(A_t|S_t)\,\big(\tilde G_t-V(S_t)\big)+\text{Huber}\big(V(S_t),\tilde G_t\big)\Big]$$

**Diferencias con la transparencia:**

| Aspecto | DRL7 p15 | Notebook (celda 9) |
|---|---|---|
| Momento de actualizar | Cada paso | Una vez, al acabar el episodio |
| Objetivo del crítico | $R+\gamma v(S')$ | $\tilde G_t$: Monte Carlo normalizado |
| *Bootstrapping* | Sí | **No** |
| Señal del actor | $\delta$ | $\tilde G_t-V(S_t)$ |
| Factor $I$ | Sí | No |
| Pérdida del crítico | Cuadrática (da $\Delta\mathbf{w}=\alpha\delta\nabla v$) | Huber |
| Parámetros | Separados, $\alpha_\theta$ y $\alpha_{\mathbf{w}}$ | Capa compartida, un solo Adam |

**Nota: qué consecuencias tiene.**

- **Según Sutton & Barto no sería actor-crítico.** Su §13.5 reserva el nombre para cuando el valor se usa para hacer *bootstrapping*; si solo es línea base, lo llaman REINFORCE con línea base. Mnih et al. usan una definición más amplia, que sí admite este caso (Mnih §3). Las dos están en uso: lo importante es saber cuál se está programando.
- **El crítico no aprende $v(s)$.** Aprende el retorno *normalizado dentro de cada episodio*, cuya escala cambia con cada episodio. Valores de episodios distintos no son comparables.
- **La normalización borra el signo global.** Por construcción, $\tilde G_t$ tiene media 0 en cada episodio: en un episodio malo, la mitad de las acciones se refuerza igualmente. En CartPole no importa mucho, porque la recompensa siempre es positiva y lo relevante es durar. **En trading sí importa**, pues la decisión central es si operar compensa; un episodio que pierde dinero tiene que poder castigar. Además, no se puede aplicar a un tramo de $t_{max}$ pasos: es otra razón para no copiarla.
- **Falta cortar el gradiente de `diff`.** `diff = ret - value` está dentro de la cinta sin `stop_gradient`, de modo que la pérdida del actor también entrena al crítico, con un término $+\ln\pi\,\nabla V$ que no tiene justificación teórica. En el Taller 2, en PyTorch, usad `advantage.detach()`.
- **Argumentos de Huber invertidos.** La firma es `huber_loss(y_true, y_pred)` y se pasa `(value, ret)`. No hay efecto, pues la función es simétrica.
- **Suma, no media.** La pérdida suma todos los pasos, así que la magnitud del gradiente crece con la duración del episodio: el mismo $\alpha$ es efectivamente mayor en los largos, aunque Adam lo compensa en parte.

**El test** (celdas 11–14): carga la política en CartPole, **muestrea** de ella (no toma la más probable) y graba un vídeo. Dura 200 pasos en ambos notebooks, el máximo de CartPole-v0. La celda 10 guarda el modelo; la 15 está vacía.

**Diferencias entre los dos notebooks.** El código solo cambia en una línea de la celda 9; el resto son rutas de avisos y resultados.

| | `3-cart_pole_actor_critic` | `Copia de 3-cart_pole_actor_critic` |
|---|---|---|
| Criterio de "resuelto" | `running_reward > 199` | `running_reward >= 195` |
| Resuelto en el episodio | 502 | 317 |
| Mejor media en los primeros 100 episodios | ≈10 | ≈137 |

con `running_reward = 0.05 * episode_reward + 0.95 * running_reward`.

**Nota: no concluyáis que uno aprende más rápido.** La diferencia no se debe sobre todo al criterio:

- El primer notebook alcanza 195 hacia el episodio **470** (195,02), así que, con el criterio de la copia, habría terminado ahí. De 470 a 317 es **variabilidad entre ejecuciones**: ya difieren en el episodio 10 (5,71 frente a 12,00). Solo se fija la semilla del entorno (`env.reset(seed=seed)`), no la de TensorFlow ni la de NumPy.
- Lo que sí aporta el criterio son unos 32 episodios más: como el máximo es 200, pasar de 195 a 199 exige $5\cdot0{,}95^n<1$, es decir, $n\approx32$ episodios perfectos. Y es lo que pasa: de 470 a 502.

**Nota: las curvas muestran dos problemas que aparecerán en trading.**

- **Meseta inicial.** En el primer notebook, la media se queda en torno a 10 durante los episodios 40–160. Es lo que dura CartPole con una política casi constante, luego es compatible con un **colapso temprano a una acción**, del que tarda en salir. No se registraron las probabilidades, así que es una inferencia. Además, no hay bonus de entropía que lo combata.
- **Inestabilidad.** Las medias retroceden mucho antes de converger: de 113,75 a unos 45 en el primero (episodios 220–270) y de 143,04 a unos 90 en la copia (episodios 180–230). Es el problema que el profesor señala como característico (T30 ~02:35).

No hay figuras guardadas en estos notebooks; el gráfico del modelo de la celda 7 no se conserva.

*Fuente: `3-cart_pole_actor_critic` y `Copia de 3-cart_pole_actor_critic`, celdas 0–15.*

---

## 5. A3C: *Asynchronous Advantage Actor-Critic*

**Nota: toda esta sección viene de Mnih et al. (2016), no de la clase.**

**Idea.** A3C combina tres ingredientes: una **ventaja** estimada con retornos de hasta $t_{max}$ pasos, un **bonus de entropía** para evitar colapsar, y **varios *workers* en paralelo** que aplican sus gradientes de forma asíncrona sobre unos pesos compartidos. Esto último ocupa el lugar que la memoria de experiencias tiene en DQN.

### 5.1 Ventaja y retorno de $n$ pasos

La ventaja de una acción es cuánto mejor es que lo esperable en ese estado, $A(s,a)=Q(s,a)-V(s)$. Restar una línea base al retorno reduce la varianza del gradiente sin sesgarlo; y como $R_t$ estima $Q$ y la línea base estima $V$, la diferencia estima la ventaja. Vista así, es una arquitectura actor-crítico donde la política es el actor y la línea base el crítico (Mnih §3).

A3C estima la ventaja con (Mnih §4):

$$A(s_t,a_t;\theta,\theta_v)=\sum_{i=0}^{k-1}\gamma^i r_{t+i}+\gamma^k V(s_{t+k};\theta_v)-V(s_t;\theta_v),\qquad k\le t_{max}$$

**$k$ cambia de un estado a otro.** Cada tramo usa el retorno más largo posible hasta el final del tramo: el último estado hace una actualización de un paso, el penúltimo de dos, y así hasta $t_{max}$ (Mnih §4). Por ejemplo, con $t_{max}=3$ y un tramo que no termina, $V_3=V(s_3)$:

$$R_2=r_2+\gamma V_3\qquad R_1=r_1+\gamma r_2+\gamma^2V_3\qquad R_0=r_0+\gamma r_1+\gamma^2 r_2+\gamma^3V_3$$

Se calculan hacia atrás con $R\leftarrow r_i+\gamma R$. Si el estado final es terminal, se parte de $R=0$.

**Por qué $n$ pasos.** En los métodos de un paso, una recompensa solo afecta directamente al par que la produjo, y a los anteriores de forma indirecta, lo que hace lento el aprendizaje. Con $n$ pasos afecta directamente a los $n$ anteriores (Mnih §3).

**Relación con los otros dos algoritmos** (**Nota** propia): con $k=1$, la ventaja es exactamente el $\delta$ de DRL7 p15; cuando $k$ llega al final del episodio, es $G_t-V(s_t)$, como en el notebook pero sin normalizar. $t_{max}$ es un mando continuo entre el sesgo del *bootstrapping* y la varianza de Monte Carlo.

### 5.2 Objetivo del actor, entropía y crítico

**Actor.** El gradiente con entropía, que se **asciende** (Mnih §4):

$$\nabla_{\theta'}\ln\pi(a_t|s_t;\theta')\,\big(R_t-V(s_t;\theta_v)\big)+\beta\,\nabla_{\theta'}H\big(\pi(s_t;\theta')\big)\qquad\text{con}\quad H(\pi(s))=-\sum_a\pi(a|s)\ln\pi(a|s)$$

**Por qué la entropía.** Añadirla "mejora la exploración al desincentivar la convergencia prematura a políticas deterministas subóptimas" (Mnih §4). $\beta$ regula su fuerza: **$\beta=0{,}01$** en Atari y TORCS (Mnih supl. §8), y $10^{-4}$ en control continuo, sobre la entropía diferencial (Mnih supl. §9).

**Crítico.** Minimiza el error cuadrático $(R_i-V(s_i;\theta_v'))^2$ (Mnih Alg. S3).

**Pérdida conjunta** (**Nota**, forma de implementarlo). El artículo acumula por separado $d\theta$ (ascenso) y $d\theta_v$ (descenso), y **no da un coeficiente** para repartir ambos en las capas compartidas. Un framework minimiza una sola cantidad, por lo que se escribe, por tramo:

$$L=\sum_i\Big[-\ln\pi(a_i|s_i)\,\underbrace{\big(R_i-V(s_i)\big)}_{\text{sin gradiente}}\;+\;c_v\big(R_i-V(s_i)\big)^2\;-\;\beta\,H(\pi(s_i))\Big]$$

$c_v$ no es del artículo: en Stable-Baselines, `vf_coef=0.25` por defecto. Con $c_v=0{,}5$, el gradiente del crítico queda $(R-V)\nabla V$, con la misma escala que el del actor.

### 5.3 Pseudocódigo de cada *worker* (Mnih Alg. S3)

```text
// θ y θ_v: vectores de parámetros globales y compartidos. T: contador global compartido, T = 0
// θ' y θ'_v: copias propias de este worker
Inicializar contador del worker t ← 1
repetir
    Reiniciar gradientes: dθ ← 0, dθ_v ← 0
    Sincronizar: θ' = θ, θ'_v = θ_v
    t_start = t
    Obtener estado s_t
    repetir
        Ejecutar a_t según π(a_t|s_t; θ')
        Recibir recompensa r_t y nuevo estado s_{t+1}
        t ← t + 1;  T ← T + 1
    hasta s_t terminal  o  t − t_start == t_max
    R = 0                si s_t es terminal
    R = V(s_t; θ'_v)     si no                     // bootstrap desde el último estado
    para i ∈ {t−1, ..., t_start}:
        R ← r_i + γR
        dθ   ← dθ   + ∇_θ' ln π(a_i|s_i; θ') · (R − V(s_i; θ'_v))
        dθ_v ← dθ_v + ∂(R − V(s_i; θ'_v))² / ∂θ'_v
    Actualización asíncrona de θ con dθ y de θ_v con dθ_v
hasta T > T_max
```

El pseudocódigo del artículo **no incluye la entropía**, que solo aparece en el texto (Mnih §4): hay que sumar $\beta\nabla_{\theta'}H$ a $d\theta$.

### 5.4 *Workers*, gradientes asíncronos y optimizador

- **Qué es asíncrono.** Cada *worker* tiene su copia del entorno y su copia de los pesos. Juega, calcula gradientes con *su copia* y los aplica sobre los pesos globales *sin esperar a nadie y sin cerrojos* (Mnih §4 y supl. §7, al estilo Hogwild!). **Nota:** un gradiente puede aplicarse sobre pesos que otro *worker* ya ha movido; se acepta esa imprecisión a cambio de velocidad.
- **Acumular antes de aplicar.** Sumar gradientes durante varios pasos es parecido a usar minibatches, y reduce la probabilidad de que los *workers* se pisen las actualizaciones (Mnih §4).
- **Red compartida.** Aunque $\theta$ y $\theta_v$ se escriben por separado, "siempre se comparte parte de los parámetros en la práctica": una salida softmax y una lineal, con todas las capas no finales compartidas (Mnih §4). En control continuo sí las separan, aunque dicen que probablemente no sea crucial (Mnih supl. §9).
- **Optimizador compartido.** Comparan SGD con momento, RMSProp por *worker* y RMSProp compartido, cuyo acumulador $g$ también es común a todos y se actualiza sin cerrojos:

  $$g=\alpha g+(1-\alpha)\Delta\theta^2\qquad\theta\leftarrow\theta-\eta\frac{\Delta\theta}{\sqrt{g+\epsilon}}$$

  **El compartido es considerablemente más robusto**, seguido del RMSProp por *worker* y del SGD con momento (Mnih §4 y supl. §7).

### 5.5 Por qué sustituye a la memoria de experiencias

Los datos de un agente en línea son no estacionarios y sus actualizaciones están muy correladas. La memoria de experiencias lo resuelve ([`03_dqn.md`](03_dqn.md)), pero gasta más memoria y cómputo por interacción y **obliga a usar algoritmos *off-policy***, capaces de aprender de datos de una política antigua (Mnih §1).

A3C elige otra vía: en cualquier momento, los *workers* están viendo estados distintos, lo que descorrela los datos. El conjunto de actualizaciones está menos correlado en el tiempo que el de un único agente, y se prescinde de la memoria (Mnih §1, §4). Esto permite usar métodos *on-policy* como el actor-crítico, que **no podría** aprovechar una memoria así sin correcciones, porque necesita datos de la política actual. **Advertencia del artículo:** esto no significa que la memoria de experiencias no sea útil; reutilizar datos puede mejorar mucho la eficiencia cuando el entorno es caro (Mnih §6).

**Nota: la consecuencia para el taller.** El mecanismo depende de que los *workers* vean cosas diferentes. Con pocos *workers* (aquí 3, no 16), la descorrelación será menor que en el artículo, y el hecho de que cada episodio sea una serie distinta cobra importancia.

### 5.6 Configuración y resultados del artículo

| Parámetro | Valor (Atari/TORCS, Mnih supl. §8) |
|---|---|
| *Workers* | 16 hilos de CPU en una máquina, sin GPU |
| $t_{max}$ | 5 |
| Optimizador | RMSProp compartido, $\alpha=0{,}99$ |
| $\gamma$ | 0,99 |
| $\beta$ | 0,01 |
| Tasa de aprendizaje | Inicial muestreada de LogUniform($10^{-4}$, $10^{-2}$), con decaimiento a 0 |
| Recorte de gradiente | Por norma; ajustado en una búsqueda (Mnih §5.1), sin dar el valor |

| Resultados en 57 juegos de Atari (Mnih Tabla 1) | Entrenamiento | Media | Mediana |
|---|---|---|---|
| DQN | 8 días con GPU | 121,9 % | 47,5 % |
| Prioritized DQN | 8 días con GPU | 463,6 % | 127,6 % |
| A3C, *feedforward* | 1 día con CPU | 344,1 % | 68,2 % |
| A3C, *feedforward* | 4 días con CPU | 496,8 % | 116,6 % |
| A3C, LSTM | 4 días con CPU | 623,0 % | 112,6 % |

(Puntuación normalizada respecto a la humana.)

- **Escalado** (Mnih Tabla 2). Aceleración media de A3C con 1, 2, 4, 8 y 16 *workers*: 1,0; 2,1; 3,7; 6,9 y 12,5. Es sublineal, a diferencia de los métodos de un paso.
- **Robustez** (Mnih §5.6). En un rango amplio de tasas de aprendizaje, todas las inicializaciones dan buen resultado, y virtualmente no hay colapsos una vez empieza a aprender. Es una observación sobre Atari: no prueba nada para este entorno.

*Fuente: Mnih et al. (2016) §1, §3–§6, supl. §7–§9, Alg. S3.*

---

## 6. A2C: la versión síncrona

**Nota: la clase solo da el nombre (DRL7 p16), y el artículo de A3C no menciona A2C.**

**Idea.** A2C es el mismo algoritmo sin la asincronía: un coordinador espera a que todos los *workers* terminen su tramo, promedia, y hace una sola actualización de los pesos.

- OpenAI la describe como una variante "síncrona y determinista" de A3C "que da el mismo rendimiento", y añade que no observó que el ruido de la asincronía beneficiara ([OpenAI Baselines](https://openai.com/index/openai-baselines-acktr-a2c)). Espera a que cada actor acabe su tramo antes de actualizar, promediando sobre todos, lo que aprovecha mejor las GPU gracias a lotes mayores. Esta fuente no se pudo abrir directamente (la web devolvió un error 403): las citas proceden de los extractos de un buscador y de la [entrada de A2C](https://ml.paperswithcode.com/method/a2c) de Papers with Code, también vista solo como extracto.
- Stable-Baselines: "variante síncrona y determinista de A3C. Usa varios *workers* para evitar la memoria de experiencias". Valores por defecto: `n_steps=5`, `vf_coef=0.25`, `ent_coef=0.01`, `max_grad_norm=0.5`, `learning_rate=0.0007`, `alpha=0.99` (de RMSProp) ([documentación](https://stable-baselines.readthedocs.io/en/master/modules/a2c.html)).

| | A3C | A2C |
|---|---|---|
| Aplicación del gradiente | Cada *worker*, cuando termina | Todas a la vez, tras esperar |
| Pesos con los que se generó el gradiente | Pueden estar desfasados | Los actuales |
| Reproducibilidad | No bit a bit | Determinista, fijadas las semillas |
| Cómputo | Multiproceso con memoria compartida | Puede simularse en un solo proceso, por lotes |

**Nota: síncrono no significa sin paralelismo.** A2C sigue necesitando varios actores para descorrelacionar. Con un único actor no se sustituiría la memoria de experiencias.

*Fuente: DRL7 p16; OpenAI Baselines; Stable-Baselines.*

---

## Aplicación al Taller 2

**Idea.** El enunciado pide A3C: programad §5, no la celda 9 del notebook. Salvo lo citado de `a3c.py` y `DECISIONES.md`, todo este apartado es **Nota**: propuestas y cálculos propios, porque ninguna fuente de clase trata A3C en este entorno. D-2.1 sigue **ABIERTA**: lo que sigue la informa, pero no la cierra.

### Lo que ya está decidido o abierto en el proyecto

- **D-2.1.** En clase, A3C solo aparece como variante, y el código disponible es un actor-crítico en CartPole. Las alternativas abiertas son A3C real con procesos asíncronos (no reproducible bit a bit, con tiempo sensible a la CPU) o A2C síncrono como comparación adicional. **Propuesta:** A3C real como entrega, pues es lo que se pide, y A2C solo como "prueba más allá de los mínimos", si queda tiempo.
- **D-0.2.** PyTorch en CPU, con `share_memory` y `spawn`, probado con 3 procesos. El arranque cuesta unos 19 s, casi todo imports: lanzad los procesos una vez por entrenamiento.
- **`a3c.py`.** `A3CConfig`: `gamma=0.99`, `lr=0.01`, `hidden=128`, `n_workers=3` (deja un núcleo libre de 4), `t_max=20`, `entropy_coef=0.01`, `n_episodes=500`. El propio archivo aclara que `gamma`, `lr` y `hidden` vienen del notebook, y que `t_max` y `entropy_coef` "son solo un punto de partida". Interfaz: `A3CAgent.act(obs, greedy)`, `worker(rank, shared_model, optimizer, cfg, market, seed, n_steps, log)`, `train_a3c(cfg, market, seed, n_steps)`.
- **D-1.1 está cerrada:** observación y recompensa se escalan con `train_scales(mercado)`, y falta confirmarlo en A3C.
- **Otras decisiones cerradas que afectan a A3C.** D-1.2 (fin de episodio: el último tramo parte de $R=0$, el `R = 0 si terminal` del pseudocódigo) y D-0.3 (convergencia medida en la curva de la política *greedy* en validación, también con episodios en paralelo).

**Tres puntos de `A3CConfig` que conviene revisar:**

1. **`lr=0.01` viene de otro contexto.** En el notebook se daba **una** actualización por episodio. Con `t_max=20` hay unas $494/20\approx25$ actualizaciones por episodio y *worker*. Además, $10^{-2}$ es el extremo superior del rango del artículo. Probad $\{10^{-4},3\cdot10^{-4},10^{-3},3\cdot10^{-3}\}$.
2. **`n_episodes` es ambiguo.** ¿Total o por *worker*? Para comparar con DQN, que sea el total, y contad pasos de entorno: 500 episodios son $\approx$247 000 pasos.
3. **`t_max=20` frente a 5 del artículo.** Probad $\{5,20,50\}$. El coste de comprar se paga en el instante y el beneficio llega después: con $t_{max}$ pequeño, la decisión se apoya casi por completo en un crítico que todavía aprende.

### De $\varepsilon$ a entropía

El punto 3 del enunciado pide ajustar "la tasa de exploración $\epsilon$ (probablemente con decaimiento temporal)". A3C no usa $\varepsilon$, porque explora al muestrear de $\pi$ (D-2.1). **El ajuste se traslada a $\beta$** y hay que justificarlo en el informe:

| DQN | A3C |
|---|---|
| $\varepsilon$ fija la exploración desde fuera | $\beta$ premia la exploración dentro del objetivo |
| Decaimiento de $\varepsilon$ | Decaimiento de $\beta$ opcional (el artículo la mantiene constante) |
| Evaluar con $\varepsilon=0$ | Evaluar con `greedy=True` (`evaluate()` ya lo hace por defecto) |

**$\beta=0{,}01$ no es un valor neutro aquí, porque se compara con la recompensa.** Si se maximiza $\sum_a\pi(a)\bar A(a)+\beta H(\pi)$ en un solo estado, la solución es $\pi(a)\propto\exp(\bar A(a)/\beta)$. Con una ventaja media de $\bar A\approx10^{-3}$ (una comisión) y $\beta=0{,}01$, la razón de probabilidades es $e^{0,1}\approx1{,}1$: **la política apenas se separa de la uniforme**, aunque haya aprendido bien. Para una razón clara, como $e^3\approx20$, hace falta $\beta\approx3\cdot10^{-4}$ o, de forma equivalente, escalar la recompensa por un factor $c$, con el que $\beta$ pesa como $\beta/c$. Es un cálculo idealizado, y por tanto una orientación: probad $\beta$ en escala logarítmica $\{0,10^{-4},10^{-3},10^{-2}\}$, **siempre junto con la decisión de D-1.1**.

### La escala de la recompensa

Medido en las 100 series de validación (semillas 50 000–50 099), retorno por paso del activo y correlación de la tendencia observada con el retorno siguiente, media por serie:

| Mercado | Media | Desv. típica | Tendencia p5 / p95 | corr(tendencia, retorno sig.) |
|---|---|---|---|---|
| 1 `tendencia_ciclo` | +0,0127 | 0,107 | −0,227 / +0,279 | +0,19 |
| 2 `browniano` | +0,00014 | 0,020 | −0,041 / +0,041 | −0,00 |
| 3 `reversion_media` | +0,00022 | 0,020 | −0,040 / +0,041 | −0,05 |
| 4 `regimenes` | −0,00031 | 0,019 | −0,044 / +0,036 | +0,00 |

**El "≈$10^{-3}$" describe la señal, no la recompensa.** Con posición abierta, cada paso trae un ruido de $\approx2\cdot10^{-2}$ en los mercados 2–4 y de $\approx0{,}1$ en el 1. La comisión y la media están en $10^{-3}$–$10^{-4}$ (salvo el mercado 1).

- **Señal y ruido.** En un tramo de 20 pasos con posición abierta, el ruido del retorno es $\approx0{,}02\sqrt{20}\approx0{,}09$, unas 90 veces la comisión. Distinguir $10^{-3}$ con esa dispersión requiere del orden de $(0{,}09/10^{-3})^2\approx8\,000$ muestras. Un presupuesto de 500 episodios no es holgado.
- **Valor.** Con $\gamma=0{,}99$, el retorno descontado con posición abierta tiene una dispersión de $\approx\sigma/\sqrt{1-\gamma^2}$: unos 0,14 en los mercados 2–4 y unos 0,76 en el 1. Por tanto, la cota $|G_t|\lesssim0{,}1$ que da `05`, Aplicación, se aplica a la señal, no a las muestras.
- **Escala distinta por mercado.** Un 0,01 de $\beta$ o un mismo `lr` no significan lo mismo en el mercado 1 que en el resto. Conviene escalar la recompensa con una constante fijada por mercado a partir de train (D-1.1), y no por episodio ni con la normalización del notebook (§4), que borra el signo global y no se puede hacer por tramos.
- **Observación.** La tendencia ronda $\pm0{,}04$ y la posición vale 0 o 1. Sin normalizar, la red apenas ve la tendencia (D-1.1).

### Qué vigilar

1. **Colapso de la política a una acción.** Si el actor llega a $\pi(a|s)\approx1$, deja de muestrear las otras acciones y $\nabla\ln\pi\to0$, y le cuesta recuperarse. La trampa aquí es **"no operar"**: si la política deja de comprar, nunca visita el estado con posición 1, el crítico deja de aprender ahí, y el resultado es una política estable con recompensa exactamente 0.
   - **Colapso frente a solución correcta.** En el browniano, no operar es lo acertado. Mirad si ocurre también en los mercados 1 y 3, donde el Taller 1 sí encontró beneficio: ahí sería un fallo.
   - **La entropía sola engaña.** Con posición 0, vender y mantener son la misma acción, y con posición 1, comprar y mantener también (`env.py`). Una política que reparte 50/50 entre mantener y vender, sin posición, tiene $H=\ln2\approx0{,}69$ y **no opera nunca**. Registrad $P(\text{comprar}|\text{posición }0)$, $P(\text{vender}|\text{posición }1)$ y las operaciones por episodio, no solo $H/\ln3$.
2. **El extremo contrario: política casi uniforme.** Unas 165 operaciones por episodio ($494/3$; el agente aleatorio hizo 164,8 en el Taller 1) cuestan $\approx0{,}165$ de recompensa. Aparece si $\beta$ es grande respecto a la recompensa. Comparad la política muestreada con la *greedy*: pueden diferir mucho.
3. **Reparto de pesos del crítico.** En la capa compartida compiten actor, crítico y entropía. El término $c_v(R-V)^2$ también depende de la escala, y un mal reparto puede hacer que el crítico domine la representación, sobre todo en el mercado 1. Registrad por separado las normas del gradiente de cada término en la capa común, y la varianza explicada del crítico, $1-\mathrm{Var}(R-V)/\mathrm{Var}(R)$. Si no converge, probad $c_v\in\{0{,}25;0{,}5;1\}$ y, luego, redes separadas: las transparencias plantean justo dos modelos (DRL7 p5), y el profesor admite ambas opciones (T30 ~02:34).
4. **Errores de implementación que no dan error.**
   - Olvidar `.detach()` en la ventaja (es lo que hace el notebook).
   - Dejar gradiente en el `V(s_last)` del *bootstrap*.
   - Hacer *bootstrap* con $R=0$ al cortar por `t_max`: es **cero solo** si terminó de verdad.
   - Repartir mal la media y la suma: si cambiáis `sum` por `mean`, cambia el `lr` efectivo.
5. **Gradientes.** El artículo recorta por norma, sin dar el valor, y conviene hacer lo mismo y registrar la norma. Comprobad también que no haya `NaN` en los *logits*: conviene usar `Categorical(logits=...)`, no `log(softmax)` a mano.

### Windows, `spawn`, hilos y semillas

- **`spawn`.** En Windows, cada proceso arranca un intérprete nuevo e importa el módulo; **`worker` tiene que estar en `a3c.py`**, porque una función definida en una celda no se puede importar (`a3c.py`). Si lo ejecutáis como script, necesitáis `if __name__ == "__main__":`, pues el código de nivel superior se ejecuta en cada subproceso (docs de PyTorch). Todo lo que se pase a `worker` debe poder serializarse.
- **`torch.set_num_threads(1)`, dentro de cada `worker`.** Si no, cada proceso lanza varios hilos, y estos compiten por los núcleos. PyTorch recomienda como máximo $\lfloor N/M\rfloor$ hilos por proceso, con $N$ CPU y $M$ procesos: $\lfloor4/3\rfloor=1$.
- **Pesos compartidos.** `shared_model.share_memory()` antes de lanzar los procesos. Si `.grad` es `None` al enviarlo, cada proceso crea su propio `.grad` no compartido (docs de PyTorch), que es lo que queremos.
- **Optimizador compartido.** El artículo muestra que compartir sus estadísticas es lo robusto. `Adam` y `RMSprop` crean su estado de forma perezosa en el primer `step()`, por lo que **no quedaría compartido** por sí solo; hay que crearlo y compartirlo antes (ver código).
- **Semilla por *worker*.** Son tres cuestiones distintas:
  - **Acciones.** `set_seed(seed * 100 + rank)` en cada proceso, como pide `a3c.py`. Con la misma semilla, los tres *workers* muestrearían igual.
  - **Series.** Han de ser disjuntas o se pierde la descorrelación, razón de ser del método: `train_series_seed(seed, ep * n_workers + rank)`. El índice debe quedarse por debajo de `TRAIN_SEED_STRIDE` (10 000).
  - **Trampa de NumPy.** `make_prices` llama a `np.random.seed(serie)`, y reinicia el generador global del proceso en cada episodio. Muestread las acciones con PyTorch, no con `np.random.choice` como en el notebook, o se correlacionarán con la serie.
- **Reproducibilidad.** El orden en que se pisan los *workers* depende del sistema: dos ejecuciones con la misma semilla **no** dan lo mismo (D-2.1). Con las 5 semillas de D-0.1, informad de la dispersión. En el informe, separad "episodios" de "pasos totales" y de "tiempo", porque tres *workers* consumen tres veces más pasos por unidad de tiempo (D-0.3).

### Esqueleto del `worker`

Ejecutado una vez como prueba de humo con PyTorch 2.15 en Windows: 3 procesos con `spawn`, browniano y 6 episodios por *worker* a partir del código de abajo, sin errores y con pesos compartidos que cambian. El contador de Adam pasó de 1 a 445, cuando se hicieron $3\times6\times25=450$ actualizaciones: se perdieron algunas cuentas, lo esperable al escribir sin cerrojos, y por eso ese contador no sirve para contar actualizaciones (en una segunda ejecución llegó a 448). **Solo comprueba la mecánica, no que aprenda.** El esqueleto trata el fin de serie como terminal ($R=0$), que es lo que decide D-1.2. `VALUE_COEF`, `REWARD_SCALE` y `MAX_GRAD_NORM` no están en `A3CConfig`; sus valores son marcadores a ajustar.

```python
import torch, torch.nn as nn, torch.nn.functional as F
from .env import SimpleTradingEnv
from .markets import make_prices
from .protocol import set_seed, train_series_seed

VALUE_COEF, REWARD_SCALE, MAX_GRAD_NORM = 0.5, 1.0, 5.0     # marcadores: ajustar (D-1.1)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden):
        super().__init__()
        self.common = nn.Linear(obs_dim, hidden)            # capa compartida, como el notebook (celda 5)
        self.pi = nn.Linear(hidden, n_actions)              # logits; la softmax la aplica Categorical
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.common(x))
        return self.pi(h), self.v(h).squeeze(-1)


def make_shared_optimizer(model, lr):
    """Crea el estado del optimizador y lo comparte; si no, sería perezoso y propio de cada proceso."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)       # o RMSprop(alpha=0.99), como el artículo
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()                                              # gradiente nulo: crea el estado sin mover pesos
    opt.zero_grad(set_to_none=True)                         # .grad = None: cada proceso tendrá el suyo
    for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                v.share_memory_()
    return opt


def worker(rank, shared_model, optimizer, cfg, market, seed, n_steps, log):
    torch.set_num_threads(1)
    set_seed(seed * 100 + rank)
    local = ActorCritic(shared_model.common.in_features, shared_model.pi.out_features, cfg.hidden)
    for ep in range(cfg.n_episodes // cfg.n_workers):       # n_episodes = total
        env = SimpleTradingEnv(make_prices(market, n_steps, train_series_seed(seed, ep * cfg.n_workers + rank)))
        obs, _ = env.reset()
        done, ep_reward, entropies = False, 0.0, []
        while not done:
            local.load_state_dict(shared_model.state_dict())        # θ' = θ
            log_probs, values, ents, rewards = [], [], [], []
            for _ in range(cfg.t_max):
                logits, v = local(torch.as_tensor(obs, dtype=torch.float32))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()                                   # con PyTorch: ver "Trampa de NumPy"
                obs, r, done, _, _ = env.step(a.item())
                log_probs.append(dist.log_prob(a)); values.append(v); ents.append(dist.entropy())
                rewards.append(r * REWARD_SCALE); ep_reward += r
                if done:
                    break
            with torch.no_grad():                                   # el bootstrap no lleva gradiente
                R = torch.zeros(()) if done else local(torch.as_tensor(obs, dtype=torch.float32))[1]
            policy_loss, value_loss = 0.0, 0.0
            for i in reversed(range(len(rewards))):
                R = rewards[i] + cfg.gamma * R
                adv = R - values[i]
                policy_loss = policy_loss - log_probs[i] * adv.detach() - cfg.entropy_coef * ents[i]
                value_loss = value_loss + adv.pow(2)
            local.zero_grad()
            (policy_loss + VALUE_COEF * value_loss).backward()
            nn.utils.clip_grad_norm_(local.parameters(), MAX_GRAD_NORM)
            for sp, lp in zip(shared_model.parameters(), local.parameters()):
                sp.grad = lp.grad                                   # gradientes de θ' aplicados sobre θ
            optimizer.step()                                        # sin cerrojos, como en el artículo
            entropies.append(torch.stack(ents).mean().item())
        log.put((rank, ep, ep_reward, sum(entropies) / len(entropies)))
```

En `train_a3c`: `shared.share_memory()`, `opt = make_shared_optimizer(shared, cfg.lr)`, `ctx = mp.get_context("spawn")`, una `ctx.Queue()` para `log`, crear los procesos con `ctx.Process(target=worker, args=(...))`, y **vaciar la cola antes de `join`** para que no se bloqueen.

### Qué esperar en cada mercado

Referencias: mediana del retorno en test del Taller 1 (`results/taller1_tabular.csv`; 1 = +100 %), con las mismas series (D-0.1) pero estado discreto. A3C ve la tendencia continua, así que no hay garantía de que las iguale.

| Mercado | Aleatorio | Buy & hold | Q-learning / SARSA | Qué es razonable esperar de A3C |
|---|---|---|---|---|
| 1 `tendencia_ciclo` | 3,30 | 32,62 | 501 / 471 (48–57 operaciones) | Batir a buy & hold: hay deriva y ciclo, y la tendencia predice algo (+0,19). Ruido alto y colas enormes: usad la mediana. Si queda en "siempre dentro", el crítico no ha captado el ciclo. |
| 2 `browniano` | −0,19 | −0,03 | 0 / 0 (no opera) | **No operar** (0). No hay señal, y la comisión hace que operar pierda. En clase ya se plantea: con precios al azar, cabría esperar no hacer nada, porque en promedio no se gana (T24 ~04:34–04:35). Aquí el colapso es la respuesta correcta. Si opera en test, es sobreajuste al ruido. |
| 3 `reversion_media` | −0,17 | −0,00 | 0,27 / 0,27 (29 operaciones) | Es donde A3C tiene que demostrar más: la señal es débil a través de esta observación (−0,05, signo contrario a seguir la tendencia), y la comisión tienta a no operar. Llegar a 0 aquí es **un posible colapso**, no un éxito: comparadlo con 0,27. |
| 4 `regimenes` | −0,22 | −0,12 | 0 / 0 (no opera) | Probablemente no operar o muy poco. El régimen no se observa directamente y la correlación con el retorno siguiente es nula (+0,00). Mejorar a no operar sería una sorpresa que hay que comprobar con cuidado. |

**Comparación con DQN** ([`03_dqn.md`](03_dqn.md)). DQN reutiliza datos y A3C no (§5.5). Cuando la señal es débil frente al ruido, la eficiencia de datos pesa, y A3C podría necesitar más pasos de entorno para el mismo resultado, aunque los consuma más rápido. Es una hipótesis que la tabla de resultados deberá confirmar o desmentir, midiendo tanto pasos como tiempo.

**Para el informe.** El profesor pide simular e interpretar si lo que hace el agente tiene sentido (T24 ~04:35). Este apartado da el criterio: para cada mercado, ¿la política aprendida es la esperable y por qué? Un 0 en el browniano y un 0 en reversión a la media se parecen en la tabla, pero significan cosas opuestas.

*Fuente: `notebooks/common/a3c.py`, `env.py`, `markets.py`, `protocol.py`; `docs/DECISIONES.md` D-0.1, D-0.2, D-0.3, D-1.1, D-1.2, D-2.1; enunciado `taller_trader2`, celda 22; DRL7 p5; T24 ~04:34–04:35; T30 ~02:34; Mnih et al. (2016); medidas propias sobre los generadores.*
