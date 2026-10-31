# Métodos basados en el gradiente de la política: teorema y REINFORCE

> **Fuentes:** DRL6 (`docs/raw/DRL/transparencias/6-Policy_gradient.pdf`, "Métodos basados en el gradiente de la política"), pp. 1–39, leídas todas sus imágenes. Transcripciones `docs/raw/transcripciones/`: T24 y T30 (clases del 24 y 30 de octubre), solo en los pasajes que aparecen al buscar términos de este tema; T23 no tiene ninguno. Cita puntual, para enlazar con lo siguiente: DRL7 (actor-crítico) p4–p5, p15–p16, y slides3 (TD) p6, p17. Enunciado: `docs/raw/taller/taller_trader2.ipynb`.
>
> **Convenciones:** "(DRL6 pN)" = página N del mazo; "(T24 ~02:15)" = hora del cue en la transcripción. `**Nota:**` = contenido que no viene del material de clase, o errata detectada en él. "S&B" = Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2.ª ed., cap. 13. **La búsqueda en las transcripciones no encontró ninguna explicación de este mazo** (solo se anuncia; son automáticas y ruidosas, y no se han leído enteras), de modo que la teoría de este documento procede casi por entero de las transparencias, y las explicaciones entre líneas son Notas.

---

## 1. Concepto: aprender la política directamente

**Idea.** Los métodos anteriores pasan por los valores; estos optimizan directamente la política (DRL6 p7):

- Q-learning estima los valores de las acciones y luego con ellos actualiza la política.
- Los algoritmos basados en la política aprenden directamente la política.
- Normalmente estiman una política estocástica.
- En MDPs siempre existe una política determinista que maximiza el valor de cada estado.

**Las tres familias, según el profesor.** Lo visto en la parte de valor (incluido Q-learning) son métodos que estiman la función de valor y de ahí sacan una política; otros buscan la política "sin preocuparse por el valor"; y una tercera familia combina ambas cosas: un **actor** realiza acciones sin preocuparse por el valor y un **crítico** le indica si lo ha hecho bien o mal. Dice que estos últimos funcionan algo mejor porque combinan elementos de los dos mundos (T24 ~04:40).

**Dónde encajan en el mapa de algoritmos.** Dentro de los métodos sin modelo, una rama es la optimización de políticas, donde están el gradiente de la política y los actor-crítico; la otra rama, la de Q-learning, contiene DQN (T30 ~01:46). **Nota:** en esa frase la transcripción dice "a 12 a 13"; interpreto que son A2C y A3C, que DRL7 p16 nombra como variantes del actor-crítico.

**Estocasticidad: no es la misma que en $\varepsilon$-greedy.** En control por Monte Carlo el profesor ya usó una política estocástica, $\varepsilon$-greedy, para poder visitar todos los estados (T24 ~02:15). Allí la aleatoriedad se añade por fuera: *on-policy* se aprende la propia $\varepsilon$-greedy, y por eso se converge a la mejor política $\varepsilon$-greedy y no a la óptima salvo que $\varepsilon$ baje (T24 ~02:17); *off-policy* una política estocástica interactúa mientras se optimiza otra, determinista (T24 ~02:15). **Nota:** en gradiente de la política la aleatoriedad forma parte de lo que se aprende: $\pi$ decide por sí misma cuánto explorar en cada estado.

**Nota (tensión aparente de p7).** Si existe una óptima determinista, ¿por qué estimar una estocástica? Porque aquí es una herramienta de aprendizaje (§2), no el objetivo. Y no impide llegar a la solución: con softmax (§4) la política se aproxima a una determinista si la diferencia de preferencias crece mucho (S&B §13.1). El enunciado de p7 es el clásico para MDP finitos con observación completa del estado.

| | Basado en valor (Q-learning, DQN) | Basado en la política (REINFORCE) |
|---|---|---|
| Qué se aprende | $Q(s,a)$ | $\pi(a\mid s,\theta)$ |
| Política | Derivada: $\arg\max_a Q$ | Explícita, normalmente estocástica |
| Exploración | Añadida ($\varepsilon$) | Muestrear de $\pi$ |
| Datos (**Nota**) | *Off-policy*: admite memoria de repetición | *On-policy*: episodios de la política actual; los antiguos no sirven |

*(La tabla es una síntesis propia a partir de DRL6 p7, p10, p30.)*

*Fuente: DRL6 p1–p7; T24 ~02:15–02:17, ~04:40; T30 ~01:46.*

---

## 2. Por qué una política estocástica

**Idea.** La política estocástica resuelve de una vez exploración y optimización por gradiente. La transparencia da estas razones (DRL6 p10):

1. **Exploración natural del espacio:** no estamos obligados a tomar solo una acción en cada estado.
2. **Coste suave:** el coste a optimizar varía suavemente si cambiamos los parámetros de la política, así que podemos usar métodos basados en gradiente para optimizarla, *aunque los refuerzos sean discretos*.
3. **Información incompleta:** "permiten abordar problemas sin información completa". **Nota (errata de la fuente):** esta tercera razón está en la capa de texto de las pp. 8–10, pero no se ve en ninguna de las tres imágenes (p8 muestra solo el encabezado, p9 añade la razón 1 y p10 la 2). Se recoge porque consta en el PDF. El mazo no la desarrolla. **Nota:** se refiere a problemas parcialmente observables: si dos situaciones distintas se observan igual, la mejor respuesta puede ser mezclar acciones, algo que un $\arg\max$ no puede expresar.

**Nota (por qué "suave").** Con $\arg\max Q$, un cambio infinitesimal en $Q$ puede cambiar de golpe la acción elegida; con softmax, las probabilidades cambian de forma continua con $\theta$ (S&B §13.2). Y lo discreto de los refuerzos no estorba porque nunca se deriva la recompensa: solo se deriva $\pi$ (§5).

**Nota (terminología).** p10 habla de "coste a optimizar" y p14 de "métrica de rendimiento". Es la misma $J(\theta)$, y se **maximiza** (§3).

*Fuente: DRL6 p8–p10.*

---

## 3. Formalización

**Idea.** Se parametriza la política y se sube por el gradiente de una medida de rendimiento.

**Política parametrizada.** La probabilidad de realizar cada acción $a$ en cada estado $s$ es una función que depende de unos parámetros $\theta$ (DRL6 p14):

$$\pi(a\mid s,\theta)=P\left(A_t=a\mid S_t=s,\theta\right)$$

**Ascenso por gradiente.** Se define una métrica de rendimiento $J(\theta)$ y se aplica ascenso por gradiente para optimizarla (DRL6 p14):

$$\theta_{t+1}=\theta_t+\alpha\cdot\widehat{\nabla J(\theta)}$$

$\alpha$ es la tasa de actualización, que multiplica a la **estimación** del gradiente de $J$ (rendimiento); el circunflejo cubre todo $\nabla J(\theta)$ (DRL6 p14). En p38 a la misma $\alpha$ se le llama "tamaño de paso".

**Nota (el mazo no define $J$).** Ninguna transparencia da la expresión de $J(\theta)$: solo se la nombra como "métrica de rendimiento". La definición habitual para tareas episódicas (S&B §13.2) es el valor del estado inicial bajo la política, es decir, el retorno esperado:

$$J(\theta)=v_{\pi_\theta}(s_0)=\mathbb{E}_{\pi_\theta}\left[G_0\right]$$

El signo es **más** porque se maximiza, al contrario que en el descenso por gradiente de DQN; ver la consecuencia en código en §6.

*Fuente: DRL6 p11–p14.*

---

## 4. Parametrización de la política: preferencias y softmax

**Idea.** Una red (o función lineal) puntúa cada acción con un número real, y softmax convierte esas puntuaciones en probabilidades.

**Preferencias.** Una manera habitual de parametrizar $\pi(a\mid s,\theta)$ es definir una función de preferencia $h$ para cada par estado-acción (DRL6 p18):

$$h(s,a,\theta)\in\mathbb{R}$$

Cuanto más probable es la acción, más grande es $h$. $h$ puede ser de muchas formas (DRL6 p18):

- Una función lineal: $h(s,a,\theta)=\theta^{T}\cdot\mathbf{x}(s,a)$, con $\mathbf{x}(s,a)$ un vector de características. **Nota:** la transparencia no dice qué es $\mathbf{x}$; es la notación de S&B.
- Una red neuronal (los $\theta$ serían los pesos).
- Etc.

**Softmax.** La política se define entonces como (DRL6 p20):

$$\pi(a\mid s,\theta)=\frac{e^{h(s,a,\theta)}}{\sum_{a'}e^{h(s,a',\theta)}}$$

Esta función garantiza que la suma de las probabilidades de todas las acciones que se pueden hacer desde el estado $s$ sea 1 (DRL6 p20).

**Nota (propiedades útiles).**
- En una red neuronal, $h$ son los *logits*: la última capa es lineal, con una salida por acción, sin softmax si se usa `Categorical(logits=...)` o `log_softmax`.
- Solo importan las diferencias de $h$: sumar la misma constante a todas las preferencias de un estado no cambia $\pi$.
- p18 dice "cuanto más probable, más grande es $h$"; la causalidad de cálculo es la inversa (mayor $h$ implica mayor probabilidad), aunque la relación es monótona y la frase es correcta como equivalencia.
- Para $h$ lineal, el gradiente que usa REINFORCE (§6) tiene forma cerrada (S&B ec. 13.9):
$$\nabla\ln\pi(a\mid s,\theta)=\mathbf{x}(s,a)-\sum_{b}\pi(b\mid s,\theta)\,\mathbf{x}(s,b)$$
es decir, las características de la acción tomada menos las de la acción media.

*Fuente: DRL6 p15–p20.*

---

## 5. Teorema del gradiente de políticas

**Idea.** Permite estimar $\nabla J$ a partir de observaciones, sin conocer el entorno. La transparencia lo califica de fundamental: permite calcular el gradiente de $J$ de manera sencilla (DRL6 p27).

$$\nabla J(\theta)\propto\sum_{s}\mu(s)\sum_{a}q_{\pi}(s,a)\,\nabla\pi(a\mid s,\theta)$$

| Símbolo | Significado (DRL6 p27) | ¿De dónde sale? |
|---|---|---|
| $\propto$ | "proporcional a" | |
| $\mu(s)$ | proporción de veces (en tanto por 1) que se visita el estado $s$ | se puede estimar con observaciones |
| $q_\pi(s,a)$ | función de acción-valor | se puede estimar con observaciones |
| $\nabla\pi(a\mid s,\theta)$ | gradiente de la política respecto a $\theta$ | lo sabemos |

**Nota (por qué es tan importante).**
- $J$ depende de $\theta$ también por la distribución de estados visitados, que depende del entorno y es desconocida. Lo notable es que la fórmula **no contiene $\nabla\mu(s)$**: toda la derivada recae sobre $\pi$, que hemos elegido nosotros (S&B §13.2). Por eso "lo sabemos", y por eso el método no necesita modelo.
- La proporcionalidad basta porque $\alpha$ absorbe la constante. En el caso episódico la constante es la longitud media del episodio (S&B §13.2).
- El mazo enuncia el teorema sin demostrarlo. S&B §13.2 tiene la demostración.

*Fuente: DRL6 p21–p27.*

---

## 6. Algoritmo REINFORCE

**Idea.** Es un algoritmo bastante conocido de gradiente de la política; genera episodios completos basándose en la política actual (DRL6 p30) y usa el retorno observado en lugar de $q_\pi$.

**Pseudocódigo** (DRL6 p38):

```text
Entrada: política parametrizada π(a|s,θ), derivable respecto a θ
Hiperparámetros: α (tamaño de paso), γ (factor de descuento)

Inicializar θ (por ejemplo, a 0)
Por cada episodio:
    Generar episodio S0, A0, R1, ..., S(T-1), A(T-1), RT siguiendo π(a|s,θ)
    Por cada paso del episodio t = 0, 1, ..., T-1:
        Se calcula Gt      (fórmula 1)
        Se actualiza θ     (fórmula 2)
```

$$G_t=\sum_{k=t+1}^{T}\gamma^{\,k-t-1}R_k\qquad\text{(1)}$$

$$\Delta\theta=\alpha\cdot\gamma^{t}\cdot G_t\cdot\nabla\ln\pi\left(A_t\mid S_t,\theta\right)\qquad\text{(2)}$$

En la transparencia, un recuadro naranja rodea la fórmula de $G_t$, con una flecha hacia la etiqueta "Retorno calculado desde el paso t"; de esa etiqueta sale otra flecha al $G_t$ de la actualización, rodeado con un círculo. Es decir: el $G_t$ de (2) es el retorno desde el paso $t$, no el del episodio entero (DRL6 p38).

**Nota de la transparencia: el gradiente del logaritmo** (DRL6 p39):

$$\nabla\ln\pi\left(A_t\mid S_t,\theta\right)=\frac{1}{\pi\left(A_t\mid S_t,\theta\right)}\cdot\nabla\pi\left(A_t\mid S_t,\theta\right)$$

**Nota (del teorema a REINFORCE; el mazo no da estos pasos).** Con la fórmula de p39 se llega a (2) (S&B §13.3). Muestrear estados siguiendo $\pi$ da los pesos $\mu(s)$, se multiplica y divide por $\pi$ para convertir la suma sobre acciones en una esperanza, y $G_t$ es una muestra de $q_\pi$ porque $\mathbb{E}[G_t\mid S_t,A_t]=q_\pi(S_t,A_t)$:

$$\begin{aligned}
\nabla J(\theta)&\propto\mathbb{E}_{\pi}\left[\sum_{a}q_\pi(S_t,a)\,\nabla\pi(a\mid S_t,\theta)\right]\\
&=\mathbb{E}_{\pi}\left[\sum_{a}\pi(a\mid S_t,\theta)\,q_\pi(S_t,a)\,\frac{\nabla\pi(a\mid S_t,\theta)}{\pi(a\mid S_t,\theta)}\right]\\
&=\mathbb{E}_{\pi}\left[q_\pi(S_t,A_t)\,\frac{\nabla\pi(A_t\mid S_t,\theta)}{\pi(A_t\mid S_t,\theta)}\right]\\
&=\mathbb{E}_{\pi}\left[G_t\,\nabla\ln\pi(A_t\mid S_t,\theta)\right]
\end{aligned}$$

**Nota (cómo leer la regla).**
- $\nabla\ln\pi$ apunta hacia donde aumenta la probabilidad de la acción tomada; $G_t$ dice cuánto y en qué sentido moverse: si es positivo, esa acción se hace más probable en ese estado, y si es negativo, menos.
- El $1/\pi$ de p39 compensa que las acciones frecuentes se muestrean más: sin él, se verían favorecidas solo por salir más veces, aunque no fuesen mejores (S&B §13.3).
- De ahí que se use el logaritmo; además, en la práctica $\ln\pi$ es numéricamente más estable y es lo que dan las bibliotecas.

**Nota (detalles que el mazo da sin comentar).**
- **El factor $\gamma^t$.** Pondera menos los pasos tardíos; corresponde a la versión con descuento (S&B §13.3), aunque el teorema de p27 se escribe sin descuento. Muchas implementaciones lo omiten, entre ellas A3C (Mnih et al., 2016). En cambio, DRL7 p15 lo mantiene en el actor-crítico como el factor $I\leftarrow\gamma I$. Las fuentes de clase lo incluyen: si lo quitáis, decidlo y justificadlo. Ver el apartado de aplicación al final.
- **"Inicializar $\theta$ a 0"** vale para $h$ lineal, y da una política uniforme. En una red neuronal **no sirve**: con todos los pesos a cero, las neuronas de una capa son simétricas y no se diferencian. Usad la inicialización aleatoria habitual; si se quiere una política inicial casi uniforme, basta con pesos pequeños en la capa de salida.
- **Actualización dentro del bucle.** Tal como está escrito, $\theta$ cambia en cada $t$, de modo que en rigor la $\pi$ de $t+1$ ya no es la que generó el episodio. Lo más común, y es lo que hace el código de abajo, es sumar los términos de todos los pasos y aplicar un único paso al final del episodio.
- **Cálculo eficiente de $G_t$:** hacia atrás, $G_t=R_{t+1}+\gamma G_{t+1}$ con $G_T=0$.

**Nota (implementación mínima, propia).** Los optimizadores minimizan, así que la pérdida lleva signo **menos**:

```python
import torch, torch.nn as nn
from torch.distributions import Categorical

politica = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 3))  # h(s,a,θ): logits
opt = torch.optim.Adam(politica.parameters(), lr=1e-3)                    # α

def episodio_reinforce(env, gamma=0.99):
    s, _ = env.reset()
    log_probs, recompensas = [], []
    terminado = False
    while not terminado:                                       # generar episodio siguiendo π
        dist = Categorical(logits=politica(torch.as_tensor(s, dtype=torch.float32)))
        a = dist.sample()
        log_probs.append(dist.log_prob(a))
        s, r, term, trunc, _ = env.step(a.item())
        recompensas.append(r)
        terminado = term or trunc

    G, retornos = 0.0, []
    for r in reversed(recompensas):                            # G_t = R_{t+1} + γ·G_{t+1}
        G = r + gamma * G
        retornos.insert(0, G)

    perdida = -sum(gamma**t * retornos[t] * log_probs[t]       # fórmula (2), con signo menos
                   for t in range(len(retornos)))
    opt.zero_grad()
    perdida.backward()
    opt.step()
```

*Fuente: DRL6 p28–p39.*

---

## 7. Qué problema de REINFORCE resuelve el crítico

**Idea.** REINFORCE es correcto en esperanza pero tiene dos defectos que vienen de usar $G_t$: **hay que esperar al final del episodio** y **$G_t$ tiene mucha varianza**. El crítico sustituye $G_t$ por una estimación aprendida, disponible en cada paso y más estable.

**Nota:** DRL6 acaba en REINFORCE y no comenta sus limitaciones. Este análisis es propio, apoyado en las citas que se indican.

**Problema 1: esperar a fin de episodio.** Para calcular $G_t$ hacen falta todas las recompensas hasta $T$ (fórmula 1), así que no se puede actualizar nada hasta que el episodio acaba (DRL6 p30, p38), y no se puede usar en tareas continuas. Es el mismo inconveniente de Monte Carlo frente a TD (slides3 p6, p17).

**Problema 2: varianza.** $G_t$ es una suma de recompensas muestreadas, de modo que acumula el azar de todas las acciones y transiciones futuras. Aunque su esperanza sea la correcta, cada muestra es ruidosa, y el gradiente estimado con ella también: hacen falta muchos episodios y $\alpha$ pequeña (slides3 p6 y p17 comparan: MC alta varianza; TD menor, porque depende de una sola transición).

**Problema 3, relacionado: falta de referencia.** $G_t$ es absoluto. Si todos los retornos son positivos, toda acción tomada se refuerza, y solo se distingue de las demás por cuánto. Lo informativo es si la acción ha sido **mejor o peor de lo esperado en ese estado**. Restar una línea base $b(s)$ no introduce sesgo (S&B §13.4), porque $\sum_a b(s)\nabla\pi(a\mid s,\theta)=b(s)\nabla\sum_a\pi(a\mid s,\theta)=b(s)\nabla 1=0$.

**La solución: un segundo modelo.** La idea de los actor-crítico es usar dos modelos, uno para la política $\pi$ ("actor") y otro para el valor ("crítico"), cada uno con sus propios parámetros (DRL7 p4–p5). Pueden ser dos redes o dos capas de salida, y muchas variantes de esta idea se plantean para estabilizar el entrenamiento (T30 ~02:34). El algoritmo básico cambia $G_t$ por el error TD (DRL7 p15):

$$\delta=R+\gamma\,v(S',\mathbf{w})-v(S,\mathbf{w})\qquad\Delta\theta=\alpha_\theta\,I\,\delta\,\nabla\ln\pi(A\mid S,\theta)\qquad\Delta\mathbf{w}=\alpha_{\mathbf{w}}\,\delta\,\nabla v(S,\mathbf{w})$$

| | REINFORCE (DRL6 p38) | Actor-crítico (DRL7 p15) |
|---|---|---|
| Qué multiplica a $\nabla\ln\pi$ | $G_t$ | $\delta$ |
| Cuándo actualiza | Al acabar el episodio | En cada paso |
| Varianza (**Nota**) | Alta: toda la trayectoria | Menor: una transición |
| Sesgo (**Nota**) | Ninguno | Sí, mientras $v$ sea imperfecto (*bootstrapping*) |
| Referencia (**Nota**) | Ninguna | $v(S,\mathbf{w})$, que actúa de línea base |
| Modelos | Uno | Dos |

**Nota:** el crítico no es gratis: cambia varianza por sesgo, y añade un segundo modelo que entrenar. A3C elige un punto intermedio con retornos de $n$ pasos y varios trabajadores en paralelo; todo ello, en [`06_actor_critico_y_a3c.md`](06_actor_critico_y_a3c.md).

*Fuente: DRL6 p30, p38; DRL7 p4–p5, p15; slides3 p6, p17; T30 ~02:34.*

---

## Aplicación al Taller 2

**Idea.** REINFORCE no es uno de los algoritmos obligatorios, pero el actor de A3C es esto mismo. Todo este apartado es **Nota**: propuestas y cálculos propios para `SimpleTradingEnv`.

- **Actor.** Red con entrada `[tendencia, posición]` y 3 *logits* (vender, mantener, comprar), más softmax (§4). Es el mismo actor en A3C, añadiendo el crítico.
- **Exploración.** En A3C no hay $\varepsilon$: se muestrea de $\pi$. El punto 3 del enunciado cita $\varepsilon$ entre los hiperparámetros a ajustar, pero en A3C solo tiene sentido para DQN; conviene explicarlo en la entrega. El riesgo es el contrario: que softmax se sature pronto en una acción (por ejemplo, no operar) y deje de explorar. A3C lo combate con entropía (ver `06`).
- **La exploración cuesta dinero.** Con una política inicial casi uniforme, la posición cambia con probabilidad $1/3$ en cada paso: unas $494/3\approx165$ operaciones por episodio. A $-10^{-3}$ de recompensa cada una, son unos $-0{,}165$ de recompensa y, en capital, $0{,}999^{165}\approx0{,}85$. Comisión y señal tienen el mismo orden ($\approx10^{-3}$), así que al principio el gradiente verá sobre todo que operar sale caro: es probable caer pronto en "no operar". Vigiladlo en el número de operaciones por episodio.
- **Un episodio, una actualización.** REINFORCE daría un solo paso cada 494 decisiones, y cada $G_t$ sumaría decenas de recompensas ruidosas, sobre todo en el browniano, donde los retornos futuros son casi puro ruido. Es el motivo práctico de usar el crítico (§7).
- **$\gamma^t$ a lo largo del episodio.** Con $\gamma=0{,}99$, $\gamma^{493}\approx0{,}007$: los pasos finales apenas actualizarían. Esto no tiene sentido si el episodio es un tramo arbitrario de mercado sin punto de partida especial; omitir $\gamma^t$ (como A3C) parece lo razonable aquí, aun alejándose de DRL6 p38 y DRL7 p15. Dejadlo razonado en la entrega.
- **Escala.** Con $r\approx10^{-3}$ y $\gamma=0{,}99$ (horizonte efectivo $\approx100$ pasos), la parte de $G_t$ debida a la señal ronda como mucho $0{,}1$, mientras que el ruido de cada muestra con posición abierta es de $\approx0{,}14$ en los mercados 2-4 y $\approx0{,}76$ en el 1 (ver `06`, Aplicación), y los gradientes son proporcionales. Si el actor no se mueve, comprobad la escala de recompensa antes de subir $\alpha$ a ciegas.
- **Información incompleta (§2, razón 3).** La observación no incluye, por ejemplo, el régimen oculto del generador 4; por tanto, una política determinista no es necesariamente la mejor en sentido estricto.
- **Acciones equivalentes.** Con posición 0, vender y mantener hacen lo mismo; con posición 1, comprar y mantener también. Sus diferencias de probabilidad solo reciben ruido: al interpretar $\pi$, agrupadlas.
- **Datos nuevos.** Como el método es *on-policy*, no hay memoria de repetición. El enunciado pide una serie de precios distinta por episodio: construid `SimpleTradingEnv(prices)` con una nueva serie antes de cada `episodio_reinforce` (§6).
- **Evaluación.** El test usa la política *greedy* ($\arg\max$; D-0.1, CERRADA): evita pagar comisión por azar y es comparable con el DQN voraz. Muestrear de $\pi$ evalúa la política realmente aprendida: usadlo como análisis adicional en validación y decid si difiere.

**Siguiente paso:** el crítico que elimina la espera a fin de episodio y reduce la varianza de $G_t$, y su versión asíncrona A3C, están en [`06_actor_critico_y_a3c.md`](06_actor_critico_y_a3c.md).
