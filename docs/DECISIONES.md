# Registro de decisiones del proyecto

Formato ligero tipo ADR, como en el Taller 1. ID `D-X.n`: `X` es el bloque (0 general, 1 entorno/estado,
2 algoritmos) y `n` un correlativo. El estado es `ABIERTA` o `CERRADA`. Las decisiones se añaden al final.

---

### D-0.1 · Protocolo común heredado del Taller 1

- **Fecha:** 2026-10-31
- **Decisión:** series de `N_STEPS = 500` pasos. En entrenamiento, una serie nueva por episodio con semilla
  `seed * 10_000 + episodio`. Validación: `N_VAL = 100` series desde `50_000`. Test: `N_TEST = 200` series desde
  `100_000`, evaluadas una sola vez con la política greedy. Resultado final con 5 semillas. El entorno y los 4
  generadores son copia literal del enunciado.
- **Justificación:** el enunciado es el mismo salvo los algoritmos, y mantener las series permite comparar DQN y A3C
  directamente con Q-Learning y SARSA. `test_common.py` comprueba que las medianas de los baselines coinciden con las
  del Taller 1 y que el código del entorno coincide con el del enunciado.
- **Estado:** CERRADA

### D-0.2 · PyTorch en lugar del Keras de clase

- **Fecha:** 2026-10-31
- **Decisión:** DQN y A3C se implementan con PyTorch en CPU. El enunciado no impone framework.
- **Evidencia (una medición en CPU de 4 núcleos, con el equipo ocupado; vale la proporción, no el valor absoluto):**
  con una red de 64 neuronas, evaluar un estado cuesta 0,3 ms en PyTorch, 10,8 ms con `model(x)` en Keras y 276 ms con
  `predict`. En un episodio de 494 pasos son ≈0,14 s, ≈5 s y ≈136 s. El paso de entrenamiento es similar (3,3 ms y
  2,8 ms, batch de 64).
- **A3C:** con procesos que comparten memoria (`share_memory`, `spawn`), probado con 3 procesos actualizando los mismos
  pesos desde un módulo importado. El arranque cuesta unos 19 s, casi todo imports, así que conviene lanzarlos una vez
  por entrenamiento.
- **Medición en el equipo actual (Ryzen 5 3600, 6 núcleos físicos y 12 lógicos; solo PyTorch, `00_entorno_y_baselines`, §7):**
  evaluar un episodio con `evaluate()` cuesta 0,05 s y un paso de entrenamiento (batch de 128), unos 0,64 ms. Lanzar 3 procesos cuesta unos
  3 s. Con todos los procesos a la vez, 3 rinden x2,5 y 11, x5,5.
- **Coste asumido:** el código de clase está en Keras y hay que traducirlo.
- **Estado:** CERRADA

### D-0.3 · Métricas y convergencia

- **Fecha:** 2026-10-31
- **Decisión:** se mantienen las métricas del Taller 1: mediana del retorno como métrica principal, media ± std, Sharpe
  por paso, max drawdown y operaciones. Se añade el baseline `no_operar`, que en el Taller 1 batió a buy & hold en los
  mercados 2-4.
- **Reglas simples como referencia:** una regla de tendencia y otra de reversión, con un solo umbral elegido en
  validación. En test baten a los tabulares del Taller 1 (x630 frente a x502 en el mercado 1, +29,2 % frente a +27,0 %
  en el 3), así que son el listón real para DQN y A3C (`00_entorno_y_baselines`, §6).
- **Problema:** `convergence_episode` (Taller 1) mide la recompensa online. No significa nada si el agente acaba
  sin operar, y no compara bien algoritmos con hiperparámetros distintos. Con A3C, además, los episodios transcurren en
  paralelo.
- **Decisión sobre la convergencia:** se registran y se pintan dos curvas, ambas con el retorno sobre el balance para que
  sean comparables. Con exploración: el retorno de cada episodio de entrenamiento, suavizado con una mediana móvil. Sin
  exploración: la mediana del retorno de la política greedy cada 25-50 episodios, sobre unas 20 series de validación
  siempre iguales. En A3C, "con exploración" es muestrear de la política y "sin exploración", la acción más probable.
- **Qué va a la tabla:** la convergencia se mide en la curva sin exploración: el primer punto que recorre el 90 % del
  camino entre su valor inicial y el final (la misma regla que `convergence_episode`), en episodios, pasos de entorno y
  segundos. Si el agente acaba sin operar, se indica "no opera", y si acaba peor que al empezar, "no mejora": en ninguno de los dos casos hay nada que haya convergido. La distancia entre las dos curvas se comenta en el análisis.
- **Estado:** CERRADA

### D-0.4 · Estructura y trabajo en paralelo

- **Fecha:** 2026-10-31
- **Decisión:** tres carpetas (`notebooks/`, `docs/`, `results/`). El código común vive en `notebooks/common/`, un
  notebook por línea de trabajo y resultados con el prefijo del notebook. `*.ipynb -merge` evita romper el JSON,
  `DECISIONES.md` usa `merge=union` y `notebooks/scratch_*.ipynb` no se versiona.
- **Por qué sin `nbstripout`:** exige configurarlo en cada equipo, falla en silencio si falta y los notebooks dejarían
  de leerse ejecutados en GitHub.
- **Caché:** depende de la configuración y de una `version` explícita. En el Taller 1 dependía de una huella del
  código, y cualquier comentario invalidaba todos los entrenamientos.
- **Estado:** CERRADA

### D-0.5 · Formato del informe

- **Fecha:** 2026-10-31
- **Decisión:** como en el Taller 1, el informe se redacta en `results/informe.md` y se exporta con
  `python results/build_informe.py`, que falla si el PDF supera 2 páginas.
- **Estado:** CERRADA

### D-1.1 · Normalización de la observación y de la recompensa

- **Fecha:** 2026-10-31
- **Contexto:** a diferencia del Taller 1, la red recibe la tendencia continua. Su escala cambia mucho entre mercados
  (percentiles 5/95 ≈ ±0,23 en el mercado 1 y ≈ ±0,04 en el resto), y la recompensa es del orden de 1e-3 por paso.
- **Alternativas:** normalizar la observación con estadísticos solo de train; escalar la recompensa; recortarla (clip),
  como propone la transparencia de mejoras de DQN.
- **Decisión:** normalizar las dos cosas con constantes por mercado, calculadas solo con series de train
  (`train_scales`): la red recibe `[tendencia / σ_tendencia, posición]` y aprende con `recompensa / σ_retorno`. Las
  métricas se siguen midiendo con el balance real. Escalar la recompensa no cambia la estrategia óptima.
- **Por qué constantes y no una ventana móvil:** las series salen de un generador estable, y al empezar el episodio una
  ventana apenas tiene datos para estimar σ. Una σ de test o de la propia serie miraría el futuro. Con el S&P 500 (03),
  cuya escala cambia con el tiempo, habría que reconsiderarlo.
- **Evidencia (`00_entorno_y_baselines`, §3):** σ de la tendencia en train de 0,153 en el mercado 1 y ≈0,024 en el
  resto, y distribuciones parecidas tras dividir por ella. Ruido del retorno entre 19 y 106 veces la comisión.
- **Evidencia (prueba con un DQN mínimo, `results/00_grid_normalizacion.csv`):** 200 episodios, 3 semillas, validación.
  Mediana en reversión a la media: sin nada 0 %, solo recompensa +1,9 %, solo observación +1,5 %, las dos +10,4 %. Solo
  con la recompensa escalada la decisión de comprar depende de la tendencia (compra tras caídas). En el mercado 1 las
  tres variantes con algún cambio se solapan (x576-x596), y en el browniano ninguna obtiene ganancia real.
- **Límite:** con 3 semillas, "las dos" no queda demostrada frente a "solo recompensa", y ninguna alcanza a la regla de
  referencia en reversión (+31,5 %). Hay que confirmarlo con el DQN definitivo, y probarlo en A3C.
- **Confirmación con el DQN definitivo (`01_dqn`, §3; 3 semillas, 500 episodios, validación):** en reversión a la media, sin
  nada +2,3 % con 3,3 operaciones, solo recompensa +28,8 %, solo observación +29,3 % (con una semilla en +1,8 %) y las dos
  +31,4 %. En el mercado 1, x370, x529, x597 y x505. Normalizar es necesario, pero "las dos" no se distingue de cada una por
  separado salvo por no tener semillas malas.
- **Confirmación en A3C (`02_a3c`, `results/02_grid_normalizacion.csv`):** sin normalizar la observación, A3C no opera en
  reversión (0 %); con las dos, +21,0 %. Solo con la recompensa escalada, el mercado 1 cae a x43,6. Se normalizan las dos.
- **Estado:** CERRADA

### D-1.2 · Fin del episodio

- **Fecha:** 2026-10-31
- **Contexto:** el entorno devuelve `terminated = True` al acabarse la serie y `truncated` siempre es `False`, pero en
  realidad es un límite de tiempo. En el Taller 1 se trató como terminal, con efecto en 1 de 494 pasos (allí, D-2.1).
- **Alternativas:** mantenerlo como terminal (objetivo `r`) o tratar el último paso como truncado y seguir estimando
  el valor de `s'`, sin tocar el entorno.
- **Decisión:** el final de la serie es terminal, tal como lo devuelve el entorno. En DQN, el objetivo del último paso es
  `y = r`; en A3C, el retorno del último tramo parte de `R = 0`. La misma regla en los dos algoritmos.
- **Justificación:** es lo explicado en clase (T30 ~04:35-04:37: en el estado final no se calcula el valor futuro), es
  más simple, coincide con el Taller 1 y mantiene la comparación con sus tabulares. Afecta a 1 de 494 pasos.
- **Límite:** es una simplificación, porque la serie se corta por tiempo y la observación no incluye cuánto queda. Si
  el agente se comporta de forma rara al final del episodio, hay que revisarla.
- **Estado:** CERRADA

### D-2.2 · Diseño del DQN

- **Fecha:** 2026-11-01
- **Decisión:** entrada `[tendencia / σ, posición]`, red 2 → 64 ReLU → 3 lineales, MSE y Adam, memoria circular de 20 000
  transiciones (unos 40 episodios) que empieza a entrenar con 1 000, minibatch de 128, un paso de gradiente cada 4 de entorno
  y copia a la red objetivo cada 500 pasos. ε parte de 1 y decae por episodio.
- **Justificación:** valores de clase (`2-cart_pole_DQN`) salvo la red, que el profesor pide lo más simple posible. Entrenar cada
  4 pasos cuesta un tercio que hacerlo en cada paso (`00_entorno_y_baselines`, §7). El agente usa su propio generador aleatorio
  porque `make_prices` reinicia el de NumPy con cada serie.
- **Límite:** la memoria, la frecuencia de entrenamiento y el tamaño de la red no se ajustaron en el 01. La red objetivo no
  demostró ayudar ni perjudicar (x505 frente a x410 y +31,4 % frente a +30,8 %, dentro del ruido). ε acaba en 0,08 con 500
  episodios, sin llegar al mínimo de 0,01.
- **Estado:** CERRADA

### D-2.3 · Ajuste por etapas con margen de ruido

- **Fecha:** 2026-11-01
- **Decisión:** un parámetro cada vez y por mercado (red objetivo, α, γ, ε, episodios), en las 100 series de validación con 3
  semillas. Un valor nuevo solo sustituye al actual si mejora más que el ruido: 10 puntos en los mercados 2 y 4, 2 puntos en el 3
  y un 25 % del múltiplo en el 1 (unos 2 errores estándar de la mediana con 100 series, `00_entorno_y_baselines`, §5).
- **Justificación:** sin margen, en los mercados sin señal se elegiría el azar más favorable.
- **Resultado en DQN:** ningún cambio supera el margen, así que los 4 mercados se quedan con la configuración de partida. Esto no
  prueba que sea óptima.
- **Límite:** las etapas no ven interacciones (ε y episodios lo están) y 3 semillas sobrestimaron el mercado 3 (+31,4 % frente a
  +27,7 % con 5 semillas, en las mismas series).
- **Estado:** CERRADA

### D-2.4 · El test ya visto no se vuelve a usar para elegir

- **Fecha:** 2026-11-01
- **Contexto:** D-0.1 dice que el test se usa una sola vez, y el resultado de DQN ya se ha visto. Interesa intentar mejorarlo.
- **Decisión:** `results/01_dqn.csv` es el resultado de DQN y no se sobrescribe. Toda mejora se elige solo en validación, y se mide
  en 400 series que no se han usado para nada (`split="nuevas"`, semillas desde 400 000). El informe debe decir que el test ya se
  había visto si cita alguna cifra posterior.
- **Estado:** CERRADA

### D-2.5 · Mejoras probadas sobre el DQN

- **Fecha:** 2026-11-01
- **Decisión:** se prueban cuatro mejoras baratas en el §7 de `01_dqn`, con 5 semillas y comparando en las mismas series de
  validación: acciones canónicas con Doble DQN, recompensa logarítmica, mejor punto en validación y media de las 5 redes. Se adoptan
  con el margen de D-2.3 expresado en logaritmo. Las cuatro son opciones de `DQNConfig` o `EnsembleAgent` que por defecto no cambian
  el agente base.
- **Resultado:** ninguna supera el margen, así que el DQN base se queda en los 4 mercados y no hay fila `dqn_mejorado`. Canónicas con
  Doble y recompensa logarítmica hacen operar más en los mercados 2 y 4 (hasta 23 operaciones frente a 0,1), contra lo esperado.
- **Comprobación en series nuevas, solo informativa:** el mejor punto da x356,7 frente a x287,4 en el mercado 1 y +29,6 % frente a
  +24,5 % en el 3, y elimina las semillas malas (peor semilla del 3: de +5,4 % a +24,5 %). No alcanza a las reglas (x530,2 y +32,7 %).
- **Límite:** la regla mira la mediana entre semillas, y por eso no recoge una mejora que actúa sobre las peores. Se decidió no cambiar
  la regla después de ver los datos: el mejor punto se presenta como la mejora más prometedora, medida una sola vez.
- **Estado:** CERRADA

### D-2.1 · Variante de A3C

- **Fecha:** 2026-10-31
- **Contexto:** el enunciado pide A3C. En clase solo aparece como variante del actor-crítico (`7-Métodos_actor_crítico`,
  p. 16) y el código disponible es un actor-crítico en CartPole. A3C no usa ε: explora porque la política es estocástica,
  así que el ajuste de ε del enunciado se traslada al coeficiente de entropía y hay que justificarlo en el informe.
- **Alternativas:** A3C real con procesos asíncronos, que no es reproducible bit a bit y da un tiempo sensible a la
  CPU; o A2C síncrono como comparación adicional.
- **Decisión:** A3C real, con 3 procesos asíncronos (`spawn`, necesario en Windows) que escriben sin cerrojos sobre pesos y
  estado de Adam compartidos. No se añade A2C. El ajuste de ε se traslada a β, el coeficiente de entropía.
- **Consecuencia:** no es reproducible bit a bit, así que se informa de la dispersión entre semillas. El tiempo solo se
  compara con DQN medido en el mismo equipo (`02_a3c.ipynb`, §6).
- **Estado:** CERRADA

### D-2.6 · Diseño y ajuste del A3C

- **Fecha:** 2026-11-03
- **Decisión:** lo posible igual que DQN: entrada `[tendencia / σ, posición]`, 64 neuronas, recompensa normalizada y fin de
  episodio terminal. `n_episodes` es el total entre procesos, para usar los mismos ≈247 000 pasos de entorno. Las series son
  disjuntas entre procesos. Red con capa compartida, retornos de hasta `t_max` pasos, Adam compartido y recorte de la
  norma del gradiente a 5. La tabla completa está en `02_a3c.ipynb`, §1.
- **Ajuste (D-2.3):** base α = 0,001, γ = 0,9, β = 0,01, `t_max` = 20, 500 episodios y sin máscara de acciones redundantes.
  Solo superan el margen α = 0,003 en el mercado 1 (x538,3 → x728,1) y β = 0 en el 3 (+21,0 % → +25,9 %). La máscara empeora
  (x491,2 y +16,0 %) y queda desactivada.
- **Resultado en test:** x374,9, −4,5 %, +13,8 % y 0 %. No bate a la mejor referencia en ningún mercado.
- **Mejor punto (informativo, D-2.4):** en las 400 series nuevas, x480,3 y +30,1 % frente a x388,2 y +15,8 % de la base,
  por encima del margen en los mercados 1 y 3. No se adopta: se midió después del test y el resultado oficial no cambia.
- **Estado:** CERRADA

### D-3.1 · Extras con un factor y evaluación posterior al test

- **Fecha:** 2026-11-04
- **Decisión:** completar `03_extras` con el plan previo de `results/03_METODOLOGIA.md`
  y `results/03_plan.json`: variantes DQN, estado ampliado, placebo, comisión,
  mezcla, generadores extra, S&P 500 y A2C frente a A3C. Cinco semillas y 500
  episodios; cada configuración DQN difiere en una sola clave de la referencia.
- **Aislamiento:** Double DQN se prueba sin las acciones canónicas de D-2.5.
  Los extras usan subclases en `common/extras.py`; el entorno, los generadores
  y los agentes originales permanecen intactos. Un test verifica que la base
  reproduce exactamente los pesos del entrenamiento DQN original.
- **Protocolo:** los cuatro resultados oficiales no se eligen de nuevo.
  Se reservan 400 series por mercado desde 600.000, distintas de 100.000 y
  400.000. Parejas con las mismas series y semillas, mediana y dispersión entre
  cinco entrenamientos, sin adoptar ni combinar variantes después de verlas.
- **Real:** train 2017–2021, validación 2022–2023, evaluación 2024–2025,
  normalización y ventanas solo de train. La trayectoria histórica es única y
  ya estaba disponible en el Taller 1; el resultado es retrospectivo.
- **Cómputo:** inferencias de evaluación agrupadas por serie, verificadas frente
  al entorno paso a paso. Se conserva el aprendizaje secuencial de DQN.
  Los tiempos con varios entrenamientos simultáneos no permiten comparar
  velocidad de variantes. A2C/A3C se miden uno a uno en el mismo equipo.
- **Estado:** CERRADA

### D-3.2 · Resultado de los extras: representación antes que arquitectura

- **Fecha:** 2026-11-04
- **Ejecución:** 260 entrenamientos DQN y 40 de A2C/A3C, cinco semillas por
  condición; 400 series nuevas por mercado sintético y una trayectoria temporal
  del S&P 500. Notebook 03 ejecutado sin errores; CSV por semilla y por serie,
  cinco figuras y conclusiones en `results/03_CONCLUSIONES.md`.
- **Resultado:** la ventana supera el margen emparejado en tendencia y reversión
  (5/5 y 4/5 semillas mejoran). MA30 y volatilidad lo superan en reversión y
  regímenes (5/5 en ambos), pero empeoran tendencia. Double DQN solo supera el
  margen en reversión (3/5). Duelo, quitar target, clipping, log y mezcla no lo
  superan. A2C tampoco lo supera frente a A3C con los parámetros de referencia.
- **Control:** el placebo sin tendencia pierde la estrategia de los mercados con
  señal. Con costes altos, el agente casi deja de operar en browniano y regímenes.
  En el histórico real, +12,1 % de mediana frente a +43,8 % de buy & hold: no se
  presenta una estrategia superior por usar datos reales.
- **Decisión:** no se cambia la referencia oficial, no se combinan mejoras ni se
  reajusta con estas series. Los márgenes son descriptivos, no significación
  estadística. A2C utiliza un proceso y A3C tres: los segundos comparan estas
  implementaciones, no idéntico uso de CPU. La trayectoria real es única.
- **Verificación:** suite de 63 pruebas sin fallos ni errores, con una omisión
  por ausencia del material bruto; controles de equivalencia, causalidad,
  comisiones, Double DQN, A2C y evaluación en lote. Auditoría del notebook:
  presupuestos, cinco semillas, parejas de configuración y 116.010 retornos
  exportados por serie, sin contar los baselines.
- **Estado:** CERRADA

### D-4.1 · Comparativa final en el notebook 04

- **Fecha:** 2026-11-05
- **Decisión:** `04_comparativa` no entrena: lee `results/` y reúne en un solo notebook las referencias, DQN, A3C, los extras,
  la tabla final de rendimiento y convergencia y el análisis crítico. Los extras del 03 se muestran aparte y no entran en la
  tabla oficial (D-2.4, D-3.2).
- **Tiempos:** los segundos de `01_dqn.csv` se midieron en otro equipo; la tabla usa los de DQN medidos en el mismo equipo
  que A3C (`02_grid_tiempo_dqn.csv`). Los tiempos de los tabulares del Taller 1 son solo orientativos.
- **Límite:** las ventajas emparejadas de las mejoras de DQN (§1.5) se copian de la salida de `01_dqn` §7, porque los retornos
  por serie solo están en la caché, que no se versiona.
- **Estado:** CERRADA

### D-4.2 · Maquetación del informe

- **Fecha:** 2026-11-05
- **Decisión:** informe de 2 páginas a dos columnas (figura y texto lado a lado, con la misma altura), en el orden diseño
  experimental → resultados por mercado → DQN frente a A3C → más allá de los mínimos → análisis crítico, y enlace al
  repositorio al final. Sin códigos de decisión en el texto: se explica lo decidido.
- **Figuras:** se eligieron de un catálogo de 44 (28 existentes y 16 dibujadas a tamaño de impresión, en
  `results/figures/informe_cand/`, con su nota en `CATALOGO.md`).
- **Límite:** las figuras existentes de 00/01/02 se dibujaron para la pantalla y en el PDF quedan con la letra pequeña.
- **Estado:** CERRADA
