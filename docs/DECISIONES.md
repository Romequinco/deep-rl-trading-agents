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
- **Coste asumido:** el código de clase está en Keras y hay que traducirlo.
- **Estado:** CERRADA

### D-0.3 · Métricas y convergencia

- **Fecha:** 2026-10-31
- **Decisión:** se mantienen las métricas del Taller 1: mediana del retorno como métrica principal, media ± std, Sharpe
  por paso, max drawdown y operaciones. Se añade el baseline `no_operar`, que en el Taller 1 batió a buy & hold en los
  mercados 2-4.
- **Problema abierto:** `convergence_episode` (Taller 1) mide la recompensa online. No significa nada si el agente acaba
  sin operar, y no compara bien algoritmos con hiperparámetros distintos. Con A3C, además, los episodios transcurren en
  paralelo. Propuesta a valorar: pasos de entorno y tiempo hasta alcanzar el resultado en validación de la política greedy.
- **Estado:** ABIERTA (criterio de convergencia para DQN y A3C)

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
- **Estado:** ABIERTA

### D-1.2 · Fin del episodio

- **Fecha:** 2026-10-31
- **Contexto:** el entorno devuelve `terminated = True` al acabarse la serie y `truncated` siempre es `False`, pero en
  realidad es un límite de tiempo. En el Taller 1 se trató como terminal, con efecto en 1 de 494 pasos (allí, D-2.1).
- **Alternativas:** mantenerlo como terminal (objetivo `r`) o tratar el último paso como truncado y seguir estimando
  el valor de `s'`, sin tocar el entorno.
- **Estado:** ABIERTA

### D-2.1 · Variante de A3C

- **Fecha:** 2026-10-31
- **Contexto:** el enunciado pide A3C. En clase solo aparece como variante del actor-crítico (`7-Métodos_actor_crítico`,
  p. 16) y el código disponible es un actor-crítico en CartPole. A3C no usa ε: explora porque la política es estocástica,
  así que el ajuste de ε del enunciado se traslada al coeficiente de entropía y hay que justificarlo en el informe.
- **Alternativas:** A3C real con procesos asíncronos, que no es reproducible bit a bit y da un tiempo sensible a la
  CPU; o A2C síncrono como comparación adicional.
- **Estado:** ABIERTA
