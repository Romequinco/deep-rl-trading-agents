# Catálogo de figuras para el informe PDF

Informe: 2 páginas A4, texto a 8,7 pt. Ancho completo ≈ 19 cm; media columna ≈ 12 cm (66 % de la rejilla 66/34); columna lateral ≈ 6,3 cm.
**Nota** = idoneidad para el informe (5 = imprescindible o lista para pegar; 1 = no usar).
Las candidatas `cand_*` están dibujadas a tamaño real (7,5 in o 4,7 in de ancho, 200 dpi), con letra de 6,5-8,5 pt y sin título general: el pie de figura va en el HTML.
Se regeneran con `python ../results/figures/informe_cand/_gen.py` desde `notebooks/`. Solo leen `results/*.csv`, `common.markets` y, en las variantes `c`, recortes de PNG existentes.

## Recomendación (top-6 para 2 páginas)

| # | Figura | Dónde | Por qué |
|---|---|---|---|
| 1 | `cand_rendimiento_a` | ancho completo, §2-3 | Es la figura central del ejercicio 4: agentes por mercado, con semillas y la mejor referencia. Ocupa 5,9 cm de alto (la original, 8 cm). |
| 2 | `cand_politica_b` | ancho completo, §2 | Muestra en una figura si cada estrategia es la esperada, con DQN y A3C superpuestos. Ocupa 6 cm, frente a los 2 × 8 cm de 01/02_politica. |
| 3 | `cand_convergencia_a` | ancho completo, §3 | Episodios por semilla y segundos en el mismo equipo: recoge la tabla de convergencia del enunciado en 5 cm. |
| 4 | `cand_extras_b` | ancho completo, §4 | Tres resultados de «más allá» (estado, comisiones y S&P 500) en 4,4 cm, con las cifras escritas. |
| 5 | `cand_mercados_a` | ancho completo, §1 | Presenta los 4 generadores en 4 cm. Si falta espacio, `cand_mercados_c` (1,7 cm) como cabecera. |
| 6 | `cand_hiperparametros_b` | ancho completo, §3 | Ajuste de α, γ, ε/β y episodios de los dos algoritmos en una figura de 5,3 cm. Muestra que solo γ alto, α bajo y pocos episodios empeoran. |

Alternativas si sobra espacio o se prefiere tabla: `cand_rendimiento_b` (heatmap de 12 cm con 34 % de texto al lado) puede sustituir a la tabla markdown de resultados; `cand_extras_a` (12 cm) sustituye a `extras_b` si se quiere la ventaja emparejada con semillas. Total del top-6: ≈ 27 cm de figuras, algo menos de una página.

## Candidatas nuevas (`results/figures/informe_cand/`)

| Ruta | Tema | Mensaje | Nota | Tamaño | Recomendación |
|---|---|---|---|---|---|
| `cand_mercados_a.png` | mercados | Una serie por generador (1×4): tendencia con ciclo, browniano sin deriva, reversión acotada y regímenes. | 4 | ancho completo (4,0 cm) | **Top-6.** Sencilla y legible. |
| `cand_mercados_b.png` | mercados | Abanico de 200 series (mediana, p25-p75 y p5-p95) y una serie de ejemplo: la dispersión y la deriva de cada mercado. | 3 | media columna (8,1 cm) | Es la más informativa, pero ocupa mucho alto para un PDF de 2 páginas. |
| `cand_mercados_c.png` | mercados | Mini cabecera: 12 series grises y una en color por mercado, sin ejes. | 3 | ancho completo (1,7 cm) | Para cuando falta espacio; sin ejes, es decorativa. |
| `cand_rendimiento_a.png` | rendimiento | Test: barras de los 9 agentes por mercado, puntos = semillas DQN/A3C, discontinua = mejor referencia. | 5 | ancho completo (5,9 cm) | **Top-6.** Versión baja de `04_p4_comparativa_mercados`. |
| `cand_rendimiento_b.png` | rendimiento | Tabla-heatmap agente × mercado con el retorno escrito; color = posición en el mercado, recuadro = mejor. | 4 | media columna (6,3 cm) | Puede sustituir a la tabla de resultados en markdown. |
| `cand_rendimiento_c.png` | rendimiento | Gráfico de puntos: las 5 semillas de DQN y de A3C frente a buy & hold, Q-Learning y la mejor referencia. | 3 | ancho completo (4,7 cm) | Muestra bien la dispersión entre semillas, pero repite a `_a`. |
| `cand_convergencia_a.png` | convergencia | Episodios hasta converger por semilla (n/5 convergen) y segundos de entrenamiento y hasta converger. | 5 | ancho completo (5,1 cm) | **Top-6.** |
| `cand_convergencia_b.png` | convergencia | Episodios frente a segundos hasta converger, por semilla (escala log): DQN, menos episodios; A3C, episodios más baratos. | 3 | columna lateral (~7 cm) | Mensaje original y compacto; necesita explicarse en el texto. |
| `cand_convergencia_c.png` | convergencia | Recorte de `01_dos_curvas` y `02_dos_curvas` (mercados 1 y 3): curvas con y sin exploración. | 2 | media columna (12,6 cm) | Demasiado alta y con letra pequeña. No hay curvas de A3C en los CSV ni en la caché, por eso se recorta. |
| `cand_politica_a.png` | política | Acción greedy por semilla y tendencia (verde comprar, rojo vender): DQN y A3C, sin y con activo. | 4 | ancho completo (6,1 cm) | Muestra cada semilla. Alternativa a `_b`. |
| `cand_politica_b.png` | política | Fracción de semillas que compran (sin activo) o venden (con activo) según la tendencia; DQN continua, A3C discontinua. | 5 | ancho completo (6,0 cm) | **Top-6.** Se lee la estrategia esperada en 1 y 3; en el 2 se ve que no la aprenden. |
| `cand_politica_c.png` | política | Recorte de `01_politica` (Q) y `02_politica` (probabilidad) en los mercados 1 y 3. | 3 | ancho completo (7,4 cm) | Muestra las curvas continuas, pero deja fuera los mercados 2 y 4. |
| `cand_extras_a.png` | extras | Ventaja log emparejada de 7 variantes clave (estado, Double, recompensa log, placebo, mezcla, A2C); recuadro = supera el margen. | 4 | media columna (6,0 cm) | Versión reducida de `04_p3_ventaja_emparejada`. |
| `cand_extras_b.png` | extras | Estado ampliado (x348 → x548; +16,8 % → +47,9 %), comisiones (69 → 26 → 1 operaciones) y S&P 500 (+12,1 % frente a +43,8 %). | 5 | ancho completo (4,4 cm) | **Top-6.** |
| `cand_hiperparametros_a.png` | hiperparámetros | Heatmap de valores probados × mercado para DQN y A3C, con el retorno escrito; recuadro = elegido. | 3 | ancho completo (9,2 cm) | Completa, pero alta para 2 páginas. |
| `cand_hiperparametros_b.png` | hiperparámetros | Δ frente al valor elegido, en márgenes de ruido (mercados 1 y 3), con la franja de ±1 margen de ruido. | 4 | ancho completo (5,3 cm) | **Top-6.** Valores < −6 recortados a −6 (γ = 0,99, α = 1e-4, 300 episodios). |

## Figuras existentes (`results/figures/`)

Todas tienen título largo incrustado y letra de 8-10 pt a unos 30 cm de ancho: a 19 cm la letra baja a ~5-6 pt.

| Ruta | Tema | Mensaje | Nota | Tamaño | Recomendación |
|---|---|---|---|---|---|
| `00_coste_por_episodio.png` | diseño/tiempo | El entrenamiento de la red domina el coste: DQN 0,34 s/episodio frente a 0,04 s de A3C. | 2 | media columna | Apoya la diferencia de tiempos, pero es secundaria. |
| `00_episodios.png` | mercados/baselines | Un episodio por mercado con la posición de la mejor regla y su balance frente a buy & hold y aleatorio. | 2 | ancho completo (8 cm) | Demasiado alta. |
| `00_escala_entradas.png` | diseño | La tendencia sin normalizar es diminuta frente a la posición; normalizada, se parece; la recompensa tiene poca señal. | 3 | ancho completo | Justifica la normalización (D-1.1); útil solo si sobra espacio. |
| `00_estabilidad_test.png` | protocolo | La mediana se estabiliza en torno a 200 series de test. | 2 | ancho completo | Justifica N_TEST; basta citarlo en el texto. |
| `00_memoria.png` | diseño | La tendencia predice en 1 (+) y 3 (−); la volatilidad de 4 tiene memoria que el estado no ve. | 3 | ancho completo | Motiva el estado ampliado; buena para el análisis crítico, pero cuesta espacio. |
| `00_mercados.png` | mercados | 30 series por generador y el histograma de retornos (σ de 10,6 % en el 1). | 3 | ancho completo (8,5 cm) | Sustituida por `cand_mercados_a/b`. |
| `00_recompensa_vs_balance.png` | diseño | La suma de recompensas no es el log del balance (sobre todo en el 1). | 2 | ancho completo | Explica el fallo en el browniano; mejor como frase. |
| `00_referencias_distribucion.png` | baselines | Distribución en test de aleatorio, B&H, no operar y reglas. | 2 | ancho completo | Redundante con la tabla. |
| `00_senal_por_decil.png` | diseño | Retorno siguiente por decil de tendencia: hay señal en 1 y 3; en 2 y 4, dentro de la comisión. | 4 | ancho completo (6 cm) | La mejor justificación de «qué estrategia se espera». Alternativa a `cand_mercados_a`. |
| `01_dos_curvas.png` | convergencia | DQN: curva con exploración frente a la greedy, y episodio de convergencia. | 3 | ancho completo (6 cm) | Buena, pero falta el par con A3C a la misma escala. |
| `01_episodios.png` | política | Un episodio por mercado con la posición del DQN y su balance frente a B&H. | 2 | ancho completo (8 cm) | Alta; ejes log ilegibles en el 1. |
| `01_politica.png` | política | Q(operar) − Q(mantener) según la tendencia, 5 semillas. | 3 | ancho completo (8 cm) | Sustituida por `cand_politica_b`. |
| `02_dos_curvas.png` | convergencia | A3C: curva con exploración frente a la greedy. | 3 | ancho completo (6 cm) | La banda del mercado 1 tapa el panel (una semilla no opera). |
| `02_episodios.png` | política | Un episodio por mercado con A3C. | 2 | ancho completo (8 cm) | Igual que `01_episodios`. |
| `02_politica.png` | política | P(operar) según la tendencia, 5 semillas. | 3 | ancho completo (8 cm) | Sustituida por `cand_politica_b`. |
| `03_arquitecturas.png` | extras | Double, Dueling, sin red objetivo y clip: ventaja emparejada (ninguna mejora clara). | 2 | media columna | Incluida en `04_p3_ventaja_emparejada`; nombres internos (`sin_objetivo`). |
| `03_comisiones.png` | extras | Log del capital según la comisión, por mercado. | 2 | media columna (2×2) | `04_p3_comisiones…` y `cand_extras_b` lo cuentan mejor. |
| `03_estados.png` | extras | Ventana, MA30/volatilidad y placebo: ventaja emparejada. | 3 | media columna | El mensaje clave del 03, pero con nombres internos y una escala de color que domina el placebo. |
| `03_objetivo_mezcla.png` | extras | Recompensa log y mezcla de mercados. | 2 | media columna | Secundaria. |
| `03_sp500.png` | extras | S&P 500 2024-25: transferencia y entrenamiento real por debajo de B&H. | 3 | media columna | Buena para el análisis crítico, pero `cand_extras_b` da la cifra en menos espacio. |
| `04_p1_mejoras.png` | DQN | Mejoras de DQN en márgenes de ruido: ninguna supera el margen salvo el mejor punto en series nuevas. | 3 | ancho completo (7 cm) | Detallada; letra pequeña. |
| `04_p1_sensibilidad.png` | hiperparámetros | Sensibilidad de DQN en validación (13 variantes): ninguna pasa de +1. | 3 | media columna (10 cm) | Alta; sustituida por `cand_hiperparametros_a/b`. |
| `04_p2_mejor_punto.png` | A3C | Mejor punto en validación frente al final (400 series nuevas): A3C +15,8 % → +30,1 % en el 3. | 3 | ancho completo (5 cm) | Buen mensaje de «estabilidad, no capacidad»; si cabe, para §4. |
| `04_p2_sensibilidad.png` | hiperparámetros | Sensibilidad de A3C por etapas (7 heatmaps). | 3 | ancho completo (4,4 cm) | Compacta, pero la letra queda en ~4 pt a 19 cm. |
| `04_p3_comisiones_y_mercados_extra.png` | extras | Operaciones según la comisión y mercados extra + S&P 500. | 4 | ancho completo (6,2 cm) | Válida tal cual; `cand_extras_b` es más baja. |
| `04_p3_ventaja_emparejada.png` | extras | Las 12 variantes del 03 frente a la referencia, en márgenes de ruido: el estado pesa más que la arquitectura. | 3 | media columna (14 cm) | Completa pero muy alta; sustituida por `cand_extras_a`. |
| `04_p4_comparativa_mercados.png` | rendimiento | Test: 9 agentes por mercado, con semillas. | 4 | ancho completo (6,4 cm) | Sustituida por `cand_rendimiento_a` (más baja y con letra mayor). |
| `04_p4_convergencia.png` | convergencia | Episodios por semilla y tiempos: A3C 3,4 veces más rápido en total, pero no llega antes. | 4 | ancho completo (6 cm) | Sustituida por `cand_convergencia_a`. |
