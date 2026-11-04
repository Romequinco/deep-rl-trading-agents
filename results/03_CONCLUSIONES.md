# 03 · Resultados y conclusiones

Experimentos posteriores al test oficial, con cinco semillas por condición y una
sola modificación respecto a la referencia. Los resultados de 01 y 02 no cambian.
Se utilizan 400 series sintéticas desde 600.000; las parejas comparten precios y
semillas. Los márgenes son descriptivos, no contrastes de significación.

### La representación aporta más que las variantes de arquitectura

Mediana del retorno entre cinco entrenamientos, medida sobre las mismas series:

| Mercado | DQN base | + 20 retornos | + MA30 y volatilidad |
|---|---:|---:|---:|
| Tendencia y ciclo | x348,4 | x547,9 | x244,9 |
| Browniano | −5,8 % | −11,3 % | −6,5 % |
| Reversión | +16,8 % | +30,7 % | +47,9 % |
| Regímenes | −7,0 % | −3,8 % | +7,6 % |

La ventana supera el margen emparejado en tendencia y reversión: ventajas log
de +0,398 y +0,094, con 5/5 y 4/5 semillas que mejoran. MA30 y volatilidad lo
superan en reversión y regímenes: +0,246 y +0,127, con 5/5 en ambos.
Las dos representaciones empeoran el browniano y MA30/volatilidad empeora la
tendencia. El resultado favorece adaptar la información al problema; añadir
variables no garantiza mejorar. También aumenta el número de pesos de entrada,
y este experimento no separa el aporte de MA30 del de volatilidad.

Double DQN supera el margen solo en reversión (+0,025, 3/5 semillas), con menos
consistencia que las ampliaciones del estado. Duelo, quitar la red objetivo y
recortar la recompensa no superan el margen en ningún mercado. El clipping
perjudica a las cinco semillas de tendencia. Esto describe esta configuración
y este presupuesto, no la utilidad general de esos métodos.

### Recompensa, placebo, mezcla y comisiones

- La recompensa logarítmica no supera el margen en ningún mercado. En tendencia
  queda cerca (+0,219 frente a log(1,25)=0,223, con 4/5 semillas que mejoran).
  No se cambia el umbral después de verlo. Que suba la mediana agregada no
  implica que mejore la comparación emparejada: la tabla muestra ambas.
- Al anular la tendencia, el agente compra y mantiene en el mercado alcista
  (x26,6) y la mediana queda en 0 % en los otros tres. Puede ganar por la deriva
  sin predecir precios. En reversión pierde la capacidad útil de la referencia.
- Mezclar los cuatro generadores con el mismo presupuesto total no mejora a la
  especialización: en reversión cae de +16,8 % a +0,1 %, y en regímenes de −7,0 %
  a −16,1 %. Cada mercado aporta 125 episodios en vez de 500. No se puede inferir
  que una política general sea imposible con otro estado o más entrenamiento.
  Se comprobó que las réplicas de una misma semilla de mezcla tienen los mismos
  pesos al evaluar los cuatro mercados: no hay adaptación oculta al mercado.
- El agente adapta su actividad al coste. En reversión pasa de 69,2 operaciones
  con comisión 0 %, a 25,9 con 0,1 % y a 1,0 con 0,5 %; el retorno mediano pasa
  de +28,2 % a +16,8 % y +2,1 %. En browniano y regímenes, con el coste alto la
  actividad es prácticamente nula. Quitar costes cambia también el aprendizaje:
  no se espera una curva de rentabilidad monótona entre políticas entrenadas por
  separado. La comisión es un cambio en la tarea, no una mejora del algoritmo.

### Mercados adicionales e histórico real

En tendencia persistente, DQN obtiene +101,5 % frente a +17,5 % de buy & hold;
el rango de las medianas entre semillas es +81,0 % a +106,9 %. En saltos queda
en −0,2 %, frente a −22,7 % de buy & hold y 0 % de no operar. La estrategia
se beneficia de tendencias observables y casi deja de operar cuando dominan
los shocks independientes; todavía queda por debajo de no operar en ese caso.

En el S&P 500, entrenar con 2017–2021 y evaluar en 2024–2025 da una mediana
de +12,1 % y 75 operaciones, frente a +43,8 % y una operación de buy & hold.
La transferencia desde entrenamiento browniano da +34,8 % de mediana, con
dispersión entre 0 % y +44,3 %. No hay evidencia de que entrenar con el histórico
real produzca una estrategia mejor en este periodo. La ventaja emparejada entre
inicializaciones y las medianas agregadas pueden tener signos distintos, por lo
que no se presenta una de ellas como prueba aislada de superioridad.

El histórico es una trayectoria única, ya disponible en el Taller 1. Cinco
inicializaciones no son cinco pruebas históricas independientes. El análisis
es retrospectivo, conserva la partición temporal y la normalización solo de
train, y no permite extrapolar a inversión real.

### A2C frente a A3C

Se mantienen la red, los parámetros de 02 y el presupuesto de 504 episodios
efectivos. Solo cambia la variante de actualización, sin reajustar A2C.

| Mercado | A3C | A2C | Ventaja log de A2C | Semillas que mejoran |
|---|---:|---:|---:|---:|
| Tendencia y ciclo | x500,3 | x509,1 | +0,085 | 3/5 |
| Browniano | +0,0% | -6,1% | -0,047 | 1/5 |
| Reversión | +14,7% | +4,6% | +0,000 | 2/5 |
| Regímenes | +0,0% | +0,0% | +0,000 | 1/5 |

A2C no supera el margen predefinido en ningún mercado. En tendencia las
medianas son próximas; en reversión su mediana es menor, aunque su mejor
semilla es mejor. No se elige una ejecución aislada como representante.

Los parámetros proceden de la referencia A3C; no se ha buscado una
configuración óptima propia para A2C. El resultado responde a esta comparación
controlada, no a una clasificación universal de algoritmos.

A2C agrupa tres entornos en un proceso y A3C utiliza tres procesos. Los
tiempos comparan estas implementaciones y no un coste intrínseco con igual
uso de CPU. Se muestran en el notebook junto al número de semillas que
cumplen el criterio de convergencia. Alcanzar el 90 % del nivel final no
demuestra que la política se haya estabilizado.

### Alcance de las conclusiones

Se completan las ideas de partida sin combinar cambios ni reajustar tras ver
las nuevas series. Las tablas conservan el detalle por semilla y los CSV por
serie permiten auditar las parejas. La variación entre semillas y las múltiples
comparaciones impiden interpretar un asterisco del margen como significación
estadística. Los tiempos DQN con varios entrenamientos simultáneos no comparan
la velocidad de las variantes. Los de actor-crítico se miden secuencialmente en
el mismo equipo, excluyendo evaluación y, en A3C, arranque de procesos.

La verificación incluye 63 pruebas sin fallos ni errores y una omisión por falta
del material bruto del enunciado. Los controles cubren equivalencia de la base,
causalidad de los indicadores, cuentas de comisión, objetivos Double DQN,
presupuesto de A2C y evaluación en lote. Los cuatro generadores y el entorno
original permanecen intactos.
