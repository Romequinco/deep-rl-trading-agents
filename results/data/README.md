# Serie real para el notebook 03

- **Serie:** SP500, cierre diario del índice S&P 500, sin dividendos.
- **Fuente:** FRED / S&P Dow Jones Indices, <https://fred.stlouisfed.org/series/SP500>.
- **Descarga:** 2026-10-28; periodo fijo 2017-01-03 a 2025-12-31 (la misma copia que en el Taller 1).
- **Ficheros:** CSV normalizado con 2 262 sesiones y JSON con URL de descarga y SHA-256.
- **Derechos:** la fuente atribuye los datos a S&P Dow Jones Indices y publica sus condiciones en las notas de la
  serie. La copia local documenta el experimento académico; no se declara que la serie sea de dominio público.

Partición del Taller 1: train 2017–2021 (1 259 sesiones), validación 2022–2023 (501) y test 2024–2025 (502). El índice
no es directamente negociable, y la simulación no incluye dividendos, spread, deslizamiento ni intereses del efectivo.
