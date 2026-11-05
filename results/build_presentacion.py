"""Genera results/presentacion/presentacion.pdf (16:9, apaisado) a partir de presentacion/slides/*.html.

Cada fichero es una diapositiva: un fragmento HTML que usa las clases de `CSS`. Se juntan en orden
alfabético, se numeran y se imprimen a PDF con Chrome sin interfaz (mismo sistema que build_informe.py).
Con --png guarda además cada diapositiva como imagen en presentacion/png/ para revisarla.

Uso: python results/build_presentacion.py [--png] [--only PREFIJO]
  --only 03 genera solo las diapositivas cuyo nombre empieza por 03, en ficheros preview_03 aparte.
"""
from pathlib import Path
import subprocess
import sys

import fitz  # PyMuPDF

from build_informe import find_chrome

ROOT = Path(__file__).resolve().parent
DIR = ROOT / "presentacion"
SLIDES, HTML, PDF, PNG = DIR / "slides", DIR / "presentacion.html", DIR / "presentacion.pdf", DIR / "png"

CSS = """
@page { size: 254mm 142.875mm; margin: 0; }
* { box-sizing: border-box; }
html, body { margin: 0; }
body { font-family: Cambria, Georgia, "Times New Roman", serif; color: #111; }
.slide { width: 254mm; height: 142.875mm; padding: 9mm 12mm 8mm; position: relative; overflow: hidden;
         break-after: page; display: flex; flex-direction: column; font-size: 13pt; line-height: 1.3; }
.slide h1 { font-size: 21pt; margin: 0 0 1.5mm; font-weight: bold; color: #102a43; }
.slide .kicker { font-size: 10.5pt; color: #2B6CB0; letter-spacing: .04em; text-transform: uppercase; margin-bottom: 1mm; }
.slide .lead { font-size: 13pt; color: #333; margin: 0 0 3mm; }
.slide p { margin: 0 0 2mm; }
.slide ul { margin: 0 0 2mm; padding-left: 6mm; }
.slide li { margin: 0 0 1.2mm; }
.slide b, .slide strong { color: #102a43; }
.body { flex: 1; min-height: 0; display: flex; gap: 6mm; }
.body > .col { flex: 1; min-width: 0; display: flex; flex-direction: column; justify-content: center; }
.body > .wide { flex: 1.6; }
.body > .narrow { flex: 0.8; }
.fig { width: 100%; max-height: 100%; object-fit: contain; display: block; }
.cap { font-size: 10pt; color: #555; margin: 1mm 0 0; }
.big { font-size: 30pt; font-weight: bold; color: #2B6CB0; line-height: 1; }
.bigl { font-size: 10.5pt; color: #555; margin: 0.5mm 0 3mm; }
.verdict { border-left: 2.2mm solid #2F9E6E; padding: 1.5mm 3mm; margin-top: 3mm; font-size: 13pt; background: #f1f8f4; }
.note { font-size: 10.5pt; color: #555; }
table { border-collapse: collapse; width: 100%; font-size: 11pt; border-top: 1pt solid #000; border-bottom: 1pt solid #000; }
thead th { border-bottom: 0.6pt solid #000; font-weight: normal; }
th, td { padding: 1mm 2mm; text-align: right; }
th:first-child, td:first-child { text-align: left; }
.num { position: absolute; right: 12mm; bottom: 4mm; font-size: 9pt; color: #888; }
.foot { position: absolute; left: 12mm; bottom: 4mm; font-size: 9pt; color: #888; }
.title { justify-content: center; }
.title h1 { font-size: 30pt; }
.annex .kicker { color: #888; }

.kpis { display: flex; gap: 3mm; margin: 1mm 0 2.5mm; }
.kpi { flex: 1; border-top: 1.2mm solid #2B6CB0; background: #f5f8fc; padding: 1.5mm 3mm 1.8mm; }
.kpi .mk { font-size: 9.5pt; font-weight: bold; letter-spacing: .02em; }
.kpi .v { font-size: 23pt; font-weight: bold; line-height: 1.05; color: #102a43; }
.kpi .s { font-size: 9.5pt; color: #555; line-height: 1.2; }
.box { border: 0.8pt solid #9fb3c8; border-top: 1.2mm solid #2B6CB0; padding: 1.5mm 3mm; font-size: 11pt; line-height: 1.25; }
.box .kicker { font-size: 9.5pt; }
.m1 { color: #6c5ce7; } .m2 { color: #1a8fe3; } .m3 { color: #00a383; } .m4 { color: #e03131; }
.small { font-size: 11pt; line-height: 1.28; }
.small li { margin-bottom: 0.9mm; }
"""

FOOT = "Deep RL para trading · Romero, Dapena y García · MIAX · github.com/Romequinco/deep-rl-trading-agents"


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    files = sorted(f for f in SLIDES.glob("*.html") if only is None or f.name.startswith(only))
    html, pdf, tag = (HTML, PDF, "diapo") if only is None else (
        DIR / f"preview_{only}.html", DIR / f"preview_{only}.pdf", f"preview_{only}")
    if not files:
        raise SystemExit("No hay diapositivas en presentacion/slides/")
    parts = []
    for i, f in enumerate(files, 1):
        body = f.read_text(encoding="utf-8")
        footer = "" if i == 1 else f'<div class="foot">{FOOT}</div><div class="num">{i}</div>'
        parts.append(body.replace("</section>", footer + "</section>", 1) if "</section>" in body else body)
    html.write_text('<!doctype html><html lang="es"><head><meta charset="utf-8">'
                    "<title>Agentes de Deep RL para trading de un activo</title>"
                    f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>", encoding="utf-8")
    subprocess.run([find_chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=True, capture_output=True, timeout=120)
    doc = fitz.open(pdf)
    if "--png" in sys.argv:
        PNG.mkdir(exist_ok=True)
        for old in PNG.glob(f"{tag}_*.png"):
            old.unlink()
        for i, page in enumerate(doc, 1):
            page.get_pixmap(dpi=80).save(PNG / f"{tag}_{i:02d}.png")
    print(f"{pdf.name}: {doc.page_count} diapositivas ({len(files)} ficheros)")
    if doc.page_count != len(files):
        raise SystemExit("Alguna diapositiva se desborda: hay más páginas que ficheros")


if __name__ == "__main__":
    main()
