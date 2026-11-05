"""Genera results/informe.pdf a partir de results/informe.md (D-0.5).

Markdown -> HTML con estilos de impresión A4 -> PDF con Chrome sin interfaz. Falla si el PDF
supera 2 páginas. Con --png guarda además cada página como imagen para revisarla.

Uso: python results/build_informe.py [--png]
"""
from pathlib import Path
import shutil
import subprocess
import sys

import fitz  # PyMuPDF
import markdown

RESULTS = Path(__file__).resolve().parent
SRC, HTML, PDF = RESULTS / "informe.md", RESULTS / "informe.html", RESULTS / "informe.pdf"
MAX_PAGES = 2
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chrome",
]

CSS = """
@page { size: A4; margin: 5.5mm 9.5mm 5mm; }
* { box-sizing: border-box; }
body { font-family: Cambria, Georgia, "Times New Roman", serif; font-size: 8.5pt; line-height: 1.17; color: #111; margin: 0; orphans: 2; widows: 2; }
h1 { font-size: 14.5pt; font-weight: bold; margin: 0 0 0.6mm; text-align: center; }
h2 { break-after: avoid; font-size: 10pt; font-weight: bold; margin: 1.8mm 0 0.7mm; }
p, li { text-align: justify; hyphens: auto; }
p { margin: 0 0 1mm; text-indent: 0; }
ul, ol { margin: 0 0 1mm; padding-left: 4mm; }
ol { padding-left: 5mm; }
img { width: 100%; display: block; break-inside: avoid; margin: 0 0 0.4mm; }
table { border-collapse: collapse; width: 100%; font-size: 8.2pt; margin: 0 0 1.4mm; border-top: 0.9pt solid #000; border-bottom: 0.9pt solid #000; }
thead th { border-bottom: 0.5pt solid #000; font-weight: normal; }
th, td { padding: 0.5mm 1mm; text-align: right; white-space: nowrap; }
th { white-space: normal; line-height: 1.12; vertical-align: bottom; }
th:first-child, td:first-child { text-align: left; }
.sub { font-size: 8.5pt; margin-bottom: 1.2mm; text-align: center; }
.cap { font-size: 7.8pt; line-height: 1.2; margin: 0 0 1mm; }
.cap b { font-weight: bold; }
.tcap { margin: 0.6mm 0 0.6mm; }
.side { display: grid; grid-template-columns: 66fr 34fr; gap: 3.5mm; align-items: center; margin: 0.4mm 0 0.6mm; break-inside: avoid; }
.side2 { grid-template-columns: 67fr 33fr; }
.side3 { grid-template-columns: 71fr 29fr; }
.side4 { grid-template-columns: 71fr 29fr; }
.side2, .side3, .side4 { margin: 2.2mm 0 2.2mm; }
.side2 p:not(.cap) { margin: 0; }
.side .cap { margin-bottom: 0.8mm; }
.pol p { margin: 0 0 0.9mm; font-size: 8.6pt; line-height: 1.22; }
.side.w62 { grid-template-columns: 62fr 38fr; }
.side.w55 { grid-template-columns: 55fr 45fr; }
.side.w50 { grid-template-columns: 1fr 1fr; }
.side.top { align-items: start; }
.banner { margin: 0 0 1mm; }
.side.s1, .side.s2, .side.s3, .side.s4, .side.s5, .side.s6, .side.s7 { align-items: start; margin: 0.6mm 0 1.2mm; }
.s1 { grid-template-columns: 56fr 44fr; } .s2 { grid-template-columns: 56fr 44fr; } .s3 { grid-template-columns: 48fr 52fr; }
.s4 { grid-template-columns: 56fr 44fr; } .s5 { grid-template-columns: 50fr 50fr; } .s6 { grid-template-columns: 50fr 50fr; }
.s7 { grid-template-columns: 42fr 58fr; }
.s6 img.small { width: 80%; margin: 0 auto; }
.side ul { margin-bottom: 0; }
.side p:last-child { margin-bottom: 0; }
.repo { text-align: center; font-size: 8.4pt; margin: 1.5mm 0 0; }
.repo a { color: #1a4fb5; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 5mm; }
.pagebreak { break-after: page; }
"""


def find_chrome():
    for c in CHROME_CANDIDATES:
        path = c if Path(c).exists() else shutil.which(c)
        if path:
            return path
    raise FileNotFoundError("No se encuentra Chrome/Edge para imprimir el PDF")


def main():
    body = markdown.markdown(SRC.read_text(encoding="utf-8"), extensions=["tables", "md_in_html", "attr_list"])
    HTML.write_text(f'<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Agentes de Deep RL para trading de un activo</title>'
                    f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    subprocess.run([find_chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, capture_output=True, timeout=120)
    doc = fitz.open(PDF)
    pages = doc.page_count
    if "--png" in sys.argv:
        for i, page in enumerate(doc):
            page.get_pixmap(dpi=110).save(RESULTS / f"informe_p{i + 1}.png")
    doc.close()
    print(f"{PDF.name}: {pages} páginas")
    if pages > MAX_PAGES:
        raise SystemExit(f"El informe ocupa {pages} páginas; el máximo es {MAX_PAGES}")


if __name__ == "__main__":
    main()
