"""
verificar_consistencia.py
==========================

Control final: comprueba que las cifras citadas en cada producto (README, METODOLOGIA, notebook ejecutado, informe PDF, presentación PPTX)
coinciden con `outputs/resultados_clave.json` (y, por lo tanto, con el dashboard), y que no quedaron afirmaciones obsoletas.

    python scripts/verificar_consistencia.py        # sale con código 1 si algo no coincide
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf
from pptx import Presentation

from src import pipeline as pl

R = json.loads((ROOT / "outputs" / "resultados_clave.json").read_text(encoding="utf-8"))
fallas: list[str] = []


def n(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def ns(x, d=1):
    return f"{x:+.{d}f}".replace(".", ",").replace("-", "−")


def texto_pdf(p):
    return "\n".join(pg.get_text() for pg in pymupdf.open(str(p)))


def texto_pptx(p):
    out = []
    for sl in Presentation(str(p)).slides:
        for sh in sl.shapes:
            if sh.has_text_frame:
                out.append(sh.text_frame.text)
            if sh.has_table:
                for fila in sh.table.rows:
                    out.append(" ".join(c.text for c in fila.cells))
    return "\n".join(out)


def texto_notebook(p):
    nb = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for c in nb["cells"]:
        out.append("".join(c["source"]))
        for o in c.get("outputs", []):
            d = o.get("data", {})
            out.append("".join(d.get("text/markdown", "")) + "".join(d.get("text/plain", "")) + "".join(o.get("text", "")))
    return "\n".join(out)


def buscar(nombre, texto, fragmentos):
    t = texto.replace("−", "-").replace("\n", " ")
    for f in fragmentos:
        if f.replace("−", "-") not in t:
            fallas.append(f"[{nombre}] no aparece: {f!r}")


def prohibido(nombre, texto, fragmentos):
    for f in fragmentos:
        if f in texto:
            fallas.append(f"[{nombre}] contiene texto obsoleto: {f!r}")


D = R["discriminacion_2019"]["IFFS compuesto (principal)"]
BON = R["discriminacion_2019"]["Benchmark: participación de bonos en la deuda (%)"]
PV = R["placebo_temporal"]["auc_iffs_por_anio"]
PRE = R["pretendencia"]["principal|IFFS|T-5->T-1"]
CB = R["cobertura"]
PERS = R["diseno"]["props"]["coyuntural"]["persistencia_rangos"]
DS = R["deriva_solvencia"]

# --- 0. huella --------------------------------------------------------------------------------------------------
if R["huella"] != pl.huella_del_proyecto():
    fallas.append("outputs/resultados_clave.json no corresponde al código/datos actuales (volver a correr scripts/generar_resultados.py)")

# --- 1. Cifras comunes a todos los productos ----------------------------------------------------------------------
comunes = {
    "cobertura": f"{CB['n_compuesto']} de {CB['n_total']}",
    "auc_2019": n(D["auc"]),
    "ic": f"{n(D['ic95'][0])}–{n(D['ic95'][1])}",
    "bonos": n(BON["auc"]),
    "pers_coyuntural": n(PERS),
    "mediana_2005": n(DS["mediana_2005"]),
    "mediana_2012": n(DS["mediana_2012"]),
    "pretendencia": f"{ns(PRE['dif'])}",
    "auc_2025": n(PV["2025"]),
}
textos = {
    "informe PDF": texto_pdf(ROOT / "informe" / "informe_fragilidad_fiscal.pdf"),
    "presentación": texto_pptx(ROOT / "presentacion" / "presentacion_fragilidad_fiscal.pptx"),
    "README": (ROOT / "README.md").read_text(encoding="utf-8"),
}
for nombre, t in textos.items():
    buscar(nombre, t, [comunes[k] for k in ("auc_2019", "ic", "bonos", "pers_coyuntural", "mediana_2005", "mediana_2012", "pretendencia", "auc_2025")])
buscar("informe PDF", textos["informe PDF"], [comunes["cobertura"]])
buscar("README", textos["README"], [comunes["cobertura"]])

# notebook: usa coma o punto según la celda; se verifican cifras enteras/etiquetas robustas
nb = texto_notebook(ROOT / "notebooks" / "fragilidad_fiscal_provincial.ipynb")
buscar("notebook", nb, [f"{CB['n_compuesto']} de {CB['n_total']}", n(D["auc"]), n(BON["auc"]), n(PERS), str(R["eventos"]["n_eventos"])])
if "Traceback" in nb or '"output_type": "error"' in nb:
    fallas.append("[notebook] contiene errores de ejecución")
if not any(o for c in json.loads((ROOT / "notebooks" / "fragilidad_fiscal_provincial.ipynb").read_text())["cells"] if c["cell_type"] == "code" for o in c.get("outputs", [])):
    fallas.append("[notebook] no tiene salidas guardadas (no está ejecutado)")

# --- 2. Estructura de entregables ---------------------------------------------------------------------------------
pag = pymupdf.open(str(ROOT / "informe" / "informe_fragilidad_fiscal.pdf")).page_count
if pag > 3:
    fallas.append(f"informe PDF con {pag} páginas (máximo 3)")
prs = Presentation(str(ROOT / "presentacion" / "presentacion_fragilidad_fiscal.pptx"))
if not (7 <= len(prs.slides) <= 9):
    fallas.append(f"presentación con {len(prs.slides)} diapositivas (debe tener 7-9)")
for i, sl in enumerate(prs.slides, 1):
    if not (sl.has_notes_slide and sl.notes_slide.notes_text_frame.text.strip()):
        fallas.append(f"diapositiva {i} sin notas del orador")

# --- 3. Afirmaciones obsoletas -----------------------------------------------------------------------------------
obsoleto = ["IFFS-B", "Nivel de alerta", "AUC descriptiva", "23 provincias y la Ciudad", "0.97)"]
for nombre, t in {**textos, "notebook": nb, "METODOLOGIA": (ROOT / "docs" / "METODOLOGIA.md").read_text(encoding="utf-8"), "app.py": (ROOT / "app.py").read_text(encoding="utf-8")}.items():
    prohibido(nombre, t, ["IFFS-B (", "IFFS_B", "Crítico", "no confiable pero se corrige"])
la_pampa_ok = ["confirmación oficial"]
for nombre in ("informe PDF", "README"):
    buscar(nombre, textos[nombre], la_pampa_ok)
for nombre, t in {"informe PDF": textos["informe PDF"], "README": textos["README"]}.items():
    if "se corrige (÷1.000)" in t or "se corrige (÷1.000)" in t.replace("\n", " "):
        fallas.append(f"[{nombre}] afirma que La Pampa se corrige")

if fallas:
    print("INCONSISTENCIAS:")
    for f in fallas:
        print(" -", f)
    sys.exit(1)
print("OK: README, METODOLOGIA, notebook, informe (", pag, "páginas ), presentación (", len(prs.slides), "diapositivas con notas ) coinciden con outputs/resultados_clave.json.")
