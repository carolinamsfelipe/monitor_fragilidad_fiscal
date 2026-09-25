"""
generar_entregables.py
=======================

Genera, a partir de `outputs/resultados_clave.json` y de las figuras de `outputs/figures/`:

  * el bloque de resultados del README,
  * `informe/informe_fragilidad_fiscal.pdf`      (máximo estricto: 3 páginas; el script FALLA si se pasa),
  * `presentacion/presentacion_fragilidad_fiscal.pptx`  (9 diapositivas para 10-15 minutos, con notas del orador).

Requiere haber corrido antes `scripts/generar_resultados.py`. Uso:  python scripts/generar_entregables.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src import pipeline as pl

FIG = ROOT / "outputs" / "figures"
R = json.loads((ROOT / "outputs" / "resultados_clave.json").read_text(encoding="utf-8"))
if R["huella"] != pl.huella_del_proyecto():
    sys.exit("outputs/ no corresponde al código/datos actuales: ejecutar scripts/generar_resultados.py")


def n(x, d=2):
    """Número con coma decimal."""
    return f"{x:.{d}f}".replace(".", ",")


def ns(x, d=1):
    """Número con signo y coma decimal (signo menos tipográfico)."""
    return f"{x:+.{d}f}".replace(".", ",").replace("-", "−")


def mil(x):
    return f"{x:,}".replace(",", ".")


D = R["discriminacion_2019"]
K = D["IFFS compuesto (principal)"]
BON = D["Benchmark: participación de bonos en la deuda (%)"]
PV = R["placebo_temporal"]
PRE = R["pretendencia"]["principal|IFFS|T-5->T-1"]
PRE_S = R["pretendencia"]["principal|S_solvencia|T-5->T-1"]
ES8, ES1 = R["estudio_eventos"]["-8"], R["estudio_eventos"]["-1"]
RD = R["diseno"]
RB = R["robustez_diseno"]
PRINC = "PRINCIPAL: corte anual · media simple de solvencia, liquidez y estructural"
RHOS = [v["spearman"] for k, v in RB.items() if k != PRINC]
W = R["pesos_aleatorios"]
COND = R["condicionado_a_bonos_2019"]
LP = R["la_pampa"]
AZ = R["azar_auc_2019"]
CB = R["cobertura"]
EVN = R["eventos"]
LR = R["la_rioja"]
BT = R["backtest_proyeccion"]
AUC_SP = R["auc_2019_especificaciones"]
PERS_COY = RD["props"]["coyuntural"]["persistencia_rangos"]
ICC_R, ICC_C = RD["props"]["estructural"]["icc"], RD["props"]["coyuntural"]["icc"]
AR1_C = RD["props"]["coyuntural"]["ar1"]
DERIVA_V1 = RB["Normalización: base fija 2005-2012 (legado)"]["deriva_mediana"]
DERIVA_V2 = RB[PRINC]["deriva_mediana"]
EF = LP["efecto_sobre_resto"]
DS = R["deriva_solvencia"]
A_2012 = PV["primer_anio_auc_ge_085"]
SU = R["servicio_unidades"]


# ======================================================================================================================
# 1. README
# ======================================================================================================================
def bloque_readme() -> str:
    return f"""
| Qué | Resultado | Tipo |
|---|---|---|
| Cobertura del compuesto | **{CB['n_compuesto']} de {CB['n_total']}** provincia-año ({n(CB['pct_compuesto'], 1)}%); faltan La Pampa 2005-2017 (stock no confiable) y 2025 (sin ejecución) | Evidencia |
| Diseño | 4 subíndices; compuesto = solvencia + liquidez + estructural. El resultado fiscal queda afuera (persistencia de rangos {n(PERS_COY)} < 0,85) | Evidencia + diseño |
| Normalización | Percentil dentro de cada año. La mediana de Deuda/Ingresos cae de {n(DS['mediana_2005'])} (2005) a {n(DS['mediana_2012'])} (2012) y {n(DS['mediana_2024'])} (2024): la base fija 2005-2012 no es estacionaria | Evidencia |
| Robustez | ρ de Spearman con el principal entre {n(min(RHOS))} y {n(max(RHOS))} frente a normalización, ventanas, pesos y agregación; con {mil(W['n_sim'])} pesos aleatorios, mediana {n(W['spearman_medio_mediana'])} | Evidencia |
| Discriminación (2019) | AUC **{n(K['auc'])}** [IC 95%: {n(K['ic95'][0])}–{n(K['ic95'][1])}] para separar a las 12 provincias con evento 2020-21 de las otras 12; el azar supera {n(AZ['azar_p95'])} solo el 5% de las veces | Evidencia |
| …pero | El % de bonos en la deuda logra AUC {n(BON['auc'])} [{n(BON['ic95'][0])}–{n(BON['ic95'][1])}]. El grupo ya se separaba en {A_2012} (AUC {n(PV['auc_iffs_por_anio'][str(A_2012)])}) y sigue separado en 2025 (AUC {n(PV['auc_iffs_por_anio']['2025'])}) | Evidencia → interpretación: rasgo persistente, no anticipación |
| Anticipación | Cambio del compuesto T−5→T−1: {ns(PRE['dif'])} puntos [{ns(PRE['ic95'][0])}; {ns(PRE['ic95'][1])}] entre provincias con evento y resto (p = {n(PRE['p'])}): sin evidencia de deterioro previo distinto | Evidencia + limitación (potencia baja) |
| Predicción | No viable: 0 eventos antes de 2020, {EVN['n_eventos']} en total, {EVN['n_covid']} simultáneos, {EVN['n_anio_ambiguo']} con año ancla convencional | Limitación |
""".strip()


def actualizar_readme() -> None:
    f = ROOT / "README.md"
    s = f.read_text(encoding="utf-8")
    s = re.sub(r"<!-- RESULTADOS:INICIO -->.*<!-- RESULTADOS:FIN -->", lambda m: f"<!-- RESULTADOS:INICIO -->\n{bloque_readme()}\n<!-- RESULTADOS:FIN -->", s, flags=re.S)
    f.write_text(s, encoding="utf-8")
    print("README actualizado.")


# ======================================================================================================================
# 2. INFORME PDF (máximo 3 páginas)
# ======================================================================================================================
def pdf() -> None:
    fdir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DV", str(fdir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DV-B", str(fdir / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DV-I", str(fdir / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")
    AZUL = colors.HexColor("#1A3C6E")
    body = ParagraphStyle("b", fontName="DV", fontSize=8.5, leading=11.1, spaceAfter=3)
    h1 = ParagraphStyle("h1", fontName="DV-B", fontSize=15, leading=18, textColor=AZUL, spaceAfter=1)
    sub = ParagraphStyle("sub", fontName="DV", fontSize=8, leading=10, textColor=colors.HexColor("#52514e"), spaceAfter=5)
    h2 = ParagraphStyle("h2", fontName="DV-B", fontSize=9.6, leading=12, textColor=AZUL, spaceBefore=4, spaceAfter=2)
    cap = ParagraphStyle("cap", fontName="DV", fontSize=7, leading=8.8, textColor=colors.HexColor("#52514e"), spaceAfter=3)
    cell = ParagraphStyle("c", fontName="DV", fontSize=7.6, leading=9.4)
    cellb = ParagraphStyle("cb", parent=cell, fontName="DV-B")

    def P(t, st=body):
        return Paragraph(t, st)

    def img(nombre, ancho_cm):
        from PIL import Image as PILImage

        w, h = PILImage.open(FIG / f"{nombre}.png").size
        return Image(str(FIG / f"{nombre}.png"), width=ancho_cm * cm, height=ancho_cm * cm * h / w)

    def dos(a, b, wa, wb):
        t = Table([[img(a, wa), img(b, wb)]], colWidths=[(wa + 0.15) * cm, (wb + 0.15) * cm])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
        return t

    def tabla(filas, anchos, cabecera=True):
        data = [[Paragraph(c, cellb if (cabecera and i == 0) else cell) for c in fila] for i, fila in enumerate(filas)]
        t = Table(data, colWidths=[a * cm for a in anchos], repeatRows=1)
        est = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c3c2b7")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
               ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
        if cabecera:
            est.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eefb")))
        t.setStyle(TableStyle(est))
        return t

    S = []
    S += [P("Monitor de Fragilidad Fiscal Provincial — IFFS v2", h1),
          P("Programación para Economistas · Segundo Parcial · UADE · septiembre de 2026 · Panel de 24 jurisdicciones, 2005-2025", sub)]

    S += [P("1. Problema, alcance y advertencia", h2),
          P(f"Ante el shock del COVID-19, <b>12 de las 24 jurisdicciones reestructuraron deuda de mercado en 2020-21</b> y La Rioja incumplió en 2025. No existe una herramienta pública que compare la fragilidad "
            "fiscal <i>relativa</i> de las provincias con datos oficiales y método transparente. El IFFS v2 ordena a cada jurisdicción frente a sus pares (percentil dentro de cada año). "
            "<b>Es un score relativo: no es una probabilidad de default</b> ni un sistema de alerta calibrado. Cada afirmación se rotula: <b>evidencia</b> (calculada), <i>interpretación</i> (económica), "
            "<b>supuesto</b>, <b>diseño</b> (decisión de construcción) o <b>limitación</b>.")]

    S += [P("2. Datos y auditoría: qué se corrigió y qué se dejó marcado", h2),
          P(f"Fuentes: DNAP (ejecución, stock y servicio de deuda) y BCRA; paneles derivados de una investigación previa (no se reprocesan los Excel originales). "
            f"<b>CABA (evidencia):</b> el servicio de deuda estaba multiplicado por 1.000 (se leyó la hoja en miles como si fuera en millones); se corrigió leyendo la hoja <i>Millones</i>, con un control que falla si el cociente entre hojas no es exactamente 1.000 "
            f"(servicio/ingresos de CABA: mediana {n(R['caba']['mediana'],3)}, máximo {n(R['caba']['max'],3)}). "
            f"<b>La Pampa 2005-2017 (decisión de diseño):</b> el stock de deuda parece estar en miles aunque la fuente declara millones (como millones sería {n(LP['deuda_ingresos_max_base_sin_tratar'],0)} veces sus ingresos), "
            f"pero <b>no hay confirmación oficial</b>: se trata como <b>faltante</b> (el valor original se conserva y se marca) y la corrección ÷1.000 solo se muestra como <b>sensibilidad</b> rotulada como supuesto. "
            f"Los saltos de servicio pendientes (Santa Cruz 2024-25, San Luis 2014 y 2022) se contrastaron con la identidad stock-flujo: son amortizaciones reales (p. ej., el FFDP de Santa Cruz pasa de " + f"{SU['santa_cruz_ffdp_stock_2023_2024'][0]:,}".replace(",", ".") + " a 0 al amortizarse " + f"{SU['santa_cruz_ffdp_amortizacion_2024']:,}".replace(",", ".") + "), y ninguna de las " + str(SU['n_obs_identidad']) + " observaciones de las otras 23 jurisdicciones es compatible con un error de unidades. "
            f"Evidencia interna sobre La Pampa (no equivale a confirmación): con el stock en miles, los intereses implican una tasa de {n(100*SU['la_pampa_tasa_2006_2015_mediana_si_miles'],1)}% anual (2006-2015); como viene, {n(100*SU['la_pampa_tasa_2006_2015_mediana_si_millones'],3)}%. "
            f"Resultado: el compuesto se calcula en <b>{CB['n_compuesto']} de {CB['n_total']}</b> provincia-año ({n(CB['pct_compuesto'],1)}%); ninguna observación se imputa.")]

    S += [P("3. Diseño metodológico (criterios que no usan los eventos de default)", h2)]
    filas = [["Decisión", "Elección", "Fundamento (tipo)"],
             ["Índice único vs. subíndices", "4 subíndices + compuesto de 3: <b>solvencia</b> (Deuda/Ingresos), <b>liquidez</b> (Servicio/Ingresos), <b>estructural</b> (Personal/Gasto); <b>coyuntural</b> (−Resultado/Ingresos) aparte",
              f"Dinámicas distintas: ICC de la rigidez {n(ICC_R)} vs. {n(ICC_C)} del resultado (78% transitorio); solvencia y liquidez muy parecidas entre provincias (ρ≈0,9), no en el tiempo (ρ≈0,7). Un factor explica solo ~{100*RD['pca_varianza'][0]:.0f}% (evidencia)"],
             ["Qué entra al compuesto", "Regla ex ante: persistencia de rangos t→t+1 ≥ 0,85", f"El resultado fiscal tiene {n(PERS_COY)}: queda afuera; los otros tres superan 0,88 (evidencia)"],
             ["Normalización", "Percentil <b>dentro de cada año</b> (v1: base fija 2005-2012)",
              f"La mediana de Deuda/Ingresos cae de {n(DS['mediana_2005'])} (2005) a {n(DS['mediana_2012'])} (2012) y {n(DS['mediana_2024'])} (2024): la base no es estacionaria (deriva del percentil mediano: {n(DERIVA_V1,0)} vs. {n(DERIVA_V2,0)} pts). Sin información futura; un error de unidades mueve al resto {n(EF['corte_anual|sin_tratar'],1)} pts con percentiles y {n(EF['zscore_base|sin_tratar'],0)} con z-score (evidencia)"],
             ["Ventana / pesos", "Sin suavizar (1 año); pesos iguales", "La regla no respalda suavizar el servicio (persistencia anual 0,89): rectifica mi intuición inicial de 3 años. Media móvil y pesos alternativos, como sensibilidad (diseño)"],
             ["Posición relativa", "Cuartil del compuesto en cada año", "Reemplaza cortes 33/66/85 arbitrarios y anclados a 2005-2012. Siempre hay 6 jurisdicciones en el cuartil superior (limitación)"]]
    S += [tabla(filas, [3.0, 5.7, 9.0]), Spacer(1, 3)]
    S += [img("deriva_normalizacion", 14.8), P("Figura 1. Deriva de la escala de solvencia (izquierda) y percentil mediano de solvencia según la normalización (derecha): por qué se abandona la base fija 2005-2012.", cap)]

    t6 = R["ranking_2024"]["top6"]
    b3 = R["ranking_2024"]["bottom3"]
    S += [P("4. Resultado descriptivo (2024) y cobertura", h2),
          P(f"<b>Evidencia.</b> En 2024 las jurisdicciones con mayor IFFS compuesto son " + ", ".join(f"{r['provincia']} ({n(r['IFFS'],0)})" for r in t6) +
            f"; las de menor, " + ", ".join(f"{r['provincia']} ({n(r['IFFS'],0)})" for r in b3) + ". "
            "Cada barra es el compuesto (media de tres percentiles) y los puntos muestran los cuatro subíndices: una jurisdicción puede ser frágil por una sola dimensión, y el coyuntural no entra a la barra. "
            f"<i>Interpretación</i>: son posiciones relativas de un año; la mediana de Deuda/Ingresos de las 24 cayó de {n(DS['mediana_2005'])} (2005) a {n(DS['mediana_2024'])} (2024), un movimiento común que el score relativo no muestra y por eso se publica el nivel común."),
          dos("ranking_iffs_2024", "cobertura_compuesto", 8.6, 8.6),
          P("Figura 2. Izquierda: IFFS 2024, compuesto (barra) y subíndices (puntos). Derecha: cobertura del compuesto (azul = calculado; gris = faltante, no imputado).", cap)]

    S += [P("5. Robustez del ordenamiento (sin eventos)", h2),
          P(f"Cada decisión se cambia una a la vez (17 alternativas: normalizaciones, ventanas, agregaciones, composición y pesos). <b>Evidencia:</b> la correlación de Spearman con el principal va de {n(min(RHOS))} a {n(max(RHOS))}; "
            f"con {mil(W['n_sim'])} vectores de pesos aleatorios la mediana es {n(W['spearman_medio_mediana'])} (percentil 5: {n(W['spearman_medio_p05'])}). Los puestos individuales sí se mueven (hay observaciones que cambian ≥ 5 puestos según la especificación): "
            "<i>no conviene leer diferencias de 1-2 puestos como fragilidad distinta</i>. Incluir el subíndice coyuntural en el compuesto empeora la persistencia del ordenamiento (0,89 vs. 0,94).")]
    S += [img("robustez_especificaciones", 13.2), P("Figura 3. Sensibilidad a cada decisión de diseño. Fuente: outputs/tables/T05.", cap)]

    S += [P("6. Validación: descripción, discriminación y anticipación son preguntas distintas", h2),
          P(f"<b>Base de eventos (supuesto):</b> {EVN['n_eventos']} eventos ({EVN['tasa_base_pct'] and n(EVN['tasa_base_pct'],1)}% de las observaciones); {EVN['n_covid']} son el mismo shock 2020-21 y para {EVN['n_anio_ambiguo']} la fuente dice “2020-2021” sin año por provincia: el 2020 es una convención "
            "(todo el análisis de fechas se repite con ancla 2021 sin cambios relevantes). En rigor son <i>un</i> shock observado en 12 unidades, no 12 observaciones independientes.")]
    disc_filas = [["Medida (año 2019 = T−1)", "AUC", "IC 95%", "Lectura"]]
    for etiqueta, clave, lectura in [
        ("IFFS compuesto", "IFFS compuesto (principal)", "Separa al grupo con evento"),
        ("Subíndice solvencia", "Subíndice solvencia", "Lo que más discrimina"),
        ("Subíndice liquidez", "Subíndice liquidez", "Discrimina"),
        ("Subíndice estructural", "Subíndice estructural", "Débil (IC incluye 0,5)"),
        ("Subíndice coyuntural", "Subíndice coyuntural", "Sin señal clara"),
        ("Benchmark: % de bonos en la deuda", "Benchmark: participación de bonos en la deuda (%)", "Casi igual que el índice: acceso al mercado")]:
        m = D[clave]
        disc_filas.append([etiqueta, n(m["auc"]), f"{n(m['ic95'][0])}–{n(m['ic95'][1])}", lectura])
    S += [tabla(disc_filas, [5.4, 1.6, 2.6, 8.1]), Spacer(1, 2),
          P(f"<b>Evidencia (discriminación):</b> el compuesto tiene AUC {n(K['auc'])} en 2019 (el azar supera {n(AZ['azar_p95'])} solo el 5% de las veces) y se mantiene en {n(min(R['rezagos'][f'2020 (convención)|IFFS|T-{k}']['auc'] for k in range(1,6)))}–{n(max(R['rezagos'][f'2020 (convención)|IFFS|T-{k}']['auc'] for k in range(1,6)))} de T−1 a T−5, con ancla 2020 o 2021. "
            f"<b>Pero es descriptivo, no anticipatorio:</b> (1) el grupo ya se separaba en {A_2012} (AUC {n(PV['auc_iffs_por_anio'][str(A_2012)])}) y sigue separado <i>después</i> de los eventos (AUC {n(PV['auc_iffs_por_anio']['2025'])} en 2025): es un rasgo persistente; "
            f"(2) solo las provincias con bonos podían reestructurarlos: entre las {COND['n_emisoras']} emisoras hay solo {COND['n_control']} controles y no hay poder para distinguir (limitación).")]
    S += [img("auc_por_anio", 13.0), P("Figura 4. Placebo temporal: AUC de las mismas 12 provincias contra el resto en cada año (IC 95% bootstrap por provincia, 500 remuestreos).", cap)]

    S += [P("7. Anticipación y pruebas no viables", h2),
          P(f"<b>Evidencia (anticipación):</b> el compuesto cambió {ns(PRE['cambio_evento'])} puntos entre T−5 y T−1 en las provincias con evento y {ns(PRE['cambio_control'])} en el resto: diferencia {ns(PRE['dif'])} [{ns(PRE['ic95'][0])}; {ns(PRE['ic95'][1])}], p = {n(PRE['p'])}; "
            f"en solvencia {ns(PRE_S['dif'])} [{ns(PRE_S['ic95'][0])}; {ns(PRE_S['ic95'][1])}] (el IC bootstrap apenas excluye el cero, pero la permutación da p = {n(PRE_S['p'])} y es una de 10 comparaciones sin corregir: indicio débil, no evidencia). La brecha ya era de {n(ES8['dif'],0)} puntos en τ = −8 y de {n(ES1['dif'],0)} en τ = −1. "
            "<b>Limitación:</b> con una sola fecha común no hay contrafactual temporal y con 12 unidades la potencia es baja: no hay evidencia de anticipación, pero tampoco puede descartarse un efecto moderado. "
            f"Un caso aparte, La Rioja 2025: en 2024 estaba en el puesto {LR['2024']['puesto']} de 24 (solvencia {n(LR['2024']['solvencia'],0)}); una observación no permite estimar aciertos."),
          P("<b>Pruebas que no se corren, y por qué:</b> modelo supervisado con corte temporal (0 eventos antes de 2020); probabilidad calibrada (12 de 13 eventos en el mismo shock); "
            "optimización de pesos o umbrales para maximizar el AUC (sobreajuste y contradice el diseño ex ante); modelos de duración y estudios de eventos escalonados (una sola cohorte); "
            f"predicción de fechas exactas (ancla convencional); comparación con calificaciones (sin datos). Extrapolar tendencias tampoco mejora a repetir el último valor (error {n(BT['mae_tendencia'],1)} vs. {n(BT['mae_ingenua'],1)} puntos).")]
    S += [img("estudio_de_eventos", 10.4)]
    S += [P("Figura 5. Estudio de eventos: IFFS medio por años respecto del ancla (τ = 0 ↔ 2020, convención).", cap)]
    S += [P("8. Limitaciones, próximos pasos y conclusión", h2),
          P("<b>Limitaciones:</b> La Pampa 2005-2017 sin confirmar (y 2025 sin ejecución); score relativo (no muestra movimientos comunes: por eso se publica también el nivel común); Deuda/Ingresos sobrestimada con inflación alta (stock a 31/12 sobre flujo anual); "
            "sin deuda flotante, garantías, vencimientos futuros ni moneda de la deuda (solo 11 de 24 provincias); una alta proporción de gasto en personal puede ser provisión de servicios y no rigidez; solo La Pampa 2024 (stock 500) sigue sin verificarse, sin efecto en el ranking. "
            "<b>Próximos pasos:</b> confirmar con la DNAP las unidades de La Pampa; incorporar deuda flotante, vencimientos y moneda; sumar eventos futuros para poder hacer validación fuera de muestra."),
          P(f"<b>Conclusión.</b> El IFFS v2 es una herramienta descriptiva y comparativa, robusta a las decisiones de diseño razonables y auditable dato por dato. Separa a las provincias que reestructuraron en 2020-21, pero lo hacía desde hace más de una década y una variable trivial lo hace casi igual: "
            "es discriminación de un rasgo persistente, no anticipación. Su valor está en mostrar <i>dónde y por qué</i> una jurisdicción es relativamente frágil y en dejar explícito lo que no se sabe. "
            "Código, datos, notebook y dashboard: carpeta del proyecto (README.md, ESTADO.md); todas las cifras salen de <i>outputs/resultados_clave.json</i>.")]

    salida = ROOT / "informe" / "informe_fragilidad_fiscal.pdf"
    salida.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(salida), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.3 * cm, bottomMargin=1.3 * cm,
                            title="Monitor de Fragilidad Fiscal Provincial — IFFS v2", author="UADE — Programación para Economistas")

    def pie(canvas, d):
        canvas.saveState()
        canvas.setFont("DV", 6.5)
        canvas.setFillColor(colors.HexColor("#898781"))
        canvas.drawString(1.6 * cm, 0.7 * cm, "Fuente: DNAP, BCRA; elaboración propia. Score relativo: no es una probabilidad de default.")
        canvas.drawRightString(A4[0] - 1.6 * cm, 0.7 * cm, f"{d.page}")
        canvas.restoreState()

    doc.build(S, onFirstPage=pie, onLaterPages=pie)
    import pymupdf

    pag = pymupdf.open(str(salida)).page_count
    print("Informe PDF:", salida, "| páginas:", pag)
    if pag > 3:
        sys.exit(f"ERROR: el informe tiene {pag} páginas (máximo estricto: 3).")


# ======================================================================================================================
# 3. PRESENTACIÓN (9 diapositivas)
# ======================================================================================================================
def pptx() -> None:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Emu, Inches, Pt

    AZ_, INK, GR = RGBColor(0x1A, 0x3C, 0x6E), RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x52, 0x51, 0x4E)
    ROJO_, VERDE_, CLARO = RGBColor(0xD0, 0x3B, 0x3B), RGBColor(0x1B, 0x8A, 0x5A), RGBColor(0xE8, 0xEE, 0xFB)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    from PIL import Image as PILImage

    def caja(sl, x, y, w, h, texto, tam=16, bold=False, color=INK, fondo=None, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
        s = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))     # cuadro de texto (sin sombra de tema)
        if fondo is not None:
            s.fill.solid()
            s.fill.fore_color.rgb = fondo
        tf = s.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Inches(0.12)
        tf.margin_top = tf.margin_bottom = Inches(0.06)
        if 14 <= tam <= 20:
            tam = round(tam * 1.18)                       # cuerpo más legible en sala
        lineas = texto if isinstance(texto, list) else [texto]
        for i, li in enumerate(lineas):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            p.space_after = Pt(5)
            partes = re.split(r"(\*\*.+?\*\*)", li)
            for pt in partes:
                if not pt:
                    continue
                r = p.add_run()
                r.text = pt.strip("*") if pt.startswith("**") else pt
                r.font.size = Pt(tam)
                r.font.bold = bold or pt.startswith("**")
                r.font.color.rgb = color
                r.font.name = "Calibri"
        return s

    def titulo(sl, t, sub=None):
        caja(sl, 0.5, 0.3, 12.3, 0.9, t, tam=24, bold=True, color=AZ_, anchor=MSO_ANCHOR.MIDDLE)
        if sub:
            caja(sl, 0.5, 1.2, 12.3, 0.6, sub, tam=14, color=GR)

    def imagen(sl, nombre, x, y, w=None, h=None):
        ww, hh = PILImage.open(FIG / f"{nombre}.png").size
        if w and not h:
            h = w * hh / ww
        elif h and not w:
            w = h * ww / hh
        elif w and h:  # ajustar dentro de la caja
            esc = min(w / ww, h / hh) * 1.0
            w, h = ww * esc, hh * esc
        sl.shapes.add_picture(str(FIG / f"{nombre}.png"), Inches(x), Inches(y), Inches(w), Inches(h))
        return w, h

    def pie(sl, t="Fuente: DNAP, BCRA; elaboración propia. IFFS v2 = posición relativa entre jurisdicciones; no es una probabilidad de default."):
        caja(sl, 0.5, 7.0, 12.3, 0.4, t, tam=9, color=RGBColor(0x89, 0x87, 0x81))

    def notas(sl, t):
        sl.notes_slide.notes_text_frame.text = t

    # 1 ---------------------------------------------------------------------------------------------------------
    sl = prs.slides.add_slide(blank)
    caja(sl, 0.7, 1.7, 12, 1.3, "Monitor de Fragilidad Fiscal Provincial", tam=36, bold=True, color=AZ_)
    caja(sl, 0.7, 3.0, 12, 0.9, "IFFS v2: ¿qué tan frágil es cada provincia respecto de sus pares?", tam=24, color=INK)
    caja(sl, 0.7, 4.2, 11.8, 1.9, ["**Es** una herramienta comparativa: posición relativa de 24 jurisdicciones (2005-2025), con cuatro subíndices.",
                                   "**No es** una probabilidad de default ni un modelo predictivo: con 13 eventos, los datos no lo permiten."], tam=20, color=GR)
    caja(sl, 0.7, 6.4, 12, 0.5, "Programación para Economistas · Segundo Parcial · UADE · Septiembre 2026", tam=14, color=GR)
    notas(sl, "Presento un índice de fragilidad fiscal provincial y, sobre todo, lo que se puede y no se puede afirmar con él. Adelanto la conclusión: describe y ordena bien, pero con los datos disponibles no se puede decir que anticipe defaults.")

    # 2 ---------------------------------------------------------------------------------------------------------
    sl = prs.slides.add_slide(blank)
    titulo(sl, "El problema: comparar la fragilidad fiscal, con honestidad")
    caja(sl, 0.6, 1.5, 6.3, 5.3, [f"**{EVN['n_covid']} de 24 jurisdicciones** reestructuraron deuda de mercado en 2020-21 (shock COVID); La Rioja incumplió en 2025.",
                                  "No hay una herramienta pública que compare la fragilidad **relativa** de las provincias con datos oficiales y método transparente.",
                                  "Objetivo: ordenar, describir y monitorear; separar **evidencia, interpretación, supuestos, decisiones de diseño y limitaciones**."], tam=18)
    caja(sl, 7.2, 1.7, 5.5, 4.6, ["**El dato clave:**", f"{EVN['n_eventos']} eventos sobre {CB['n_total']} observaciones ({n(EVN['tasa_base_pct'],1)}%).",
                                  f"{EVN['n_covid']} son el mismo shock; para {EVN['n_anio_ambiguo']} el año 2020 es una convención (la fuente dice “2020-2021”).",
                                  "→ No hay eventos antes de 2020: **no se puede entrenar ni calibrar un modelo predictivo.**"], tam=19, fondo=CLARO)
    pie(sl)
    notas(sl, "El motivo de todo el trabajo es el shock de 2020: 12 provincias reestructuraron. Pero justamente por eso la evidencia es pobre: es un solo shock común observado en 12 unidades, y con año exacto convencional en 11. Eso condiciona qué validación es honesta.")

    # 3 ---------------------------------------------------------------------------------------------------------
    sl = prs.slides.add_slide(blank)
    titulo(sl, "Datos y auditoría: una corrección y una decisión", f"Compuesto calculado en {CB['n_compuesto']} de {CB['n_total']} provincia-año ({n(CB['pct_compuesto'],1)}%); nada se imputa")
    caja(sl, 0.6, 1.9, 5.6, 5.0, ["**CABA (evidencia).** El servicio de deuda estaba x1.000: se leyó la hoja en miles como millones. Corregido con la hoja Millones; el código falla si el cociente entre hojas no es 1.000.",
                                  "**La Pampa 2005-2017 (decisión).** El stock parece estar en miles, pero **sin confirmación oficial**: se trata como **faltante** (el original se conserva). La corrección ÷1.000 es solo una sensibilidad, rotulada como supuesto.",
                                  "Los saltos de servicio (Santa Cruz, San Luis) son amortizaciones reales: identidad stock-flujo. Intereses de La Pampa: tasa de 4-7% solo si el stock está en miles (evidencia interna, no confirmación)."], tam=14)
    imagen(sl, "cobertura_compuesto", 6.4, 1.8, w=6.5)
    pie(sl)
    notas(sl, "Dos hallazgos de auditoría. En CABA había un error de unidades verificable y se corrigió con control automático. En La Pampa hay sospecha de error, pero no confirmación oficial: no la corrijo, la marco como faltante y muestro la corrección solo como sensibilidad. El gráfico muestra qué falta y por qué.")

    # 4 ---------------------------------------------------------------------------------------------------------
    sl = prs.slides.add_slide(blank)
    titulo(sl, "Diseño: cuatro subíndices y un compuesto de tres", "Criterios sin usar los eventos de default")
    filas = [("Subíndice", "Variable", "Compuesto"), ("Solvencia", "Deuda / Ingresos", "Sí"), ("Liquidez", "Servicio de deuda / Ingresos", "Sí"),
             ("Estructural", "Personal / Gasto total", "Sí"), ("Coyuntural", "−Resultado / Ingresos", "No")]
    tb = sl.shapes.add_table(5, 3, Inches(0.6), Inches(1.9), Inches(6.4), Inches(2.6)).table
    for i, f in enumerate(filas):
        for j, v in enumerate(f):
            c = tb.cell(i, j)
            c.text = v
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(18)
                    r.font.bold = i == 0
    tb.columns[0].width, tb.columns[1].width, tb.columns[2].width = Inches(1.7), Inches(3.4), Inches(1.3)
    caja(sl, 7.3, 1.8, 5.5, 4.9, [f"**¿Por qué subíndices?** Dinámicas distintas: la rigidez es casi un rasgo fijo (ICC {n(ICC_R)}); el resultado fiscal es 78% transitorio (ICC {n(ICC_C)}).",
                                  f"**Regla ex ante:** entra al compuesto lo que tiene persistencia de rangos ≥ 0,85. El resultado fiscal tiene {n(PERS_COY)}: queda **aparte**.",
                                  "**Rectificación:** mi idea inicial de suavizar el servicio 3 años no pasa la regla; se aplica la regla."], tam=17)
    caja(sl, 0.6, 4.8, 6.4, 1.9, "Compuesto = media simple de solvencia, liquidez y estructural. Pesos iguales, sin optimizar. Posición = cuartil del año.", tam=16, color=GR, fondo=CLARO, anchor=MSO_ANCHOR.MIDDLE)
    pie(sl)
    notas(sl, "Cuatro dimensiones con lectura económica distinta. El resultado fiscal es demasiado ruidoso para promediarlo con variables persistentes; lo dejo como señal coyuntural aparte. La regla se fijó antes de mirar eventos y, de hecho, contradijo mi intuición sobre el suavizado: gana la regla.")

    # 5 ---------------------------------------------------------------------------------------------------------
    sl = prs.slides.add_slide(blank)
    titulo(sl, "Normalización: percentil dentro de cada año", f"La mediana de Deuda/Ingresos cae de {n(DS['mediana_2005'])} a {n(DS['mediana_2012'])} dentro de la propia base 2005-2012")
    imagen(sl, "deriva_normalizacion", 0.5, 2.0, w=8.4)
    caja(sl, 9.0, 1.9, 3.9, 5.0, [f"Contra la base fija, el percentil mediano deriva **{n(DERIVA_V1,0)} puntos**; con corte anual, {n(DERIVA_V2,0)}.",
                                  f"Un error de unidades mueve al resto **{n(EF['corte_anual|sin_tratar'],1)} pts** (percentil) vs. **{n(EF['zscore_base|sin_tratar'],0)}** (z-score).",
                                  "Sin información futura.", "**Costo:** no muestra movimientos comunes → se publica el nivel común."], tam=15)
    pie(sl)
    notas(sl, "El score es relativo por diseño: qué tan frágil es cada provincia respecto de sus pares ese año. Elegí el percentil anual por tres razones que no dependen de eventos: la escala de la base no es estacionaria, no usa información futura y es inmune a outliers. El costo: si todas mejoran a la vez, el score no lo muestra, por eso mostramos también el nivel común.")

    # 6 ---------------------------------------------------------------------------------------------------------
    sl = prs.slides.add_slide(blank)
    titulo(sl, "Robustez: el ordenamiento aguanta las decisiones razonables", f"ρ de Spearman con el principal: {n(min(RHOS))} a {n(max(RHOS))}; pesos aleatorios: mediana {n(W['spearman_medio_mediana'])}")
    imagen(sl, "robustez_especificaciones", 0.5, 1.75, w=8.6)
    caja(sl, 9.3, 1.9, 3.7, 5.0, ["Se cambia **una decisión a la vez**: normalización, ventana, pesos, agregación, composición.",
                                  "Los **puestos individuales sí se mueven**: no leer diferencias de 1-2 puestos como fragilidad distinta.",
                                  "Todas las especificaciones se reportan; ninguna se elige por su desempeño con los eventos."], tam=15)
    pie(sl)
    notas(sl, "Cada punto es una alternativa evaluada una a la vez. El ordenamiento global es estable, pero conviene no sobreinterpretar diferencias pequeñas de posición. Importante: reporto todas, incluidas las que dan peor resultado con los eventos.")

    # 7 ---------------------------------------------------------------------------------------------------------
    sl = prs.slides.add_slide(blank)
    titulo(sl, "Discriminación: en 2019 el grupo con evento puntuaba más alto", f"AUC {n(K['auc'])} [IC 95%: {n(K['ic95'][0])}–{n(K['ic95'][1])}]; el azar supera {n(AZ['azar_p95'])} solo el 5% de las veces")
    filas = [("Medida (2019)", "AUC", "IC 95%")]
    for et, cl in [("IFFS compuesto", "IFFS compuesto (principal)"), ("Solvencia", "Subíndice solvencia"), ("Liquidez", "Subíndice liquidez"), ("Estructural", "Subíndice estructural"),
                   ("Coyuntural", "Subíndice coyuntural"), ("% de bonos en la deuda", "Benchmark: participación de bonos en la deuda (%)")]:
        m = D[cl]
        filas.append((et, n(m["auc"]), f"{n(m['ic95'][0])}–{n(m['ic95'][1])}"))
    tb = sl.shapes.add_table(len(filas), 3, Inches(0.6), Inches(1.9), Inches(6.6), Inches(3.9)).table
    for i, f in enumerate(filas):
        for j, v in enumerate(f):
            c = tb.cell(i, j)
            c.text = v
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(18)
                    r.font.bold = i in (0, 1)
    tb.columns[0].width, tb.columns[1].width, tb.columns[2].width = Inches(3.4), Inches(1.2), Inches(2.0)
    caja(sl, 7.5, 1.9, 5.3, 4.9, ["**Evidencia:** el índice separa a las 12 provincias con evento de las otras 12, con cualquier ancla (2020 o 2021) y de T−1 a T−5.",
                                  f"**Pero** una variable trivial —el % de bonos en la deuda— logra {n(BON['auc'])}: solo quien tenía bonos podía reestructurarlos.",
                                  f"Entre emisoras hay {COND['n_control']} controles: **no hay poder** para distinguir."], tam=17, fondo=CLARO)
    pie(sl)
    notas(sl, "Discriminación es una pregunta descriptiva: en una fecha, ¿el grupo que después reestructuró tenía un score más alto? Sí. Pero el porcentaje de bonos en la deuda hace casi lo mismo, porque para reestructurar bonos hay que tenerlos. Entre emisoras casi no hay grupo de control, así que no puedo distinguir.")

    # 8 ---------------------------------------------------------------------------------------------------------
    sl = prs.slides.add_slide(blank)
    titulo(sl, "Anticipación: sin evidencia de una señal previa", f"El grupo ya estaba separado en {A_2012} (AUC {n(PV['auc_iffs_por_anio'][str(A_2012)])}) y sigue separado en 2025 (AUC {n(PV['auc_iffs_por_anio']['2025'])})")
    imagen(sl, "auc_por_anio", 0.5, 1.85, w=8.3)
    caja(sl, 9.0, 1.9, 4.0, 5.0, [f"**Placebo temporal:** el AUC del mismo grupo ya es alto años antes del shock y después de él → rasgo **persistente**.",
                                  f"**Pretendencia T−5→T−1:** {ns(PRE['dif'])} pts [{ns(PRE['ic95'][0])}; {ns(PRE['ic95'][1])}], p = {n(PRE['p'])}.",
                                  "Con una sola fecha común y 12 unidades, tampoco se descarta un efecto moderado."], tam=15)
    pie(sl)
    notas(sl, "Anticipar es otra cosa: que el score suba antes del evento más que en el resto. El gráfico muestra que el grupo ya estaba separado desde alrededor de 2012 y lo sigue estando en 2025, después de los eventos. El cambio entre cinco años y un año antes no es distinto. Con estos datos: no hay evidencia de anticipación, pero tampoco es una prueba de que no exista.")

    # 9 ---------------------------------------------------------------------------------------------------------
    sl = prs.slides.add_slide(blank)
    titulo(sl, "Conclusiones, límites y próximos pasos")
    caja(sl, 0.6, 1.4, 6.2, 5.5, ["**Qué sí se puede decir**", "• Ordena y describe la fragilidad relativa, con cobertura visible y datos auditados.",
                                  "• Es robusto a las decisiones de diseño razonables.", "• Separa al grupo que reestructuró, desde hace más de una década.",
                                  f"**Qué no se puede decir**", "• Que anticipe defaults ni que dé probabilidades.", f"• Nada predictivo con {EVN['n_eventos']} eventos ({EVN['n_covid']} simultáneos)."], tam=18)
    caja(sl, 7.0, 1.4, 5.8, 5.5, ["**Pendientes**", "• Confirmar con la DNAP las unidades de La Pampa 2005-2017.", "• Incorporar deuda flotante, vencimientos y moneda de la deuda.",
                                  "• Sumar eventos futuros para validar fuera de muestra.", "**Producto**", "• Dashboard, notebook e informe con las mismas cifras (outputs/resultados_clave.json)."], tam=18, fondo=CLARO)
    pie(sl)
    notas(sl, "Cierro con lo que sí y lo que no. El valor está en la transparencia: mostrar dónde y por qué una provincia es relativamente frágil, y dejar explícito lo que no se sabe. Pendientes: confirmar La Pampa, sumar variables y esperar más eventos para poder validar.")

    salida = ROOT / "presentacion" / "presentacion_fragilidad_fiscal.pptx"
    salida.parent.mkdir(exist_ok=True)
    prs.save(str(salida))
    print("Presentación:", salida, "| diapositivas:", len(prs.slides))


if __name__ == "__main__":
    actualizar_readme()
    pdf()
    pptx()
