"""
FASE 1 — Auditoría de datos y de construcción del IFFS
========================================================

Script reproducible: `python auditoria/fase1_auditoria.py`
Escribe tablas en outputs/auditoria/ y un resumen por consola. NO modifica datos ni código de producción.

Cada bloque responde una pregunta de auditoría concreta y deja evidencia numérica:

  A. ¿Hay cambios de unidades / quiebres de escala en las series de la DNAP?   (La Pampa, CABA)
  B. ¿Cómo se comporta cada variable? (asimetría, outliers, persistencia, parte estructural vs. coyuntural)
  C. ¿Cuánto distorsiona la escala 2005-2012 (ventana base) y la inclusión de La Pampa?
  D. ¿Son consistentes los denominadores del índice?
  E. ¿La razón Deuda/Ingresos está contaminada por el efecto de valuación cambiaria?
  F. ¿Qué falta, dónde y por qué? (cobertura y missing)
  G. ¿Cuánto sobrestima Deuda/Ingresos el desajuste stock (31/12) / flujo (promedio anual)?
  H. Integridad: grilla 24x21, duplicados, rangos, consistencia entre paneles y chequeo de unidades de CABA.

NOTA (2026-09-25): el tratamiento de La Pampa 2005-2017 cambió. La corrección ÷1.000 NO cuenta con confirmación oficial, así que el
análisis principal trata esos años como faltantes y la corrección queda solo como escenario de sensibilidad (`la_pampa="sensibilidad_miles"`).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data_processing as dp          # noqa: E402
from src import indicators as ind              # noqa: E402
from src import validation as val         # noqa: E402

RAW = ROOT / "data" / "raw" / "base_investigacion"
DNAP = ROOT / "data" / "raw" / "fuentes_dnap"
OUT = ROOT / "outputs" / "auditoria"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

VARS = {
    "deuda_sobre_ingresos": "Deuda / Ingresos corrientes",
    "servicio_sobre_ingresos": "Servicio / Ingresos corrientes",
    "rigidez_gastototal": "Personal / Gasto total",
    "fragilidad_resultado": "-(Resultado financiero / Ingresos totales)",
    "dependencia_transferencias": "Ingresos de origen nacional / Ingresos corrientes",
}


def titulo(s: str) -> None:
    print("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100)


# ---------------------------------------------------------------------------------------------
# A. Unidades y quiebres de escala
# ---------------------------------------------------------------------------------------------
def bloque_A() -> None:
    titulo("A. UNIDADES Y QUIEBRES DE ESCALA")
    deuda = pd.read_csv(RAW / "deuda_provincial_panel_1996_2025.csv")
    stock = deuda[deuda.variable == "deuda_total_sin_flotante"].pivot(
        index="anio", columns="provincia", values="valor_millones_pesos_corrientes")
    serv = pd.read_csv(RAW / "servicio_deuda_provincial_panel_2005_2025.csv")
    serv = serv.groupby(["anio", "provincia"]).valor_millones_pesos_corrientes.sum().unstack()
    ingr = pd.read_csv(RAW / "panel_fiscal_benchmarks.csv").pivot(
        index="anio", columns="provincia", values="ingresos_corrientes")

    filas = []
    for nombre, df in [("stock_deuda", stock), ("servicio_deuda", serv), ("ingresos_corrientes", ingr)]:
        ld = np.log10(df.replace(0, np.nan)).diff()
        rel = ld.sub(ld.median(axis=1), axis=0)          # descuenta la inflación común a todas las provincias
        for prov in df.columns:
            for anio in df.index[1:]:
                v = rel.loc[anio, prov]
                if pd.notna(v) and abs(v) > 1.0:          # salto > 10x respecto del comportamiento mediano
                    filas.append({"serie": nombre, "provincia": prov, "anio": int(anio), "log10_relativo": round(v, 2),
                                  "valor_previo": df.loc[anio - 1, prov], "valor": df.loc[anio, prov]})
    saltos = pd.DataFrame(filas).sort_values("log10_relativo", key=abs, ascending=False)
    saltos.to_csv(OUT / "A1_saltos_de_escala.csv", index=False)
    print("Saltos de escala > 10x (relativos a la mediana provincial):\n", saltos.head(12).to_string(index=False))

    # A2. Test de plausibilidad de La Pampa: ¿puede ser mayor que las otras 23 provincias juntas?
    otros = stock.drop(columns="La Pampa").sum(axis=1)
    a2 = pd.DataFrame({"La_Pampa_tal_como_esta": stock["La Pampa"], "suma_otras_23": otros,
                       "cociente": stock["La Pampa"] / otros}).loc[1996:2020]
    a2.round(3).to_csv(OUT / "A2_la_pampa_vs_resto.csv")
    print("\nLa Pampa vs. suma de las otras 23 (M$ como cargado):\n", a2.loc[[1996, 2005, 2010, 2012, 2015, 2016, 2017, 2018]].round(2))

    # A3. CABA: hoja 'CABA' (miles de pesos) vs hoja 'Millones'
    f = DNAP / "servicio_caba.xlsx"
    def total(sheet):
        # 8 categorías de acreedor (SIN la columna TOTAL, que duplicaría) — mismo criterio que las otras 23 provincias
        df = pd.read_excel(f, sheet_name=sheet, header=None)
        cats = df.iloc[5, 2:].ffill().astype(str).str.strip().str.upper()
        body = df.iloc[7:, 1:]
        body = body[pd.to_numeric(body.iloc[:, 0], errors="coerce").between(2000, 2030)]
        vals = body.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        vals.columns = cats.to_numpy()[: vals.shape[1]]
        vals = vals.loc[:, [not c.startswith("TOTAL") for c in vals.columns]]
        return vals.sum(axis=1).set_axis(body.iloc[:, 0].astype(int))
    miles, mill = total("CABA"), total("Millones")
    caba = pd.DataFrame({"servicio_hoja_CABA_miles": miles, "servicio_hoja_Millones": mill,
                         "cociente": miles / mill})
    caba["stock"] = stock["Ciudad"]
    caba["ingresos"] = ingr["Ciudad"]
    caba["servicio_sobre_ingresos_corregido"] = caba.servicio_hoja_Millones / caba.ingresos
    caba.round(4).to_csv(OUT / "A3_caba_servicio_corregido.csv")
    print("\nCABA — cociente hoja miles / hoja millones (todas las años):", caba.cociente.round(3).unique())
    print("CABA servicio/ingresos corregido: min %.3f | mediana %.3f | max %.3f" % (
        caba.servicio_sobre_ingresos_corregido.min(), caba.servicio_sobre_ingresos_corregido.median(),
        caba.servicio_sobre_ingresos_corregido.max()))
    ratios = pd.read_csv(RAW / "deuda_ratios_provincia_anio.csv")
    otras = ratios[(ratios.provincia != "Ciudad") & ratios.servicio_confiable].servicio_sobre_ingresos
    print("Otras 23 provincias servicio/ingresos: p5 %.3f | p50 %.3f | p95 %.3f" % tuple(otras.quantile([.05, .5, .95])))


# ---------------------------------------------------------------------------------------------
# B. Propiedades estadísticas de cada variable
# ---------------------------------------------------------------------------------------------
def _icc_ar1(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """
    ICC(1) por ANOVA de un factor: (MSB - MSW) / (MSB + (k0 - 1) MSW), con k0 = tamaño medio efectivo de grupo.
    Interpreta la parte de la varianza que es diferencia PERMANENTE entre provincias. Y AR(1) intra-provincia.
    """
    d = df[["provincia", "anio", col]].dropna()
    g = d.groupby("provincia")[col]
    n_i, m_i, N, k = g.size(), g.mean(), len(d), d.provincia.nunique()
    gm = d[col].mean()
    msb = (n_i * (m_i - gm) ** 2).sum() / (k - 1)
    msw = ((d[col] - g.transform("mean")) ** 2).sum() / (N - k)
    k0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    icc = float((msb - msw) / (msb + (k0 - 1) * msw))
    within = d.assign(x=d[col] - g.transform("mean"))
    within["lag"] = within.groupby("provincia")["x"].shift(1)
    w = within.dropna(subset=["lag"])
    ar1 = float(np.corrcoef(w["x"], w["lag"])[0, 1]) if len(w) > 5 else np.nan
    return icc, ar1


def bloque_B(panel: pd.DataFrame) -> None:
    titulo("B. PROPIEDADES DE CADA VARIABLE")
    filas = []
    for v, nombre in {**VARS, "IFFS": "IFFS v2 (compuesto)"}.items():
        s = panel[v]
        sin_pampa = panel.loc[panel.provincia != "La Pampa", v]
        icc, ar1 = _icc_ar1(panel, v)
        filas.append({
            "variable": v, "n": int(s.notna().sum()), "pct_missing": round(100 * s.isna().mean(), 1),
            "mediana": s.median(), "p95": s.quantile(.95), "maximo": s.max(),
            "max_sobre_mediana": s.max() / s.median() if s.median() else np.nan,
            "asimetria": s.skew(), "asimetria_sin_LaPampa": sin_pampa.skew(),
            "ICC_entre_provincias": icc, "AR1_intra_provincia": ar1,
        })
    tab = pd.DataFrame(filas).round(3)
    tab.to_csv(OUT / "B1_propiedades_variables.csv", index=False)
    print(tab.to_string(index=False))
    print("\nLectura: ICC = parte de la varianza que es diferencia PERMANENTE entre provincias (estructural);"
          "\n1-ICC = movimiento en el tiempo. AR1 intra = persistencia de la desviación respecto del promedio de la provincia.")

    # B2. Redundancia: correlaciones pooled, entre provincias y dentro de provincia
    cols = list(VARS)
    pooled = panel[cols].corr(method="spearman")
    medias = panel.groupby("provincia")[cols].mean().corr(method="spearman")
    dem = panel[cols] - panel.groupby("provincia")[cols].transform("mean")
    within = dem.corr(method="spearman")
    for nombre, m in [("B2_corr_pooled", pooled), ("B2_corr_entre_provincias", medias), ("B2_corr_intra_provincia", within)]:
        m.round(3).to_csv(OUT / f"{nombre}.csv")
    print("\nSpearman deuda-servicio: pooled %.2f | entre provincias %.2f | intra-provincia %.2f" % (
        pooled.loc[cols[0], cols[1]], medias.loc[cols[0], cols[1]], within.loc[cols[0], cols[1]]))


# ---------------------------------------------------------------------------------------------
# C. Escala base 2005-2012 y La Pampa
# ---------------------------------------------------------------------------------------------
def bloque_C() -> None:
    titulo("C. ESCALA Y LA PAMPA: tres tratamientos del stock 2005-2017 (sin tratar / excluido / supuesto en miles)")
    v = "deuda_sobre_ingresos"
    p_sin = dp.construir_panel_maestro(la_pampa="sin_tratar")
    p_exc = dp.construir_panel_maestro(la_pampa="excluir")
    p_mil = dp.construir_panel_maestro(la_pampa="sensibilidad_miles")
    base = p_sin["anio"].between(*ind.BASE_WINDOW)
    print("Techo de Deuda/Ingresos en 2005-2012: SIN TRATAR %.1fx | EXCLUIDO %.2fx | SUPUESTO EN MILES %.2fx" % (
        p_sin.loc[base, v].max(), p_exc.loc[base, v].max(), p_mil.loc[base, v].max()))
    tab = val.sensibilidad_la_pampa(p_exc, p_mil, p_sin)
    tab.round(3).to_csv(OUT / "C4_sensibilidad_la_pampa_por_normalizacion.csv", index=False)
    print("\nCambio del subíndice de solvencia del RESTO de las provincias (2005-2017) por normalización:\n", tab.round(2).to_string(index=False))
    lp = p_sin[p_sin.provincia == "La Pampa"][["anio", v]].rename(columns={v: "ratio_sin_tratar"}).merge(
        p_mil[p_mil.provincia == "La Pampa"][["anio", v]].rename(columns={v: "ratio_supuesto_miles"}), on="anio")
    lp.round(4).to_csv(OUT / "C2_la_pampa_percentiles.csv", index=False)

    # Deriva de la escala de solvencia (datos con La Pampa excluida): por qué la ventana base fija no es estacionaria
    drift = p_exc.groupby("anio")[v].median().rename("mediana_deuda_ingresos").to_frame()
    en_base = p_exc.loc[p_exc["anio"].between(*ind.BASE_WINDOW), v]
    drift["percentil_mediano_vs_base_2005_2012"] = p_exc.assign(_p=ind._percentil_ecdf(p_exc[v], en_base)).groupby("anio")["_p"].median()
    drift.round(3).to_csv(OUT / "C1_deriva_escala_deuda.csv")
    print("\nDeriva de la escala de solvencia (La Pampa excluida):\n", drift.loc[[2005, 2008, 2012, 2016, 2019, 2021, 2024, 2025]].round(2))


# ---------------------------------------------------------------------------------------------
# D. Denominadores
# ---------------------------------------------------------------------------------------------
def bloque_D() -> None:
    titulo("D. DENOMINADORES DEL ÍNDICE")
    b = pd.read_csv(RAW / "panel_fiscal_benchmarks.csv")
    b = b[b.ingresos_corrientes > 0].copy()
    b["res_sobre_totales"] = b.resultado_financiero / b.ingresos_totales
    b["res_sobre_corrientes"] = b.resultado_financiero / b.ingresos_corrientes
    b["peso_capital"] = (b.ingresos_totales - b.ingresos_corrientes) / b.ingresos_totales
    b["personal_sobre_ingresos"] = b.gasto_personal / b.ingresos_corrientes
    b["personal_sobre_gasto"] = b.gasto_personal / b.gastos_totales
    print("Recursos de capital / ingresos totales: mediana %.3f | p95 %.3f | máx %.3f" % (
        b.peso_capital.median(), b.peso_capital.quantile(.95), b.peso_capital.max()))
    print("Resultado financiero: dif. absoluta media entre denominadores (totales vs corrientes): %.4f (%.0f%% de |resultado|)" % (
        (b.res_sobre_totales - b.res_sobre_corrientes).abs().mean(),
        100 * (b.res_sobre_totales - b.res_sobre_corrientes).abs().mean() / b.res_sobre_totales.abs().mean()))
    print("Spearman entre ambas versiones: %.3f" % b[["res_sobre_totales", "res_sobre_corrientes"]].corr("spearman").iloc[0, 1])
    print("Spearman Personal/Gasto total vs Personal/Ingresos corrientes: %.3f" % b[["personal_sobre_gasto", "personal_sobre_ingresos"]].corr("spearman").iloc[0, 1])
    print("Observaciones con ingresos corrientes = 0 (dato no reportado codificado como 0):",
          b_zero := int((pd.read_csv(RAW / 'panel_fiscal_benchmarks.csv').ingresos_corrientes == 0).sum()))
    pd.read_csv(RAW / "panel_fiscal_benchmarks.csv").query("ingresos_corrientes == 0")[["provincia", "anio"]].to_csv(
        OUT / "D1_ceros_como_faltantes.csv", index=False)


# ---------------------------------------------------------------------------------------------
# E. Efecto de valuación cambiaria en Deuda/Ingresos
# ---------------------------------------------------------------------------------------------
def bloque_E(panel: pd.DataFrame) -> None:
    titulo("E. EFECTO DE VALUACIÓN CAMBIARIA (Deuda/Ingresos)")
    import statsmodels.formula.api as smf
    tc = pd.read_csv(DNAP / "tipo_cambio_mensual_bcra_92_1.csv", parse_dates=["indice_tiempo"])
    dic = tc[tc.indice_tiempo.dt.month == 12].assign(anio=lambda d: d.indice_tiempo.dt.year).set_index("anio")["tipo_cambio_valuacion"]
    dlog_tc = np.log(dic).diff().rename("dlog_tc")

    d = panel.sort_values(["provincia", "anio"]).copy()
    d["dlog_ratio"] = d.groupby("provincia")["deuda_sobre_ingresos"].transform(lambda s: np.log(s.replace(0, np.nan)).diff())
    d["fx_prev"] = d.groupby("provincia")["pct_moneda_extranjera"].shift(1) / 100
    d = d.merge(dlog_tc, left_on="anio", right_index=True, how="left")
    d = d.dropna(subset=["dlog_ratio", "fx_prev", "dlog_tc"])
    d["exposicion_x_tc"] = d.fx_prev * d.dlog_tc
    m = smf.ols("dlog_ratio ~ exposicion_x_tc + C(provincia) + C(anio)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["provincia"]})
    b, se = m.params["exposicion_x_tc"], m.bse["exposicion_x_tc"]
    print("N = %d obs, %d provincias (solo las 11 con composición por moneda, 2015-2020)" % (len(d), d.provincia.nunique()))
    print("Δlog(Deuda/Ingresos) sobre (participación en USD(t-1) × Δlog TC): coef = %.3f  (EE cluster %.3f, p = %.3f)" % (b, se, m.pvalues["exposicion_x_tc"]))
    print("Interpretación: un coef. cercano a 1 indica que la razón se mueve 1 a 1 con la valuación de la deuda en USD.")
    pd.DataFrame({"coef": [b], "ee_cluster": [se], "p": [m.pvalues["exposicion_x_tc"]], "n": [len(d)]}).to_csv(OUT / "E1_efecto_valuacion.csv", index=False)


# ---------------------------------------------------------------------------------------------
# F. Cobertura
# ---------------------------------------------------------------------------------------------
def bloque_F(panel: pd.DataFrame) -> None:
    titulo("F. COBERTURA Y MISSING")
    cols = list(VARS) + ["IFFS", "pct_moneda_extranjera", "capacidadoperativa"]
    cob = panel.groupby("provincia")[cols].agg(lambda s: round(100 * s.notna().mean()))
    cob.to_csv(OUT / "F1_cobertura_por_provincia.csv")
    print(cob[(cob < 100).any(axis=1)].to_string())
    prov_fx = panel.groupby("provincia").pct_moneda_extranjera.apply(lambda s: s.notna().any())
    print("\nProvincias con composición por moneda: %d de %d" % (prov_fx.sum(), len(prov_fx)))


# ---------------------------------------------------------------------------------------------
# G. Desajuste temporal stock/flujo en Deuda/Ingresos
# ---------------------------------------------------------------------------------------------
def bloque_G(panel: pd.DataFrame) -> None:
    titulo("G. DESAJUSTE STOCK (31/12) / FLUJO (promedio del año) EN DEUDA/INGRESOS")
    ing = panel.pivot(index="anio", columns="provincia", values="ingresos_corrientes")
    pi = (ing / ing.shift(1)).median(axis=1) - 1            # PROXY de inflación: crecimiento nominal mediano de ingresos
    # Con inflación pi durante el año, el flujo anual está en precios medios y el stock en precios de fin de año:
    # la razón queda sobrestimada por ~(1 + pi/2) respecto de una razón 'consistente' (aproximación de primer orden).
    fac = 1 + pi / 2
    med = panel.groupby("anio")["deuda_sobre_ingresos"].median()
    tab = pd.DataFrame({"proxy_inflacion": pi, "factor_sobrestimacion": fac, "mediana_ratio": med,
                        "mediana_ratio_ajustada": med / fac}).round(3).loc[2012:2025]
    tab.to_csv(OUT / "G1_desajuste_stock_flujo.csv")
    print("PROXY (crecimiento nominal mediano de ingresos; NO es inflación medida — supuesto):\n", tab)


def bloque_H(panel_maestro: pd.DataFrame) -> None:
    titulo("H. INTEGRIDAD: grilla, duplicados, rangos, consistencia entre paneles, unidades de CABA")
    p = panel_maestro
    filas = []
    def chk(nombre, ok, detalle=""):
        filas.append({"control": nombre, "resultado": "OK" if ok else "REVISAR", "detalle": detalle})
        print(f"[{'OK' if ok else 'REVISAR'}] {nombre} {detalle}")
    chk("Grilla completa 24 jurisdicciones x 21 años", len(p) == 24 * 21 and p.provincia.nunique() == 24, f"({len(p)} filas)")
    chk("Sin duplicados (provincia, año)", p.duplicated(["provincia", "anio"]).sum() == 0)
    for f in ["panel_fiscal_benchmarks.csv", "deuda_ratios_provincia_anio.csv", "capacidad_respuesta_panel.csv"]:
        d = pd.read_csv(RAW / f)
        dup = d.duplicated([d.columns[0], d.columns[1]]).sum()
        chk(f"Sin duplicados en {f}", dup == 0, f"({dup})")
    for c in ["deuda_sobre_ingresos", "servicio_sobre_ingresos", "rigidez_gastototal"]:
        chk(f"Sin valores negativos en {c}", int((p[c] < 0).sum()) == 0)
    chk("Rigidez del gasto en [0,1]", bool(p.rigidez_gastototal.dropna().between(0, 1).all()))
    chk("Ingresos corrientes estrictamente positivos donde hay dato", bool((p.ingresos_corrientes.dropna() > 0).all()))
    # consistencia stock: panel de ratios vs. panel de deuda por acreedor
    d = pd.read_csv(RAW / "deuda_provincial_panel_1996_2025.csv")
    d = d[d.variable == "deuda_total_sin_flotante"].copy()
    d["provincia"] = dp._normalizar_provincia(d["provincia"])
    m = p.merge(d[["provincia", "anio", "valor_millones_pesos_corrientes"]], on=["provincia", "anio"], how="inner")
    m = m.dropna(subset=["deuda_stock_total_fuente"])
    rel = ((m.deuda_stock_total_fuente - m.valor_millones_pesos_corrientes).abs() / m.valor_millones_pesos_corrientes.abs()).max()
    chk("Stock del panel de ratios = deuda_total_sin_flotante del panel por acreedor", rel < 1e-6, f"(dif. relativa máx {rel:.1e})")
    # suma de acreedores = total
    w2 = pd.read_csv(RAW / "deuda_provincial_panel_1996_2025.csv")
    piv = w2.pivot_table(index=["provincia", "anio"], columns="variable", values="valor_millones_pesos_corrientes")
    comp = [c for c in piv.columns if c != "deuda_total_sin_flotante"]
    dif = ((piv[comp].sum(axis=1) - piv["deuda_total_sin_flotante"]).abs() / piv["deuda_total_sin_flotante"].abs())
    chk("Total de deuda = suma de las 7 categorías por acreedor (incluye 'consolidada')", bool(dif.max() < 1e-3), f"(dif. relativa máx {dif.max():.1e})")
    # CABA: cociente de hojas exactamente 1000 (falla con ValueError si no)
    dp.cargar_servicio_caba_millones()
    chk("CABA: hoja `CABA` (miles) / hoja `Millones` = 1.000 en todos los años", True)
    caba = p[p.provincia == "CABA"].servicio_sobre_ingresos
    chk("CABA servicio/ingresos dentro del rango del resto (<= máx del resto)", bool(caba.max() <= p[p.provincia != "CABA"].servicio_sobre_ingresos.max()),
        f"(CABA máx {caba.max():.3f})".replace(".", ","))
    pd.DataFrame(filas).to_csv(OUT / "H1_controles_integridad.csv", index=False)


def main() -> None:
    panel_maestro = dp.construir_panel_maestro()
    panel = ind.construir_indice(panel_maestro)
    bloque_A()
    bloque_B(panel)
    bloque_C()
    bloque_D()
    bloque_E(panel)
    bloque_F(panel)
    bloque_G(panel)
    bloque_H(panel_maestro)
    print(f"\nTablas guardadas en {OUT}")


if __name__ == "__main__":
    main()
