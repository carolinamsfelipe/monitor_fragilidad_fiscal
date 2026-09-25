"""
FASE 2 — Saltos de servicio y unidades: identidad stock-flujo y tasa de interés implícita
==========================================================================================

Reproducible con `python auditoria/fase2_servicio_y_unidades.py` (tablas en outputs/auditoria/, prefijo I).

Dos pruebas INTERNAS (no requieren fuentes externas) que permiten detectar errores de unidades entre los paneles de stock y de servicio:

  1. Identidad stock-flujo:  ΔStock_t = Endeudamiento_t − Amortización_t (+ valuación). Cuando el stock cae, la caída no puede ser
     órdenes de magnitud distinta de la amortización pagada. Un cociente amortización/stock del año previo cercano a 1.000 (o a 0,001)
     delata un cambio de unidad entre archivos.
  2. Tasa de interés implícita:  intereses_t / stock_{t-1}. Debe ser plausible (unos pocos puntos a decenas de puntos porcentuales);
     valores de 0,005% o de 300% indican unidades distintas.

Etiquetas: [EVIDENCIA] medido en los datos · [SUPUESTO] no verificado · [LIMITACIÓN] no resoluble con los datos actuales.
Ninguna de estas pruebas equivale a una confirmación oficial: son evidencia interna que orienta la decisión.
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

from src import data_processing as dp  # noqa: E402
from src import indicators as ind      # noqa: E402

RAW = ROOT / "data" / "raw" / "base_investigacion"
OUT = ROOT / "outputs" / "auditoria"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def flujos() -> pd.DataFrame:
    """Amortización, intereses y stock por provincia-año (millones de pesos corrientes). CABA se lleva a millones (÷1.000, error ya auditado)."""
    s = pd.read_csv(RAW / "servicio_deuda_provincial_panel_2005_2025.csv")
    d = pd.read_csv(RAW / "deuda_provincial_panel_1996_2025.csv")
    s["provincia"], d["provincia"] = dp._normalizar_provincia(s["provincia"]), dp._normalizar_provincia(d["provincia"])
    am = s[s.variable.str.endswith("amortizacion")].groupby(["provincia", "anio"]).valor_millones_pesos_corrientes.sum().rename("amortizacion")
    it = s[s.variable.str.endswith("interes")].groupby(["provincia", "anio"]).valor_millones_pesos_corrientes.sum().rename("intereses")
    st = d[d.variable == "deuda_total_sin_flotante"].set_index(["provincia", "anio"]).valor_millones_pesos_corrientes.rename("stock_fuente")
    x = pd.concat([am, it, st], axis=1).reset_index().sort_values(["provincia", "anio"])
    caba = x["provincia"] == "CABA"
    x.loc[caba, ["amortizacion", "intereses"]] = x.loc[caba, ["amortizacion", "intereses"]] / 1000.0   # hoja en miles -> millones
    x["stock_prev_fuente"] = x.groupby("provincia").stock_fuente.shift(1)
    return x[x.anio >= 2006].reset_index(drop=True)


def con_unidad_stock(x: pd.DataFrame, la_pampa: str) -> pd.DataFrame:
    """Aplica al stock de La Pampa 2005-2017 el factor de unidad de cada hipótesis: 'millones' (como viene) o 'miles' (÷1.000)."""
    x = x.copy()
    f = np.where((x.provincia == "La Pampa") & (x.anio <= dp.LA_PAMPA_STOCK_DUDOSO_HASTA) & (la_pampa == "miles"), 1000.0, 1.0)
    fp = np.where((x.provincia == "La Pampa") & (x.anio - 1 <= dp.LA_PAMPA_STOCK_DUDOSO_HASTA) & (la_pampa == "miles"), 1000.0, 1.0)
    x["stock"], x["stock_prev"] = x.stock_fuente / f, x.stock_prev_fuente / fp
    x["delta_stock"] = x.stock - x.stock_prev
    x["amort_sobre_stock_prev"] = x.amortizacion / x.stock_prev
    x["tasa_implicita"] = x.intereses / x.stock_prev
    x["amort_sobre_caida"] = np.where(x.delta_stock < 0, x.amortizacion / (-x.delta_stock), np.nan)
    return x


def main() -> None:
    x = con_unidad_stock(flujos(), "millones")

    # ------------------------------------------------------------------ 1. Panel completo: ¿hay otros errores de unidad?
    print("=" * 100 + "\n1. IDENTIDAD STOCK-FLUJO Y TASA IMPLÍCITA EN LAS 24 JURISDICCIONES (2006-2025)\n" + "=" * 100)
    otros = x[(x.provincia != "La Pampa") & (x.stock_prev > 0)]
    print("Fuera de La Pampa: amortización / stock del año previo: p5 %.3f · mediana %.3f · p95 %.3f · máx %.2f" % (
        *otros.amort_sobre_stock_prev.quantile([.05, .5, .95]), otros.amort_sobre_stock_prev.max()))
    print("Fuera de La Pampa: tasa de interés implícita: p5 %.3f · mediana %.3f · p95 %.3f · máx %.2f" % (
        *otros.tasa_implicita.quantile([.05, .5, .95]), otros.tasa_implicita.max()))
    # Cocientes compatibles con un error x1000: amortización > 50 veces el stock previo, o tasa implícita > 20 veces el stock
    sospechosos = otros[(otros.amort_sobre_stock_prev > 50) | (otros.tasa_implicita > 20)]
    print(f"Observaciones fuera de La Pampa compatibles con un error de unidades (amort/stock > 50 o tasa > 2.000%): {len(sospechosos)}")
    if len(sospechosos):
        print(sospechosos[["provincia", "anio", "amortizacion", "intereses", "stock_prev"]].round(3).to_string(index=False))
    x.round(4).to_csv(OUT / "I1_identidad_stock_flujo.csv", index=False)

    # ------------------------------------------------------------------ 2. La Pampa: qué unidad explica mejor los flujos
    print("\n" + "=" * 100 + "\n2. LA PAMPA: TASA DE INTERÉS IMPLÍCITA SEGÚN LA HIPÓTESIS DE UNIDAD DEL STOCK 2005-2017\n" + "=" * 100)
    lp = []
    for h in ("millones", "miles"):
        y = con_unidad_stock(flujos(), h)
        y = y[y.provincia == "La Pampa"][["anio", "amortizacion", "intereses", "stock_prev", "tasa_implicita", "amort_sobre_stock_prev"]].copy()
        y["hipotesis_stock_2005_2017"] = "como viene (millones)" if h == "millones" else "en miles (÷1.000) [SUPUESTO]"
        lp.append(y)
    lp = pd.concat(lp)
    tab = lp.pivot(index="anio", columns="hipotesis_stock_2005_2017", values="tasa_implicita")
    tab = tab.rename(columns={"como viene (millones)": "tasa_si_stock_en_millones", "en miles (÷1.000) [SUPUESTO]": "tasa_si_stock_en_miles"})[
        ["tasa_si_stock_en_miles", "tasa_si_stock_en_millones"]]
    tab["tasa_referencia_resto_p5"] = float(otros.tasa_implicita.quantile(.05))
    tab["tasa_referencia_resto_p95"] = float(otros.tasa_implicita.quantile(.95))
    print(tab.round(4).to_string())
    tab.round(5).to_csv(OUT / "I2_la_pampa_tasa_implicita_por_hipotesis.csv")
    pre = tab.loc[2006:2015]
    print("\n[EVIDENCIA] 2006-2015: tasa implícita mediana %.1f%% si el stock estuviera en miles (÷1.000) vs. %.4f%% si estuviera en millones (como viene). "
          "Rango del resto de las provincias (p5-p95): %.2f%%-%.1f%%." % (
              100 * pre.tasa_si_stock_en_miles.median(), 100 * pre.tasa_si_stock_en_millones.median(),
              100 * otros.tasa_implicita.quantile(.05), 100 * otros.tasa_implicita.quantile(.95)))
    print("[EVIDENCIA] 2017 (intereses 108) es compatible con un stock 2016 de 555.836 EN MILES (tasa 19,5%) pero NO con el stock 2017 = 33.436: en 2018 la tasa implícita sería de 480% "
          "si ese valor estuviera en miles y de 0,5% si estuviera en millones; ninguna es plausible. El stock 2017 es inconsistente con los intereses de 2017 y 2018 bajo CUALQUIER unidad.")
    print("[EVIDENCIA] 2018-2023 (stock en millones): tasas implícitas plausibles (7%-28%). 2024: stock = 500 (cifra redonda) con una caída de 1.742 y amortización de 82: no cierra.")

    # ------------------------------------------------------------------ 3. Resolución de los saltos de servicio pendientes
    print("\n" + "=" * 100 + "\n3. RESOLUCIÓN DE LOS SALTOS DE SERVICIO PENDIENTES\n" + "=" * 100)
    casos = [("Santa Cruz", 2024, "bullet real"), ("Santa Cruz", 2025, "consecuencia del bullet 2024"), ("San Luis", 2014, "amortización gradual sobre base ínfima"),
             ("San Luis", 2022, "amortización a Nación consistente con el stock"), ("La Pampa", 2024, "no verificable")]
    filas = []
    for prov, anio, _ in casos:
        r = x[(x.provincia == prov) & (x.anio == anio)].iloc[0]
        filas.append({"provincia": prov, "anio": anio, "amortizacion": r.amortizacion, "intereses": r.intereses, "stock_prev": r.stock_prev, "stock": r.stock,
                      "caida_stock": -r.delta_stock, "amort_sobre_caida": r.amort_sobre_caida, "amort_sobre_stock_prev": r.amort_sobre_stock_prev, "tasa_implicita": r.tasa_implicita})
    res = pd.DataFrame(filas)
    # San Luis: la amortización coincide con la caída del stock?
    sl = x[(x.provincia == "San Luis") & (x.anio.isin([2019, 2020, 2021, 2022]))][["anio", "amortizacion", "delta_stock"]]
    sl["diferencia_amort_vs_caida"] = sl.amortizacion + sl.delta_stock
    print("San Luis: amortización vs. caída del stock (la diferencia debería ser 0 si no hay endeudamiento nuevo):\n", sl.round(4).to_string(index=False))
    sc = pd.read_csv(RAW / "servicio_deuda_provincial_panel_2005_2025.csv")
    sc = sc[(sc.provincia == "Santa Cruz") & (sc.anio.isin([2023, 2024])) & (sc.variable == "servicio_ffdp_amortizacion")].set_index("anio").valor_millones_pesos_corrientes
    dd = pd.read_csv(RAW / "deuda_provincial_panel_1996_2025.csv")
    ff = dd[(dd.provincia == "Santa Cruz") & (dd.variable == "deuda_ffdp") & dd.anio.isin([2023, 2024])].set_index("anio").valor_millones_pesos_corrientes
    print("\nSanta Cruz, Fondo Fiduciario de Desarrollo Provincial (FFDP): stock 2023 = %.0f -> 2024 = %.0f; amortización 2024 = %.0f."
          % (ff.loc[2023], ff.loc[2024], sc.loc[2024]))
    res.round(4).to_csv(OUT / "I3_resolucion_saltos_de_servicio.csv", index=False)

    # ------------------------------------------------------------------ 4. ¿Importa La Pampa 2024 para el índice?
    print("\n" + "=" * 100 + "\n4. SENSIBILIDAD: LA PAMPA 2024 CON EL STOCK DE 2023 EN LUGAR DE 500\n" + "=" * 100)
    p = dp.construir_panel_maestro()
    a = ind.construir_indice(p)
    q = p.copy()
    m = (q.provincia == "La Pampa") & (q.anio == 2024)
    q.loc[m, "deuda_stock_total"] = float(p.loc[(p.provincia == "La Pampa") & (p.anio == 2023), "deuda_stock_total"].iloc[0])
    q.loc[m, "deuda_sobre_ingresos"] = q.loc[m, "deuda_stock_total"] / q.loc[m, "ingresos_corrientes"]
    b = ind.construir_indice(q)
    for nombre, df in [("stock 2024 = 500 (como viene)", a), ("stock 2024 = 2.242 (valor 2023)", b)]:
        r = df[(df.provincia == "La Pampa") & (df.anio == 2024)].iloc[0]
        print(f"{nombre}: Deuda/Ingresos = {r.deuda_sobre_ingresos:.5f} · S_solvencia = {r.S_solvencia:.1f} · IFFS = {r.IFFS:.1f} · puesto (1 = más frágil) = "
              f"{int((df[df.anio == 2024].IFFS > r.IFFS).sum() + 1)} de {int(df[df.anio == 2024].IFFS.notna().sum())}")
    dif = (a[a.anio == 2024].set_index("provincia").IFFS - b[b.anio == 2024].set_index("provincia").IFFS).drop("La Pampa").abs().max()
    print(f"Cambio máximo del IFFS 2024 en las demás jurisdicciones: {dif:.2f} puntos")
    print(f"\nTablas guardadas en {OUT}")


def resumen() -> dict:
    """Cifras clave de la Fase 2 (las consume scripts/generar_resultados.py para el JSON único de resultados)."""
    x = con_unidad_stock(flujos(), "millones")
    otros = x[(x.provincia != "La Pampa") & (x.stock_prev > 0)]
    sosp = otros[(otros.amort_sobre_stock_prev > 50) | (otros.tasa_implicita > 20)]
    xm = con_unidad_stock(flujos(), "miles")
    lp_m = xm[(xm.provincia == "La Pampa") & xm.anio.between(2006, 2015)].tasa_implicita
    lp_c = x[(x.provincia == "La Pampa") & x.anio.between(2006, 2015)].tasa_implicita
    sl = x[(x.provincia == "San Luis") & x.anio.isin([2019, 2020, 2021, 2022])]
    sc = pd.read_csv(RAW / "servicio_deuda_provincial_panel_2005_2025.csv")
    sc = sc[(sc.provincia == "Santa Cruz") & (sc.anio == 2024) & (sc.variable == "servicio_ffdp_amortizacion")].valor_millones_pesos_corrientes.iloc[0]
    dd = pd.read_csv(RAW / "deuda_provincial_panel_1996_2025.csv")
    ff = dd[(dd.provincia == "Santa Cruz") & (dd.variable == "deuda_ffdp")].set_index("anio").valor_millones_pesos_corrientes
    return {
        "n_obs_identidad": int(len(otros)), "n_sospechosas_fuera_la_pampa": int(len(sosp)),
        "amort_sobre_stock_p5_p95": [round(float(v), 3) for v in otros.amort_sobre_stock_prev.quantile([.05, .95])],
        "tasa_implicita_p5_p95": [round(float(v), 3) for v in otros.tasa_implicita.quantile([.05, .95])],
        "la_pampa_tasa_2006_2015_mediana_si_miles": round(float(lp_m.median()), 3),
        "la_pampa_tasa_2006_2015_mediana_si_millones": round(float(lp_c.median()), 5),
        "san_luis_max_diferencia_amort_vs_caida": round(float((sl.amortizacion + sl.delta_stock).abs().max()), 6),
        "santa_cruz_ffdp_stock_2023_2024": [round(float(ff.loc[2023])), round(float(ff.loc[2024]))],
        "santa_cruz_ffdp_amortizacion_2024": round(float(sc)),
    }


if __name__ == "__main__":
    main()
