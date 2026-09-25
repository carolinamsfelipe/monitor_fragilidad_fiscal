"""
generar_resultados.py
======================

FUENTE ÚNICA DE CIFRAS del proyecto. Calcula toda la evidencia del IFFS v2 y la escribe en:

  outputs/tables/*.csv          tablas (las mismas que lee el dashboard y cita el informe)
  outputs/resultados_clave.json todas las cifras citadas en texto (informe, presentación, README, dashboard)
  outputs/figures/*.png         figuras
  data/processed/panel_iffs_scores.csv   panel completo con subíndices y compuesto

Uso:  python scripts/generar_resultados.py           (semilla fija; ~3-6 minutos por los bootstraps)
Todas las rutas son relativas a la raíz del proyecto.
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import analytics as an                    # noqa: E402
from src import data_processing as dp              # noqa: E402
from src import indicators as ind                  # noqa: E402
from src import modeling as mdl                    # noqa: E402
from src import pipeline as pl                     # noqa: E402
from src import validation as val                  # noqa: E402
from src import visualization as viz               # noqa: E402

TABLAS = ROOT / "outputs" / "tables"
OUT = ROOT / "outputs"
PROCESSED = ROOT / "data" / "processed"
for d in (TABLAS, PROCESSED):
    d.mkdir(parents=True, exist_ok=True)


def _r(x, n=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def main() -> dict:
    R: dict = {"huella": pl.huella_del_proyecto(), "semilla": val.RANDOM_SEED, "n_boot": val.N_BOOT, "n_perm": val.N_PERM}
    T = pl.construir_todo("excluir")
    panel, eventos = T["panel"], T["eventos"]
    panel.to_csv(PROCESSED / "panel_iffs_scores.csv", index=False)

    # ------------------------------------------------------------------ 1. Cobertura y descripción
    cob = val.cobertura(panel)
    R["cobertura"] = {"n_total": cob["n_total"], "n_compuesto": cob["n_compuesto"], "pct_compuesto": _r(100 * cob["pct_compuesto"], 1),
                      "por_variable_pct": {k: _r(100 * v, 1) for k, v in cob["por_variable"].items()},
                      "faltantes_compuesto": cob["faltantes_compuesto"], "n_provincias": int(panel["provincia"].nunique()),
                      "anios": [int(panel["anio"].min()), int(panel["anio"].max())]}
    cp = panel.groupby("provincia").agg(
        anios_con_compuesto=("IFFS", "count"), anios_solvencia=("S_solvencia", "count"), anios_liquidez=("S_liquidez", "count"),
        anios_estructural=("S_estructural", "count"), anios_coyuntural=("S_coyuntural", "count"))
    cp["anios_totales"] = panel["anio"].nunique()
    cp.reset_index().to_csv(TABLAS / "T01_cobertura_por_provincia.csv", index=False)
    R["avisos_datos"] = T["avisos"]
    R["nivel_comun"] = {k: {str(a): _r(v, 3) for a, v in s.items()} for k, s in val.nivel_comun(panel).items()}
    val.nivel_comun(panel).round(4).to_csv(TABLAS / "T15_nivel_comun.csv")

    # ------------------------------------------------------------------ 2. Criterios de diseño (sin eventos)
    props = val.propiedades_variables(panel)
    props.to_csv(TABLAS / "T02_propiedades_variables.csv", index=False)
    regla = val.criterio_ventanas_y_composicion(panel)
    regla.to_csv(TABLAS / "T03_regla_ventanas_composicion.csv", index=False)
    red = val.redundancia_subindices(panel)
    red.to_csv(TABLAS / "T04_redundancia_subindices.csv", index=False)
    pca = val.pca_subindices(panel)
    R["diseno"] = {
        "props": {r.dimension: {"icc": _r(r.icc), "ar1": _r(r.ar1_intra), "persistencia_rangos": _r(r.persistencia_rangos)} for r in props.itertuples()},
        "regla_umbral_persistencia": val.UMBRAL_PERSISTENCIA,
        "entra_al_compuesto": {d: bool(regla[(regla.dimension == d)]["entra_al_compuesto"].iloc[0]) for d in ind.DIMS},
        "ventana_elegida": {d: int(regla[(regla.dimension == d)]["ventana_elegida"].iloc[0]) for d in ind.DIMS},
        "redundancia": {r.par: {"pooled": _r(r.pooled), "entre": _r(r.entre_provincias), "intra": _r(r.intra_provincia)} for r in red.itertuples()},
        "pca_varianza": [_r(x) for x in pca["varianza_explicada"]],
    }
    spec = val.comparar_especificaciones(panel)
    spec.to_csv(TABLAS / "T05_comparacion_especificaciones.csv", index=False)
    R["robustez_diseno"] = {r.especificacion: {"spearman": _r(r.spearman_vs_principal), "deriva_mediana": _r(r.deriva_mediana_anual, 1),
                                                "persistencia": _r(r.persistencia_rangos), "cambio_rango_medio": _r(r.cambio_rango_medio, 2),
                                                "n_ge5": int(r.obs_con_cambio_rango_ge5), "top6_2024": int(r.solapamiento_top6_2024)}
                            for r in spec.itertuples()}
    R["pesos_aleatorios"] = {k: (_r(v) if isinstance(v, float) else v) for k, v in val.sensibilidad_a_pesos(panel).items()}

    # cambios de ranking entre especificaciones (2019 y 2024)
    base = panel.drop(columns=[c for c in panel.columns if c.startswith(("S_", "x_", "IFFS", "LEG_"))], errors="ignore")
    alt = {"principal": ind.ESPEC_PRINCIPAL,
           "base_fija_2005_12 (v1)": replace(ind.ESPEC_PRINCIPAL, normalizacion="base_fija"),
           "ventana_movil_8a": replace(ind.ESPEC_PRINCIPAL, normalizacion="ventana_movil"),
           "zscore_robusto": replace(ind.ESPEC_PRINCIPAL, normalizacion="zscore_robusto_base"),
           "4_dimensiones": replace(ind.ESPEC_PRINCIPAL, dims_compuesto=tuple(ind.DIMS))}
    filas = []
    for a in (2019, 2024):
        dd = pd.DataFrame({"provincia": base.loc[base["anio"] == a, "provincia"].to_numpy()})
        for nombre, sp in alt.items():
            r = ind.construir_indice(base, sp, sufijo="_z")
            x = r.loc[r["anio"] == a, "IFFS_z"]
            dd[nombre] = x.rank(ascending=False, method="min").to_numpy()
        dd.insert(0, "anio", a)
        filas.append(dd)
    rk = pd.concat(filas).sort_values(["anio", "principal"])
    rk.to_csv(TABLAS / "T14_ranking_comparado.csv", index=False)

    # ------------------------------------------------------------------ 3. Eventos: definición y pruebas
    tev = val.tabla_eventos(eventos)
    tev.to_csv(TABLAS / "T18_eventos.csv", index=False)
    provs, anclas = val.grupos_covid(eventos)
    n_amb = int((~tev.loc[tev["bloque"] == "COVID 2020-21", "fecha_puntual"]).sum())
    R["eventos"] = {"n_eventos": int(len(tev)), "n_covid": int((tev["bloque"] == "COVID 2020-21").sum()), "provincias_evento_covid": provs,
                    "n_provincias_evento_covid": len(provs), "n_anio_ambiguo": n_amb, "eventos_por_anio": {str(k): int(v) for k, v in tev.groupby("anio_evento").size().items()},
                    "tasa_base_pct": _r(100 * len(tev) / cob["n_total"], 1)}

    disc = val.discriminacion_transversal(panel, provs, 2019, val.COLUMNAS_DISCRIMINACION)
    disc.to_csv(TABLAS / "T07_discriminacion_2019.csv", index=False)
    R["discriminacion_2019"] = {r.medida: {"auc": _r(r.auc), "ic95": [_r(r.ic95_inf), _r(r.ic95_sup)], "p_perm": _r(r.p_perm_unilateral, 4),
                                            "n_evento": int(r.n_evento), "n_control": int(r.n_control)} for r in disc.itertuples()}
    R["azar_auc_2019"] = {k: _r(v) for k, v in val.placebo_grupos_aleatorios(panel, provs).items()}
    R["condicionado_a_bonos_2019"] = {k: (_r(v) if isinstance(v, float) else v) for k, v in val.discriminacion_condicionada_a_bonos(panel, provs).items()}

    filas = []
    for esc in ("principal", "alt"):
        for etiqueta, col in {"IFFS": "IFFS", "solvencia": "S_solvencia", "liquidez": "S_liquidez", "estructural": "S_estructural",
                              "coyuntural": "S_coyuntural", "bonos_pct": "pct_bonos", "IFFS_v1": "LEG_B"}.items():
            t = val.discriminacion_por_rezago(panel, provs, anclas[esc], col)
            t.insert(0, "medida", etiqueta)
            t.insert(0, "ancla", "2020 (convención)" if esc == "principal" else "2021 para los 11 de fecha ambigua")
            filas.append(t)
    rez = pd.concat(filas)
    rez.to_csv(TABLAS / "T08_discriminacion_por_rezago.csv", index=False)
    R["rezagos"] = {f"{r.ancla}|{r.medida}|T-{r.rezago}": {"auc": _r(r.auc), "ic95": [_r(r.ic95_inf), _r(r.ic95_sup)], "anio_ref": int(r.anio_ref_mediano)}
                    for r in rez.itertuples()}

    pt = {c: val.placebo_temporal(panel, provs, c) for c in ("IFFS", "pct_bonos", "LEG_B", "S_solvencia", "S_liquidez")}
    out = []
    for c, t in pt.items():
        out.append(t.assign(medida=c))
    pd.concat(out).to_csv(TABLAS / "T09_placebo_temporal.csv", index=False)
    piff = pt["IFFS"].set_index("anio")
    R["placebo_temporal"] = {"auc_iffs_por_anio": {str(a): _r(v, 3) for a, v in piff["auc"].items()},
                             "ic95_iffs": {str(a): [_r(piff.loc[a, "ic95_inf"]), _r(piff.loc[a, "ic95_sup"])] for a in piff.index},
                             "auc_bonos_por_anio": {str(a): _r(v, 3) for a, v in pt["pct_bonos"].set_index("anio")["auc"].items()},
                             "primer_anio_ic_sobre_azar": int(piff.index[piff["ic95_inf"] > 0.5].min()),
                             "primer_anio_auc_ge_085": int(piff.index[piff["auc"] >= 0.85].min())}

    pre = []
    for esc in ("principal", "alt"):
        for col in ("IFFS", "S_solvencia", "S_liquidez", "S_estructural", "S_coyuntural"):
            for desde in (5, 3):
                r = val.pretendencia(panel, provs, anclas[esc], col, desde, 1)
                pre.append({"ancla": esc, **r})
    prd = pd.DataFrame(pre)
    prd.to_csv(TABLAS / "T10_pretendencia.csv", index=False)
    R["pretendencia"] = {f"{r.ancla}|{r.col}|{r.desde}->{r.hasta}": {"dif": _r(r.diferencia, 2), "ic95": [_r(r.ic95_inf, 2), _r(r.ic95_sup, 2)],
                                                                  "p": _r(r.p_perm_bilateral), "cambio_evento": _r(r.cambio_medio_evento, 2),
                                                                  "cambio_control": _r(r.cambio_medio_control, 2)} for r in prd.itertuples()}
    es = val.estudio_de_eventos(panel, provs, anclas["principal"], "IFFS")
    es.to_csv(TABLAS / "T11_estudio_eventos.csv", index=False)
    R["estudio_eventos"] = {str(int(r.tau)): {"evento": _r(r.media_evento, 1), "control": _r(r.media_control, 1), "dif": _r(r.diferencia, 1),
                                               "ic95": [_r(r.ic95_inf, 1), _r(r.ic95_sup, 1)]} for r in es.itertuples()}
    lr = val.caso_la_rioja_2025(panel)
    lr.to_csv(TABLAS / "T12_la_rioja_2025.csv", index=False)
    R["la_rioja"] = {str(int(r.anio)): {"IFFS": _r(r.IFFS, 1), "puesto": int(r.puesto_compuesto), "solvencia": _r(r.S_solvencia, 1)} for r in lr.itertuples()}
    nv = val.pruebas_no_viables(panel, eventos)
    nv.to_csv(TABLAS / "T13_pruebas_no_viables.csv", index=False)
    R["split_temporal"] = mdl.intento_de_split_temporal(mdl.construir_target_estres(panel, eventos))
    R["split_temporal"]["eventos_por_anio"] = {str(k): int(v) for k, v in R["split_temporal"]["eventos_por_anio"].items()}

    # AUC 2019 de todas las especificaciones alternativas (TODAS reportadas; no se elige ninguna)
    filas = []
    for etiqueta, sp in [("PRINCIPAL", ind.ESPEC_PRINCIPAL)] + val.alternativas_de_especificacion():
        r = ind.construir_indice(base, sp, sufijo="_z")
        d = r[r["anio"] == 2019][["provincia", "IFFS_z"]].dropna()
        x1, x0 = d.loc[d.provincia.isin(provs), "IFFS_z"].to_numpy(), d.loc[~d.provincia.isin(provs), "IFFS_z"].to_numpy()
        filas.append({"especificacion": etiqueta, "auc_2019": val.auc_rangos(x1, x0), "n_evento": len(x1), "n_control": len(x0)})
    auc_spec = pd.DataFrame(filas)
    auc_spec.to_csv(TABLAS / "T19_auc_2019_todas_las_especificaciones.csv", index=False)
    R["auc_2019_especificaciones"] = {"min": _r(auc_spec["auc_2019"].min()), "max": _r(auc_spec["auc_2019"].max()), "n": int(len(auc_spec)),
                                      "detalle": {r.especificacion: _r(r.auc_2019) for r in auc_spec.itertuples()}}

    # ------------------------------------------------------------------ 4. Sensibilidad a La Pampa (SUPUESTO)
    T_m = pl.construir_todo("sensibilidad_miles")
    T_s = dp.construir_panel_maestro(la_pampa="sin_tratar")
    p_m = T_m["panel"]
    slp = val.sensibilidad_la_pampa(dp.construir_panel_maestro(), dp.construir_panel_maestro(la_pampa="sensibilidad_miles"), T_s)
    slp.to_csv(TABLAS / "T17_sensibilidad_la_pampa.csv", index=False)
    pt_m = val.placebo_temporal(p_m, provs, "IFFS", anios=range(2005, 2018))
    dif_auc = (pt_m.set_index("anio")["auc"] - piff.loc[2005:2017, "auc"]).abs()
    lp_sup = p_m[(p_m.provincia == "La Pampa") & (p_m.anio <= 2017)][["anio", "IFFS"]]
    R["la_pampa"] = {
        "tratamiento_principal": "stock 2005-2017 tratado como faltante (no confiable, sin confirmación oficial)",
        "obs_afectadas": int((panel["stock_estado"] == "no_confiable_excluido").sum()),
        "escenario_supuesto_miles": {"n_compuesto": int(p_m["IFFS"].notna().sum()), "dif_max_auc_placebo_2005_2017": _r(dif_auc.max(), 3),
                                     "iffs_la_pampa_min": _r(lp_sup["IFFS"].min(), 1), "iffs_la_pampa_max": _r(lp_sup["IFFS"].max(), 1)},
        "efecto_sobre_resto": {f"{r.normalizacion}|{r.tratamiento_vs_excluir}": _r(r.cambio_medio_abs_subindice_solvencia_resto, 2) for r in slp.itertuples()},
        "deuda_ingresos_max_base_sin_tratar": _r(T_s.loc[T_s["anio"].between(2005, 2012), "deuda_sobre_ingresos"].max(), 1),
        "deuda_ingresos_max_base_excluido": _r(panel.loc[panel["anio"].between(2005, 2012), "deuda_sobre_ingresos"].max(), 2),
    }

    # ------------------------------------------------------------------ 4b. Saltos de servicio y unidades (Fase 2: identidad stock-flujo)
    import contextlib
    import io

    sys.path.insert(0, str(ROOT / "auditoria"))
    import fase2_servicio_y_unidades as f2

    with contextlib.redirect_stdout(io.StringIO()):
        f2.main()                                    # escribe outputs/auditoria/I1-I3
    R["servicio_unidades"] = f2.resumen()

    # ------------------------------------------------------------------ 5. Datos clave del último año / CABA / legado
    u = 2024
    r24 = panel[panel["anio"] == u].dropna(subset=["IFFS"]).sort_values("IFFS", ascending=False)
    R["ranking_2024"] = {"top6": r24.head(6)[["provincia", "IFFS"]].round(1).to_dict("records"),
                         "bottom3": r24.tail(3)[["provincia", "IFFS"]].round(1).to_dict("records"), "n": int(len(r24))}
    caba = panel[panel.provincia == "CABA"]["servicio_sobre_ingresos"]
    R["caba"] = {"servicio_ingresos_min": _r(caba.min(), 3), "mediana": _r(caba.median(), 3), "max": _r(caba.max(), 3),
                 "otras_p50": _r(panel[panel.provincia != "CABA"]["servicio_sobre_ingresos"].median(), 3)}
    R["perfiles"] = {"k": T["info_perfiles"]["k"], "silhouette": _r(T["info_perfiles"]["silhouette"], 2)}
    bt = T["backtest"]
    R["backtest_proyeccion"] = {"n": bt["n"], "mae_tendencia": _r(bt["mae_tendencia"], 1), "mae_ingenua": _r(bt["mae_ingenua"], 1),
                                "tendencia_gana_pct": _r(bt["tendencia_gana_pct"], 0)}
    R["legado"] = {"varianza_pc1": _r(T["meta_legado"]["varianza_explicada_pc1"], 2)}
    an_d = panel.groupby("anio")["deuda_sobre_ingresos"].median()
    R["deriva_solvencia"] = {"mediana_2005": _r(an_d.loc[2005], 2), "mediana_2012": _r(an_d.loc[2012], 2), "mediana_2024": _r(an_d.loc[2024], 2)}

    # ------------------------------------------------------------------ 6. Figuras
    viz.fig_cobertura(panel)
    viz.fig_ranking(panel, 2024)
    viz.fig_ranking(panel, 2019)
    viz.fig_deriva(panel)
    viz.fig_robustez(spec)
    viz.fig_auc_por_anio(pt["IFFS"], pt["pct_bonos"], pt["LEG_B"])
    viz.fig_estudio_eventos(es)
    viz.fig_heatmap(panel, "IFFS")
    viz.fig_evolucion(panel, ["La Rioja", "Buenos Aires", "CABA", "Santiago del Estero", "Chubut"])
    viz.fig_subindices_provincia(panel, "La Rioja", "subindices_La_Rioja")

    (OUT / "resultados_clave.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("OK: resultados_clave.json, tablas y figuras escritas. Huella:", R["huella"])
    return R


if __name__ == "__main__":
    main()
