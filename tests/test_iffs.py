"""Pruebas automáticas del IFFS v2: datos (CABA, La Pampa), construcción del índice, reglas de diseño y consistencia de resultados.

Ejecutar desde la raíz del proyecto:  python -m pytest -q
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import analytics as an                 # noqa: E402
from src import data_processing as dp           # noqa: E402
from src import indicators as ind               # noqa: E402
from src import pipeline as pl                  # noqa: E402
from src import validation as val               # noqa: E402


@pytest.fixture(scope="module")
def panel_maestro():
    return dp.construir_panel_maestro()


@pytest.fixture(scope="module")
def todo():
    return pl.construir_todo("excluir")


# ------------------------------------------------------------------ datos
def test_grilla_completa_sin_duplicados(panel_maestro):
    assert len(panel_maestro) == 24 * 21
    assert panel_maestro["provincia"].nunique() == 24
    assert not panel_maestro.duplicated(["provincia", "anio"]).any()
    assert "Santiago del Estero" in set(panel_maestro["provincia"])       # espacios normalizados


def test_caba_servicio_en_millones_y_validacion_de_hojas():
    serv = dp.cargar_servicio_caba_millones()             # falla con ValueError si el cociente miles/millones != 1000
    assert serv.notna().all() and (serv > 0).all()


def test_caba_servicio_sobre_ingresos_en_rango_del_resto(panel_maestro):
    caba = panel_maestro[panel_maestro.provincia == "CABA"]["servicio_sobre_ingresos"]
    resto = panel_maestro[panel_maestro.provincia != "CABA"]["servicio_sobre_ingresos"]
    assert caba.notna().all()
    assert caba.max() <= resto.max()                       # antes del arreglo era ~75x los ingresos
    assert 0.02 < caba.median() < 0.10


def test_la_pampa_excluida_por_defecto_y_original_preservado(panel_maestro):
    lp = panel_maestro[(panel_maestro.provincia == "La Pampa") & (panel_maestro.anio <= 2017)]
    assert lp["deuda_sobre_ingresos"].isna().all() and lp["deuda_stock_total"].isna().all()
    assert lp["deuda_stock_total_fuente"].notna().all()    # el dato original queda como rastro
    assert (lp["stock_estado"] == "no_confiable_excluido").all()


def test_la_pampa_sensibilidad_y_sin_tratar():
    exc = dp.construir_panel_maestro(la_pampa="excluir")
    mil = dp.construir_panel_maestro(la_pampa="sensibilidad_miles")
    raw = dp.construir_panel_maestro(la_pampa="sin_tratar")
    m = (raw.provincia == "La Pampa") & (raw.anio <= 2017)
    assert np.allclose(mil.loc[m, "deuda_stock_total"], raw.loc[m, "deuda_stock_total"] / 1000)
    assert (mil.loc[m, "stock_estado"] == "supuesto_miles").all()
    assert raw.loc[m, "deuda_sobre_ingresos"].max() > 100
    # fuera de La Pampa 2005-2017 los tres tratamientos coinciden
    pd.testing.assert_frame_equal(exc.loc[~m, ["deuda_sobre_ingresos"]], raw.loc[~m, ["deuda_sobre_ingresos"]])


def test_ingresos_cero_son_faltantes(panel_maestro):
    assert (panel_maestro["ingresos_corrientes"].dropna() > 0).all()
    x = panel_maestro[(panel_maestro.provincia == "La Pampa") & (panel_maestro.anio == 2025)]
    assert x[["rigidez_gastototal", "fragilidad_resultado", "servicio_sobre_ingresos"]].isna().all(axis=None)


# ------------------------------------------------------------------ índice
def test_percentiles_corte_anual_media_50_y_rango(todo):
    p = todo["panel"]
    for d in ind.DIMS:
        m = p.groupby("anio")[f"S_{d}"].mean()
        assert np.allclose(m, 50.0)
        assert p[f"S_{d}"].dropna().between(0, 100).all()
    assert p["IFFS"].dropna().between(0, 100).all()


def test_compuesto_exige_tres_dimensiones_y_no_incluye_coyuntural(todo):
    p = todo["panel"]
    faltan = p[COLS := ind.COLS_S_NUCLEO].isna().any(axis=1)
    assert p.loc[faltan, "IFFS"].isna().all()
    ok = ~faltan
    assert np.allclose(p.loc[ok, "IFFS"], p.loc[ok, COLS].mean(axis=1))
    lp = p[(p.provincia == "La Pampa") & (p.anio <= 2017)]
    assert lp["IFFS"].isna().all() and lp["S_liquidez"].notna().all()      # subíndices con dato se conservan


def test_regla_ex_ante_coincide_con_la_especificacion_principal(todo):
    t = val.criterio_ventanas_y_composicion(todo["panel"])
    elegidas = t.groupby("dimension")["ventana_elegida"].first()
    assert elegidas["liquidez"] == ind.ESPEC_PRINCIPAL.suavizado_liquidez
    entran = t.groupby("dimension")["entra_al_compuesto"].first()
    assert {d for d, v in entran.items() if v} == set(ind.ESPEC_PRINCIPAL.dims_compuesto)


def test_ventana_movil_no_mira_hacia_adelante(panel_maestro):
    spec = replace(ind.ESPEC_PRINCIPAL, normalizacion="ventana_movil")
    a = ind.construir_indice(panel_maestro, spec)
    alt = panel_maestro.copy()
    alt.loc[alt.anio >= 2016, "deuda_sobre_ingresos"] *= 7          # cambiar el futuro
    b = ind.construir_indice(alt, spec)
    hasta = a["anio"] <= 2015
    pd.testing.assert_series_equal(a.loc[hasta, "S_solvencia"], b.loc[hasta, "S_solvencia"])


def test_percentil_robusto_a_error_de_unidades_pero_zscore_no(panel_maestro):
    raw = dp.construir_panel_maestro(la_pampa="sin_tratar")
    res = val.sensibilidad_la_pampa(panel_maestro, dp.construir_panel_maestro(la_pampa="sensibilidad_miles"), raw)
    z = res[(res.normalizacion == "zscore_base") & (res.tratamiento_vs_excluir == "sin_tratar")]["cambio_medio_abs_subindice_solvencia_resto"].iloc[0]
    c = res[(res.normalizacion == "corte_anual") & (res.tratamiento_vs_excluir == "sin_tratar")]["cambio_medio_abs_subindice_solvencia_resto"].iloc[0]
    assert z > 5 * c


def test_simulador_escenario_nulo_reproduce_el_indice(todo):
    p = todo["panel"]
    for prov in ["La Rioja", "CABA", "Chubut"]:
        r = an.simular_escenario(p, prov, 2023)
        assert r["iffs_base"] == pytest.approx(p[(p.provincia == prov) & (p.anio == 2023)]["IFFS"].iloc[0])
        assert r["iffs_escenario"] == pytest.approx(r["iffs_base"])


# ------------------------------------------------------------------ estadística
def test_auc_rangos_igual_a_mann_whitney():
    rng = np.random.default_rng(3)
    x1, x0 = rng.normal(0.8, 1, 12), rng.normal(0, 1, 14)
    assert val.auc_rangos(x1, x0) == pytest.approx(stats.mannwhitneyu(x1, x0)[0] / (12 * 14))
    assert val.auc_rangos(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5


def test_eventos_tabla(todo):
    t = val.tabla_eventos(todo["eventos"])
    assert len(t) == 13 and (t["anio_evento"] == 2020).sum() == 12
    assert t.loc[t.bloque == "COVID 2020-21", "provincia"].nunique() == 12


def test_split_temporal_no_es_viable(todo):
    from src import modeling as mdl
    r = mdl.intento_de_split_temporal(mdl.construir_target_estres(todo["panel"], todo["eventos"]))
    assert r["eventos_entrenamiento"] == 0 and not r["viable"]


# ------------------------------------------------------------------ consistencia de productos
def test_resultados_clave_corresponden_al_codigo_actual():
    f = ROOT / "outputs" / "resultados_clave.json"
    if not f.exists():
        pytest.skip("Aún no se ejecutó scripts/generar_resultados.py")
    R = json.loads(f.read_text(encoding="utf-8"))
    assert R["huella"] == pl.huella_del_proyecto(), "outputs/ está desactualizado: volver a correr scripts/generar_resultados.py"


def test_panel_procesado_coincide_con_pipeline(todo):
    f = ROOT / "data" / "processed" / "panel_iffs_scores.csv"
    if not f.exists():
        pytest.skip("Sin panel procesado")
    d = pd.read_csv(f)
    assert len(d) == len(todo["panel"])
    a = d.sort_values(["provincia", "anio"])["IFFS"].to_numpy()
    b = todo["panel"].sort_values(["provincia", "anio"])["IFFS"].to_numpy()
    assert np.allclose(a, b, equal_nan=True)


# ------------------------------------------------------------------ formato y auditoría Fase 2
def test_formato_castellano():
    from src.formato import es
    assert es("AUC 0.93, 165.7x, 1.000 sorteos, 555.836, 0.020") == "AUC 0,93, 165,7x, 1.000 sorteos, 555.836, 0,020"
    assert es("T-5->T-1 v2 archivo.csv") == "T-5->T-1 v2 archivo.csv"


def test_fase2_saltos_de_servicio_sin_errores_de_unidades():
    sys.path.insert(0, str(ROOT / "auditoria"))
    import fase2_servicio_y_unidades as f2
    r = f2.resumen()
    assert r["n_sospechosas_fuera_la_pampa"] == 0                    # ninguna otra jurisdicción con error x1000
    assert r["san_luis_max_diferencia_amort_vs_caida"] < 1e-3        # amortización = caída del stock
    assert r["santa_cruz_ffdp_stock_2023_2024"][1] == 0 and r["santa_cruz_ffdp_amortizacion_2024"] > r["santa_cruz_ffdp_stock_2023_2024"][0]
    assert r["la_pampa_tasa_2006_2015_mediana_si_miles"] > 100 * r["la_pampa_tasa_2006_2015_mediana_si_millones"]
