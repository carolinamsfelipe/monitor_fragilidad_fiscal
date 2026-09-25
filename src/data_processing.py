"""
data_processing.py
===================

Carga, validación y armonización de los paneles fiscales/de deuda provinciales
que alimentan el Monitor de Fragilidad Fiscal Provincial.

Los datos base (en data/raw/base_investigacion/) provienen de un pipeline de
investigación previo y ya auditado (Taller de Indicadores Económicos, UADE,
2026), que procesó y verificó contra fuentes oficiales (DNAP, BCRA, Ejecución
Presupuestaria Provincial) los siguientes paneles:

- panel_fiscal_provincial.csv       -> ejecución presupuestaria (ingresos, gastos)
- panel_fiscal_benchmarks.csv       -> ratios fiscales derivados (dependencia,
                                        rigidez, resultado)
- deuda_provincial_panel_1996_2025.csv -> stock de deuda por acreedor
- servicio_deuda_provincial_panel_2005_2025.csv -> servicio de deuda (amortización+interés)
- deuda_ratios_provincia_anio.csv   -> deuda/ingresos y servicio/ingresos ya cruzados
- deuda_moneda_panel.csv            -> composición de deuda por moneda (cobertura parcial)
- capacidad_respuesta_panel.csv     -> liquidez y flexibilidad de gasto (BCRA)
- eventos_estres_provincial.csv     -> base de eventos de default/reestructuración
                                        (construida para este proyecto a partir de
                                        docs 08 y 11 de la investigación original;
                                        no existía como archivo tabular)

Este módulo NO reconstruye esos paneles desde el Excel crudo de la DNAP/BCRA
(ese trabajo, ya hecho y verificado con <0.4% de diferencia contra el consolidado
oficial, está documentado en el proyecto de investigación original) -- reutiliza
esos outputs como insumo, verificándolos con validaciones de esquema explícitas,
y a partir de ahí construye el panel maestro que usa el resto de la aplicación.

Todas las rutas son relativas a la raíz del proyecto.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw" / "base_investigacion"
DNAP_DIR = ROOT_DIR / "data" / "raw" / "fuentes_dnap"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# STOCK de deuda de La Pampa 2005-2017 -- DATO NO CONFIABLE (auditoría Fase 1, hallazgo H2).
# [EVIDENCIA] Cargado como millones, La Pampa sería 6x la deuda de las otras 23 provincias juntas en 1996 y 165x sus
# ingresos en 2005: imposible. La serie además salta de forma errática (2015: 79.418 -> 2016: 555.836 -> 2017: 33.436 ->
# 2018: 1.343) y el archivo de la DNAP declara "millones" y "miles" a la vez.
# [SUPUESTO, NO CONFIRMADO] Que el archivo esté en miles hasta 2017 y en millones desde 2018. No hay confirmación oficial
# de la DNAP ni de la cuenta de inversión provincial.
# [DECISIÓN DE DISEÑO] En el análisis principal esos años se tratan como FALTANTES (no se corrigen ni se descartan en
# silencio: el valor original se conserva en `deuda_stock_total_fuente` y se marca en `stock_estado`). La corrección
# ÷1.000 existe SOLO como escenario de sensibilidad, rotulado como supuesto (`la_pampa="sensibilidad_miles"`).
LA_PAMPA_STOCK_DUDOSO_HASTA = 2017
TRATAMIENTOS_LA_PAMPA = ("excluir", "sensibilidad_miles", "sin_tratar")  # 'sin_tratar' = SOLO auditoría (usa el dato tal cual viene)
ANIOS_PANEL = tuple(range(2005, 2026))

PROVINCIAS_24 = [
    "Buenos Aires", "Catamarca", "Chaco", "Chubut", "Ciudad", "Córdoba",
    "Corrientes", "Entre Ríos", "Formosa", "Jujuy", "La Pampa", "La Rioja",
    "Mendoza", "Misiones", "Neuquén", "Río Negro", "Salta", "San Juan",
    "San Luis", "Santa Cruz", "Santa Fe", "Santiago del Estero",
    "Tierra del Fuego", "Tucumán",
]

# Nombre "oficial" usado en la app para la Ciudad de Buenos Aires.
CABA_LABEL = "CABA"


# ---------------------------------------------------------------------------
# Utilidades de validación
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    """Lanza un error claro si faltan columnas esperadas -- evita fallas silenciosas."""
    faltantes = set(columns) - set(df.columns)
    if faltantes:
        raise ValueError(
            f"[{source}] Faltan columnas esperadas: {sorted(faltantes)}. "
            f"Columnas disponibles: {sorted(df.columns)}"
        )


def _normalizar_provincia(serie: pd.Series) -> pd.Series:
    """Uniformiza el nombre de la Ciudad de Buenos Aires ('Ciudad' -> 'CABA')."""
    serie = serie.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()  # 'Santiago del  Estero' -> 'Santiago del Estero'
    return serie.replace({"Ciudad": CABA_LABEL, "Ciudad de Buenos Aires": CABA_LABEL})


def _safe_divide(numerador: pd.Series, denominador: pd.Series) -> pd.Series:
    """División que evita errores por cero: devuelve NaN en vez de inf/-inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        resultado = numerador / denominador
    return resultado.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Loaders individuales
# ---------------------------------------------------------------------------

def load_panel_fiscal(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "panel_fiscal_provincial.csv")
    _require_columns(df, ["provincia", "anio", "variable", "valor_millones_pesos"], "panel_fiscal_provincial")
    df["provincia"] = _normalizar_provincia(df["provincia"])
    df["anio"] = df["anio"].astype(int)
    return df


def load_panel_fiscal_benchmarks(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "panel_fiscal_benchmarks.csv")
    _require_columns(
        df,
        ["provincia", "anio", "B_dependencia_transferencias", "C_rigidez_personal",
         "D_resultado_fiscal_pct_ingresos", "ingresos_corrientes", "gastos_totales"],
        "panel_fiscal_benchmarks",
    )
    df["provincia"] = _normalizar_provincia(df["provincia"])
    df["anio"] = df["anio"].astype(int)
    return df


def load_deuda_ratios(raw_dir: Path = RAW_DIR, la_pampa: str = "excluir") -> pd.DataFrame:
    """
    Panel de deuda/servicio sobre ingresos con los tratamientos de la auditoría ya aplicados:

    - CABA: servicio corregido desde la hoja `Millones` de la DNAP (error de unidades x1000).
    - La Pampa 2005-2017: stock no confiable -> NaN (`la_pampa="excluir"`, por defecto) o, solo como SENSIBILIDAD
      etiquetada como supuesto, dividido por 1.000 (`la_pampa="sensibilidad_miles"`). `"sin_tratar"` deja el dato tal
      cual viene y existe únicamente para medir en la auditoría cuánto distorsionaría un error de unidades.
    - Ingresos corrientes <= 0 (dato no reportado codificado como cero) -> NaN.
    """
    if la_pampa not in TRATAMIENTOS_LA_PAMPA:
        raise ValueError(f"la_pampa debe ser uno de {TRATAMIENTOS_LA_PAMPA}, no {la_pampa!r}")
    df = pd.read_csv(raw_dir / "deuda_ratios_provincia_anio.csv")
    _require_columns(
        df,
        ["provincia", "anio", "deuda_stock_total", "ingresos_corrientes", "servicio_total",
         "deuda_sobre_ingresos", "servicio_sobre_ingresos", "servicio_confiable"],
        "deuda_ratios_provincia_anio",
    )
    df["provincia"] = _normalizar_provincia(df["provincia"])
    df["anio"] = df["anio"].astype(int)
    df["servicio_confiable"] = df["servicio_confiable"].astype(bool)
    df = _corregir_servicio_caba(df)
    # Cualquier otra observación marcada como no confiable se anula explícitamente (no se deja un número dudoso).
    mask_no_confiable = ~df["servicio_confiable"]
    df.loc[mask_no_confiable, "servicio_sobre_ingresos"] = np.nan
    df.loc[mask_no_confiable, "servicio_total"] = np.nan
    # Ingresos <= 0: dato no reportado (p. ej. La Pampa 2025), no un cero real.
    sin_ingresos = ~(df["ingresos_corrientes"] > 0)
    df.loc[sin_ingresos, ["ingresos_corrientes", "deuda_sobre_ingresos", "servicio_sobre_ingresos"]] = np.nan
    df = _tratar_stock_la_pampa(df, la_pampa)
    return df


def _tratar_stock_la_pampa(df: pd.DataFrame, modo: str) -> pd.DataFrame:
    """
    Trata el stock de La Pampa 2005-2017 (ver constantes al inicio del módulo). Siempre conserva el valor original en
    `deuda_stock_total_fuente` y deja rastro en `stock_estado` ('confiable' | 'no_confiable_excluido' | 'supuesto_miles').
    """
    df = df.copy()
    df["deuda_stock_total_fuente"] = df["deuda_stock_total"]
    df["stock_estado"] = "confiable"
    m = (df["provincia"] == "La Pampa") & (df["anio"] <= LA_PAMPA_STOCK_DUDOSO_HASTA)
    if modo == "excluir":
        df.loc[m, ["deuda_stock_total", "deuda_sobre_ingresos"]] = np.nan
        df.loc[m, "stock_estado"] = "no_confiable_excluido"
    elif modo == "sin_tratar":                     # solo para medir cuánto distorsionaría el dato sin tratar
        df.loc[m, "stock_estado"] = "sin_tratar_dudoso"
    else:
        df.loc[m, "deuda_stock_total"] = df.loc[m, "deuda_stock_total"] / 1000.0
        df.loc[m, "deuda_sobre_ingresos"] = _safe_divide(df.loc[m, "deuda_stock_total"], df.loc[m, "ingresos_corrientes"])
        df.loc[m, "stock_estado"] = "supuesto_miles"
    return df


def cargar_servicio_caba_millones(dnap_dir: Path = DNAP_DIR) -> pd.Series:
    """
    Servicio de deuda de CABA (amortización + interés, millones de pesos corrientes) leído de la hoja `Millones`
    del archivo de la DNAP.

    CORRECCIÓN DE UN ERROR DE UNIDADES (auditoría Fase 1, hallazgo H1): el pipeline original leía la primera hoja
    (`CABA`, expresada en MILES de pesos) y la trataba como millones, inflando el servicio x1000 y llevando a
    excluir a CABA del índice. La hoja `Millones` es la misma información en millones.

    Se suman las 8 categorías de acreedor (mismo criterio que para las otras 23 provincias) y se EXCLUYE la
    columna `TOTAL (5)` para no duplicar. Se valida contra la hoja en miles: el cociente debe ser exactamente 1000.
    """
    archivo = dnap_dir / "servicio_caba.xlsx"
    def _leer(hoja):
        raw = pd.read_excel(archivo, sheet_name=hoja, header=None)
        categorias = raw.iloc[5, 2:].ffill().astype(str).str.strip().str.upper()
        cuerpo = raw.iloc[7:, 1:]
        cuerpo = cuerpo[pd.to_numeric(cuerpo.iloc[:, 0], errors="coerce").between(2000, 2100)]
        anios = cuerpo.iloc[:, 0].astype(int).to_numpy()
        vals = cuerpo.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        vals.columns = categorias.to_numpy()[: vals.shape[1]]
        vals = vals.loc[:, [not c.startswith("TOTAL") for c in vals.columns]]
        return vals.sum(axis=1).set_axis(anios)

    millones, miles = _leer("Millones"), _leer("CABA")
    cociente = (miles / millones).round(3)
    if not (cociente == 1000).all():
        raise ValueError(f"Las hojas 'CABA' (miles) y 'Millones' de CABA no son consistentes: {cociente.unique()}")
    return millones.rename("servicio_total")


def _corregir_servicio_caba(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza el servicio de CABA por el valor corregido en millones y lo marca como confiable."""
    df = df.copy()
    es_caba = df["provincia"] == CABA_LABEL
    if not es_caba.any() or not (DNAP_DIR / "servicio_caba.xlsx").exists():
        return df  # sin la fuente primaria disponible se conserva el tratamiento previo (CABA anulada)
    serv = cargar_servicio_caba_millones()
    nuevo = df.loc[es_caba, "anio"].map(serv)
    df.loc[es_caba, "servicio_total"] = nuevo
    df.loc[es_caba, "servicio_sobre_ingresos"] = _safe_divide(nuevo, df.loc[es_caba, "ingresos_corrientes"])
    df.loc[es_caba, "servicio_confiable"] = nuevo.notna()
    return df


def load_deuda_moneda(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "deuda_moneda_panel.csv")
    _require_columns(
        df,
        ["provincia", "anio", "deuda_moneda_nacional", "deuda_moneda_extranjera",
         "deuda_total_moneda", "pct_moneda_extranjera"],
        "deuda_moneda_panel",
    )
    df["provincia"] = _normalizar_provincia(df["provincia"])
    df["anio"] = df["anio"].astype(int)
    return df


def load_capacidad_respuesta(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "capacidad_respuesta_panel.csv")
    df.columns = [c.lower() for c in df.columns]
    _require_columns(
        df, ["provincia", "anio", "liquidezmeses", "flexibilidad", "capacidadoperativa"],
        "capacidad_respuesta_panel",
    )
    df["provincia"] = _normalizar_provincia(df["provincia"])
    df["anio"] = df["anio"].astype(int)
    return df


def load_eventos_estres(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "eventos_estres_provincial.csv")
    _require_columns(df, ["provincia", "anio_evento", "tipo_evento"], "eventos_estres_provincial")
    df["provincia"] = _normalizar_provincia(df["provincia"])
    df["anio_evento"] = df["anio_evento"].astype(int)
    return df


def load_composicion_deuda(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Participación de los títulos públicos (bonos) en el stock de deuda (%). Es un cociente entre columnas del mismo
    archivo, por lo que no depende de la unidad del archivo (sirve incluso para La Pampa 2005-2017)."""
    d = pd.read_csv(raw_dir / "deuda_provincial_panel_1996_2025.csv")
    _require_columns(d, ["provincia", "anio", "variable", "valor_millones_pesos_corrientes"], "deuda_provincial_panel")
    d["provincia"] = _normalizar_provincia(d["provincia"])
    w = d.pivot_table(index=["provincia", "anio"], columns="variable", values="valor_millones_pesos_corrientes").reset_index()
    w["pct_bonos"] = 100.0 * _safe_divide(w["deuda_titulos_publicos_bonos"], w["deuda_total_sin_flotante"])
    return w[["provincia", "anio", "pct_bonos"]]


# ---------------------------------------------------------------------------
# Panel maestro
# ---------------------------------------------------------------------------

def construir_panel_maestro(raw_dir: Path = RAW_DIR, la_pampa: str = "excluir") -> pd.DataFrame:
    """
    Arma el panel Provincia x Año COMPLETO (24 jurisdicciones x 2005-2025 = 504 filas) con todas las variables fiscales,
    de deuda y de capacidad de respuesta necesarias para construir el IFFS.

    Los valores faltantes se dejan como NaN explícito -- ninguna imputación. Las filas que la fuente no trae (p. ej.
    La Pampa 2025 en el panel de deuda) se crean vacías para que la cobertura sea visible y contable.
    `la_pampa` controla el tratamiento del stock dudoso de La Pampa (ver `TRATAMIENTOS_LA_PAMPA`).
    """
    ratios = load_deuda_ratios(raw_dir, la_pampa=la_pampa)
    benchmarks = load_panel_fiscal_benchmarks(raw_dir)
    capacidad = load_capacidad_respuesta(raw_dir)
    moneda = load_deuda_moneda(raw_dir)
    comp = load_composicion_deuda(raw_dir)

    provincias = sorted(set(ratios["provincia"]) | set(benchmarks["provincia"]))
    grilla = pd.MultiIndex.from_product([provincias, ANIOS_PANEL], names=["provincia", "anio"]).to_frame(index=False)
    panel = grilla.merge(ratios, on=["provincia", "anio"], how="left")
    panel["stock_estado"] = panel["stock_estado"].fillna("sin_dato")

    # Un cero en ejecución presupuestaria es un dato no reportado (La Pampa 2025), no un valor real.
    cols_bench = ["provincia", "anio", "B_dependencia_transferencias", "C_rigidez_personal",
                  "D_resultado_fiscal_pct_ingresos", "gastos_totales", "gasto_personal", "ingresos_totales"]
    panel = panel.merge(benchmarks[cols_bench], on=["provincia", "anio"], how="left")
    sin_ejec = ~(panel["gastos_totales"] > 0)
    panel.loc[sin_ejec, ["gastos_totales", "gasto_personal", "ingresos_totales"]] = np.nan

    # Rigidez del gasto = personal / gasto total (se prefiere sobre Personal/Ingresos para no duplicar el shock de
    # ingresos ya capturado en Deuda/Ingresos).
    panel["rigidez_gastototal"] = _safe_divide(panel["gasto_personal"], panel["gastos_totales"])

    # Fragilidad del resultado = -(resultado financiero / ingresos), transformado para que "mayor = más frágil".
    panel["fragilidad_resultado"] = -panel["D_resultado_fiscal_pct_ingresos"]

    panel = panel.merge(
        capacidad[["provincia", "anio", "liquidezmeses", "flexibilidad", "capacidadoperativa"]],
        on=["provincia", "anio"], how="left",
    )
    panel = panel.merge(moneda[["provincia", "anio", "pct_moneda_extranjera"]], on=["provincia", "anio"], how="left")
    panel = panel.merge(comp, on=["provincia", "anio"], how="left")

    panel = panel.rename(columns={"B_dependencia_transferencias": "dependencia_transferencias"})
    panel = panel.sort_values(["provincia", "anio"]).reset_index(drop=True)
    return panel


def validar_panel_maestro(panel: pd.DataFrame) -> list[str]:
    """
    Batería de controles sobre el panel maestro. Devuelve avisos en texto plano (no lanza excepción: el panel se sabe
    incompleto en partes y eso está documentado como limitación).
    """
    avisos = []

    n_prov = panel["provincia"].nunique()
    if n_prov != 24:
        avisos.append(f"Se esperaban 24 jurisdicciones, se encontraron {n_prov}.")
    dup = panel.duplicated(subset=["provincia", "anio"]).sum()
    if dup:
        avisos.append(f"{dup} filas duplicadas en (provincia, año).")
    if len(panel) != 24 * len(ANIOS_PANEL):
        avisos.append(f"El panel tiene {len(panel)} filas; se esperaban {24 * len(ANIOS_PANEL)}.")

    for col in ["deuda_sobre_ingresos", "servicio_sobre_ingresos", "rigidez_gastototal"]:
        negativos = (panel[col] < 0).sum()
        if negativos:
            avisos.append(f"{negativos} valores negativos en '{col}' (revisar).")
    fuera = panel["pct_moneda_extranjera"].dropna().pipe(lambda s: ((s < 0) | (s > 100)).sum())
    if fuera:
        avisos.append(f"{fuera} valores fuera de [0,100] en 'pct_moneda_extranjera'.")

    if "stock_estado" in panel:
        n_excl = int((panel["stock_estado"] == "no_confiable_excluido").sum())
        n_sup = int((panel["stock_estado"] == "supuesto_miles").sum())
        if n_excl:
            avisos.append(f"La Pampa 2005-{LA_PAMPA_STOCK_DUDOSO_HASTA} ({n_excl} obs.): stock de deuda con posible error de "
                          "unidades y SIN confirmación oficial -> tratado como faltante en el análisis principal "
                          "(no se corrige ni se borra el original: ver `deuda_stock_total_fuente`).")
        if n_sup:
            avisos.append(f"ESCENARIO DE SENSIBILIDAD: La Pampa 2005-{LA_PAMPA_STOCK_DUDOSO_HASTA} ({n_sup} obs.) dividido por 1.000 "
                          "bajo el SUPUESTO (no confirmado) de que la fuente está en miles.")
    altos = panel[panel["deuda_sobre_ingresos"] > 5]
    if not altos.empty:
        avisos.append(f"{len(altos)} observaciones con Deuda/Ingresos > 5x ({', '.join(sorted(altos['provincia'].unique()))}): "
                      "posible error de unidades en la fuente (ver docs/FASE1_AUDITORIA.md).")
    serv = panel["servicio_sobre_ingresos"].dropna()
    implausibles = int((serv > 0.5).sum())
    if implausibles:
        avisos.append(f"{implausibles} observaciones con servicio/ingresos > 50%: posible error de unidades, revisar.")
    if panel.loc[panel["provincia"] == CABA_LABEL, "servicio_sobre_ingresos"].isna().all():
        avisos.append("CABA sin servicio de deuda: no se encontró la fuente `data/raw/fuentes_dnap/servicio_caba.xlsx`.")
    sin_2025 = panel[(panel["anio"] == 2025) & panel["ingresos_corrientes"].isna()]["provincia"].tolist()
    if sin_2025:
        avisos.append(f"Sin ejecución presupuestaria 2025 ({', '.join(sin_2025)}): dato no reportado, queda como faltante.")
    return avisos
