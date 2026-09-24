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
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

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


def load_deuda_ratios(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
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
    # CABA: el servicio de deuda no reconcilia con el consolidado oficial (ver
    # 07-auditoria-deuda-provincial.md de la investigación original). Se fuerza
    # a NaN explícitamente en vez de dejar un número que se sabe incorrecto.
    mask_no_confiable = ~df["servicio_confiable"]
    df.loc[mask_no_confiable, "servicio_sobre_ingresos"] = np.nan
    df.loc[mask_no_confiable, "servicio_total"] = np.nan
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


# ---------------------------------------------------------------------------
# Panel maestro
# ---------------------------------------------------------------------------

def construir_panel_maestro(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Arma el panel Provincia x Año (2005-2025, 24 jurisdicciones) con todas las
    variables fiscales, de deuda y de capacidad de respuesta necesarias para
    construir el IFFS y el resto de los indicadores del dashboard.

    Devuelve un DataFrame con una fila por (provincia, año) y columnas anchas.
    Los valores faltantes se dejan como NaN explícito -- ninguna imputación.
    """
    ratios = load_deuda_ratios(raw_dir)
    benchmarks = load_panel_fiscal_benchmarks(raw_dir)
    capacidad = load_capacidad_respuesta(raw_dir)
    moneda = load_deuda_moneda(raw_dir)

    panel = ratios.merge(
        benchmarks[[
            "provincia", "anio", "B_dependencia_transferencias", "C_rigidez_personal",
            "D_resultado_fiscal_pct_ingresos", "gastos_totales", "gasto_personal",
        ]],
        on=["provincia", "anio"], how="left",
    )

    # Rigidez del gasto = personal / gasto total (variable núcleo del IFFS-B,
    # ver 16-especificacion-iffs-v0.1.md: se prefiere sobre Personal/Ingresos
    # para no duplicar el shock de ingresos ya capturado en Deuda/Ingresos).
    panel["rigidez_gastototal"] = _safe_divide(panel["gasto_personal"], panel["gastos_totales"])

    # Fragilidad del resultado = -(resultado financiero / ingresos corrientes),
    # transformado ANTES de percentilar para que "mayor = más frágil" en las
    # cuatro dimensiones (ver 16-especificacion-iffs-v0.1.md).
    panel["fragilidad_resultado"] = -panel["D_resultado_fiscal_pct_ingresos"]

    panel = panel.merge(
        capacidad[["provincia", "anio", "liquidezmeses", "flexibilidad", "capacidadoperativa"]],
        on=["provincia", "anio"], how="left",
    )

    panel = panel.merge(
        moneda[["provincia", "anio", "pct_moneda_extranjera"]],
        on=["provincia", "anio"], how="left",
    )

    panel = panel.rename(columns={"B_dependencia_transferencias": "dependencia_transferencias"})
    panel = panel.sort_values(["provincia", "anio"]).reset_index(drop=True)
    return panel


def validar_panel_maestro(panel: pd.DataFrame) -> list[str]:
    """
    Corre una batería mínima de sanity checks sobre el panel maestro y
    devuelve una lista de advertencias en texto plano (no lanza excepción:
    el panel se sabe incompleto en partes, documentado como limitación).
    """
    avisos = []

    n_prov = panel["provincia"].nunique()
    if n_prov != 24:
        avisos.append(f"Se esperaban 24 jurisdicciones, se encontraron {n_prov}.")

    dup = panel.duplicated(subset=["provincia", "anio"]).sum()
    if dup:
        avisos.append(f"{dup} filas duplicadas en (provincia, año).")

    for col in ["deuda_sobre_ingresos", "servicio_sobre_ingresos", "rigidez_gastototal"]:
        negativos = (panel[col] < 0).sum()
        if negativos:
            avisos.append(f"{negativos} valores negativos en '{col}' (revisar).")

    for col in ["pct_moneda_extranjera"]:
        fuera_rango = panel[col].dropna().pipe(lambda s: ((s < 0) | (s > 100)).sum())
        if fuera_rango:
            avisos.append(f"{fuera_rango} valores fuera de [0,100] en '{col}'.")

    caba_servicio = panel.loc[panel["provincia"] == CABA_LABEL, "servicio_sobre_ingresos"]
    if caba_servicio.notna().any():
        avisos.append("CABA tiene valores de servicio_sobre_ingresos no nulos: revisar exclusión.")

    return avisos
