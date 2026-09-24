"""
indicators.py
==============

Construcción del Índice de Fragilidad Fiscal Subnacional (IFFS) y de los
niveles de alerta derivados.

Metodología (replica, con código propio, la especificación congelada en
16-especificacion-iffs-v0.1.md del proyecto de investigación original;
ver también 12-estrategia-modelado-fragilidad.md y 22-validacion-interna-iffs.md):

1. Se seleccionan 4 variables núcleo, una por dimensión de fragilidad, cada
   una con evidencia de calidad de datos suficiente (>90% de cobertura en el
   panel 2005-2025) y fundamento económico directo:
     - Solvencia:          Deuda / Ingresos corrientes
     - Liquidez:           Servicio de deuda / Ingresos corrientes
     - Rigidez del gasto:  Gasto en personal / Gasto total
     - Desempeño fiscal:   -(Resultado financiero / Ingresos corrientes)

   Quedan FUERA del núcleo (y se documentan como limitación, no se ocultan):
     - Dependencia de transferencias nacionales: ambigüedad teórica genuina
       (puede ser estabilizador o fuente de vulnerabilidad según el diseño
       institucional) no resuelta sin mirar el outcome -> se reporta aparte,
       nunca como parte del índice.
     - Composición de deuda en moneda extranjera: cobertura de solo 11-13 de
       24 provincias, en dos ventanas discontinuas -> insuficiente para un
       índice comparable de las 24 jurisdicciones.

2. Cada variable se normaliza a un percentil 0-100 usando una ventana base
   FIJA (2005-2012, pre-muestra de eventos de estrés) -- no z-score, no
   normalización año a año. Esto evita que el propio evento que se quiere
   anticipar (2020-2021) contamine la escala de comparación.

3. Se calculan tres especificaciones para exponer la sensibilidad del
   resultado a decisiones metodológicas (ninguna se presenta como "la
   verdadera"):
     - IFFS-A: promedio simple de percentiles de Deuda y Servicio (exige 2/2).
     - IFFS-B: promedio simple de las 4 dimensiones núcleo (índice principal).
     - IFFS-C: primer componente principal (PCA) de las mismas 4 variables,
       con cargas estimadas únicamente en la ventana base 2005-2012.

4. Pesos iguales entre dimensiones en IFFS-A/B (decisión ex ante, documentada,
   nunca ajustada mirando el outcome de estrés -- evita data snooping).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BASE_WINDOW = (2005, 2012)

DIMENSIONES_NUCLEO = {
    "deuda_sobre_ingresos": "Solvencia (Deuda / Ingresos)",
    "servicio_sobre_ingresos": "Liquidez (Servicio deuda / Ingresos)",
    "rigidez_gastototal": "Rigidez del gasto (Personal / Gasto total)",
    "fragilidad_resultado": "Desempeño fiscal (-Resultado / Ingresos)",
}


def _percentil_ecdf(valores: pd.Series, base: pd.Series) -> pd.Series:
    """
    P(x) = 100 * |{x_j en la base : x_j <= x}| / N_base

    Percentil empírico de cada observación respecto de una muestra base FIJA
    (no respecto de sí misma), evitando que el propio período de interés
    (2013-2025) redefina la escala.
    """
    base_valida = base.dropna().sort_values().to_numpy()
    n_base = len(base_valida)
    if n_base == 0:
        return pd.Series(np.nan, index=valores.index)

    def _pct(x):
        if pd.isna(x):
            return np.nan
        return 100.0 * np.searchsorted(base_valida, x, side="right") / n_base

    return valores.apply(_pct)


def calcular_percentiles(panel: pd.DataFrame, base_window: tuple[int, int] = BASE_WINDOW) -> pd.DataFrame:
    """Agrega columnas P_<variable> con el percentil de cada variable núcleo."""
    panel = panel.copy()
    en_base = panel["anio"].between(*base_window)
    for var in DIMENSIONES_NUCLEO:
        base_valores = panel.loc[en_base, var]
        panel[f"P_{var}"] = _percentil_ecdf(panel[var], base_valores)
    return panel


def _pca_primer_componente(base_df: pd.DataFrame, columnas: list[str]) -> pd.Series:
    """
    PCA de 1 componente estimado SOLO en la ventana base, devuelve las cargas
    (una por columna) orientadas para que 'más frágil' sea positivo.
    """
    from sklearn.decomposition import PCA

    X = base_df[columnas].dropna()
    if len(X) < 5:
        raise ValueError("Muy pocas observaciones en la ventana base para estimar PCA.")
    X_std = (X - X.mean()) / X.std(ddof=0)
    pca = PCA(n_components=1, random_state=42)
    pca.fit(X_std)
    cargas = pd.Series(pca.components_[0], index=columnas)
    # Convención de signo: la carga de Deuda/Ingresos debe ser positiva
    # (a mayor deuda relativa, mayor fragilidad).
    if cargas["P_deuda_sobre_ingresos"] < 0:
        cargas = -cargas
    varianza_explicada = float(pca.explained_variance_ratio_[0])
    return cargas, varianza_explicada


def construir_iffs(panel: pd.DataFrame, base_window: tuple[int, int] = BASE_WINDOW) -> tuple[pd.DataFrame, dict]:
    """
    Devuelve (panel_con_iffs, metadata) donde metadata incluye las cargas del
    PCA y la varianza explicada, para reportarlas en el notebook/dashboard.
    """
    panel = calcular_percentiles(panel, base_window)

    cols_p = [f"P_{v}" for v in DIMENSIONES_NUCLEO]

    # --- IFFS-A: Deuda + Servicio (exige las dos) ---
    cov_a = panel[["P_deuda_sobre_ingresos", "P_servicio_sobre_ingresos"]].notna().mean(axis=1)
    panel["Coverage_A"] = cov_a
    panel["IFFS_A"] = np.where(
        cov_a == 1.0,
        panel[["P_deuda_sobre_ingresos", "P_servicio_sobre_ingresos"]].mean(axis=1),
        np.nan,
    )

    # --- IFFS-B: promedio de las 4 dimensiones núcleo, missing granular ---
    cov_b = panel[cols_p].notna().mean(axis=1)
    panel["Coverage_B"] = cov_b
    panel["IFFS_B"] = panel[cols_p].mean(axis=1, skipna=True)
    panel.loc[cov_b < 0.75, "IFFS_B"] = np.nan  # menos de 3/4 dimensiones -> NaN
    panel["IFFS_B_estado"] = np.select(
        [cov_b == 1.0, cov_b >= 0.75],
        ["completo", "provisional"],
        default="insuficiente",
    )

    # --- IFFS-C: PCA, ventana base ---
    en_base = panel["anio"].between(*base_window)
    cargas, varianza_explicada = _pca_primer_componente(panel.loc[en_base], cols_p)
    cov_c = panel[cols_p].notna().all(axis=1)
    panel["Coverage_C"] = cov_c.astype(float)
    z = (panel[cols_p] - panel.loc[en_base, cols_p].mean()) / panel.loc[en_base, cols_p].std(ddof=0)
    panel["IFFS_C_raw"] = z.mul(cargas.values, axis=1).sum(axis=1)
    panel.loc[~cov_c, "IFFS_C_raw"] = np.nan
    # Reescalado a 0-100 vía percentil ECDF sobre la misma ventana base, para
    # que sea comparable en escala con IFFS-A/B.
    base_c = panel.loc[en_base, "IFFS_C_raw"]
    panel["IFFS_C"] = _percentil_ecdf(panel["IFFS_C_raw"], base_c)

    metadata = {
        "base_window": base_window,
        "cargas_pca": cargas.to_dict(),
        "varianza_explicada_pc1": varianza_explicada,
        "n_obs": len(panel),
    }
    return panel, metadata


# ---------------------------------------------------------------------------
# Niveles de alerta (reglas transparentes, NO umbrales universales)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UmbralAlerta:
    etiqueta: str
    minimo: float
    color: str


UMBRALES_ALERTA_IFFS_B = [
    UmbralAlerta("Bajo", 0, "#2E7D32"),
    UmbralAlerta("Medio", 33, "#F9A825"),
    UmbralAlerta("Alto", 66, "#EF6C00"),
    UmbralAlerta("Crítico", 85, "#C62828"),
]
"""
Umbrales de corte del IFFS-B para clasificar el nivel de alerta.

IMPORTANTE (documentado explícitamente para el dashboard y el informe):
son puntos de corte metodológicos elegidos por practicidad expositiva sobre
una escala de percentiles (33/66/85), NO una calibración estadística contra
el outcome de estrés -- el propio backtest de este proyecto muestra que el
IFFS no tiene, en la evidencia disponible, poder de anticipación lineal
fuerte y generalizado. Se presentan como una ayuda de lectura, no como una
predicción de default.
"""


def asignar_nivel_alerta(iffs_b: pd.Series, umbrales: list[UmbralAlerta] = UMBRALES_ALERTA_IFFS_B) -> pd.Series:
    umbrales_ordenados = sorted(umbrales, key=lambda u: u.minimo)
    bins = [u.minimo for u in umbrales_ordenados] + [100.0001]
    labels = [u.etiqueta for u in umbrales_ordenados]
    return pd.cut(iffs_b, bins=bins, labels=labels, right=False, include_lowest=True)


def descomponer_contribucion(fila: pd.Series) -> pd.DataFrame:
    """
    Devuelve, para una fila (provincia, año) del panel IFFS, la contribución
    de cada dimensión al IFFS-B -- insumo de la explicación automática.
    """
    filas = []
    for var, nombre in DIMENSIONES_NUCLEO.items():
        p = fila.get(f"P_{var}")
        filas.append({"dimension": nombre, "variable": var, "percentil": p})
    return pd.DataFrame(filas).sort_values("percentil", ascending=False)
