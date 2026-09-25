"""
indicators.py
==============

Construcción del Índice de Fragilidad Fiscal Subnacional (IFFS) — versión 2 (rediseño metodológico, ver
`docs/METODOLOGIA.md`).

Etiquetas usadas en la documentación:  [EVIDENCIA] calculado en los datos · [ECONÓMICO] interpretación económica ·
[SUPUESTO] no verificado · [DISEÑO] decisión de construcción · [LIMITACIÓN] lo que no se puede resolver.

Estructura del IFFS v2
----------------------
[DISEÑO] Cuatro SUBÍNDICES separados (una variable por dimensión, cada una con significado económico distinto):

  solvencia    Deuda / Ingresos corrientes                          (stock vs. capacidad de pago)
  liquidez     Servicio de deuda / Ingresos corrientes                (presión de caja; lumpy por amortizaciones *bullet*)
  estructural  Personal / Gasto total                               (rigidez del gasto; rasgo casi fijo)
  coyuntural   −Resultado financiero / Ingresos                     (estrés de corto plazo; ruidoso)

y un COMPUESTO `IFFS` = media simple de solvencia, liquidez y estructural (el "núcleo persistente").
El estrés coyuntural NO entra al compuesto: se reporta aparte.

[EVIDENCIA] Justificación sin mirar ningún evento de default (auditoría Fase 1 y `src/validation.py:criterios_diseno`):
el resultado fiscal es 78% transitorio (ICC 0,22) y tiene AR(1) intra-provincia de 0,36; incluirlo en un promedio con
variables persistentes mezcla una señal lenta con ruido. Además el PCA muestra que 1 solo factor explica ~47% de la
varianza: los indicadores NO son intercambiables, por lo que el resumen en un número oculta información.

Normalización [DISEÑO]
----------------------
Por defecto, PERCENTIL DENTRO DE CADA AÑO entre las jurisdicciones con dato ("corte anual"): responde a la pregunta
"¿qué tan frágil es la provincia respecto de sus pares ESE año?". Es una posición RELATIVA (media 50 por construcción);
NO es una probabilidad de default ni una medida absoluta de solvencia. Motivos (no relacionados con eventos):
  1. [EVIDENCIA] La ventana base fija 2005-2012 no es estacionaria (mediana de Deuda/Ingresos: 1,09 en 2005 -> 0,29 en 2012
     -> 0,15 en 2024), de modo que el score por percentil contra esa ventana pierde poder discriminante por construcción.
  2. Es inmune a shocks comunes (inflación, tipo de cambio, COVID): estos se reportan aparte como "nivel común".
  3. No usa información futura ni una ventana arbitraria.
Costo: [LIMITACIÓN] si todas las provincias empeoran a la vez, el score no lo muestra (por eso se informa el nivel común).

Las demás normalizaciones (base fija 2005-2012 = IFFS legado, ventana móvil, z-score, z-score robusto) se calculan como
ESPECIFICACIONES ALTERNATIVAS para el análisis de robustez; ninguna se eligió mirando eventos.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.stats import norm

BASE_WINDOW = (2005, 2012)

# --- Dimensiones -----------------------------------------------------------------------------------------------
DIMENSIONES = {
    "solvencia": {
        "variable": "deuda_sobre_ingresos",
        "nombre": "Solvencia",
        "detalle": "Deuda / Ingresos corrientes",
        "tipo": "stock",
        "descripcion": "Stock de deuda a 31/12 respecto de los ingresos corrientes del año.",
    },
    "liquidez": {
        "variable": "servicio_sobre_ingresos",
        "nombre": "Liquidez",
        "detalle": "Servicio de deuda / Ingresos corrientes",
        "tipo": "flujo",
        "descripcion": "Presión de caja por amortización e intereses. Puede tener picos por vencimientos tipo bullet (ver sensibilidad con media móvil).",
    },
    "estructural": {
        "variable": "rigidez_gastototal",
        "nombre": "Estructural",
        "detalle": "Gasto en personal / Gasto total",
        "tipo": "estructural",
        "descripcion": "Rigidez del gasto: cuánto del gasto es difícil de ajustar. Rasgo casi permanente de cada provincia.",
    },
    "coyuntural": {
        "variable": "fragilidad_resultado",
        "nombre": "Coyuntural",
        "detalle": "−Resultado financiero / Ingresos",
        "tipo": "coyuntura",
        "descripcion": "Déficit del año. Volátil: 78% de su varianza es transitoria (ICC 0,22).",
    },
}
DIMS = list(DIMENSIONES)
DIMS_NUCLEO = ["solvencia", "liquidez", "estructural"]
VARIABLE_DE = {d: v["variable"] for d, v in DIMENSIONES.items()}
NOMBRE_DIM = {d: v["nombre"] for d, v in DIMENSIONES.items()}
COLS_S = [f"S_{d}" for d in DIMS]                 # subíndices (0-100)
COLS_S_NUCLEO = [f"S_{d}" for d in DIMS_NUCLEO]

# Compatibilidad con el IFFS legado (v1): 4 variables, base fija 2005-2012, promedio simple de percentiles.
DIMENSIONES_NUCLEO = {
    "deuda_sobre_ingresos": "Solvencia (Deuda / Ingresos)",
    "servicio_sobre_ingresos": "Liquidez (Servicio deuda / Ingresos)",
    "rigidez_gastototal": "Rigidez del gasto (Personal / Gasto total)",
    "fragilidad_resultado": "Desempeño fiscal (-Resultado / Ingresos)",
}

NORMALIZACIONES = ("corte_anual", "base_fija", "ventana_movil", "zscore_base", "zscore_robusto_base", "pooled_total")
AGREGACIONES = ("media", "geometrica", "potencia2", "maximo")


@dataclass(frozen=True)
class Especificacion:
    """Una especificación completa del índice. La principal es `ESPEC_PRINCIPAL`; el resto son alternativas de robustez."""
    nombre: str = "IFFS v2 (principal)"
    normalizacion: str = "corte_anual"
    base_window: tuple = BASE_WINDOW
    ventana_movil: int = 8          # años previos usados como referencia en 'ventana_movil'
    min_hist_movil: int = 4         # años previos mínimos exigidos en 'ventana_movil'
    suavizado_liquidez: int = 1     # media móvil (años, hacia atrás) de Servicio/Ingresos; 1 = sin suavizar (regla ex ante)
    suavizado_resultado: int = 1
    agregacion: str = "media"
    dims_compuesto: tuple = tuple(DIMS_NUCLEO)
    pesos: tuple = ()               # vacío = pesos iguales entre dims_compuesto; si no, un peso por dimensión
    winsor: float | None = None     # p. ej. 0.05 recorta a los percentiles 5/95 (solo afecta a z-scores)

    def pesos_dict(self) -> dict:
        if self.pesos:
            w = dict(zip(self.dims_compuesto, self.pesos))
        else:
            w = {d: 1.0 for d in self.dims_compuesto}
        tot = sum(w.values())
        return {d: v / tot for d, v in w.items()}


ESPEC_PRINCIPAL = Especificacion()


# --- Normalizaciones ---------------------------------------------------------------------------------------------

def _pct_ecdf_contra(valores: np.ndarray, base: np.ndarray) -> np.ndarray:
    """P(x) = 100 * |{x_j en base : x_j <= x}| / N_base  (mismo criterio que el IFFS legado). NaN si x es NaN."""
    base = np.sort(base[~np.isnan(base)])
    out = np.full(len(valores), np.nan)
    if len(base) == 0:
        return out
    ok = ~np.isnan(valores)
    out[ok] = 100.0 * np.searchsorted(base, valores[ok], side="right") / len(base)
    return out


def _percentil_ecdf(valores: pd.Series, base: pd.Series) -> pd.Series:
    """Percentil ECDF de `valores` respecto de una muestra base fija (uso: IFFS legado y simulador legado)."""
    return pd.Series(_pct_ecdf_contra(valores.to_numpy(float), base.to_numpy(float)), index=valores.index)


def _pct_corte_anual(df: pd.DataFrame, col: str) -> pd.Series:
    """Percentil (rango medio) dentro de cada año: 100*(rango-0.5)/n entre las jurisdicciones con dato. Media exacta = 50."""
    g = df.groupby("anio")[col]
    return 100.0 * (g.rank(method="average") - 0.5) / g.transform("count")


def _pct_ventana_movil(df: pd.DataFrame, col: str, k: int, min_hist: int) -> pd.Series:
    """Percentil ECDF de x_t contra las observaciones de los `k` años ANTERIORES (sin mirar hacia adelante)."""
    out = pd.Series(np.nan, index=df.index)
    for t in sorted(df["anio"].unique()):
        previos = df["anio"].between(t - k, t - 1)
        if df.loc[previos, "anio"].nunique() < min_hist:
            continue
        actual = df["anio"] == t
        out[actual] = _pct_ecdf_contra(df.loc[actual, col].to_numpy(float), df.loc[previos, col].to_numpy(float))
    return out


def _z_a_percentil(z: pd.Series) -> pd.Series:
    return 100.0 * pd.Series(norm.cdf(z), index=z.index).where(z.notna())


def normalizar(df: pd.DataFrame, col: str, spec: Especificacion) -> pd.Series:
    """Devuelve la variable `col` normalizada a 0-100 (mayor = más frágil) según la especificación."""
    m = spec.normalizacion
    if m not in NORMALIZACIONES:
        raise ValueError(f"normalizacion desconocida: {m}")
    x = df[col]
    en_base = df["anio"].between(*spec.base_window)
    if m == "corte_anual":
        return _pct_corte_anual(df, col)
    if m == "base_fija":
        return pd.Series(_pct_ecdf_contra(x.to_numpy(float), x[en_base].to_numpy(float)), index=df.index)
    if m == "pooled_total":
        return pd.Series(_pct_ecdf_contra(x.to_numpy(float), x.to_numpy(float)), index=df.index)
    if m == "ventana_movil":
        return _pct_ventana_movil(df, col, spec.ventana_movil, spec.min_hist_movil)
    base = x[en_base].dropna()
    if spec.winsor:
        lo, hi = base.quantile(spec.winsor), base.quantile(1 - spec.winsor)
        x = x.clip(lo, hi)
        base = base.clip(lo, hi)
    if m == "zscore_base":
        return _z_a_percentil((x - base.mean()) / base.std(ddof=0))
    # zscore_robusto_base: mediana y MAD de la base, recorte en ±3
    mad = 1.4826 * (base - base.median()).abs().median()
    return _z_a_percentil(((x - base.median()) / mad).clip(-3, 3))


# --- Variables transformadas -------------------------------------------------------------------------------------

def _media_movil_hacia_atras(df: pd.DataFrame, col: str, ventana: int) -> pd.Series:
    """Media de los últimos `ventana` años (incluye el actual), por provincia; exige >= 1 dato (ventana 1 = sin suavizar)."""
    if ventana <= 1:
        return df[col]
    d = df.sort_values(["provincia", "anio"])
    r = d.groupby("provincia")[col].transform(lambda s: s.rolling(ventana, min_periods=1).mean())
    return r.reindex(df.index)


def preparar_variables(panel: pd.DataFrame, spec: Especificacion) -> pd.DataFrame:
    """Agrega las columnas `x_<dim>` (variable de cada subíndice ya suavizada) al panel."""
    p = panel.copy()
    ventanas = {"solvencia": 1, "liquidez": spec.suavizado_liquidez, "estructural": 1, "coyuntural": spec.suavizado_resultado}
    for d, v in DIMENSIONES.items():
        p[f"x_{d}"] = _media_movil_hacia_atras(p, v["variable"], ventanas[d])
    return p


# --- Agregación --------------------------------------------------------------------------------------------------

def agregar(S: pd.DataFrame, pesos: dict, como: str) -> pd.Series:
    """Compuesto de subíndices 0-100. Exige TODAS las dimensiones del compuesto (sin imputar)."""
    cols = list(pesos)
    w = np.array([pesos[c] for c in cols])
    X = S[cols].to_numpy(float)
    completo = ~np.isnan(X).any(axis=1)
    out = np.full(len(S), np.nan)
    Xc = X[completo]
    if como == "media":
        v = Xc @ w
    elif como == "geometrica":
        v = np.exp(np.log(np.clip(Xc, 1.0, None) / 100.0) @ w) * 100.0
    elif como == "potencia2":
        v = np.sqrt((Xc ** 2) @ w)
    elif como == "maximo":
        v = Xc.max(axis=1)
    else:
        raise ValueError(f"agregacion desconocida: {como}")
    out[completo] = v
    return pd.Series(out, index=S.index)


def construir_indice(panel: pd.DataFrame, spec: Especificacion = ESPEC_PRINCIPAL, sufijo: str = "") -> pd.DataFrame:
    """
    Agrega al panel: `x_<dim>`, `S_<dim>` (subíndices 0-100), `IFFS<sufijo>` (compuesto) y `IFFS_estado`.
    `sufijo` permite convivir con otras especificaciones en el mismo DataFrame (robustez).
    """
    p = preparar_variables(panel, spec)
    for d in DIMS:
        p[f"S_{d}{sufijo}"] = normalizar(p, f"x_{d}", spec)
    pesos = spec.pesos_dict()
    S = pd.DataFrame({d: p[f"S_{d}{sufijo}"] for d in pesos})
    p[f"IFFS{sufijo}"] = agregar(S, pesos, spec.agregacion)
    if not sufijo:
        n_dims = S.notna().sum(axis=1)
        p["IFFS_estado"] = np.select([n_dims == len(pesos), n_dims > 0], ["completo", "incompleto"], default="sin_dato")
    return p


# --- IFFS legado (v1) --------------------------------------------------------------------------------------------

def _pca_primer_componente(base_df: pd.DataFrame, columnas: list[str]) -> tuple[pd.Series, float]:
    """PCA de 1 componente estimado SOLO en la ventana base. Devuelve (cargas orientadas, varianza explicada)."""
    from sklearn.decomposition import PCA

    X = base_df[columnas].dropna()
    if len(X) < 5:
        raise ValueError("Muy pocas observaciones en la ventana base para estimar PCA.")
    X_std = (X - X.mean()) / X.std(ddof=0)
    pca = PCA(n_components=1, random_state=42)
    pca.fit(X_std)
    cargas = pd.Series(pca.components_[0], index=columnas)
    if cargas[columnas[0]] < 0:                       # convención de signo: la primera variable (deuda) carga positivo
        cargas = -cargas
    return cargas, float(pca.explained_variance_ratio_[0])


def construir_iffs_legado(panel: pd.DataFrame, base_window: tuple = BASE_WINDOW) -> tuple[pd.DataFrame, dict]:
    """
    IFFS v1, conservado SOLO como especificación de comparación (`LEG_A`, `LEG_B`, `LEG_C`):
    percentil ECDF contra la ventana fija 2005-2012 + promedio de 4 dimensiones (B), de 2 (A) o 1er componente principal (C).
    """
    p = panel.copy()
    en_base = p["anio"].between(*base_window)
    cols_p = []
    for var in DIMENSIONES_NUCLEO:
        p[f"LP_{var}"] = _percentil_ecdf(p[var], p.loc[en_base, var])
        cols_p.append(f"LP_{var}")
    dv, sv = "LP_deuda_sobre_ingresos", "LP_servicio_sobre_ingresos"
    p["LEG_A"] = np.where(p[[dv, sv]].notna().all(axis=1), p[[dv, sv]].mean(axis=1), np.nan)
    cov = p[cols_p].notna().mean(axis=1)
    p["LEG_B"] = p[cols_p].mean(axis=1, skipna=True).where(cov >= 0.75)
    cargas, var_exp = _pca_primer_componente(p.loc[en_base], cols_p)
    completo = p[cols_p].notna().all(axis=1)
    z = (p[cols_p] - p.loc[en_base, cols_p].mean()) / p.loc[en_base, cols_p].std(ddof=0)
    raw = z.mul(cargas.values, axis=1).sum(axis=1).where(completo)
    p["LEG_C"] = _percentil_ecdf(raw, raw[en_base])
    meta = {"base_window": base_window, "cargas_pca": cargas.to_dict(), "varianza_explicada_pc1": var_exp}
    return p.drop(columns=cols_p), meta


# --- Posición relativa (reemplaza a los "niveles de alerta" con umbrales 33/66/85) ----------------------------------

ETIQUETAS_CUARTIL = ["Cuartil 1 (menor fragilidad relativa)", "Cuartil 2", "Cuartil 3", "Cuartil 4 (mayor fragilidad relativa)"]
COLORES_CUARTIL = {ETIQUETAS_CUARTIL[0]: "#2E7D32", ETIQUETAS_CUARTIL[1]: "#9CB84A", ETIQUETAS_CUARTIL[2]: "#EF9C00",
                   ETIQUETAS_CUARTIL[3]: "#C62828", "Sin dato": "#9E9E9E"}


def asignar_posicion(panel: pd.DataFrame, col: str = "IFFS") -> pd.Series:
    """
    [DISEÑO] Cuartil del compuesto DENTRO de cada año. Es una lectura de posición relativa (siempre hay ~6 jurisdicciones en el
    cuartil superior aunque todas estén sanas); NO es una alerta absoluta ni una calibración contra eventos. Reemplaza a los
    cortes 33/66/85 del IFFS v1, que eran arbitrarios y estaban anclados a la ventana 2005-2012.
    """
    q = panel.groupby("anio")[col].rank(pct=True, method="first")   # rango / n con dato
    pos = np.ceil(q * 4).clip(1, 4)
    out = pos.map({1.0: ETIQUETAS_CUARTIL[0], 2.0: ETIQUETAS_CUARTIL[1], 3.0: ETIQUETAS_CUARTIL[2], 4.0: ETIQUETAS_CUARTIL[3]})
    return out.fillna("Sin dato")


def descomponer_contribucion(fila: pd.Series) -> pd.DataFrame:
    """
    Subíndices de una fila (provincia, año). Como el compuesto es un promedio simple del núcleo, la contribución de cada
    dimensión nuclear al compuesto es S/3 (`aporte_al_compuesto`); la dimensión coyuntural no aporta al compuesto.
    """
    filas = []
    for d in DIMS:
        s = fila.get(f"S_{d}")
        filas.append({
            "dimension": DIMENSIONES[d]["nombre"], "detalle": DIMENSIONES[d]["detalle"], "subindice": s,
            "en_compuesto": d in DIMS_NUCLEO,
            "aporte_al_compuesto": (s / len(DIMS_NUCLEO)) if (d in DIMS_NUCLEO and pd.notna(s)) else np.nan,
        })
    return pd.DataFrame(filas).sort_values("subindice", ascending=False)
