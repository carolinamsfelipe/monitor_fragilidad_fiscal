"""
validation.py
==============

Criterios de diseño (SIN eventos), robustez y validación (CON eventos) del IFFS v2.

Separación explícita de tres preguntas distintas (no se mezclan en ningún producto):

  DESCRIPCIÓN     ¿qué muestra el índice? cobertura, distribución, cambios de ranking, cómo se mueven los subíndices.
  DISCRIMINACIÓN  ¿en una fecha dada, las provincias que luego reestructuraron tenían un score más alto que las demás?
                  (AUC por rangos, IC bootstrap, prueba de permutación). NO implica anticipar: un nivel más alto puede ser un
                  rasgo permanente de esas provincias (p. ej. haber emitido bonos).
  ANTICIPACIÓN    ¿el score de las provincias con evento SUBIÓ antes del evento más que el de las demás? (pretendencia,
                  estudio de eventos). Con una sola fecha común de evento (COVID) y 12 unidades, solo admite lectura exploratoria.

Reglas de honestidad implementadas:
  * Ningún peso, ventana ni umbral se elige mirando los eventos. Las especificaciones alternativas se reportan TODAS.
  * Las pruebas no viables (poder nulo o supuestos que no se cumplen) se declaran en `pruebas_no_viables`, con la razón.
  * Semilla fija (RANDOM_SEED = 42) en todo lo aleatorio.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats

from . import indicators as ind

RANDOM_SEED = 42
N_BOOT = 2000
N_PERM = 10000


# ===================================================================================================================
# 1. CRITERIOS DE DISEÑO (no usan eventos)
# ===================================================================================================================

def icc_ar1(panel: pd.DataFrame, col: str) -> dict:
    """
    ICC(1) = parte de la varianza que es diferencia PERMANENTE entre provincias (ANOVA de un factor, panel balanceado aprox.)
    y AR(1) intra-provincia (autocorrelación de los desvíos respecto de la media de cada provincia).
    """
    d = panel[["provincia", "anio", col]].dropna()
    g = d.groupby("provincia")[col]
    k = g.size().mean()
    ms_b = ((g.mean() - d[col].mean()) ** 2 * g.size()).sum() / (d["provincia"].nunique() - 1)
    ms_w = ((d[col] - g.transform("mean")) ** 2).sum() / (len(d) - d["provincia"].nunique())
    icc = (ms_b - ms_w) / (ms_b + (k - 1) * ms_w)
    dev = d.assign(dev=d[col] - g.transform("mean")).sort_values(["provincia", "anio"])
    dev["dev_prev"] = dev.groupby("provincia")["dev"].shift(1)
    dev["gap"] = dev.groupby("provincia")["anio"].diff()
    dd = dev[dev["gap"] == 1].dropna(subset=["dev_prev"])
    ar1 = float(np.corrcoef(dd["dev"], dd["dev_prev"])[0, 1]) if len(dd) > 10 else np.nan
    return {"icc": float(icc), "ar1_intra": ar1, "n": int(len(d))}


def persistencia_de_rangos(panel: pd.DataFrame, col: str) -> float:
    """Media (sobre años) de la correlación de Spearman entre los rangos de t y de t+1: estabilidad del ordenamiento."""
    w = panel.pivot(index="provincia", columns="anio", values=col)
    rs = []
    for t in w.columns[:-1]:
        par = w[[t, t + 1]].dropna() if (t + 1) in w.columns else pd.DataFrame()
        if len(par) >= 8:
            rs.append(stats.spearmanr(par[t], par[t + 1])[0])
    return float(np.mean(rs)) if rs else np.nan


def propiedades_variables(panel: pd.DataFrame) -> pd.DataFrame:
    """Persistencia (ICC, AR1) y asimetría de las cuatro variables núcleo y de la dependencia de transferencias."""
    filas = []
    extra = {"dependencia_transferencias": "Ingresos de origen nacional / Ingresos corrientes (no entra al índice)"}
    for d, v in ind.DIMENSIONES.items():
        filas.append((d, v["variable"], v["detalle"]))
    for var, det in extra.items():
        filas.append(("contexto", var, det))
    out = []
    for dim, col, det in filas:
        m = icc_ar1(panel, col)
        x = panel[col].dropna()
        out.append({"dimension": dim, "variable": col, "detalle": det, **m,
                    "mediana": float(x.median()), "asimetria": float(stats.skew(x)),
                    "persistencia_rangos": persistencia_de_rangos(panel.assign(_p=ind._pct_corte_anual(panel, col)), "_p")})
    return pd.DataFrame(out)


UMBRAL_PERSISTENCIA = 0.85   # regla ex ante (fijada antes de mirar eventos): correlación de rangos t vs t+1


def criterio_ventanas_y_composicion(panel: pd.DataFrame, ventanas=(1, 3, 5)) -> pd.DataFrame:
    """
    REGLA DE DISEÑO EX ANTE, sin eventos (se aplica literalmente, aunque contradiga la intuición previa):
      (a) ventana de suavizado de cada variable = la MÁS CORTA de {1, 3, 5} años con persistencia de rangos >= 0,85;
      (b) una dimensión entra al compuesto si su forma anual (1 año) ya tiene persistencia de rangos >= 0,85; si no,
          se reporta como subíndice separado (señal coyuntural), porque mezclar ruido con rasgos persistentes deteriora el ordenamiento.
    """
    filas = []
    for d, v in ind.DIMENSIONES.items():
        col, elegida = v["variable"], None
        for w in ventanas:
            x = ind._media_movil_hacia_atras(panel, col, w)
            tmp = panel.assign(_v=x)
            tmp["_p"] = ind._pct_corte_anual(tmp, "_v")
            pers = persistencia_de_rangos(tmp, "_p")
            m = icc_ar1(tmp, "_v")
            if elegida is None and pers >= UMBRAL_PERSISTENCIA:
                elegida = w
            filas.append({"dimension": d, "variable": col, "ventana_anios": w, "persistencia_rangos": pers,
                          "icc": m["icc"], "ar1_intra": m["ar1_intra"]})
    t = pd.DataFrame(filas)
    elegidas = {d: t[(t["dimension"] == d) & (t["persistencia_rangos"] >= UMBRAL_PERSISTENCIA)]["ventana_anios"].min() for d in ind.DIMS}
    t["ventana_elegida"] = t["dimension"].map(elegidas)
    t["entra_al_compuesto"] = t["dimension"].map(
        {d: bool(t[(t["dimension"] == d) & (t["ventana_anios"] == 1)]["persistencia_rangos"].iloc[0] >= UMBRAL_PERSISTENCIA) for d in ind.DIMS})
    return t


def redundancia_subindices(panel: pd.DataFrame) -> pd.DataFrame:
    """Correlación de Spearman entre variables/subíndices: total (pooled), entre provincias (medias) e intra-provincia."""
    cols = [ind.VARIABLE_DE[d] for d in ind.DIMS]
    nombres = {ind.VARIABLE_DE[d]: ind.NOMBRE_DIM[d] for d in ind.DIMS}
    filas = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            d = panel[["provincia", "anio", a, b]].dropna()
            pooled = stats.spearmanr(d[a], d[b])[0]
            m = d.groupby("provincia")[[a, b]].mean()
            entre = stats.spearmanr(m[a], m[b])[0]
            dev = d[[a, b]] - d.groupby("provincia")[[a, b]].transform("mean")
            intra = stats.spearmanr(dev[a], dev[b])[0]
            filas.append({"par": f"{nombres[a]} – {nombres[b]}", "pooled": pooled, "entre_provincias": entre,
                          "intra_provincia": intra, "n": len(d)})
    return pd.DataFrame(filas)


def pca_subindices(panel: pd.DataFrame) -> dict:
    """Cuántos factores necesita el conjunto de subíndices: varianza explicada por cada componente (estandarizado)."""
    from sklearn.decomposition import PCA

    X = panel[ind.COLS_S].dropna()
    pca = PCA(random_state=RANDOM_SEED).fit((X - X.mean()) / X.std(ddof=0))
    return {"varianza_explicada": pca.explained_variance_ratio_.tolist(), "n": int(len(X)),
            "cargas_pc1": dict(zip(ind.COLS_S, pca.components_[0].tolist()))}


def _resumen_especificacion(base: pd.DataFrame, spec: ind.Especificacion, etiqueta: str, ref: pd.Series) -> dict:
    """Métricas de diseño, sin eventos, de una especificación (referencia: el IFFS principal)."""
    r = ind.construir_indice(base, spec, sufijo="_x")
    x = r["IFFS_x"]
    med = r.groupby("anio")["IFFS_x"].median()
    ok = x.notna() & ref.notna()
    top = lambda s: set(base.loc[s[(base["anio"] == 2024)].nlargest(6).index, "provincia"]) if s.notna().any() else set()
    # cambios de ranking respecto del principal: rango dentro de cada año
    rk = r.assign(_r=x).groupby("anio")["_r"].rank(ascending=False)
    rk_ref = base.assign(_r=ref).groupby("anio")["_r"].rank(ascending=False)
    dif = (rk - rk_ref).abs()[ok]
    return {
        "especificacion": etiqueta,
        "n_obs": int(x.notna().sum()),
        "spearman_vs_principal": float(stats.spearmanr(x[ok], ref[ok])[0]),
        "deriva_mediana_anual": float(med.max() - med.min()),
        "persistencia_rangos": persistencia_de_rangos(r.assign(_i=x), "_i"),
        "cambio_rango_medio": float(dif.mean()),
        "obs_con_cambio_rango_ge5": int((dif >= 5).sum()),
        "solapamiento_top6_2024": len(top(x) & top(ref)),
    }


def alternativas_de_especificacion() -> list[tuple[str, ind.Especificacion]]:
    """
    Especificaciones alternativas evaluadas UNA A LA VEZ respecto de la principal (análisis de sensibilidad ceteris paribus).
    La lista es fija y se reporta completa: no hay selección posterior.
    """
    E = ind.Especificacion
    b = ind.ESPEC_PRINCIPAL
    alts = [
        ("Normalización: base fija 2005-2012 (legado)", replace(b, normalizacion="base_fija")),
        ("Normalización: base fija 2005-2019", replace(b, normalizacion="base_fija", base_window=(2005, 2019))),
        ("Normalización: ventana móvil 8 años (sin mirar adelante)", replace(b, normalizacion="ventana_movil")),
        ("Normalización: ventana móvil 5 años", replace(b, normalizacion="ventana_movil", ventana_movil=5, min_hist_movil=3)),
        ("Normalización: pooled todos los años", replace(b, normalizacion="pooled_total")),
        ("Normalización: z-score base 2005-2012", replace(b, normalizacion="zscore_base")),
        ("Normalización: z-score robusto (mediana/MAD, recorte ±3)", replace(b, normalizacion="zscore_robusto_base")),
        ("Suavizado liquidez: media móvil 3 años", replace(b, suavizado_liquidez=3)),
        ("Suavizado liquidez: 5 años", replace(b, suavizado_liquidez=5)),
        ("Agregación: media geométrica", replace(b, agregacion="geometrica")),
        ("Agregación: media cuadrática (penaliza extremos)", replace(b, agregacion="potencia2")),
        ("Agregación: máximo de dimensiones (eslabón débil)", replace(b, agregacion="maximo")),
        ("Compuesto con las 4 dimensiones (incluye coyuntural)", replace(b, dims_compuesto=tuple(ind.DIMS))),
        ("Compuesto sin rigidez (solvencia + liquidez)", replace(b, dims_compuesto=("solvencia", "liquidez"))),
        ("Pesos 50% solvencia / 25% / 25%", replace(b, pesos=(0.5, 0.25, 0.25))),
        ("Pesos 50% liquidez / 25% / 25%", replace(b, pesos=(0.25, 0.5, 0.25))),
        ("Pesos 50% estructural / 25% / 25%", replace(b, pesos=(0.25, 0.25, 0.5))),
    ]
    return alts


def comparar_especificaciones(panel: pd.DataFrame) -> pd.DataFrame:
    """Tabla de diseño (sin eventos): cada alternativa vs. la principal en persistencia, deriva, cambios de ranking y solapamiento."""
    base = panel.drop(columns=[c for c in panel.columns if c.startswith(("S_", "x_", "IFFS"))], errors="ignore")
    ref_full = ind.construir_indice(base, ind.ESPEC_PRINCIPAL)
    ref = ref_full["IFFS"]
    filas = [_resumen_especificacion(base, ind.ESPEC_PRINCIPAL, "PRINCIPAL: corte anual · media simple de solvencia, liquidez y estructural", ref)]
    for etiqueta, spec in alternativas_de_especificacion():
        filas.append(_resumen_especificacion(base, spec, etiqueta, ref))
    return pd.DataFrame(filas)


def sensibilidad_a_pesos(panel: pd.DataFrame, n_sim: int = 2000) -> dict:
    """
    Pesos aleatorios uniformes en el simplex (Dirichlet(1,1,1)) sobre las 3 dimensiones nucleares: ¿cuánto cambia el ordenamiento?
    Devuelve la distribución de la correlación de Spearman (dentro de cada año) con el compuesto de pesos iguales y la
    estabilidad de pertenencia al cuartil superior de los pares provincia-año que lo integran con pesos iguales.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    d = panel.dropna(subset=ind.COLS_S_NUCLEO + ["IFFS"]).copy()
    W = rng.dirichlet(np.ones(3), size=n_sim)
    S = d[ind.COLS_S_NUCLEO].to_numpy()
    anios = d["anio"].to_numpy()
    top_eq = d.groupby("anio")["IFFS"].transform(lambda s: s >= s.quantile(0.75)).to_numpy()
    rhos = np.zeros(n_sim)
    mantiene = np.zeros((n_sim, int(top_eq.sum())))
    for i, w in enumerate(W):
        sc = S @ w
        tmp = pd.DataFrame({"anio": anios, "sc": sc, "eq": d["IFFS"].to_numpy()})
        rhos[i] = tmp.groupby("anio").apply(lambda g: stats.spearmanr(g["sc"], g["eq"])[0], include_groups=False).mean()
        q75 = tmp.groupby("anio")["sc"].transform(lambda s: s.quantile(0.75)).to_numpy()
        mantiene[i] = (sc >= q75)[top_eq]
    estab = mantiene.mean(axis=0)
    return {
        "n_sim": n_sim,
        "spearman_medio_p05": float(np.quantile(rhos, 0.05)), "spearman_medio_mediana": float(np.median(rhos)),
        "spearman_medio_min": float(rhos.min()),
        "cuartil_superior_estable_ge80": float((estab >= 0.8).mean()),
        "cuartil_superior_estable_mediana": float(np.median(estab)),
        "n_obs_cuartil_superior": int(top_eq.sum()),
    }


def sensibilidad_la_pampa(panel_excluir: pd.DataFrame, panel_miles: pd.DataFrame, panel_sin_tratar: pd.DataFrame) -> pd.DataFrame:
    """
    Efecto del tratamiento de La Pampa 2005-2017 sobre el ORDENAMIENTO DEL RESTO de las provincias (sin La Pampa), con tres
    normalizaciones. Muestra por qué el percentil es robusto a un error de unidades y el z-score no.
    """
    filas = []
    for norm_ in ("corte_anual", "base_fija", "zscore_base", "zscore_robusto_base"):
        spec = replace(ind.ESPEC_PRINCIPAL, normalizacion=norm_)
        res = {}
        for nombre, p in [("excluir", panel_excluir), ("supuesto_miles", panel_miles), ("sin_tratar", panel_sin_tratar)]:
            r = ind.construir_indice(p.drop(columns=[c for c in p.columns if c.startswith(("S_", "x_", "IFFS"))], errors="ignore"), spec)
            res[nombre] = r
        base = res["excluir"]
        resto = (base["provincia"] != "La Pampa") & (base["anio"] <= 2017)
        for nombre in ("supuesto_miles", "sin_tratar"):
            a, b = base.loc[resto, "S_solvencia"], res[nombre].loc[resto, "S_solvencia"]
            ok = a.notna() & b.notna()
            filas.append({
                "normalizacion": norm_, "tratamiento_vs_excluir": nombre,
                "cambio_medio_abs_subindice_solvencia_resto": float((a[ok] - b[ok]).abs().mean()),
                "cambio_max_abs": float((a[ok] - b[ok]).abs().max()),
                "spearman_solvencia_resto": float(stats.spearmanr(a[ok], b[ok])[0]),
            })
    return pd.DataFrame(filas)


# ===================================================================================================================
# 2. EVENTOS: definición de grupos y anclajes
# ===================================================================================================================

def tabla_eventos(eventos: pd.DataFrame) -> pd.DataFrame:
    """
    Una fila por evento con su fecha documentada o convenida:
      anio_ancla       año usado en el análisis principal (el del archivo de eventos).
      fecha_puntual    True si la fuente documenta el año exacto; False si solo dice '2020-2021' y el año 2020 es una CONVENCIÓN.
      anio_ancla_alt   año alternativo (2021 para los ambiguos) usado como sensibilidad.
    """
    e = eventos.copy()
    e["fecha_puntual"] = e["nota_convencion_anio"].str.startswith("Fecha puntual")
    e["anio_ancla"] = e["anio_evento"]
    e["anio_ancla_alt"] = np.where(e["fecha_puntual"], e["anio_evento"], e["anio_evento"] + 1)
    e["bloque"] = np.where(e["anio_evento"].between(2020, 2021), "COVID 2020-21", "otro")
    return e


def grupos_covid(eventos: pd.DataFrame) -> tuple[list[str], dict]:
    """Provincias con evento en el bloque 2020-21 y su año ancla por escenario ('principal' / 'alt')."""
    t = tabla_eventos(eventos)
    t = t[t["bloque"] == "COVID 2020-21"]
    provs = sorted(t["provincia"].unique())
    return provs, {"principal": dict(zip(t["provincia"], t["anio_ancla"])), "alt": dict(zip(t["provincia"], t["anio_ancla_alt"]))}


# ===================================================================================================================
# 3. DISCRIMINACIÓN (AUC por rangos, IC bootstrap, permutación)
# ===================================================================================================================

def auc_rangos(x1: np.ndarray, x0: np.ndarray) -> float:
    """AUC = P(score evento > score no-evento) + 0.5 P(empate) (Mann-Whitney)."""
    n1, n0 = len(x1), len(x0)
    r = stats.rankdata(np.concatenate([x1, x0]))          # rangos promedio: los empates cuentan 0,5
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_con_incertidumbre(x1: np.ndarray, x0: np.ndarray, n_boot: int = N_BOOT, n_perm: int = N_PERM, seed: int = RANDOM_SEED) -> dict:
    """AUC, IC 95% bootstrap estratificado (remuestrea provincias dentro de cada grupo) y p de permutación unilateral."""
    rng = np.random.default_rng(seed)
    a = auc_rangos(x1, x0)
    boots = np.array([auc_rangos(rng.choice(x1, len(x1)), rng.choice(x0, len(x0))) for _ in range(n_boot)])
    allx = np.concatenate([x1, x0])
    n1 = len(x1)
    perm = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(allx))
        perm[i] = auc_rangos(allx[idx[:n1]], allx[idx[n1:]])
    return {"auc": a, "ic95_inf": float(np.quantile(boots, 0.025)), "ic95_sup": float(np.quantile(boots, 0.975)),
            "p_perm_unilateral": float((1 + (perm >= a - 1e-12).sum()) / (n_perm + 1)), "n_evento": int(len(x1)), "n_control": int(len(x0))}


def discriminacion_transversal(panel: pd.DataFrame, provs_evento: list[str], anio_ref: int, columnas: dict[str, str],
                               n_boot: int = N_BOOT, n_perm: int = N_PERM) -> pd.DataFrame:
    """AUC de cada columna en un corte transversal `anio_ref`: provincias con evento vs. el resto."""
    d = panel[panel["anio"] == anio_ref]
    filas = []
    for etiqueta, col in columnas.items():
        v = d[["provincia", col]].dropna()
        x1 = v.loc[v["provincia"].isin(provs_evento), col].to_numpy(float)
        x0 = v.loc[~v["provincia"].isin(provs_evento), col].to_numpy(float)
        if len(x1) < 3 or len(x0) < 3:
            filas.append({"medida": etiqueta, "anio_ref": anio_ref, "nota": "muestra insuficiente"})
            continue
        filas.append({"medida": etiqueta, "anio_ref": anio_ref, **auc_con_incertidumbre(x1, x0, n_boot, n_perm)})
    return pd.DataFrame(filas)


COLUMNAS_DISCRIMINACION = {
    "IFFS compuesto (principal)": "IFFS",
    "Subíndice solvencia": "S_solvencia",
    "Subíndice liquidez": "S_liquidez",
    "Subíndice estructural": "S_estructural",
    "Subíndice coyuntural": "S_coyuntural",
    "IFFS legado B (base fija 2005-12, 4 dims)": "LEG_B",
    "Benchmark: participación de bonos en la deuda (%)": "pct_bonos",
    "Benchmark: dependencia de transferencias": "dependencia_transferencias",
}


def discriminacion_por_rezago(panel: pd.DataFrame, provs: list[str], anclas: dict, col: str, rezagos=(1, 2, 3, 4, 5),
                              n_boot: int = N_BOOT, n_perm: int = N_PERM) -> pd.DataFrame:
    """
    AUC de `col` medido k años ANTES del año ancla (T−k), con el ancla principal o alternativa. Para cada provincia con evento el año de
    referencia es su propio ancla − k, de modo que el análisis respeta un posible escalonamiento (aquí casi no lo hay).
    Las provincias control usan el año de referencia mediano de las tratadas.
    """
    filas = []
    ancla_med = int(np.median(list(anclas.values())))
    for k in rezagos:
        v1, v0 = [], []
        for prov, g in panel.groupby("provincia"):
            if prov in provs:
                a = anclas[prov] - k
            else:
                a = ancla_med - k
            r = g.loc[g["anio"] == a, col].dropna()
            if len(r):
                (v1 if prov in provs else v0).append(float(r.iloc[0]))
        if len(v1) >= 3 and len(v0) >= 3:
            filas.append({"rezago": k, "anio_ref_mediano": ancla_med - k, **auc_con_incertidumbre(np.array(v1), np.array(v0), n_boot, n_perm)})
    return pd.DataFrame(filas)


def placebo_temporal(panel: pd.DataFrame, provs: list[str], col: str = "IFFS", anios=range(2005, 2026),
                     n_boot: int = 500, n_perm: int = 2000) -> pd.DataFrame:
    """
    Placebo por fecha: el AUC del MISMO grupo de 12 provincias contra las demás en TODOS los años. Si el grupo ya se
    separaba en años muy anteriores al evento (p. ej. 2005-2012) el nivel más alto es un rasgo persistente, no una señal
    que aparece antes del shock. (Menos réplicas que el análisis principal para acotar el tiempo de cómputo.)
    """
    filas = []
    for a in anios:
        v = panel.loc[panel["anio"] == a, ["provincia", col]].dropna()
        x1 = v.loc[v["provincia"].isin(provs), col].to_numpy(float)
        x0 = v.loc[~v["provincia"].isin(provs), col].to_numpy(float)
        if len(x1) >= 3 and len(x0) >= 3:
            filas.append({"anio": a, **auc_con_incertidumbre(x1, x0, n_boot, n_perm)})
    return pd.DataFrame(filas)


def placebo_grupos_aleatorios(panel: pd.DataFrame, provs: list[str], col: str = "IFFS", anio: int = 2019, n: int = 5000,
                              seed: int = RANDOM_SEED) -> dict:
    """Distribución del AUC si el grupo de 'eventos' fuera un sorteo de igual tamaño: qué AUC produce el azar con 24 provincias."""
    rng = np.random.default_rng(seed)
    v = panel.loc[panel["anio"] == anio, ["provincia", col]].dropna()
    x = v[col].to_numpy(float)
    n1 = int(v["provincia"].isin(provs).sum())
    aucs = np.array([auc_rangos(x[i[:n1]], x[i[n1:]]) for i in (rng.permutation(len(x)) for _ in range(n))])
    obs = auc_rangos(x[v["provincia"].isin(provs).to_numpy()], x[~v["provincia"].isin(provs).to_numpy()])
    return {"auc_observado": obs, "azar_p50": float(np.quantile(aucs, .5)), "azar_p95": float(np.quantile(aucs, .95)),
            "azar_p99": float(np.quantile(aucs, .99)), "n_sorteos": n}


def discriminacion_condicionada_a_bonos(panel: pd.DataFrame, provs: list[str], anio_ref: int = 2019, min_bonos: float = 25.0) -> dict:
    """
    Control de acceso a mercado: solo las provincias con bonos podían reestructurar bonos. Repite el AUC (compuesto y subíndices)
    solo entre provincias con participación de bonos >= `min_bonos`% (corte fijado ex ante, no optimizado).
    """
    d = panel[(panel["anio"] == anio_ref) & (panel["pct_bonos"] >= min_bonos)]
    out = {"anio_ref": anio_ref, "min_bonos_pct": min_bonos, "n_emisoras": int(len(d)),
           "n_evento": int(d["provincia"].isin(provs).sum()), "n_control": int((~d["provincia"].isin(provs)).sum()),
           "controles": sorted(d.loc[~d["provincia"].isin(provs), "provincia"])}
    for etiqueta, col in {"IFFS": "IFFS", "solvencia": "S_solvencia", "liquidez": "S_liquidez",
                          "estructural": "S_estructural", "coyuntural": "S_coyuntural"}.items():
        v = d[["provincia", col]].dropna()
        x1, x0 = v.loc[v["provincia"].isin(provs), col].to_numpy(float), v.loc[~v["provincia"].isin(provs), col].to_numpy(float)
        out[f"auc_{etiqueta}"] = auc_rangos(x1, x0) if len(x1) >= 2 and len(x0) >= 2 else np.nan
    return out


# ===================================================================================================================
# 4. ANTICIPACIÓN: estudio de eventos y pretendencia
# ===================================================================================================================

def estudio_de_eventos(panel: pd.DataFrame, provs: list[str], anclas: dict, col: str = "IFFS", ventana=(-8, 3),
                       n_boot: int = N_BOOT, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Promedio de `col` por tiempo relativo τ = año − ancla, provincias con evento vs. control (que usa el ancla mediano).
    Diferencia con IC 95% bootstrap (remuestreo de provincias dentro de cada grupo).
    """
    rng = np.random.default_rng(seed)
    ancla_med = int(np.median(list(anclas.values())))
    d = panel[["provincia", "anio", col]].copy()
    d["ancla"] = d["provincia"].map(lambda p: anclas.get(p, ancla_med))
    d["tau"] = d["anio"] - d["ancla"]
    d["grupo"] = np.where(d["provincia"].isin(provs), "evento", "control")
    filas = []
    for tau in range(ventana[0], ventana[1] + 1):
        dt = d[d["tau"] == tau].dropna(subset=[col])
        g1 = dt.loc[dt["grupo"] == "evento", col].to_numpy(float)
        g0 = dt.loc[dt["grupo"] == "control", col].to_numpy(float)
        if len(g1) < 3 or len(g0) < 3:
            continue
        dif = g1.mean() - g0.mean()
        b = np.array([rng.choice(g1, len(g1)).mean() - rng.choice(g0, len(g0)).mean() for _ in range(n_boot)])
        filas.append({"tau": tau, "media_evento": g1.mean(), "media_control": g0.mean(), "diferencia": dif,
                      "ic95_inf": float(np.quantile(b, .025)), "ic95_sup": float(np.quantile(b, .975)),
                      "n_evento": len(g1), "n_control": len(g0)})
    return pd.DataFrame(filas)


def pretendencia(panel: pd.DataFrame, provs: list[str], anclas: dict, col: str = "IFFS", desde: int = 5, hasta: int = 1,
                 n_boot: int = N_BOOT, n_perm: int = N_PERM, seed: int = RANDOM_SEED) -> dict:
    """
    ANTICIPACIÓN propiamente dicha: cambio del score entre T−`desde` y T−`hasta`, provincias con evento vs. control. Un score que
    ya era más alto pero no sube ANTES del evento no anticipa nada (discrimina por nivel). Devuelve diferencia de medias de los
    cambios, IC 95% bootstrap y p de permutación bilateral.
    """
    rng = np.random.default_rng(seed)
    ancla_med = int(np.median(list(anclas.values())))
    cam1, cam0 = [], []
    for prov, g in panel.groupby("provincia"):
        a = anclas.get(prov, ancla_med)
        s = g.set_index("anio")[col]
        try:
            v = float(s.loc[a - hasta]) - float(s.loc[a - desde])
        except KeyError:
            continue
        if np.isnan(v):
            continue
        (cam1 if prov in provs else cam0).append(v)
    c1, c0 = np.array(cam1), np.array(cam0)
    if len(c1) < 3 or len(c0) < 3:
        return {"nota": "muestra insuficiente"}
    dif = c1.mean() - c0.mean()
    boots = np.array([rng.choice(c1, len(c1)).mean() - rng.choice(c0, len(c0)).mean() for _ in range(n_boot)])
    allc, n1 = np.concatenate([c1, c0]), len(c1)
    perm = np.array([(lambda i: allc[i[:n1]].mean() - allc[i[n1:]].mean())(rng.permutation(len(allc))) for _ in range(n_perm)])
    return {"col": col, "desde": f"T-{desde}", "hasta": f"T-{hasta}", "cambio_medio_evento": float(c1.mean()),
            "cambio_medio_control": float(c0.mean()), "diferencia": float(dif), "ic95_inf": float(np.quantile(boots, .025)),
            "ic95_sup": float(np.quantile(boots, .975)), "p_perm_bilateral": float((1 + (np.abs(perm) >= abs(dif) - 1e-12).sum()) / (n_perm + 1)),
            "n_evento": int(len(c1)), "n_control": int(len(c0))}


def caso_la_rioja_2025(panel: pd.DataFrame, provincia: str = "La Rioja", anio_evento: int = 2025) -> pd.DataFrame:
    """Trayectoria de La Rioja antes del incumplimiento de feb-2025: descriptivo, UN solo caso (no permite inferencia)."""
    d = panel[(panel["provincia"] == provincia) & (panel["anio"] < anio_evento) & (panel["anio"] >= anio_evento - 6)]
    out = d[["anio"] + ind.COLS_S + ["IFFS"]].copy()
    n = panel[panel["anio"].isin(d["anio"])].groupby("anio")["IFFS"].count()
    out["puesto_compuesto"] = [int((panel.loc[panel["anio"] == a, "IFFS"] > v).sum() + 1) for a, v in zip(out["anio"], out["IFFS"])]
    out["n_jurisdicciones_con_dato"] = out["anio"].map(n)
    return out


def pruebas_no_viables(panel: pd.DataFrame, eventos: pd.DataFrame) -> pd.DataFrame:
    """
    Pruebas que NO se corren, con la razón cuantificada a partir de los datos (no es una opinión).
    """
    t = tabla_eventos(eventos)
    pre2020 = int((t["anio_evento"] < 2020).sum())
    por_anio = t.groupby("anio_evento").size().to_dict()
    n_obs = int(panel[["provincia", "anio"]].drop_duplicates().shape[0])
    n_covid = int((t["bloque"] == "COVID 2020-21").sum())
    n_punt = int(t.loc[t["bloque"] == "COVID 2020-21", "fecha_puntual"].sum())
    return pd.DataFrame([
        {"prueba": "Modelo supervisado con corte temporal (entrenar antes de 2020, evaluar después)",
         "por_que_no_es_viable": f"{pre2020} eventos antes de 2020: no hay ninguna observación positiva para entrenar. Eventos por año: {por_anio}."},
        {"prueba": "Probabilidad de default calibrada (logit/probit, curva de calibración)",
         "por_que_no_es_viable": f"{len(t)} eventos sobre {n_obs} observaciones provincia-año ({100*len(t)/n_obs:.1f}%), {n_covid} de ellos en un mismo shock: "
                                 "sin variación temporal independiente no se identifica una probabilidad; un score relativo (media 50 por construcción) no es una probabilidad."},
        {"prueba": "Optimización de pesos o umbrales para maximizar el AUC / la tasa de acierto",
         "por_que_no_es_viable": "Con 13 eventos (12 simultáneos) cualquier optimización sobreajusta y el resultado no generaliza; además contradice la regla de diseño ex ante. No se hace."},
        {"prueba": "Modelo de duración / riesgo proporcional (Cox)",
         "por_que_no_es_viable": f"Los {n_covid} eventos del bloque COVID comparten fecha (o un año de diferencia): no hay escalonamiento temporal que identifique el efecto de las covariables."},
        {"prueba": "Estudio de eventos con efectos de tratamiento escalonados (Callaway-Sant'Anna, Sun-Abraham)",
         "por_que_no_es_viable": "Requiere cohortes de adopción en años distintos; aquí hay una única cohorte de 12 unidades. Se reporta solo la comparación de trayectorias promedio, con IC bootstrap, como exploración."},
        {"prueba": "Estimación del año exacto de cada evento (timing)",
         "por_que_no_es_viable": f"De los {n_covid} eventos del bloque 2020-21, solo {n_punt} tienen fecha documentada; en los otros {n_covid - n_punt} el año 2020 es una convención. Se reportan ambos anclajes (2020 y 2021)."},
        {"prueba": "Validación fuera de muestra con el evento de La Rioja 2025",
         "por_que_no_es_viable": "Es UNA observación: sirve para mostrar la trayectoria (descriptivo), no para estimar aciertos ni errores."},
        {"prueba": "Comparación con calificaciones crediticias",
         "por_que_no_es_viable": "No hay una base pública gratuita y consistente de calificaciones provinciales en los datos disponibles."},
    ])


# ===================================================================================================================
# 5. Cobertura y descripción
# ===================================================================================================================

def _rangos(anios: list[int]) -> str:
    """[2005, 2006, 2007, 2025] -> '2005-2007 y 2025'."""
    tramos, ini = [], 0
    for i in range(1, len(anios) + 1):
        if i == len(anios) or anios[i] != anios[i - 1] + 1:
            tramos.append(str(anios[ini]) if ini == i - 1 else f"{anios[ini]}-{anios[i - 1]}")
            ini = i
    return " y ".join(tramos)


def cobertura(panel: pd.DataFrame) -> dict:
    """Cobertura de cada variable, del compuesto y por provincia (sin imputar)."""
    n = len(panel)
    por_var = {d: float(panel[ind.VARIABLE_DE[d]].notna().mean()) for d in ind.DIMS}
    por_var_subidx = {d: float(panel[f"S_{d}"].notna().mean()) for d in ind.DIMS}
    faltantes = panel[panel["IFFS"].isna()][["provincia", "anio"]]
    return {"n_total": n, "n_compuesto": int(panel["IFFS"].notna().sum()), "pct_compuesto": float(panel["IFFS"].notna().mean()),
            "por_variable": por_var, "por_subindice": por_var_subidx,
            "faltantes_compuesto": faltantes.groupby("provincia")["anio"].agg(lambda s: f"{_rangos(sorted(s))} ({len(s)} años)").to_dict()}


def nivel_comun(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Nivel COMÚN (mediana entre provincias) de cada variable por año. Complemento indispensable del score relativo: si todas las provincias
    empeoran o mejoran a la vez, el percentil no lo muestra pero estas medianas sí.
    """
    cols = {"deuda_sobre_ingresos": "Deuda/Ingresos", "servicio_sobre_ingresos": "Servicio/Ingresos",
            "rigidez_gastototal": "Personal/Gasto", "fragilidad_resultado": "-Resultado/Ingresos"}
    g = panel.groupby("anio")[list(cols)].median()
    return g.rename(columns=cols)
