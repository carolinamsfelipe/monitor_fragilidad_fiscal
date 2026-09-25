"""
analytics.py
=============

Capa de analítica descriptiva ("IA sin costo") sobre el IFFS v2.

Todo se resuelve con machine learning clásico (scikit-learn / numpy), sin APIs externas ni claves, y es 100% reproducible
(semilla fija). NINGUNA de estas piezas valida al IFFS como predictor de default (ver `src/validation.py`):

1. Perfiles fiscales (k-means) sobre los 4 subíndices: agrupan según la FORMA de la fragilidad (qué dimensiones pesan).
   Se ajustan solo con la ventana 2005-2012 y se asignan a los demás años (sin usar información futura para definir grupos).
2. Detección de anomalías (Isolation Forest): marca observaciones con combinaciones o saltos inusuales. Descriptivo.
3. Proyección exploratoria + backtest contra la regla ingenua (repetir el último valor). Resultado informado tal cual sale.
4. Simulador de escenarios: recalcula subíndices y compuesto ante shocks mecánicos. Es sensibilidad, NO un pronóstico.
5. Ranking de cambios: quién subió o bajó de cuartil y qué dimensión lo explica.
6. Generador de análisis: motor de reglas que solo redacta cifras efectivamente calculadas (no es IA generativa).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .formato import es
from .indicators import (
    COLS_S,
    COLS_S_NUCLEO,
    DIMENSIONES,
    DIMS,
    DIMS_NUCLEO,
    ETIQUETAS_CUARTIL,
    VARIABLE_DE,
    asignar_posicion,
)

RANDOM_SEED = 42
VENTANA_AJUSTE_PERFILES = (2005, 2012)
NOMBRE_DIM = {f"S_{d}": DIMENSIONES[d]["nombre"] for d in DIMS}
ORDEN_NIVELES = {e: i for i, e in enumerate(ETIQUETAS_CUARTIL)}


# ---------------------------------------------------------------------------
# 1. Perfiles fiscales (k-means)
# ---------------------------------------------------------------------------

def ajustar_perfiles(panel: pd.DataFrame, ks=(3, 4, 5)) -> tuple[pd.DataFrame, dict]:
    """
    Ajusta k-means sobre los 4 subíndices en la ventana 2005-2012, elige k por silhouette y asigna un perfil a cada
    observación con los 4 subíndices disponibles, de todos los años. Las observaciones con algún subíndice faltante
    (p. ej. La Pampa 2005-2017 en solvencia) quedan sin perfil: no se imputan.
    """
    panel = panel.copy()
    completo = panel[COLS_S].notna().all(axis=1)
    en_base = panel["anio"].between(*VENTANA_AJUSTE_PERFILES) & completo
    X_base = panel.loc[en_base, COLS_S].to_numpy()

    mejor, silhouettes = None, {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED).fit(X_base)
        s = silhouette_score(X_base, km.labels_)
        silhouettes[k] = float(s)
        if mejor is None or s > mejor[1]:
            mejor = (km, s, k)
    km, sil, k = mejor

    centroides = pd.DataFrame(km.cluster_centers_, columns=COLS_S)
    orden = centroides.mean(axis=1).sort_values().index.tolist()      # de menor a mayor fragilidad media del centroide
    mapa = {old: new for new, old in enumerate(orden)}
    centroides = centroides.loc[orden].reset_index(drop=True)

    nombres = {}
    for i, fila in centroides.iterrows():
        altas = [NOMBRE_DIM[c].lower() for c in fila.sort_values(ascending=False).index if fila[c] >= 60]
        if len(altas) == 0:
            desc = "Intermedio, sin dimensión dominante" if fila.mean() >= 35 else "Holgado"
        elif len(altas) >= 3:
            desc = "Tensión generalizada"
        else:
            desc = "Tensión en " + " y ".join(altas)
        nombres[i] = f"Perfil {i + 1} · {desc}"

    panel["perfil"] = np.nan
    etiquetas = km.predict(panel.loc[completo, COLS_S].to_numpy())
    panel.loc[completo, "perfil"] = [mapa[e] for e in etiquetas]
    panel["perfil_nombre"] = panel["perfil"].map(nombres).fillna("Sin perfil (datos incompletos)")

    info = {
        "k": k, "silhouette": float(sil), "silhouettes_por_k": silhouettes,
        "centroides": centroides.rename(columns=NOMBRE_DIM), "nombres": nombres, "n_base": int(len(X_base)),
    }
    return panel, info


# ---------------------------------------------------------------------------
# 2. Anomalías (Isolation Forest) + cambios interanuales
# ---------------------------------------------------------------------------

def agregar_cambios(panel: pd.DataFrame) -> pd.DataFrame:
    """Variación de 1 y 3 años del compuesto y de 1 año de cada subíndice."""
    panel = panel.sort_values(["provincia", "anio"]).copy()
    g = panel.groupby("provincia")
    panel["d_iffs_1a"] = g["IFFS"].diff(1)
    panel["d_iffs_3a"] = g["IFFS"].diff(3)
    for c in COLS_S:
        panel[f"d_{c}_1a"] = g[c].diff(1)
    return panel


def detectar_anomalias(panel: pd.DataFrame, contaminacion: float = 0.04) -> pd.DataFrame:
    """
    Isolation Forest sobre (4 subíndices + cambio interanual del compuesto). Marca `anomalia=True` en el `contaminacion` % más
    aislado. Descriptivo, no predictivo.
    """
    panel = panel.copy()
    feats = COLS_S + ["d_iffs_1a"]
    ok = panel[feats].notna().all(axis=1)
    iso = IsolationForest(n_estimators=300, contamination=contaminacion, random_state=RANDOM_SEED)
    iso.fit(panel.loc[ok, feats])
    panel["anomalia_score"] = np.nan
    panel.loc[ok, "anomalia_score"] = -iso.score_samples(panel.loc[ok, feats])
    panel["anomalia"] = False
    panel.loc[ok, "anomalia"] = iso.predict(panel.loc[ok, feats]) == -1
    return panel


def construir_panel_analitico(panel_iffs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Panel del IFFS + posición relativa (cuartil) + perfiles + cambios + anomalías."""
    p = panel_iffs.copy()
    p["posicion"] = asignar_posicion(p, "IFFS")
    p, info_perfiles = ajustar_perfiles(p)
    p = agregar_cambios(p)
    p = detectar_anomalias(p)
    return p, info_perfiles


# ---------------------------------------------------------------------------
# 3. Proyección exploratoria + backtest honesto
# ---------------------------------------------------------------------------

def _tendencia(anios: np.ndarray, valores: np.ndarray, horizonte: int):
    """OLS lineal; devuelve predicciones a `horizonte` años, desvío residual y pendiente."""
    A = np.vstack([anios, np.ones_like(anios)]).T
    coef, *_ = np.linalg.lstsq(A, valores, rcond=None)
    resid = valores - A @ coef
    sigma = float(resid.std(ddof=2)) if len(valores) > 2 else float("nan")
    futuros = anios.max() + np.arange(1, horizonte + 1)
    return futuros, coef[0] * futuros + coef[1], sigma, coef[0]


def proyectar_iffs(panel: pd.DataFrame, provincia: str, horizonte: int = 2, ventana: int = 8) -> pd.DataFrame:
    """Tendencia lineal de los últimos `ventana` años válidos del compuesto, banda ~80% (±1.28σ, acotada a 0-100)."""
    d = panel[(panel["provincia"] == provincia)].dropna(subset=["IFFS"]).sort_values("anio").tail(ventana)
    if len(d) < 5:
        return pd.DataFrame()
    futuros, pred, sigma, pendiente = _tendencia(d["anio"].to_numpy(float), d["IFFS"].to_numpy(float), horizonte)
    ancho = 1.28 * sigma * np.sqrt(1 + np.arange(1, horizonte + 1) / len(d))
    return pd.DataFrame({
        "anio": futuros.astype(int), "proyeccion": np.clip(pred, 0, 100),
        "inferior": np.clip(pred - ancho, 0, 100), "superior": np.clip(pred + ancho, 0, 100), "pendiente_anual": pendiente,
    })


def backtest_proyeccion(panel: pd.DataFrame, ventana: int = 8, origen_min: int = 2012, origen_max: int = 2023) -> dict:
    """
    Pseudo-fuera de muestra a 1 año: para cada origen t y provincia, proyecta t+1 con datos <= t y compara el error absoluto
    contra la regla ingenua (compuesto de t). Sin look-ahead.
    """
    filas = []
    for prov, d in panel.dropna(subset=["IFFS"]).groupby("provincia"):
        d = d.sort_values("anio")
        for t in range(origen_min, origen_max + 1):
            hist = d[d["anio"] <= t].tail(ventana)
            real = d.loc[d["anio"] == t + 1, "IFFS"]
            if len(hist) < 5 or real.empty or hist["anio"].max() != t:
                continue
            _, pred, _, _ = _tendencia(hist["anio"].to_numpy(float), hist["IFFS"].to_numpy(float), 1)
            filas.append({
                "provincia": prov, "origen": t,
                "err_tendencia": abs(float(np.clip(pred[0], 0, 100)) - float(real.iloc[0])),
                "err_ingenua": abs(float(hist["IFFS"].iloc[-1]) - float(real.iloc[0])),
            })
    df = pd.DataFrame(filas)
    if df.empty:
        return {"n": 0}
    return {
        "n": int(len(df)), "mae_tendencia": float(df["err_tendencia"].mean()), "mae_ingenua": float(df["err_ingenua"].mean()),
        "tendencia_gana_pct": float((df["err_tendencia"] < df["err_ingenua"]).mean() * 100), "detalle": df,
    }


# ---------------------------------------------------------------------------
# 4. Simulador de escenarios (sensibilidad mecánica)
# ---------------------------------------------------------------------------

def _percentil_en_corte(valores_otros: pd.Series, valor: float) -> float:
    """Percentil de rango medio de `valor` insertado entre los valores de las demás jurisdicciones del mismo año."""
    if pd.isna(valor):
        return np.nan
    v = valores_otros.dropna().to_numpy(float)
    n = len(v) + 1
    rango = (v < valor).sum() + 1 + 0.5 * (v == valor).sum()          # rango medio (empates cuentan 0,5)
    return float(100.0 * (rango - 0.5) / n)


def _subindices_desde_ratios(panel: pd.DataFrame, provincia: str, anio: int, vals: dict) -> tuple[float, dict]:
    """Subíndices y compuesto de una provincia con `vals` (ratios contrafácticos) frente a las demás jurisdicciones del año."""
    corte = panel[(panel["anio"] == anio) & (panel["provincia"] != provincia)]
    percs = {f"S_{d}": _percentil_en_corte(corte[VARIABLE_DE[d]], vals[d]) for d in DIMS}
    nucleo = [percs[f"S_{d}"] for d in DIMS_NUCLEO]
    iffs = float(np.mean(nucleo)) if not any(pd.isna(v) for v in nucleo) else np.nan
    return iffs, percs


def simular_escenario(panel: pd.DataFrame, provincia: str, anio: int,
                      caida_ingresos: float = 0.0, depreciacion: float = 0.0, suba_servicio: float = 0.0) -> dict:
    """
    Recalcula subíndices y compuesto con shocks mecánicos sobre el año elegido, dejando FIJAS a las demás jurisdicciones
    (la posición es relativa a los pares de ese año).

    - caida_ingresos (0-1): ingresos corrientes caen ese %, gasto constante -> el resultado empeora.
    - depreciacion (>=0): revaloriza la parte de la deuda en moneda extranjera. Solo para provincias con composición por moneda
      (último dato disponible 2014-2020); si no hay dato, no se aplica y se avisa.
    - suba_servicio (>=0): el servicio total sube ese % (cota superior: incluye amortización de capital).
    """
    fila = panel[(panel["provincia"] == provincia) & (panel["anio"] == anio)]
    if fila.empty:
        return {"error": "Sin datos para esa provincia y año."}
    f = fila.iloc[0]

    fx_serie = panel.loc[(panel["provincia"] == provincia) & panel["pct_moneda_extranjera"].notna()].sort_values("anio")
    fx_share = (float(fx_serie["pct_moneda_extranjera"].iloc[-1]) / 100.0) if not fx_serie.empty else np.nan
    fx_anio = int(fx_serie["anio"].iloc[-1]) if not fx_serie.empty else None
    fx_aplicado = depreciacion > 0 and not np.isnan(fx_share)

    ing0, deuda0, serv0 = f["ingresos_corrientes"], f["deuda_stock_total"], f["servicio_total"]
    res_ing0 = f["D_resultado_fiscal_pct_ingresos"]
    ing1 = ing0 * (1 - caida_ingresos)
    deuda1 = deuda0 * (1 + (fx_share * depreciacion if fx_aplicado else 0.0))
    serv1 = serv0 * (1 + suba_servicio) if pd.notna(serv0) else np.nan
    res_ing1 = (res_ing0 - caida_ingresos) / (1 - caida_ingresos) if caida_ingresos < 1 else np.nan

    base_vals = {d: f[VARIABLE_DE[d]] for d in DIMS}
    nuevo_vals = {"solvencia": deuda1 / ing1, "liquidez": serv1 / ing1 if pd.notna(serv1) else np.nan,
                  "estructural": f["rigidez_gastototal"], "coyuntural": -res_ing1}
    base_iffs, base_p = _subindices_desde_ratios(panel, provincia, anio, base_vals)
    nuevo_iffs, nuevo_p = _subindices_desde_ratios(panel, provincia, anio, nuevo_vals)

    def cuartil(x):  # posición del valor entre los compuestos de las demás jurisdicciones ese año
        otros = panel.loc[(panel["anio"] == anio) & (panel["provincia"] != provincia), "IFFS"].dropna()
        if pd.isna(x) or otros.empty:
            return "Sin dato"
        rango = int(np.searchsorted(np.sort(otros.to_numpy()), x, side="right")) + 1      # misma regla que asignar_posicion
        return ETIQUETAS_CUARTIL[int(min(4, np.ceil(4 * rango / (len(otros) + 1)))) - 1]

    return {
        "provincia": provincia, "anio": anio, "iffs_base": base_iffs, "iffs_escenario": nuevo_iffs,
        "delta": nuevo_iffs - base_iffs, "posicion_base": cuartil(base_iffs), "posicion_escenario": cuartil(nuevo_iffs),
        "subindices_base": base_p, "subindices_escenario": nuevo_p,
        "ratios": {"deuda_sobre_ingresos": (base_vals["solvencia"], nuevo_vals["solvencia"]),
                   "servicio_sobre_ingresos": (base_vals["liquidez"], nuevo_vals["liquidez"])},
        "fx_share": fx_share, "fx_anio": fx_anio, "fx_aplicado": fx_aplicado,
        "fx_solicitado_sin_dato": depreciacion > 0 and np.isnan(fx_share),
    }


# ---------------------------------------------------------------------------
# 5. Ranking de cambios
# ---------------------------------------------------------------------------

def ranking_cambios(panel: pd.DataFrame, anio: int) -> pd.DataFrame:
    """Variación del compuesto respecto del año anterior, cambio de cuartil y subíndice que más se movió."""
    d = panel[panel["anio"] == anio].dropna(subset=["d_iffs_1a"]).copy()
    ant = panel[panel["anio"] == anio - 1].set_index("provincia")["posicion"]
    d["posicion_anterior"] = d["provincia"].map(ant)
    cols_d = {f"d_{c}_1a": NOMBRE_DIM[c] for c in COLS_S}
    d["dimension_motor"] = d[list(cols_d)].abs().idxmax(axis=1).map(cols_d)
    d["cambio_cuartil"] = [
        "Subió de cuartil" if ORDEN_NIVELES.get(str(a), -1) > ORDEN_NIVELES.get(str(b), -1) and pd.notna(b)
        else "Bajó de cuartil" if ORDEN_NIVELES.get(str(a), -1) < ORDEN_NIVELES.get(str(b), -1) and pd.notna(b)
        else "Sin cambio"
        for a, b in zip(d["posicion"], d["posicion_anterior"])
    ]
    return d[["provincia", "IFFS", "d_iffs_1a", "d_iffs_3a", "posicion_anterior", "posicion",
              "cambio_cuartil", "dimension_motor", "anomalia"]].sort_values("d_iffs_1a", ascending=False)


# ---------------------------------------------------------------------------
# 6. Generador de análisis (motor de reglas)
# ---------------------------------------------------------------------------

def _fmt(x, dec=1):
    return "s/d" if pd.isna(x) else f"{x:.{dec}f}"


def generar_analisis(panel: pd.DataFrame, provincia: str, anio: int, eventos: pd.DataFrame,
                     info_perfiles: dict, backtest: dict | None = None) -> str:
    """Análisis en Markdown de una provincia-año. Solo usa números calculados; cada bloque rotula su tipo de afirmación."""
    d = panel[(panel["provincia"] == provincia) & (panel["anio"] == anio)]
    if d.empty:
        return f"No hay datos para **{provincia}** en **{anio}**."
    f = d.iloc[0]
    partes = []
    del_anio = panel[panel["anio"] == anio].dropna(subset=["IFFS"])

    # 1. Diagnóstico [EVIDENCIA]
    if pd.isna(f["IFFS"]):
        faltan = [DIMENSIONES[x]["nombre"].lower() for x in DIMS_NUCLEO if pd.isna(f[f"S_{x}"])]
        motivo = ("el stock de deuda de La Pampa 2005-2017 es un dato no confiable (posible error de unidades, sin confirmación oficial)"
                  if provincia == "La Pampa" and anio <= 2017 else "no hay datos de " + ", ".join(faltan))
        partes.append(f"**1. Diagnóstico [evidencia].** No se calcula el compuesto de {provincia} en {anio}: {motivo}. "
                      "Los subíndices con dato se muestran igual.")
    else:
        puesto = int((del_anio["IFFS"] > f["IFFS"]).sum()) + 1
        partes.append(
            f"**1. Diagnóstico [evidencia].** En {anio}, {provincia} tiene un IFFS de **{f['IFFS']:.1f}** (posición: **{f['posicion']}**), "
            f"puesto **{puesto} de {len(del_anio)}** en fragilidad relativa (1 = más frágil). Mediana ese año: {del_anio['IFFS'].median():.1f}."
        )

    # 2. Qué lo explica [EVIDENCIA]
    percs = f[COLS_S].dropna().sort_values(ascending=False)
    if len(percs):
        top = ", ".join(f"{NOMBRE_DIM[k].lower()} ({v:.0f})" for k, v in percs.head(2).items())
        partes.append(
            f"**2. Qué lo explica [evidencia].** Subíndices más altos (0-100, relativos a las demás jurisdicciones ese año): {top}. "
            f"El más holgado es {NOMBRE_DIM[percs.index[-1]].lower()} ({percs.iloc[-1]:.0f}). "
            f"Deuda/Ingresos = {_fmt(f['deuda_sobre_ingresos'], 2)}x; Servicio/Ingresos = {_fmt(f['servicio_sobre_ingresos'], 3)}."
        )

    # 3. Trayectoria [EVIDENCIA]
    d1, d3 = f.get("d_iffs_1a"), f.get("d_iffs_3a")
    if pd.notna(d1):
        sentido = "subió" if d1 > 2 else "bajó" if d1 < -2 else "se mantuvo estable"
        t = f"**3. Trayectoria [evidencia].** Frente al año anterior el compuesto {sentido} ({d1:+.1f} puntos)"
        if pd.notna(d3):
            t += f"; en tres años, {d3:+.1f} puntos"
        partes.append(t + ". Un cambio de posición relativa no implica un cambio en las razones fiscales de la provincia: puede deberse al movimiento de las demás.")

    # 4. Perfil (ML) [EVIDENCIA descriptiva]
    if pd.notna(f.get("perfil")):
        pares = panel[(panel["anio"] == anio) & (panel["perfil"] == f["perfil"]) & (panel["provincia"] != provincia)]["provincia"].tolist()
        s = (f"**4. Perfil fiscal [evidencia descriptiva] (k-means, k={info_perfiles['k']}, silhouette {info_perfiles['silhouette']:.2f}: "
             f"grupos de contornos moderados).** Pertenece al *{f['perfil_nombre']}*")
        s += f", junto a {', '.join(pares[:6])}{'…' if len(pares) > 6 else ''}." if pares else "."
        partes.append(s)
    else:
        partes.append("**4. Perfil fiscal.** No se asigna perfil: falta algún subíndice (no se imputan datos).")

    # 5. Anomalía
    if bool(f.get("anomalia")):
        partes.append("**5. Señal estadística [evidencia descriptiva].** Marcada como **inusual** por Isolation Forest (combinación o salto poco "
                      "frecuente en el panel). Es una señal para revisar, no un veredicto.")

    # 6. Proyección
    proy = proyectar_iffs(panel[panel["anio"] <= anio], provincia)
    if not proy.empty:
        r = proy.iloc[0]
        txt = (f"**6. Extrapolación de tendencia [supuesto: la tendencia continúa].** El compuesto de {int(r['anio'])} rondaría "
               f"**{r['proyeccion']:.0f}** (banda ~80%: {r['inferior']:.0f}–{r['superior']:.0f}).")
        if backtest and backtest.get("n"):
            txt += (f" Precaución: en un backtest a 1 año ({backtest['n']} casos) el error medio de la tendencia fue "
                    f"{backtest['mae_tendencia']:.1f} puntos vs. {backtest['mae_ingenua']:.1f} de repetir el último valor.")
        partes.append(txt)

    # 7. Eventos
    ev = eventos[eventos["provincia"] == provincia]
    if not ev.empty:
        lista = "; ".join(f"{int(r.anio_evento)} ({r.tipo_evento.lower()})" for r in ev.itertuples())
        partes.append(f"**7. Antecedentes [evidencia].** Eventos de estrés en la base: {lista}. (Para 2020 el año es una convención; ver limitaciones.)")
    else:
        partes.append("**7. Antecedentes [evidencia].** Sin default ni reestructuración de deuda de mercado registrados en 2005-2025.")

    # 8. Cómo leerlo [limitación]
    partes.append(
        "**8. Cómo leerlo [limitación].** Es una posición *relativa* frente a las 24 jurisdicciones ese año; no es una probabilidad de default. "
        "Con 13 eventos (12 en el mismo shock de 2020-21) no se puede validar capacidad predictiva. Las cifras de discriminación y sus "
        "límites (el índice ya separaba a ese grupo años antes del shock; la participación de bonos separa casi igual) están en la sección Validación."
    )
    return es("\n\n".join(partes))
