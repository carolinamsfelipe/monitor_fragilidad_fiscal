"""
analytics.py
=============

Capa de analítica avanzada ("IA sin costo") sobre el panel del IFFS.

Todo se resuelve con machine learning clásico (scikit-learn / numpy), sin APIs
externas ni claves, y es 100% reproducible (semilla fija):

1. Perfiles fiscales (k-means): agrupa las jurisdicciones según la forma de su
   fragilidad (qué dimensiones pesan), no solo según su nivel. Se ajusta SOLO
   con la ventana base 2005-2012 y se asigna al resto de los años, para no
   usar información futura al definir los grupos.
2. Detección de anomalías (Isolation Forest): marca observaciones provincia-año
   con combinaciones o saltos inusuales. Es descriptivo/retrospectivo.
3. Proyección exploratoria a corto plazo: tendencia lineal de los últimos años
   con banda de incertidumbre. Se valida con un backtest contra una regla ingenua
   (repetir el último valor) y se informa el resultado tal cual sale.
4. Simulador de escenarios: recalcula el IFFS-B ante shocks mecánicos de
   ingresos, tipo de cambio y servicio de deuda. Es análisis de sensibilidad,
   NO un pronóstico.
5. Ranking de cambios y alertas: qué provincias se deterioraron o mejoraron y
   qué dimensión explica el movimiento.
6. Generador de análisis exhaustivo: arma un texto en lenguaje claro combinando
   todo lo anterior. Es un motor de reglas/plantillas: solo redacta números que
   efectivamente se calcularon (no inventa ni "alucina").

Nada de esto valida al IFFS como predictor de default: ver notebook, Sección 13.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .indicators import (
    BASE_WINDOW,
    DIMENSIONES_NUCLEO,
    _percentil_ecdf,
    asignar_nivel_alerta,
)

RANDOM_SEED = 42
COLS_P = [f"P_{v}" for v in DIMENSIONES_NUCLEO]
NOMBRE_DIM = {f"P_{v}": n.split(" (")[0] for v, n in DIMENSIONES_NUCLEO.items()}
ORDEN_NIVELES = {"Bajo": 0, "Medio": 1, "Alto": 2, "Crítico": 3}


# ---------------------------------------------------------------------------
# 1. Perfiles fiscales (k-means)
# ---------------------------------------------------------------------------

def ajustar_perfiles(panel: pd.DataFrame, ks=(3, 4, 5)) -> tuple[pd.DataFrame, dict]:
    """
    Ajusta k-means sobre los percentiles de las 4 dimensiones en la ventana base
    (2005-2012), elige k por silhouette y asigna un perfil a cada observación
    completa (las 4 dimensiones disponibles) de todos los años.

    Observaciones con alguna dimensión faltante (p. ej. La Pampa 2025, sin datos reportados) quedan sin perfil:
    no se imputan.
    """
    panel = panel.copy()
    completo = panel[COLS_P].notna().all(axis=1)
    en_base = panel["anio"].between(*BASE_WINDOW) & completo
    X_base = panel.loc[en_base, COLS_P].to_numpy()

    mejor = None
    silhouettes = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED).fit(X_base)
        s = silhouette_score(X_base, km.labels_)
        silhouettes[k] = float(s)
        if mejor is None or s > mejor[1]:
            mejor = (km, s, k)
    km, sil, k = mejor

    # Reordenar clusters de menor a mayor fragilidad media del centroide
    centroides = pd.DataFrame(km.cluster_centers_, columns=COLS_P)
    orden = centroides.mean(axis=1).sort_values().index.tolist()
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
    etiquetas = km.predict(panel.loc[completo, COLS_P].to_numpy())
    panel.loc[completo, "perfil"] = [mapa[e] for e in etiquetas]
    panel["perfil_nombre"] = panel["perfil"].map(nombres).fillna("Sin perfil (datos incompletos)")

    info = {
        "k": k,
        "silhouette": float(sil),
        "silhouettes_por_k": silhouettes,
        "centroides": centroides.rename(columns=NOMBRE_DIM),
        "nombres": nombres,
        "n_base": int(len(X_base)),
    }
    return panel, info


# ---------------------------------------------------------------------------
# 2. Anomalías (Isolation Forest) + cambios interanuales
# ---------------------------------------------------------------------------

def agregar_cambios(panel: pd.DataFrame) -> pd.DataFrame:
    """Agrega variación de 1 y 3 años del IFFS-B, y de cada dimensión (1 año)."""
    panel = panel.sort_values(["provincia", "anio"]).copy()
    g = panel.groupby("provincia")
    panel["d_iffs_1a"] = g["IFFS_B"].diff(1)
    panel["d_iffs_3a"] = g["IFFS_B"].diff(3)
    for c in COLS_P:
        panel[f"d_{c}_1a"] = g[c].diff(1)
    return panel


def detectar_anomalias(panel: pd.DataFrame, contaminacion: float = 0.04) -> pd.DataFrame:
    """
    Isolation Forest sobre (percentiles de las 4 dimensiones + cambio interanual del IFFS-B).
    Marca `anomalia=True` en el `contaminacion` % más aislado. Descriptivo, no predictivo.
    """
    panel = panel.copy()
    feats = COLS_P + ["d_iffs_1a"]
    ok = panel[feats].notna().all(axis=1)
    iso = IsolationForest(n_estimators=300, contamination=contaminacion, random_state=RANDOM_SEED)
    iso.fit(panel.loc[ok, feats])
    panel["anomalia_score"] = np.nan
    panel.loc[ok, "anomalia_score"] = -iso.score_samples(panel.loc[ok, feats])  # mayor = más raro
    panel["anomalia"] = False
    panel.loc[ok, "anomalia"] = iso.predict(panel.loc[ok, feats]) == -1
    return panel


def construir_panel_analitico(panel_iffs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Panel del IFFS + niveles de alerta + perfiles + cambios + anomalías."""
    p = panel_iffs.copy()
    p["nivel_alerta"] = asignar_nivel_alerta(p["IFFS_B"])
    p, info_perfiles = ajustar_perfiles(p)
    p = agregar_cambios(p)
    p = detectar_anomalias(p)
    return p, info_perfiles


# ---------------------------------------------------------------------------
# 3. Proyección exploratoria + backtest honesto
# ---------------------------------------------------------------------------

def _tendencia(anios: np.ndarray, valores: np.ndarray, horizonte: int):
    """OLS lineal; devuelve predicciones a `horizonte` años y desvío residual."""
    A = np.vstack([anios, np.ones_like(anios)]).T
    coef, *_ = np.linalg.lstsq(A, valores, rcond=None)
    resid = valores - A @ coef
    sigma = float(resid.std(ddof=2)) if len(valores) > 2 else float("nan")
    futuros = anios.max() + np.arange(1, horizonte + 1)
    return futuros, coef[0] * futuros + coef[1], sigma, coef[0]


def proyectar_iffs(panel: pd.DataFrame, provincia: str, horizonte: int = 2, ventana: int = 8) -> pd.DataFrame:
    """Tendencia lineal de los últimos `ventana` años válidos, banda ~80% (±1.28σ, acotada a 0-100)."""
    d = panel[(panel["provincia"] == provincia)].dropna(subset=["IFFS_B"]).sort_values("anio").tail(ventana)
    if len(d) < 5:
        return pd.DataFrame()
    futuros, pred, sigma, pendiente = _tendencia(d["anio"].to_numpy(float), d["IFFS_B"].to_numpy(float), horizonte)
    ancho = 1.28 * sigma * np.sqrt(1 + np.arange(1, horizonte + 1) / len(d))
    return pd.DataFrame({
        "anio": futuros.astype(int),
        "proyeccion": np.clip(pred, 0, 100),
        "inferior": np.clip(pred - ancho, 0, 100),
        "superior": np.clip(pred + ancho, 0, 100),
        "pendiente_anual": pendiente,
    })


def backtest_proyeccion(panel: pd.DataFrame, ventana: int = 8, origen_min: int = 2012, origen_max: int = 2023) -> dict:
    """
    Pseudo-fuera de muestra a 1 año: para cada origen t y provincia, proyecta t+1 con datos <= t
    y compara el error absoluto contra la regla ingenua (IFFS-B de t). Sin look-ahead.
    """
    filas = []
    for prov, d in panel.dropna(subset=["IFFS_B"]).groupby("provincia"):
        d = d.sort_values("anio")
        for t in range(origen_min, origen_max + 1):
            hist = d[d["anio"] <= t].tail(ventana)
            real = d.loc[d["anio"] == t + 1, "IFFS_B"]
            if len(hist) < 5 or real.empty or hist["anio"].max() != t:
                continue
            _, pred, _, _ = _tendencia(hist["anio"].to_numpy(float), hist["IFFS_B"].to_numpy(float), 1)
            filas.append({
                "provincia": prov, "origen": t,
                "err_tendencia": abs(float(np.clip(pred[0], 0, 100)) - float(real.iloc[0])),
                "err_ingenua": abs(float(hist["IFFS_B"].iloc[-1]) - float(real.iloc[0])),
            })
    df = pd.DataFrame(filas)
    if df.empty:
        return {"n": 0}
    return {
        "n": int(len(df)),
        "mae_tendencia": float(df["err_tendencia"].mean()),
        "mae_ingenua": float(df["err_ingenua"].mean()),
        "tendencia_gana_pct": float((df["err_tendencia"] < df["err_ingenua"]).mean() * 100),
        "detalle": df,
    }


# ---------------------------------------------------------------------------
# 4. Simulador de escenarios (sensibilidad mecánica)
# ---------------------------------------------------------------------------

def _pct_vs_base(panel: pd.DataFrame, variable: str, valor: float) -> float:
    base = panel.loc[panel["anio"].between(*BASE_WINDOW), variable]
    if pd.isna(valor):
        return np.nan
    return float(_percentil_ecdf(pd.Series([valor]), base).iloc[0])


def _iffs_desde_ratios(panel: pd.DataFrame, deuda_ing, serv_ing, rigidez, fragilidad_res) -> tuple[float, dict]:
    percs = {
        "P_deuda_sobre_ingresos": _pct_vs_base(panel, "deuda_sobre_ingresos", deuda_ing),
        "P_servicio_sobre_ingresos": _pct_vs_base(panel, "servicio_sobre_ingresos", serv_ing),
        "P_rigidez_gastototal": _pct_vs_base(panel, "rigidez_gastototal", rigidez),
        "P_fragilidad_resultado": _pct_vs_base(panel, "fragilidad_resultado", fragilidad_res),
    }
    validos = [v for v in percs.values() if not pd.isna(v)]
    iffs = float(np.mean(validos)) if len(validos) >= 3 else np.nan  # misma regla que IFFS-B (>= 3/4)
    return iffs, percs


def simular_escenario(panel: pd.DataFrame, provincia: str, anio: int,
                      caida_ingresos: float = 0.0, depreciacion: float = 0.0,
                      suba_servicio: float = 0.0) -> dict:
    """
    Recalcula el IFFS-B con shocks mecánicos sobre el año base elegido.

    - caida_ingresos (0-1): ingresos corrientes caen ese %, gasto constante -> resultado empeora.
    - depreciacion (>=0): la parte de la deuda en moneda extranjera se revaloriza. Solo disponible
      para provincias con composición por moneda (último dato 2014-2020); si no, no se aplica.
    - suba_servicio (>=0): el servicio de deuda total sube ese % (cota superior: aplica también al capital).
    Los percentiles se calculan contra la MISMA ventana base fija del IFFS (2005-2012).
    """
    fila = panel[(panel["provincia"] == provincia) & (panel["anio"] == anio)]
    if fila.empty:
        return {"error": "Sin datos para esa provincia y año."}
    f = fila.iloc[0]

    fx_serie = panel.loc[(panel["provincia"] == provincia) & panel["pct_moneda_extranjera"].notna()]\
        .sort_values("anio")
    fx_share = (float(fx_serie["pct_moneda_extranjera"].iloc[-1]) / 100.0) if not fx_serie.empty else np.nan
    fx_anio = int(fx_serie["anio"].iloc[-1]) if not fx_serie.empty else None
    fx_aplicado = depreciacion > 0 and not np.isnan(fx_share)

    ing0, deuda0, serv0 = f["ingresos_corrientes"], f["deuda_stock_total"], f["servicio_total"]
    res_ing0 = f["D_resultado_fiscal_pct_ingresos"]  # resultado financiero / ingresos corrientes

    ing1 = ing0 * (1 - caida_ingresos)
    deuda1 = deuda0 * (1 + (fx_share * depreciacion if fx_aplicado else 0.0))
    serv1 = serv0 * (1 + suba_servicio) if pd.notna(serv0) else np.nan
    res_ing1 = (res_ing0 - caida_ingresos) / (1 - caida_ingresos) if caida_ingresos < 1 else np.nan

    base_iffs, base_p = _iffs_desde_ratios(
        panel, f["deuda_sobre_ingresos"], f["servicio_sobre_ingresos"], f["rigidez_gastototal"], f["fragilidad_resultado"])
    nuevo_iffs, nuevo_p = _iffs_desde_ratios(
        panel, deuda1 / ing1, serv1 / ing1 if pd.notna(serv1) else np.nan, f["rigidez_gastototal"], -res_ing1)

    nivel = lambda x: str(asignar_nivel_alerta(pd.Series([x])).iloc[0]) if pd.notna(x) else "Sin datos"
    return {
        "provincia": provincia, "anio": anio,
        "iffs_base": base_iffs, "iffs_escenario": nuevo_iffs,
        "delta": nuevo_iffs - base_iffs,
        "nivel_base": nivel(base_iffs), "nivel_escenario": nivel(nuevo_iffs),
        "percentiles_base": base_p, "percentiles_escenario": nuevo_p,
        "ratios": {
            "deuda_sobre_ingresos": (f["deuda_sobre_ingresos"], deuda1 / ing1),
            "servicio_sobre_ingresos": (f["servicio_sobre_ingresos"], serv1 / ing1 if pd.notna(serv1) else np.nan),
        },
        "fx_share": fx_share, "fx_anio": fx_anio, "fx_aplicado": fx_aplicado,
        "fx_solicitado_sin_dato": depreciacion > 0 and np.isnan(fx_share),
    }


# ---------------------------------------------------------------------------
# 5. Ranking de cambios y alertas
# ---------------------------------------------------------------------------

def ranking_cambios(panel: pd.DataFrame, anio: int) -> pd.DataFrame:
    """Variación del IFFS-B respecto del año anterior + dimensión que más se movió."""
    d = panel[panel["anio"] == anio].dropna(subset=["d_iffs_1a"]).copy()
    ant = panel[panel["anio"] == anio - 1].set_index("provincia")["nivel_alerta"]
    d["nivel_anterior"] = d["provincia"].map(ant)
    cols_d = {f"d_{c}_1a": NOMBRE_DIM[c] for c in COLS_P}
    d["dimension_motor"] = d[list(cols_d)].abs().idxmax(axis=1).map(cols_d)
    d["cambio_nivel"] = [
        "Empeoró" if ORDEN_NIVELES.get(str(a), -1) > ORDEN_NIVELES.get(str(b), -1) and pd.notna(b)
        else "Mejoró" if ORDEN_NIVELES.get(str(a), -1) < ORDEN_NIVELES.get(str(b), -1) and pd.notna(b)
        else "Sin cambio"
        for a, b in zip(d["nivel_alerta"], d["nivel_anterior"])
    ]
    return d[["provincia", "IFFS_B", "d_iffs_1a", "d_iffs_3a", "nivel_anterior", "nivel_alerta",
              "cambio_nivel", "dimension_motor", "anomalia"]].sort_values("d_iffs_1a", ascending=False)


# ---------------------------------------------------------------------------
# 6. Generador de análisis exhaustivo (motor de reglas)
# ---------------------------------------------------------------------------

def _fmt(x, dec=1):
    return "s/d" if pd.isna(x) else f"{x:.{dec}f}"


def generar_analisis(panel: pd.DataFrame, provincia: str, anio: int, eventos: pd.DataFrame,
                     info_perfiles: dict, backtest: dict | None = None) -> str:
    """Texto en Markdown con el análisis de una provincia-año. Solo usa números calculados."""
    d = panel[(panel["provincia"] == provincia) & (panel["anio"] == anio)]
    if d.empty or pd.isna(d["IFFS_B"].iloc[0]):
        return f"No hay datos suficientes para **{provincia}** en **{anio}**."
    f = d.iloc[0]
    partes = []

    # 1. Diagnóstico
    del_anio = panel[(panel["anio"] == anio)].dropna(subset=["IFFS_B"])
    puesto = int((del_anio["IFFS_B"] > f["IFFS_B"]).sum()) + 1
    partes.append(
        f"**1. Diagnóstico.** En {anio}, {provincia} tiene un IFFS-B de **{f['IFFS_B']:.1f}/100** "
        f"(nivel **{f['nivel_alerta']}**), puesto **{puesto} de {len(del_anio)}** en fragilidad relativa "
        f"(1 = más frágil). Mediana de las jurisdicciones ese año: {del_anio['IFFS_B'].median():.1f}."
    )
    if f["IFFS_B_estado"] != "completo":
        partes[-1] += (" El índice es **provisional**: falta una de las cuatro dimensiones "
                       "(dato no disponible), por lo que no es plenamente comparable con las demás.")

    # 2. Qué lo explica
    percs = f[COLS_P].dropna().sort_values(ascending=False)
    top = ", ".join(f"{NOMBRE_DIM[k].lower()} (percentil {v:.0f})" for k, v in percs.head(2).items())
    bajo = percs.index[-1]
    partes.append(
        f"**2. Qué lo explica.** Lo que más pesa: {top}. La dimensión más holgada es "
        f"{NOMBRE_DIM[bajo].lower()} (percentil {percs.iloc[-1]:.0f}). "
        f"Deuda/Ingresos = {_fmt(f['deuda_sobre_ingresos'], 2)}x; Servicio/Ingresos = {_fmt(f['servicio_sobre_ingresos'], 3)}."
    )

    # 3. Trayectoria
    d1, d3 = f.get("d_iffs_1a"), f.get("d_iffs_3a")
    if pd.notna(d1):
        sentido = "se deterioró" if d1 > 2 else "mejoró" if d1 < -2 else "se mantuvo estable"
        t = f"**3. Trayectoria.** Frente al año anterior {sentido} ({d1:+.1f} puntos)"
        if pd.notna(d3):
            t += f"; en tres años, {d3:+.1f} puntos"
        partes.append(t + ".")

    # 4. Perfil (ML)
    if pd.notna(f.get("perfil")):
        pares = panel[(panel["anio"] == anio) & (panel["perfil"] == f["perfil"]) & (panel["provincia"] != provincia)]["provincia"].tolist()
        s = f"**4. Perfil fiscal (k-means, k={info_perfiles['k']}, silhouette {info_perfiles['silhouette']:.2f}: grupos de contornos moderados).** Pertenece al *{f['perfil_nombre']}*"
        s += f", junto a {', '.join(pares[:6])}{'…' if len(pares) > 6 else ''}." if pares else "."
        partes.append(s)
    else:
        partes.append("**4. Perfil fiscal.** No se asigna perfil: faltan dimensiones (no se imputan datos).")

    # 5. Anomalía
    if bool(f.get("anomalia")):
        partes.append(
            "**5. Señal de alerta estadística.** Esta observación fue marcada como **inusual** por Isolation Forest "
            "(combinación o salto poco frecuente en el panel). Es una señal para revisar, no un veredicto."
        )

    # 6. Proyección
    proy = proyectar_iffs(panel[panel["anio"] <= anio], provincia)
    if not proy.empty:
        r = proy.iloc[0]
        txt = (f"**6. Proyección exploratoria.** Extrapolando la tendencia reciente, el IFFS-B de {int(r['anio'])} rondaría "
               f"**{r['proyeccion']:.0f}** (banda ~80%: {r['inferior']:.0f}–{r['superior']:.0f}).")
        if backtest and backtest.get("n"):
            txt += (f" Precaución: en un backtest a 1 año ({backtest['n']} casos) el error medio de la tendencia fue "
                    f"{backtest['mae_tendencia']:.1f} puntos vs. {backtest['mae_ingenua']:.1f} de la regla ingenua "
                    f"(repetir el último valor); la tendencia fue mejor en el {backtest['tendencia_gana_pct']:.0f}% de los casos.")
        partes.append(txt)

    # 7. Historial de eventos
    ev = eventos[eventos["provincia"] == provincia]
    if not ev.empty:
        lista = "; ".join(f"{int(r.anio_evento)} ({r.tipo_evento.lower()})" for r in ev.itertuples())
        partes.append(f"**7. Antecedentes.** Registra eventos de estrés en la base: {lista}.")
    else:
        partes.append("**7. Antecedentes.** Sin default ni reestructuración de deuda de mercado registrados en 2005-2025.")

    # 8. Advertencia
    partes.append(
        "**8. Cómo leerlo.** Es una lectura *relativa* al comportamiento histórico de las 24 jurisdicciones (ventana base "
        "2005-2012), no una probabilidad de default. Con 13 eventos concentrados en 2020-2021, la validación predictiva del "
        "índice es débil; el hallazgo más sólido es descriptivo (en 2019, deuda y servicio sobre ingresos separaban a quienes "
        "luego reestructuraron)."
    )
    return "\n\n".join(partes)
