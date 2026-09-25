"""
Monitor de Fragilidad Fiscal Provincial — Dashboard (Streamlit), IFFS v2
=========================================================================

Ejecutar con:   streamlit run app.py

Herramienta de apoyo al análisis: muestra POSICIONES RELATIVAS entre jurisdicciones (percentil dentro de cada año), NO probabilidades
de default. Todas las cifras de validación provienen de `outputs/` (generado por scripts/generar_resultados.py), las mismas que usan
el notebook, el informe y la presentación. Si el código o los datos cambiaron desde esa corrida, la app lo avisa.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import analytics as an
from src import indicators as ind
from src import pipeline as pl
from src import validation as val
from src.formato import es as _es

# Formato castellano (coma decimal) para todo el texto que la app escribe; se parchea una sola vez por proceso.
from streamlit.delta_generator import DeltaGenerator as _DG

if not getattr(_DG, "_es_parcheado", False):
    for _nombre in ("markdown", "caption", "info", "warning", "error", "success", "write"):
        def _envolver(orig):
            def _f(self, cuerpo=None, *a, **k):
                return orig(self, _es(cuerpo), *a, **k)
            return _f
        setattr(_DG, _nombre, _envolver(getattr(_DG, _nombre)))
    _orig_metric = _DG.metric
    _DG.metric = lambda self, label, value, delta=None, *a, **k: _orig_metric(self, _es(label), _es(value), _es(delta), *a, **k)
    _DG._es_parcheado = True

st.set_page_config(page_title="Monitor de Fragilidad Fiscal Provincial", page_icon="📊", layout="wide")

COLOR_DIM = {"solvencia": "#2a78d6", "liquidez": "#eb6834", "estructural": "#1baf7a", "coyuntural": "#4a3aa7"}
COLOR_POS = {**ind.COLORES_CUARTIL}
TABLAS = ROOT / "outputs" / "tables"


# ---------------------------------------------------------------------------
# Carga (cacheada)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Construyendo el panel del IFFS v2...")
def cargar(huella: str, la_pampa: str):
    return pl.construir_todo(la_pampa)


@st.cache_data(show_spinner=False)
def cargar_resultados(mtime: float):
    f = ROOT / "outputs" / "resultados_clave.json"
    R = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    tablas = {p.stem: pd.read_csv(p) for p in sorted(TABLAS.glob("T*.csv"))} if TABLAS.exists() else {}
    return R, tablas


HUELLA = pl.huella_del_proyecto()
st.sidebar.title("📊 Monitor de Fragilidad Fiscal")
st.sidebar.caption("IFFS v2 · 24 jurisdicciones · 2005-2025")
escenario = st.sidebar.radio(
    "Tratamiento de La Pampa (stock 2005-2017)",
    ["excluir", "sensibilidad_miles"],
    format_func=lambda x: {"excluir": "Principal: dato no confiable → faltante",
                           "sensibilidad_miles": "Sensibilidad: ÷1.000 (SUPUESTO sin confirmar)"}[x],
    help="El stock de deuda de La Pampa 2005-2017 puede tener un error de unidades y NO hay confirmación oficial. "
         "El análisis principal lo trata como faltante; la opción de sensibilidad aplica ÷1.000 solo como ejercicio.",
)
T = cargar(HUELLA, escenario)
R, TAB = cargar_resultados((ROOT / 'outputs' / 'resultados_clave.json').stat().st_mtime if (ROOT / 'outputs' / 'resultados_clave.json').exists() else 0.0)
panel, eventos, info_perfiles, backtest = T["panel"], T["eventos"], T["info_perfiles"], T["backtest"]
PROVINCIAS = sorted(panel["provincia"].unique())
ANIOS = sorted(panel["anio"].unique())

provincia_foco = st.sidebar.selectbox("Provincia a analizar", PROVINCIAS, index=PROVINCIAS.index("La Rioja"))
periodo = st.sidebar.slider("Período", min(ANIOS), max(ANIOS), (min(ANIOS), max(ANIOS)))
provincias_comparar = st.sidebar.multiselect("Provincias a comparar", PROVINCIAS,
                                             default=[p for p in ["La Rioja", "Buenos Aires", "CABA", "Chubut"] if p in PROVINCIAS])
with st.sidebar.expander("⚠️ Avisos de validación de datos", expanded=False):
    for a in T["avisos"]:
        st.write("-", a)

st.title("Monitor de Fragilidad Fiscal Provincial")
st.caption("Posición **relativa** de cada jurisdicción frente a sus pares (percentil dentro de cada año). "
           "**No es una probabilidad de default** ni una medida absoluta de solvencia.")
if escenario != "excluir":
    st.warning("**Escenario de sensibilidad activo.** La Pampa 2005-2017 se calcula dividiendo su stock de deuda por 1.000 bajo el SUPUESTO "
               "(sin confirmación oficial) de que la fuente está en miles. Las tablas de validación de la pestaña ✅ corresponden al análisis principal.")
if R is None:
    st.error("Falta `outputs/resultados_clave.json`: ejecutá `python scripts/generar_resultados.py`. La pestaña de validación no está disponible.")
elif R.get("huella") != HUELLA:
    st.warning("Las tablas de `outputs/` se generaron con una versión anterior del código o de los datos: volvé a ejecutar "
               "`python scripts/generar_resultados.py` para que la validación coincida con lo que se ve acá.")

panel_f = panel[panel["anio"].between(*periodo)]
(t_res, t_rank, t_evo, t_det, t_val, t_rob, t_perf, t_sim, t_dat) = st.tabs(
    ["🏠 Resumen", "🏆 Ranking", "📈 Evolución", "🔎 Detalle", "✅ Validación", "🧪 Robustez", "🧩 Perfiles y cambios", "🎛️ Simulador", "📥 Datos y cobertura"])

TIPOS = ("**Evidencia** = calculado en los datos · **Interpretación** = lectura económica · **Supuesto** = no verificado · "
         "**Diseño** = decisión de construcción · **Limitación** = lo que no se puede concluir.")


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def fmt_auc(m):
    return f"{m['auc']:.2f} [IC95%: {m['ic95'][0]:.2f}–{m['ic95'][1]:.2f}]"


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
with t_res:
    st.markdown("### Qué muestra y qué no")
    c1, c2, c3 = st.columns(3)
    if R:
        c1.metric("Cobertura del compuesto", f"{R['cobertura']['n_compuesto']} de {R['cobertura']['n_total']}", f"{R['cobertura']['pct_compuesto']}%", delta_color="off")
        d = R["discriminacion_2019"]["IFFS compuesto (principal)"]
        c2.metric("AUC en 2019 (discriminación)", f"{d['auc']:.2f}", f"IC95% {d['ic95'][0]:.2f}–{d['ic95'][1]:.2f}", delta_color="off")
        c3.metric("Eventos (default/reestructuración)", f"{R['eventos']['n_eventos']}", f"{R['eventos']['n_covid']} en el mismo shock 2020-21", delta_color="off")
    st.info(TIPOS)
    st.markdown("""
**Estructura (Diseño).** Cuatro **subíndices** (0-100, mayor = más frágil *respecto de los pares ese año*) y un **compuesto**:

| Subíndice | Variable | Entra al compuesto |
|---|---|---|
| **Solvencia** | Deuda / Ingresos corrientes | Sí |
| **Liquidez** | Servicio de deuda / Ingresos corrientes | Sí |
| **Estructural** | Gasto en personal / Gasto total | Sí |
| **Coyuntural** | −Resultado financiero / Ingresos | **No** (regla ex ante: su persistencia de rangos, __PERS__, es menor que 0,85; se reporta aparte) |

**Compuesto (IFFS)** = media simple de solvencia, liquidez y estructural. Se calcula solo si están las tres.
""".replace("__PERS__", f"{R['diseno']['props']['coyuntural']['persistencia_rangos']:.2f}".replace(".", ",") if R else "n/d"))
    if R:
        vt = R["diseno"]
        pv = R["placebo_temporal"]["auc_iffs_por_anio"]
        bo = R["discriminacion_2019"]["Benchmark: participación de bonos en la deuda (%)"]
        pre = R["pretendencia"]["principal|IFFS|T-5->T-1"]
        st.markdown("### Lo que la evidencia sostiene… y lo que no")
        st.success(f"**Evidencia — descripción y discriminación.** En 2019, las 12 provincias que luego reestructuraron tenían un IFFS más alto que las otras 12: "
                   f"AUC {fmt_auc(R['discriminacion_2019']['IFFS compuesto (principal)'])}; el azar produce AUC ≤ {R['azar_auc_2019']['azar_p95']:.2f} el 95% de las veces.")
        st.warning(f"**Limitación — no es anticipación.** (1) Ese grupo ya se separaba en {R['placebo_temporal']['primer_anio_auc_ge_085']} (AUC {pv[str(R['placebo_temporal']['primer_anio_auc_ge_085'])]:.2f}) "
                   f"y sigue separado después de los eventos (AUC {pv['2025']:.2f} en 2025): es un rasgo persistente del grupo. "
                   f"(2) El % de bonos en la deuda —una variable trivial— discrimina casi igual: AUC {fmt_auc(bo)}; solo las provincias con bonos podían reestructurarlos. "
                   f"(3) El cambio del IFFS entre T−5 y T−1 no fue significativamente mayor en las provincias con evento: {pre['dif']:+.1f} puntos "
                   f"[{pre['ic95'][0]:+.1f}; {pre['ic95'][1]:+.1f}].")
        st.error("**Limitación — validación predictiva no viable.** Hay 13 eventos; 12 ocurren en el mismo shock (2020-21) y, para 11 de ellos, el año exacto es una convención "
                 "(la fuente dice “2020-2021”). No hay eventos antes de 2020 para entrenar, no se puede calibrar una probabilidad y no se optimizaron pesos ni umbrales. Ver ✅ Validación.")
    st.markdown("### Datos: una corrección y una decisión")
    st.markdown(
        "- **CABA** (*evidencia*): el servicio de deuda estaba x1.000 por un error de unidades del procesamiento; se lee la hoja `Millones` de la DNAP, con validación automática.\n"
        "- **La Pampa 2005-2017** (*decisión de diseño*): posible error de unidades **sin confirmación oficial** → se trata como faltante (no se corrige ni se borra el original). "
        "La corrección ÷1.000 se muestra solo como sensibilidad (barra lateral).")


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
with t_rank:
    a = st.selectbox("Año", sorted(ANIOS, reverse=True), key="anio_rank")
    d = panel[panel["anio"] == a].dropna(subset=["IFFS"]).sort_values("IFFS")
    fig = go.Figure()
    fig.add_bar(y=d["provincia"], x=d["IFFS"], orientation="h", name="IFFS compuesto", marker_color="#d9d8d1",
                hovertemplate="%{y}: %{x:.1f}<extra>Compuesto</extra>")
    for dim, sym in zip(ind.DIMS, ["circle", "square", "triangle-up", "diamond"]):
        fig.add_scatter(y=d["provincia"], x=d[f"S_{dim}"], mode="markers", name=ind.NOMBRE_DIM[dim] + (" (fuera del compuesto)" if dim == "coyuntural" else ""),
                        marker=dict(color=COLOR_DIM[dim], symbol=sym, size=9, line=dict(color="white", width=1)),
                        hovertemplate="%{y}: %{x:.0f}<extra>" + ind.NOMBRE_DIM[dim] + "</extra>")
    fig.update_layout(title=f"IFFS {a}: compuesto (barra) y subíndices (puntos), 0-100 relativo a los pares de ese año", height=680,
                      xaxis=dict(range=[0, 100], title="Posición relativa 0-100"), legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig, width="stretch")
    st.caption("Los subíndices se leen contra las demás jurisdicciones del mismo año. Una provincia puede ser 'frágil' por una sola dimensión: mirar los puntos, no solo la barra.")
    sin = panel[(panel["anio"] == a) & panel["IFFS"].isna()]["provincia"].tolist()
    if sin:
        st.info(f"Sin compuesto en {a}: {', '.join(sin)} (falta al menos un subíndice; ver 📥 Datos y cobertura).")
    piv = panel_f.pivot_table(index="provincia", columns="anio", values="IFFS")
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fh = px.imshow(piv, color_continuous_scale="YlOrRd", zmin=0, zmax=100, aspect="auto",
                   title="Mapa de calor del compuesto (celdas vacías = sin dato)", labels=dict(x="Año", y="Provincia", color="IFFS"))
    fh.update_layout(height=620)
    st.plotly_chart(fh, width="stretch")


# ---------------------------------------------------------------------------
# Evolución
# ---------------------------------------------------------------------------
with t_evo:
    g = panel[(panel.provincia == provincia_foco) & panel.anio.between(*periodo)]
    fig = go.Figure()
    for dim in ind.DIMS:
        fig.add_scatter(x=g["anio"], y=g[f"S_{dim}"], mode="lines", name=ind.NOMBRE_DIM[dim], line=dict(color=COLOR_DIM[dim], width=2))
    fig.add_scatter(x=g["anio"], y=g["IFFS"], mode="lines+markers", name="Compuesto", line=dict(color="#0b0b0b", width=3.4))
    fig.add_vrect(x0=2019.5, x1=2021.5, fillcolor="grey", opacity=0.12, line_width=0, annotation_text="eventos 2020-21")
    fig.update_layout(title=f"{provincia_foco}: subíndices y compuesto (0-100, relativo a los pares de cada año)", height=440, yaxis=dict(range=[0, 100]),
                      xaxis_title="Año", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    ev = eventos[eventos.provincia == provincia_foco]
    if not ev.empty:
        st.warning(f"**{provincia_foco}** tiene {len(ev)} evento(s) en la base: " + "; ".join(f"{r.anio_evento} ({r.tipo_evento})" for r in ev.itertuples())
                   + ". Para 2020 el año es una convención cuando la fuente dice “2020-2021”.")
    if provincia_foco == "La Pampa":
        st.info("La Pampa: el stock de deuda 2005-2017 no es confiable (posible error de unidades, sin confirmación oficial); el compuesto no se calcula esos años.")
    st.markdown("**Razones que componen el índice (valores, no percentiles)**")
    gr = g.melt(id_vars="anio", value_vars=["deuda_sobre_ingresos", "servicio_sobre_ingresos", "rigidez_gastototal", "fragilidad_resultado"],
                var_name="variable", value_name="valor")
    gr["variable"] = gr["variable"].map({"deuda_sobre_ingresos": "Deuda/Ingresos", "servicio_sobre_ingresos": "Servicio/Ingresos",
                                         "rigidez_gastototal": "Personal/Gasto", "fragilidad_resultado": "−Resultado/Ingresos"})
    st.plotly_chart(px.line(gr, x="anio", y="valor", color="variable", markers=True).update_layout(height=340, xaxis_title="Año", yaxis_title="Razón"),
                    width="stretch")
    nc = val.nivel_comun(panel).loc[periodo[0]:periodo[1]]
    st.markdown("**Nivel común (mediana de las 24 jurisdicciones)**: lo que el score relativo no puede mostrar")
    st.plotly_chart(px.line(nc.reset_index().melt(id_vars="anio"), x="anio", y="value", color="variable", facet_col="variable", facet_col_wrap=4)
                    .update_yaxes(matches=None, showticklabels=True).update_layout(height=280, showlegend=False)
                    .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])), width="stretch")
    st.caption("Si todas las provincias mejoran o empeoran a la vez, el percentil no cambia pero estas medianas sí (por ejemplo, la mediana de Deuda/Ingresos cae de "
               f"{R['deriva_solvencia']['mediana_2005']:.2f} en 2005 a {R['deriva_solvencia']['mediana_2024']:.2f} en 2024)." if R else "")
    if provincias_comparar:
        dc = panel[panel.provincia.isin(provincias_comparar) & panel.anio.between(*periodo)]
        st.plotly_chart(px.line(dc, x="anio", y="IFFS", color="provincia", markers=True, title="Comparación del compuesto").update_layout(height=380),
                        width="stretch")


# ---------------------------------------------------------------------------
# Detalle
# ---------------------------------------------------------------------------
with t_det:
    a = st.select_slider("Año", options=ANIOS, value=max(ANIOS), key="anio_det")
    fila = panel[(panel.provincia == provincia_foco) & (panel.anio == a)]
    f = fila.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("IFFS compuesto", "s/d" if pd.isna(f["IFFS"]) else f"{f['IFFS']:.1f}")
    c2.metric("Posición (cuartil del año)", str(f["posicion"]).split(" (")[0])
    c3.metric("Deuda / Ingresos", "s/d" if pd.isna(f["deuda_sobre_ingresos"]) else f"{f['deuda_sobre_ingresos']:.2f}x")
    c4.metric("Estado del stock de deuda", {"confiable": "confiable", "no_confiable_excluido": "no confiable (excluido)", "supuesto_miles": "SUPUESTO ÷1.000", "sin_dato": "sin dato"}.get(f["stock_estado"], f["stock_estado"]))
    dc = ind.descomponer_contribucion(f)
    figd = px.bar(dc, x="subindice", y="dimension", orientation="h", range_x=[0, 100], text_auto=".0f", color="en_compuesto",
                  color_discrete_map={True: "#2a78d6", False: "#b3b0a6"},
                  title=f"Subíndices de {provincia_foco} en {a} (azul = entra al compuesto; gris = solo informativo)", labels={"subindice": "0-100", "dimension": ""})
    figd.update_layout(height=300, showlegend=False)
    st.plotly_chart(figd, width="stretch")
    st.caption("En el compuesto cada dimensión nuclear aporta S/3 (media simple). La dimensión coyuntural no aporta.")
    st.markdown("#### Análisis automático (motor de reglas: solo redacta cifras calculadas)")
    st.markdown(an.generar_analisis(panel, provincia_foco, a, eventos, info_perfiles, backtest))


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------
with t_val:
    if R is None:
        st.stop()
    st.markdown("## Validación: tres preguntas distintas")
    st.info("**Descripción** (¿qué muestra?) · **Discriminación** (¿el grupo con evento tenía un score más alto en una fecha?) · **Anticipación** (¿el score *subió* antes del evento?). "
            "Que valga la segunda no implica la tercera. Todo es exploratorio: 13 eventos, 12 en el mismo shock, año ancla convencional para 11.")
    st.markdown("### 1. Discriminación en 2019 (T−1 del ancla 2020)")
    disc = TAB["T07_discriminacion_2019"].copy()
    disc["AUC"] = disc["auc"].round(2)
    disc["IC 95%"] = disc.apply(lambda r: f"{r['ic95_inf']:.2f}–{r['ic95_sup']:.2f}", axis=1)
    disc["p permutación (unilateral)"] = disc["p_perm_unilateral"].map(lambda p: "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ","))
    st.dataframe(disc[["medida", "AUC", "IC 95%", "p permutación (unilateral)", "n_evento", "n_control"]].rename(columns={"medida": "Medida"}), hide_index=True, width="stretch")
    st.caption(f"AUC = probabilidad de que una provincia con evento tenga un score mayor que una sin evento (0,5 = azar). Con grupos de 12 vs. 12, sorteando los grupos al azar el AUC supera "
               f"{R['azar_auc_2019']['azar_p95']:.2f} solo el 5% de las veces. **El % de bonos en la deuda (variable trivial) discrimina casi igual que el índice.** "
               f"Solo entre emisoras de bonos (≥{R['condicionado_a_bonos_2019']['min_bonos_pct']:.0f}%: {R['condicionado_a_bonos_2019']['n_evento']} con evento vs. {R['condicionado_a_bonos_2019']['n_control']} sin evento: "
               f"{', '.join(R['condicionado_a_bonos_2019']['controles'])}) el AUC del compuesto es {R['condicionado_a_bonos_2019']['auc_IFFS']:.2f}, con poder casi nulo por el tamaño del grupo control.")
    st.markdown("### 2. ¿Cuándo aparece la separación? (placebo temporal)")
    pt = TAB["T09_placebo_temporal"]
    fig = go.Figure()
    p1 = pt[pt.medida == "IFFS"]
    fig.add_scatter(x=list(p1["anio"]) + list(p1["anio"])[::-1], y=list(p1["ic95_sup"]) + list(p1["ic95_inf"])[::-1], fill="toself", fillcolor="rgba(42,120,214,.15)",
                    line=dict(width=0), name="IC 95% IFFS", hoverinfo="skip")
    fig.add_scatter(x=p1["anio"], y=p1["auc"], mode="lines+markers", name="IFFS compuesto", line=dict(color="#2a78d6", width=3))
    p2 = pt[pt.medida == "pct_bonos"]
    fig.add_scatter(x=p2["anio"], y=p2["auc"], mode="lines+markers", name="% de bonos en la deuda", line=dict(color="#d03b3b", dash="dash"))
    fig.add_hline(y=0.5, line_dash="dot", line_color="grey", annotation_text="azar")
    fig.add_vrect(x0=2019.5, x1=2021.5, fillcolor="grey", opacity=0.12, line_width=0, annotation_text="eventos")
    fig.update_layout(title="AUC año por año: las mismas 12 provincias vs. el resto", height=400, xaxis_title="Año en que se mide el score", yaxis=dict(range=[0.3, 1.02], title="AUC"))
    st.plotly_chart(fig, width="stretch")
    pv = R["placebo_temporal"]
    st.caption(f"**Lectura.** El AUC ya vale {pv['auc_iffs_por_anio']['2012']:.2f} en 2012, {pv['auc_iffs_por_anio']['2015']:.2f} en 2015 y {pv['auc_iffs_por_anio']['2019']:.2f} en 2019, "
               f"y llega a {pv['auc_iffs_por_anio']['2025']:.2f} en 2025, *después* de los eventos. El grupo es persistentemente más frágil desde hace más de una década: "
               "**eso es discriminación de un rasgo persistente, no una señal de alerta que aparezca antes del shock.**")
    st.markdown("### 3. Rezagos T−1 … T−5 (y sensibilidad al año ancla)")
    rez = TAB["T08_discriminacion_por_rezago"]
    rz = rez[rez.medida.isin(["IFFS", "solvencia", "liquidez", "bonos_pct"])].copy()
    rz["AUC [IC95%]"] = rz.apply(lambda r: f"{r['auc']:.2f} [{r['ic95_inf']:.2f}–{r['ic95_sup']:.2f}]", axis=1)
    tabla_rz = rz.pivot_table(index=["ancla", "medida"], columns="rezago", values="AUC [IC95%]", aggfunc="first")
    tabla_rz.columns = [f"T−{c}" for c in tabla_rz.columns]
    st.dataframe(tabla_rz, width="stretch")
    st.caption("Anclaje 2020 = convención (fuente: “2020-2021”); anclaje alternativo = 2021 para los 11 eventos de fecha ambigua. La conclusión no cambia con el anclaje.")
    st.markdown("### 4. Anticipación: ¿el score subió antes del evento más que en el resto?")
    pre = TAB["T10_pretendencia"]
    pre = pre[pre.ancla == "principal"].copy()
    pre["Medida"] = pre["col"].map({"IFFS": "Compuesto", "S_solvencia": "Solvencia", "S_liquidez": "Liquidez", "S_estructural": "Estructural", "S_coyuntural": "Coyuntural"})
    pre["Ventana"] = pre["desde"] + " → " + pre["hasta"]
    pre["Diferencia de cambios (pts) [IC95%]"] = pre.apply(lambda r: f"{r['diferencia']:+.1f} [{r['ic95_inf']:+.1f}; {r['ic95_sup']:+.1f}]", axis=1)
    pre["p (permutación)"] = pre["p_perm_bilateral"].map(lambda p: f"{p:.2f}".replace(".", ","))
    st.dataframe(pre[["Medida", "Ventana", "cambio_medio_evento", "cambio_medio_control", "Diferencia de cambios (pts) [IC95%]", "p (permutación)"]]
                 .rename(columns={"cambio_medio_evento": "Δ con evento", "cambio_medio_control": "Δ resto"}).round(2), hide_index=True, width="stretch")
    es = TAB["T11_estudio_eventos"]
    fe = go.Figure()
    fe.add_scatter(x=es["tau"], y=es["media_evento"], name="Con evento 2020-21", line=dict(color="#d03b3b", width=3), mode="lines+markers")
    fe.add_scatter(x=es["tau"], y=es["media_control"], name="Resto", line=dict(color="#52514e", width=3), mode="lines+markers")
    fe.add_vline(x=0, line_dash="dot", line_color="grey")
    fe.update_layout(title="Estudio de eventos: IFFS medio por años respecto del ancla (τ = 0 ↔ 2020, convención)", height=380, xaxis_title="τ", yaxis_title="IFFS medio")
    st.plotly_chart(fe, width="stretch")
    ps = R["pretendencia"]["principal|S_solvencia|T-5->T-1"]
    st.caption(f"Ninguna diferencia es significativa por permutación al 5%. El IC bootstrap de solvencia T−5→T−1 apenas excluye el cero ({ps['ic95'][0]:+.1f}) pero p = {ps['p']:.2f}, y es una de 10 comparaciones sin corregir: indicio débil, no evidencia. Con una sola fecha común no hay contrafactual temporal: "
               "no se puede separar 'ya era más frágil' de 'se estaba deteriorando'.")
    st.markdown("### 5. Un caso fuera del bloque COVID: La Rioja 2025 (default del Bono Verde, feb-2025)")
    lr = TAB["T12_la_rioja_2025"].round(1)
    lr.columns = ["Año", "Solvencia", "Liquidez", "Estructural", "Coyuntural", "IFFS", "Puesto (1 = más frágil)", "Jurisdicciones con dato"]
    st.dataframe(lr, hide_index=True, width="stretch")
    st.caption("Una sola observación: sirve para mostrar la trayectoria, no para estimar aciertos. En 2024 La Rioja estaba en el puesto 7 de 24 con solvencia alta.")
    st.markdown("### 6. Pruebas que NO se corren y por qué")
    st.dataframe(TAB["T13_pruebas_no_viables"].rename(columns={"prueba": "Prueba", "por_que_no_es_viable": "Por qué no es viable"}), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Robustez
# ---------------------------------------------------------------------------
with t_rob:
    if R is None:
        st.stop()
    st.markdown("### Sensibilidad del ordenamiento a cada decisión de diseño (una a la vez)")
    sp = TAB["T05_comparacion_especificaciones"].copy()
    sp = sp.merge(TAB["T19_auc_2019_todas_las_especificaciones"][["especificacion", "auc_2019"]], on="especificacion", how="left")
    st.dataframe(sp.rename(columns={
        "especificacion": "Especificación", "n_obs": "Obs.", "spearman_vs_principal": "ρ de Spearman vs. principal", "deriva_mediana_anual": "Deriva de la mediana anual (pts)",
        "persistencia_rangos": "Persistencia de rangos", "cambio_rango_medio": "Cambio medio de puesto", "obs_con_cambio_rango_ge5": "Obs. con cambio ≥ 5 puestos",
        "solapamiento_top6_2024": "Top-6 de 2024 en común", "auc_2019": "AUC 2019 (info. no usada para elegir)"}).round(3), hide_index=True, width="stretch")
    a1 = R["auc_2019_especificaciones"]
    st.caption(f"Las {a1['n']} especificaciones se reportan completas; ninguna se eligió por su AUC (rango {a1['min']:.2f}–{a1['max']:.2f}). "
               "La normalización principal se eligió por estacionariedad, ausencia de información futura y robustez a outliers (ver docs/METODOLOGIA.md).")
    st.plotly_chart(px.bar(sp.iloc[1:].sort_values("spearman_vs_principal"), x="spearman_vs_principal", y="especificacion", orientation="h",
                           range_x=[0.8, 1.0], labels={"spearman_vs_principal": "ρ de Spearman vs. principal", "especificacion": ""})
                    .update_layout(height=520), width="stretch")
    w = R["pesos_aleatorios"]
    st.markdown(f"**Pesos aleatorios ({w['n_sim']:,} sorteos uniformes sobre las 3 dimensiones):** correlación media con el ordenamiento de pesos iguales, mediana {w['spearman_medio_mediana']:.2f} "
                f"(percentil 5: {w['spearman_medio_p05']:.2f}); el {100 * w['cuartil_superior_estable_ge80']:.0f}% de las observaciones del cuartil superior se mantiene ahí en ≥80% de los sorteos.")
    st.markdown("### La Pampa: efecto del tratamiento del stock 2005-2017 sobre las demás provincias")
    slp = TAB["T17_sensibilidad_la_pampa"].round(2)
    st.dataframe(slp.rename(columns={"normalizacion": "Normalización", "tratamiento_vs_excluir": "Tratamiento (vs. excluir)",
                                     "cambio_medio_abs_subindice_solvencia_resto": "Cambio medio del subíndice de solvencia del resto (pts)",
                                     "cambio_max_abs": "Máx.", "spearman_solvencia_resto": "ρ de Spearman"}), hide_index=True, width="stretch")
    lp = R["la_pampa"]
    ef = lp["efecto_sobre_resto"]
    st.caption(f"Con percentiles (corte anual), un error de unidades como el de La Pampa sin tratar mueve el subíndice de solvencia del resto {ef['corte_anual|sin_tratar']:.1f} puntos en promedio; "
               f"con z-score de base fija, {ef['zscore_base|sin_tratar']:.1f}. Ésa es una de las razones para no usar z-scores.")
    st.markdown(f"Escenario ÷1.000 (**supuesto**): el IFFS de La Pampa 2005-2017 quedaría entre {lp['escenario_supuesto_miles']['iffs_la_pampa_min']} y {lp['escenario_supuesto_miles']['iffs_la_pampa_max']}, "
                f"y el AUC año por año cambia como máximo {lp['escenario_supuesto_miles']['dif_max_auc_placebo_2005_2017']:.2f}.")


# ---------------------------------------------------------------------------
# Perfiles y cambios
# ---------------------------------------------------------------------------
with t_perf:
    st.subheader("Perfiles fiscales (k-means, descriptivo)")
    st.caption(f"k={info_perfiles['k']} por silhouette ({info_perfiles['silhouette']:.2f}: estructura moderada), ajustado con 2005-2012 sobre los 4 subíndices. Sin perfil si falta algún subíndice.")
    a = st.select_slider("Año", options=ANIOS, value=max(ANIOS), key="anio_perf")
    cent = info_perfiles["centroides"].copy()
    cent.index = [info_perfiles["nombres"][i] for i in cent.index]
    st.plotly_chart(px.imshow(cent.round(0), color_continuous_scale="YlOrRd", zmin=0, zmax=100, text_auto=True, aspect="auto",
                              title="Centroides: subíndice típico de cada perfil"), width="stretch")
    tp = panel[panel.anio == a][["provincia", "perfil_nombre", "IFFS", "posicion"]].sort_values(["perfil_nombre", "IFFS"], ascending=[True, False]).round(1)
    st.dataframe(tp.rename(columns={"provincia": "Provincia", "perfil_nombre": "Perfil", "posicion": "Posición"}), hide_index=True, width="stretch")
    st.subheader("Cambios de posición")
    a2 = st.select_slider("Año a analizar", options=ANIOS[1:], value=max(ANIOS), key="anio_cambio")
    cam = an.ranking_cambios(panel, a2)
    sube = cam[cam["cambio_cuartil"] == "Subió de cuartil"]
    if len(sube):
        st.error("**Subieron de cuartil:** " + "; ".join(f"{r.provincia} ({r.d_iffs_1a:+.1f} pts, motor: {r.dimension_motor})" for r in sube.itertuples()))
    n_emp = int((cam["d_iffs_1a"] > 0).sum())
    if n_emp >= 0.75 * len(cam):
        st.warning(f"{n_emp} de {len(cam)} jurisdicciones suben a la vez: el compuesto es relativo, así que un movimiento común apunta a un cambio en la escala del cuartil más que en cada provincia.")
    st.dataframe(cam.round(1).rename(columns={"provincia": "Provincia", "d_iffs_1a": "Δ 1 año", "d_iffs_3a": "Δ 3 años", "posicion_anterior": "Posición anterior",
                                              "posicion": "Posición", "cambio_cuartil": "Cambio", "dimension_motor": "Subíndice que más se movió"}),
                 hide_index=True, width="stretch")
    st.subheader("Observaciones inusuales (Isolation Forest, descriptivo)")
    an_df = panel[panel["anomalia"]].sort_values("anomalia_score", ascending=False)[["provincia", "anio", "IFFS", "d_iffs_1a", "anomalia_score"]].round(2)
    st.dataframe(an_df, hide_index=True, width="stretch")
    if backtest.get("n"):
        st.caption(f"Extrapolar tendencias no mejora a repetir el último valor (backtest a 1 año, {backtest['n']} casos: error medio {backtest['mae_tendencia']:.1f} vs. {backtest['mae_ingenua']:.1f} puntos). "
                   "Por eso la proyección se muestra solo en el análisis automático, con esa advertencia.")


# ---------------------------------------------------------------------------
# Simulador
# ---------------------------------------------------------------------------
with t_sim:
    st.subheader(f"Simulador de escenarios — {provincia_foco}")
    st.caption("Sensibilidad **mecánica** (no pronóstico): recalcula los subíndices con el shock y compara contra las demás jurisdicciones del mismo año, que quedan fijas.")
    a = st.select_slider("Año base", options=ANIOS, value=max(ANIOS), key="anio_sim")
    s1, s2, s3 = st.columns(3)
    caida = s1.slider("Caída de ingresos corrientes (%)", 0, 40, 10)
    deprec = s2.slider("Depreciación del peso (%)", 0, 200, 0, help="Solo afecta a provincias con dato de composición por moneda (11 de 24; último dato 2014-2020).")
    serv = s3.slider("Suba del servicio de deuda (%)", 0, 100, 0, help="Cota superior: se aplica al servicio total (capital + interés).")
    r = an.simular_escenario(panel, provincia_foco, a, caida / 100, deprec / 100, serv / 100)
    if "error" in r or pd.isna(r.get("iffs_base")):
        st.warning("No hay datos suficientes para simular esta provincia y año (falta algún subíndice del compuesto).")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("IFFS base", f"{r['iffs_base']:.1f}", str(r["posicion_base"]).split(" (")[0], delta_color="off")
        m2.metric("IFFS escenario", f"{r['iffs_escenario']:.1f}", f"{r['delta']:+.1f} pts", delta_color="inverse")
        m3.metric("Posición", str(r["posicion_escenario"]).split(" (")[0], "sin cambio" if r["posicion_escenario"] == r["posicion_base"] else f"desde {str(r['posicion_base']).split(' (')[0]}", delta_color="off")
        dims = pd.DataFrame({"Subíndice": list(an.NOMBRE_DIM.values()), "Base": list(r["subindices_base"].values()), "Escenario": list(r["subindices_escenario"].values())}) \
            .melt(id_vars="Subíndice", var_name="Situación", value_name="0-100")
        st.plotly_chart(px.bar(dims, x="0-100", y="Subíndice", color="Situación", barmode="group", orientation="h", range_x=[0, 100], text_auto=".0f"), width="stretch")
        st.write(f"Deuda/Ingresos: **{r['ratios']['deuda_sobre_ingresos'][0]:.2f}x → {r['ratios']['deuda_sobre_ingresos'][1]:.2f}x**")
        if r["fx_solicitado_sin_dato"]:
            st.warning("Esta provincia no tiene composición de deuda por moneda: la depreciación no se aplicó.")
        elif r["fx_aplicado"]:
            st.info(f"Depreciación aplicada con {r['fx_share'] * 100:.0f}% de deuda en moneda extranjera (último dato: {r['fx_anio']}).")


# ---------------------------------------------------------------------------
# Datos y cobertura
# ---------------------------------------------------------------------------
with t_dat:
    st.subheader("Cobertura: qué hay y qué falta")
    if R:
        c = R["cobertura"]
        st.markdown(f"Compuesto calculado en **{c['n_compuesto']} de {c['n_total']}** provincia-año ({c['pct_compuesto']}%). Faltantes: "
                    + "; ".join(f"**{k}** {v}" for k, v in c["faltantes_compuesto"].items()) + ". Ninguno se imputa.")
    ok = panel.pivot(index="provincia", columns="anio", values="IFFS").notna().astype(int)
    st.plotly_chart(px.imshow(ok, color_continuous_scale=[[0, "#c3c2b7"], [1, "#2a78d6"]], aspect="auto", zmin=0, zmax=1,
                              title="Compuesto disponible (azul) vs. faltante (gris)").update_layout(coloraxis_showscale=False, height=520), width="stretch")
    if "T01_cobertura_por_provincia" in TAB:
        st.dataframe(TAB["T01_cobertura_por_provincia"].rename(columns={"provincia": "Provincia"}), hide_index=True, width="stretch")
    st.subheader("Tabla filtrada")
    cols = ["provincia", "anio", "deuda_sobre_ingresos", "servicio_sobre_ingresos", "rigidez_gastototal", "fragilidad_resultado",
            "S_solvencia", "S_liquidez", "S_estructural", "S_coyuntural", "IFFS", "posicion", "stock_estado"]
    tabla = panel_f[cols].sort_values(["provincia", "anio"]).round(3)
    st.dataframe(tabla, hide_index=True, width="stretch")
    st.download_button("⬇️ Descargar tabla filtrada (CSV)", tabla.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"iffs_v2_{periodo[0]}_{periodo[1]}.csv", mime="text/csv")
    st.subheader("Limitaciones")
    st.markdown(
        "- **La Pampa 2005-2017**: stock de deuda no confiable, sin confirmación oficial (faltante en el principal; sensibilidad ÷1.000 = supuesto).\n"
        "- **Score relativo**: si todas mejoran o empeoran a la vez, no se ve (mirar el nivel común). No es probabilidad de default.\n"
        "- **13 eventos**, 12 en el mismo shock; 11 con año ancla convencional. No hay validación predictiva.\n"
        "- **Stock/flujo**: Deuda a 31/12 sobre ingresos del año sobrestima la razón con inflación alta.\n"
        "- **Faltan**: deuda flotante, garantías, vencimientos futuros, moneda de la deuda (solo 11 de 24 provincias).\n"
        "- **Saltos de servicio** (Santa Cruz 2024-25, San Luis) resueltos con la identidad stock-flujo: son amortizaciones reales. Solo La Pampa 2024 (stock = 500) no es verificable y no cambia ningún puesto.\n"
        "- Los paneles base son derivados de las fuentes de la DNAP y el BCRA; no se reprocesan aquí.")
    st.caption(f"Metodología: docs/METODOLOGIA.md · Notebook: notebooks/fragilidad_fiscal_provincial.ipynb · Huella del código y datos: {HUELLA[:10]}")
