"""
Monitor de Fragilidad Fiscal Provincial — Dashboard (Streamlit)
==================================================================

Prototipo funcional para el Segundo Parcial de Programación para Economistas.

Ejecutar con:
    streamlit run app.py

Este dashboard es una herramienta de apoyo al análisis, no un predictor de
default: los umbrales de alerta son una convención metodológica (ver
src/indicators.py) y el propio notebook (notebooks/fragilidad_fiscal_provincial.ipynb)
documenta que el índice no tiene, con la evidencia disponible, poder de
anticipación confiable fuera de muestra.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import data_processing as dp
from src import indicators as ind
from src import modeling as mdl
from src import analytics as an

st.set_page_config(
    page_title="Monitor de Fragilidad Fiscal Provincial",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Carga de datos (cacheada)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Cargando y procesando datos fiscales provinciales...")
def cargar_datos():
    panel = dp.construir_panel_maestro()
    avisos = dp.validar_panel_maestro(panel)
    panel_iffs, meta = ind.construir_iffs(panel)
    panel_iffs, info_perfiles = an.construir_panel_analitico(panel_iffs)  # + nivel de alerta, perfiles, cambios, anomalías
    eventos = dp.load_eventos_estres()
    panel_target = mdl.construir_target_estres(panel_iffs, eventos)
    backtest = an.backtest_proyeccion(panel_iffs)
    return panel_iffs, meta, avisos, eventos, panel_target, info_perfiles, backtest


panel_iffs, meta_iffs, avisos_validacion, eventos, panel_target, info_perfiles, backtest_proy = cargar_datos()

PROVINCIAS = sorted(panel_iffs["provincia"].unique())
ANIOS = sorted(panel_iffs["anio"].unique())


ETIQUETAS = {
    "provincia": "Provincia", "anio": "Año", "IFFS_A": "IFFS-A", "IFFS_B": "IFFS-B", "IFFS_C": "IFFS-C",
    "nivel_alerta": "Nivel de alerta", "perfil_nombre": "Perfil", "d_iffs_1a": "Δ IFFS-B (1 año)",
    "d_iffs_3a": "Δ IFFS-B (3 años)", "dimension_motor": "Dimensión motor", "anomalia_score": "Puntaje de rareza",
    "deuda_sobre_ingresos": "Deuda / Ingresos", "servicio_sobre_ingresos": "Servicio / Ingresos",
    "rigidez_gastototal": "Rigidez del gasto", "fragilidad_resultado": "Fragilidad del resultado",
    "nivel_anterior": "Nivel anterior",
}
DECIMALES = {
    "IFFS_A": 0, "IFFS_B": 0, "IFFS_C": 0, "d_iffs_1a": 0, "d_iffs_3a": 0, "anomalia_score": 2,
    "deuda_sobre_ingresos": 2, "servicio_sobre_ingresos": 3, "rigidez_gastototal": 2, "fragilidad_resultado": 3,
}


def vista(df: pd.DataFrame) -> pd.DataFrame:
    """Versión para mostrar: nombres legibles (Año, Provincia...) y sin decimales innecesarios."""
    df = df.copy()
    for col, dec in DECIMALES.items():
        if col in df.columns:
            df[col] = df[col].round(dec)
    return df.rename(columns=ETIQUETAS)


GLOSARIO = """
**IFFS = Índice de Fragilidad Fiscal Subnacional.** Va de 0 a 100 y compara a cada provincia-año con la
historia de referencia 2005-2012 (percentil): **más alto = más frágil = menos margen para absorber un shock.**

| Versión | Qué combina | Para qué sirve |
|---|---|---|
| **IFFS-A** | Solvencia (Deuda / Ingresos) + Liquidez (Servicio de deuda / Ingresos), con igual peso | Versión simple y parsimoniosa. Exige tener las 2 dimensiones. |
| **IFFS-B** | Las 4 dimensiones con igual peso: solvencia, liquidez, **rigidez del gasto** (Personal / Gasto total) y **desempeño fiscal** (−Resultado / Ingresos) | **Índice principal**: lo usan el ranking, las alertas y los perfiles. Si falta una dimensión (CABA, sin servicio de deuda confiable) queda como *provisional*. |
| **IFFS-C** | Primer componente principal (PCA) de las mismas 4 dimensiones; los pesos se estiman solo con datos 2005-2012 | Chequeo de robustez: pesos que salen de los datos en vez de fijarse a mano. |

**Niveles de alerta (sobre IFFS-B):** Bajo < 33 · Medio 33-66 · Alto 66-85 · Crítico ≥ 85. Son cortes de lectura, no una calibración estadística.
"""


# ---------------------------------------------------------------------------
# Sidebar — controles
# ---------------------------------------------------------------------------

st.sidebar.title("📊 Monitor de Fragilidad Fiscal")
st.sidebar.caption("Sistema de evaluación y alerta temprana — provincias argentinas")

provincia_foco = st.sidebar.selectbox("Provincia a analizar", PROVINCIAS, index=PROVINCIAS.index("La Rioja") if "La Rioja" in PROVINCIAS else 0)
periodo = st.sidebar.slider("Período", min_value=min(ANIOS), max_value=max(ANIOS), value=(min(ANIOS), max(ANIOS)))
provincias_comparar = st.sidebar.multiselect(
    "Provincias a comparar",
    PROVINCIAS,
    default=[p for p in ["La Rioja", "Buenos Aires", "CABA", "Santiago del Estero"] if p in PROVINCIAS],
)

with st.sidebar.expander("ℹ️ Sobre el índice y sus límites"):
    st.markdown(
        "El **IFFS-B** combina, con igual ponderación, cuatro dimensiones: "
        "solvencia (deuda/ingresos), liquidez (servicio/ingresos), rigidez del "
        "gasto y desempeño fiscal, normalizadas por percentil sobre una ventana "
        "base fija (2005-2012). Los umbrales de alerta (33/66/85) son una "
        "convención expositiva, **no** una calibración estadística contra "
        "episodios reales de default. CABA no tiene servicio de deuda "
        "confiable (excluida de ese componente). Ver el notebook para el "
        "detalle metodológico completo y sus limitaciones."
    )

if avisos_validacion:
    with st.sidebar.expander("⚠️ Avisos de validación de datos"):
        for a in avisos_validacion:
            st.write("-", a)


panel_filtrado = panel_iffs[panel_iffs["anio"].between(*periodo)]

st.title("Monitor de Fragilidad Fiscal Provincial")
st.caption("Es solo **UNA** herramienta de apoyo al análisis económico, no un mecanismo de predicción de default.")
with st.expander("ℹ️ ¿Qué significan IFFS-A, IFFS-B e IFFS-C?"):
    st.markdown(GLOSARIO)

(tab_ranking, tab_evolucion, tab_comparacion, tab_detalle,
 tab_perfiles, tab_alertas, tab_simulador, tab_datos) = st.tabs(
    ["🏆 Ranking / mapa", "📈 Evolución histórica", "⚖️ Comparación", "🔎 Detalle y análisis",
     "🧩 Perfiles (ML)", "🚨 Alertas y cambios", "🎛️ Simulador", "📥 Datos"]
)

# ---------------------------------------------------------------------------
# Tab 1: Ranking / mapa de calor
# ---------------------------------------------------------------------------
with tab_ranking:
    col1, col2 = st.columns([1, 2])
    with col1:
        anio_ranking = st.selectbox("Año del ranking", sorted(ANIOS, reverse=True), key="anio_ranking")
    df_ranking = panel_iffs[panel_iffs["anio"] == anio_ranking].dropna(subset=["IFFS_B"]).sort_values("IFFS_B", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            df_ranking, x="IFFS_B", y="provincia", orientation="h",
            color="nivel_alerta",
            color_discrete_map={"Bajo": "#2E7D32", "Medio": "#F9A825", "Alto": "#EF6C00", "Crítico": "#C62828"},
            category_orders={"provincia": df_ranking["provincia"].tolist()[::-1]},
            title=f"Ranking de fragilidad fiscal (IFFS-B) — {anio_ranking}",
            labels={"IFFS_B": "IFFS-B (0-100)", "provincia": "Provincia", "nivel_alerta": "Nivel de alerta"},
            hover_data={"IFFS_B": ":.0f"},
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, width="stretch")
    with c2:
        pivot = panel_filtrado.pivot_table(index="provincia", columns="anio", values="IFFS_B")
        fig_heat = px.imshow(
            pivot, color_continuous_scale="RdYlGn_r", aspect="auto",
            title="Mapa de calor — IFFS-B por provincia y año",
            labels={"x": "Año", "y": "Provincia", "color": "IFFS-B"},
        )
        fig_heat.update_traces(hovertemplate="%{y}<br>Año %{x}<br>IFFS-B: %{z:.0f}<extra></extra>")
        fig_heat.update_layout(height=650)
        st.plotly_chart(fig_heat, width="stretch")

    st.caption(
        "Fuente: DNAP (ejecución presupuestaria, deuda y servicio de deuda), BCRA "
        "(capacidad de respuesta). Elaboración propia sobre paneles auditados. "
        f"Período: {min(ANIOS)}-{max(ANIOS)}."
    )

# ---------------------------------------------------------------------------
# Tab 2: Evolución histórica
# ---------------------------------------------------------------------------
with tab_evolucion:
    st.subheader(f"Evolución histórica — {provincia_foco}")
    df_prov = panel_iffs[(panel_iffs.provincia == provincia_foco) & (panel_iffs.anio.between(*periodo))]

    fig_evol = go.Figure()
    for col, nombre in [("IFFS_A", "IFFS-A (deuda + servicio)"), ("IFFS_B", "IFFS-B (4 dimensiones, principal)"),
                        ("IFFS_C", "IFFS-C (componente principal / PCA)")]:
        fig_evol.add_trace(go.Scatter(x=df_prov["anio"], y=df_prov[col], mode="lines+markers", name=nombre,
                                      hovertemplate="Año %{x}: %{y:.0f}<extra>%{fullData.name}</extra>"))
    fig_evol.add_vrect(x0=2020, x1=2021, fillcolor="grey", opacity=0.15, line_width=0, annotation_text="Shock COVID-19")
    fig_evol.update_layout(
        title=f"Evolución de las tres especificaciones del IFFS — {provincia_foco}",
        xaxis_title="Año", yaxis_title="IFFS (0-100)", height=420,
    )
    st.plotly_chart(fig_evol, width="stretch")

    st.markdown("**Ratios que componen el índice:**")
    fig_ratios = go.Figure()
    for col, nombre in [
        ("deuda_sobre_ingresos", "Deuda / Ingresos"),
        ("servicio_sobre_ingresos", "Servicio deuda / Ingresos"),
        ("rigidez_gastototal", "Personal / Gasto total"),
    ]:
        fig_ratios.add_trace(go.Scatter(x=df_prov["anio"], y=df_prov[col], mode="lines+markers", name=nombre,
                                        hovertemplate="Año %{x}: %{y:.2f}<extra>%{fullData.name}</extra>"))
    fig_ratios.update_layout(xaxis_title="Año", yaxis_title="Ratio (veces)", height=380)
    st.plotly_chart(fig_ratios, width="stretch")

    eventos_prov = eventos[eventos.provincia == provincia_foco]
    if not eventos_prov.empty:
        st.warning(
            f"**{provincia_foco}** registra {len(eventos_prov)} evento(s) de estrés fiscal en la base: "
            + "; ".join(f"{r.anio_evento} ({r.tipo_evento})" for r in eventos_prov.itertuples())
        )
    else:
        st.info(f"**{provincia_foco}** no registra eventos de default/reestructuración en la base 2005-2025.")

# ---------------------------------------------------------------------------
# Tab 3: Comparación
# ---------------------------------------------------------------------------
with tab_comparacion:
    st.subheader("Comparación entre provincias")
    if not provincias_comparar:
        st.info("Elegí una o más provincias en la barra lateral para comparar.")
    else:
        df_comp = panel_iffs[
            (panel_iffs.provincia.isin(provincias_comparar)) & (panel_iffs.anio.between(*periodo))
        ]
        fig_comp = px.line(
            df_comp, x="anio", y="IFFS_B", color="provincia", markers=True,
            title="IFFS-B — comparación entre provincias seleccionadas",
            labels={"IFFS_B": "IFFS-B (0-100)", "anio": "Año", "provincia": "Provincia"},
            hover_data={"IFFS_B": ":.0f"},
        )
        fig_comp.add_vrect(x0=2020, x1=2021, fillcolor="grey", opacity=0.15, line_width=0)
        fig_comp.update_layout(height=450)
        st.plotly_chart(fig_comp, width="stretch")

        anio_snapshot = st.select_slider("Año del corte transversal", options=sorted(ANIOS), value=max(ANIOS))
        snapshot = df_comp[df_comp.anio == anio_snapshot][
            ["provincia", "IFFS_B", "nivel_alerta", "deuda_sobre_ingresos", "servicio_sobre_ingresos"]
        ].sort_values("IFFS_B", ascending=False)
        st.dataframe(vista(snapshot), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Tab 4: Detalle y explicación automática
# ---------------------------------------------------------------------------
with tab_detalle:
    st.subheader(f"Detalle — {provincia_foco}")
    anio_detalle = st.select_slider("Año", options=sorted(ANIOS), value=max(ANIOS), key="anio_detalle")

    fila = panel_iffs[(panel_iffs.provincia == provincia_foco) & (panel_iffs.anio == anio_detalle)]
    if fila.empty or pd.isna(fila["IFFS_B"].iloc[0]):
        st.warning("No hay datos suficientes para esta combinación de provincia y año.")
    else:
        fila = fila.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("IFFS-B", f"{fila['IFFS_B']:.1f} / 100")
        col2.metric("Nivel de alerta", fila["nivel_alerta"])
        col3.metric("Deuda / Ingresos", f"{fila['deuda_sobre_ingresos']:.2f}x")

        descomp = ind.descomponer_contribucion(fila)
        fig_desc = px.bar(
            descomp, x="percentil", y="dimension", orientation="h",
            range_x=[0, 100],
            title=f"Descomposición del IFFS-B — {provincia_foco} ({anio_detalle})",
            labels={"percentil": "Percentil (0-100, vs. ventana base 2005-2012)", "dimension": "Dimensión"},
        )
        fig_desc.update_traces(hovertemplate="%{y}: percentil %{x:.0f}<extra></extra>")
        st.plotly_chart(fig_desc, width="stretch")

        st.markdown("### 🗣️ Análisis exhaustivo automático")
        st.caption("Generado por reglas transparentes sobre resultados de ML (k-means, Isolation Forest, tendencia). "
                   "No usa IA generativa ni APIs: solo redacta cifras efectivamente calculadas.")
        st.markdown(an.generar_analisis(panel_iffs, provincia_foco, anio_detalle, eventos, info_perfiles, backtest_proy))

        proy = an.proyectar_iffs(panel_iffs[panel_iffs["anio"] <= anio_detalle], provincia_foco)
        if not proy.empty:
            hist = panel_iffs[(panel_iffs.provincia == provincia_foco) & (panel_iffs.anio <= anio_detalle)]
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=hist["anio"], y=hist["IFFS_B"], mode="lines+markers", name="IFFS-B observado"))
            fig_p.add_trace(go.Scatter(x=proy["anio"], y=proy["proyeccion"], mode="lines+markers", name="Proyección (tendencia)",
                                       line=dict(dash="dash")))
            fig_p.add_trace(go.Scatter(x=list(proy["anio"]) + list(proy["anio"])[::-1],
                                       y=list(proy["superior"]) + list(proy["inferior"])[::-1],
                                       fill="toself", fillcolor="rgba(120,120,120,0.2)", line=dict(width=0), name="Banda ~80%"))
            fig_p.update_layout(title="Proyección exploratoria a 2 años (no es un pronóstico validado)",
                                xaxis_title="Año", yaxis_title="IFFS-B (0-100)", height=380)
            fig_p.update_traces(hovertemplate="Año %{x}: %{y:.0f}<extra>%{fullData.name}</extra>")
            st.plotly_chart(fig_p, width="stretch")

# ---------------------------------------------------------------------------
# Tab 5: Perfiles (k-means) y anomalías
# ---------------------------------------------------------------------------
with tab_perfiles:
    st.subheader("Perfiles fiscales (k-means)")
    st.caption(
        f"Agrupa observaciones según *qué dimensiones* pesan, no solo su nivel. k={info_perfiles['k']} elegido por silhouette "
        f"({info_perfiles['silhouette']:.2f}, estructura moderada) y ajustado solo con la ventana 2005-2012, para no usar "
        "información futura. CABA queda sin perfil (falta servicio de deuda confiable; no se imputan datos)."
    )
    anio_perfil = st.select_slider("Año", options=sorted(ANIOS), value=max(ANIOS), key="anio_perfil")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        cent = info_perfiles["centroides"].copy()
        cent.index = [info_perfiles["nombres"][i] for i in cent.index]
        fig_c = px.imshow(cent.round(0), color_continuous_scale="RdYlGn_r", zmin=0, zmax=100, aspect="auto", text_auto=True,
                          title="Centroides: percentil típico de cada perfil por dimensión")
        st.plotly_chart(fig_c, width="stretch")
    with col_b:
        tabla_perfil = panel_iffs[panel_iffs["anio"] == anio_perfil][["provincia", "perfil_nombre", "IFFS_B", "nivel_alerta"]]\
            .sort_values(["perfil_nombre", "IFFS_B"], ascending=[True, False])
        st.dataframe(vista(tabla_perfil), width="stretch", hide_index=True, height=420)

    st.markdown("**Observaciones inusuales (Isolation Forest, descriptivo):**")
    anom = panel_iffs[panel_iffs["anomalia"]].sort_values("anomalia_score", ascending=False)[
        ["provincia", "anio", "IFFS_B", "d_iffs_1a", "nivel_alerta", "anomalia_score"]]
    st.dataframe(vista(anom), width="stretch", hide_index=True)
    st.caption("Marca el ~4% de observaciones con combinaciones o saltos poco frecuentes. Es una señal para revisar, no un veredicto.")

# ---------------------------------------------------------------------------
# Tab 6: Alertas y cambios
# ---------------------------------------------------------------------------
with tab_alertas:
    st.subheader("Alertas tempranas: qué cambió")
    anio_alerta = st.select_slider("Año a analizar", options=sorted(ANIOS)[1:], value=max(ANIOS), key="anio_alerta")
    cambios = an.ranking_cambios(panel_iffs, anio_alerta)
    empeoraron = cambios[cambios["cambio_nivel"] == "Empeoró"]
    if not empeoraron.empty:
        st.error("**Subieron de nivel de alerta:** " + "; ".join(
            f"{r.provincia} ({r.nivel_anterior} → {r.nivel_alerta}, {r.d_iffs_1a:+.1f} pts, motor: {r.dimension_motor})"
            for r in empeoraron.itertuples()))
    else:
        st.success("Ninguna jurisdicción subió de nivel de alerta este año.")
    n_emp = int((cambios["d_iffs_1a"] > 0).sum())
    med = lambda y: panel_iffs.loc[panel_iffs["anio"] == y, "D_resultado_fiscal_pct_ingresos"].median() * 100
    if n_emp >= 0.75 * len(cambios):
        st.warning(
            f"**Ojo con la lectura:** {n_emp} de {len(cambios)} jurisdicciones empeoran a la vez, lo que apunta a un movimiento común "
            f"y no a {n_emp} problemas individuales. La mediana de resultado/ingresos pasó de {med(anio_alerta - 1):+.1f}% a "
            f"{med(anio_alerta):+.1f}%: buena parte del cambio viene del resultado fiscal (un año previo atípico) y no de la deuda. "
            "Los datos del último año pueden ser preliminares."
        )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Mayor deterioro (Δ IFFS-B, 1 año)**")
        st.dataframe(vista(cambios.head(6)[["provincia", "IFFS_B", "d_iffs_1a", "nivel_alerta", "dimension_motor"]]),
                     width="stretch", hide_index=True)
    with c2:
        st.markdown("**Mayor mejora**")
        st.dataframe(vista(cambios.tail(6).iloc[::-1][["provincia", "IFFS_B", "d_iffs_1a", "nivel_alerta", "dimension_motor"]]),
                     width="stretch", hide_index=True)
    d_ch = cambios.sort_values("d_iffs_1a").copy()
    d_ch["etiqueta"] = d_ch["d_iffs_1a"].map(lambda v: "0" if round(v) == 0 else f"{v:+.0f}")  # evita "-0" / "+0"
    fig_ch = px.bar(d_ch, x="d_iffs_1a", y="provincia", orientation="h", text="etiqueta", custom_data=["etiqueta"],
                    color="d_iffs_1a", color_continuous_scale="RdYlGn_r", color_continuous_midpoint=0,
                    title=f"Variación del IFFS-B respecto de {anio_alerta - 1}",
                    labels={"d_iffs_1a": "Δ puntos", "provincia": "Provincia"})
    fig_ch.update_traces(hovertemplate="%{y}: %{customdata[0]} puntos<extra></extra>", textposition="outside")
    fig_ch.update_layout(height=620, coloraxis_showscale=False)
    st.plotly_chart(fig_ch, width="stretch")
    st.caption("Δ = IFFS-B(t) − IFFS-B(t−1). \"Motor\" = la dimensión cuyo percentil más se movió. "
               "Variaciones grandes pueden deberse a cambios en un solo componente (p. ej. el resultado fiscal).")

# ---------------------------------------------------------------------------
# Tab 7: Simulador de escenarios
# ---------------------------------------------------------------------------
with tab_simulador:
    st.subheader(f"Simulador de escenarios — {provincia_foco}")
    st.caption("Sensibilidad **mecánica** del IFFS-B ante shocks sobre el año elegido. No es un pronóstico: "
               "mantiene constante el gasto y compara contra la misma escala base 2005-2012.")
    anio_sim = st.select_slider("Año base", options=sorted(ANIOS), value=max(ANIOS), key="anio_sim")
    s1, s2, s3 = st.columns(3)
    caida = s1.slider("Caída de ingresos corrientes (%)", 0, 40, 10)
    deprec = s2.slider("Depreciación del peso (%)", 0, 200, 0, help="Solo afecta a provincias con dato de composición por moneda (último dato 2014-2020).")
    serv = s3.slider("Suba del servicio de deuda (%)", 0, 100, 0, help="Cota superior: se aplica al servicio total (capital + interés).")
    r = an.simular_escenario(panel_iffs, provincia_foco, anio_sim, caida / 100, deprec / 100, serv / 100)
    if "error" in r or pd.isna(r.get("iffs_base")):
        st.warning("No hay datos suficientes para simular esta provincia y año.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("IFFS-B base", f"{r['iffs_base']:.1f}", r["nivel_base"])
        m2.metric("IFFS-B escenario", f"{r['iffs_escenario']:.1f}", f"{r['delta']:+.1f} pts", delta_color="inverse")
        m3.metric("Nivel de alerta", r["nivel_escenario"], "sin cambio" if r["nivel_escenario"] == r["nivel_base"] else f"desde {r['nivel_base']}", delta_color="off")
        dims = pd.DataFrame({
            "Dimensión": [NOMBRE for NOMBRE in an.NOMBRE_DIM.values()],
            "Base": list(r["percentiles_base"].values()),
            "Escenario": list(r["percentiles_escenario"].values()),
        }).melt(id_vars="Dimensión", var_name="Situación", value_name="Percentil")
        fig_s = px.bar(dims, x="Percentil", y="Dimensión", color="Situación", barmode="group", orientation="h", range_x=[0, 100],
                       title="Percentil por dimensión: base vs. escenario", text_auto=".0f")
        fig_s.update_traces(hovertemplate="%{y} (%{fullData.name}): percentil %{x:.0f}<extra></extra>")
        st.plotly_chart(fig_s, width="stretch")
        rat = r["ratios"]
        st.write(f"Deuda/Ingresos: **{rat['deuda_sobre_ingresos'][0]:.2f}x → {rat['deuda_sobre_ingresos'][1]:.2f}x**")
        if r["fx_solicitado_sin_dato"]:
            st.warning("Esta provincia no tiene composición de deuda por moneda en la base: la depreciación no se aplicó.")
        elif r["fx_aplicado"]:
            st.info(f"Depreciación aplicada con {r['fx_share']*100:.0f}% de deuda en moneda extranjera (último dato: {r['fx_anio']}).")

# ---------------------------------------------------------------------------
# Tab 8: Datos y descarga
# ---------------------------------------------------------------------------
with tab_datos:
    st.subheader("Tabla filtrada")
    columnas_mostrar = [
        "provincia", "anio", "deuda_sobre_ingresos", "servicio_sobre_ingresos",
        "rigidez_gastototal", "fragilidad_resultado", "IFFS_A", "IFFS_B", "IFFS_C", "nivel_alerta",
    ]
    tabla = panel_filtrado[columnas_mostrar].sort_values(["provincia", "anio"])
    st.dataframe(vista(tabla), width="stretch", hide_index=True)
    st.download_button(
        "⬇️ Descargar tabla filtrada (CSV)",
        data=vista(tabla).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"iffs_filtrado_{periodo[0]}_{periodo[1]}.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.caption(
        "Metodología completa, validación y limitaciones documentadas en "
        "`notebooks/fragilidad_fiscal_provincial.ipynb` e `informe/informe_fragilidad_fiscal.pdf`. "
        f"Varianza explicada del PCA (IFFS-C): {meta_iffs['varianza_explicada_pc1']*100:.1f}%."
    )
