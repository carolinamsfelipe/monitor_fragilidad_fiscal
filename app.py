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
    panel_iffs["nivel_alerta"] = ind.asignar_nivel_alerta(panel_iffs["IFFS_B"])
    eventos = dp.load_eventos_estres()
    panel_target = mdl.construir_target_estres(panel_iffs, eventos)
    return panel_iffs, meta, avisos, eventos, panel_target


panel_iffs, meta_iffs, avisos_validacion, eventos, panel_target = cargar_datos()

PROVINCIAS = sorted(panel_iffs["provincia"].unique())
ANIOS = sorted(panel_iffs["anio"].unique())


def explicar_alerta(provincia: str, anio: int) -> str:
    fila = panel_iffs[(panel_iffs.provincia == provincia) & (panel_iffs.anio == anio)]
    if fila.empty or pd.isna(fila["IFFS_B"].iloc[0]):
        return f"No hay datos suficientes para **{provincia}** en **{anio}**."
    fila = fila.iloc[0]
    nivel = fila["nivel_alerta"]
    descomp = ind.descomponer_contribucion(fila)
    principal = descomp.iloc[0]
    secundaria = descomp.iloc[1]
    return (
        f"**{provincia} ({anio})** — nivel de alerta **{nivel}** "
        f"(IFFS-B = {fila['IFFS_B']:.1f} sobre 100).\n\n"
        f"La dimensión que más empuja el resultado es **{principal['dimension']}** "
        f"(percentil {principal['percentil']:.0f} respecto de la ventana histórica base "
        f"2005-2012), seguida por **{secundaria['dimension']}** "
        f"(percentil {secundaria['percentil']:.0f}).\n\n"
        f"Esta lectura es **relativa** a la historia reciente de las 24 provincias "
        f"argentinas, no una probabilidad estimada de default. El análisis de "
        f"validación del proyecto (ver notebook) muestra que el índice no tiene, "
        f"con la evidencia disponible, capacidad de anticipación confiable."
    )


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
st.caption(
    "Prototipo — Segundo Parcial, Programación para Economistas. "
    "Herramienta de apoyo al análisis económico, **no** un mecanismo de predicción de default."
)

tab_ranking, tab_evolucion, tab_comparacion, tab_detalle, tab_datos = st.tabs(
    ["🏆 Ranking / mapa", "📈 Evolución histórica", "⚖️ Comparación", "🔎 Detalle y explicación", "📥 Datos"]
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
            labels={"IFFS_B": "IFFS-B (0-100)", "provincia": "", "nivel_alerta": "Nivel de alerta"},
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, width="stretch")
    with c2:
        pivot = panel_filtrado.pivot_table(index="provincia", columns="anio", values="IFFS_B")
        fig_heat = px.imshow(
            pivot, color_continuous_scale="RdYlGn_r", aspect="auto",
            title="Mapa de calor — IFFS-B por provincia y año",
            labels={"color": "IFFS-B"},
        )
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
    for col, nombre in [("IFFS_A", "IFFS-A"), ("IFFS_B", "IFFS-B"), ("IFFS_C", "IFFS-C")]:
        fig_evol.add_trace(go.Scatter(x=df_prov["anio"], y=df_prov[col], mode="lines+markers", name=nombre))
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
        fig_ratios.add_trace(go.Scatter(x=df_prov["anio"], y=df_prov[col], mode="lines+markers", name=nombre))
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
        )
        fig_comp.add_vrect(x0=2020, x1=2021, fillcolor="grey", opacity=0.15, line_width=0)
        fig_comp.update_layout(height=450)
        st.plotly_chart(fig_comp, width="stretch")

        anio_snapshot = st.select_slider("Año del corte transversal", options=sorted(ANIOS), value=max(ANIOS))
        snapshot = df_comp[df_comp.anio == anio_snapshot][
            ["provincia", "IFFS_B", "nivel_alerta", "deuda_sobre_ingresos", "servicio_sobre_ingresos"]
        ].sort_values("IFFS_B", ascending=False)
        st.dataframe(snapshot, width="stretch", hide_index=True)

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
            labels={"percentil": "Percentil (0-100, vs. ventana base 2005-2012)", "dimension": ""},
        )
        st.plotly_chart(fig_desc, width="stretch")

        st.markdown("### 🗣️ Explicación automática")
        st.info(explicar_alerta(provincia_foco, anio_detalle))

# ---------------------------------------------------------------------------
# Tab 5: Datos y descarga
# ---------------------------------------------------------------------------
with tab_datos:
    st.subheader("Tabla filtrada")
    columnas_mostrar = [
        "provincia", "anio", "deuda_sobre_ingresos", "servicio_sobre_ingresos",
        "rigidez_gastototal", "fragilidad_resultado", "IFFS_A", "IFFS_B", "IFFS_C", "nivel_alerta",
    ]
    tabla = panel_filtrado[columnas_mostrar].sort_values(["provincia", "anio"])
    st.dataframe(tabla, width="stretch", hide_index=True)
    st.download_button(
        "⬇️ Descargar tabla filtrada (CSV)",
        data=tabla.to_csv(index=False).encode("utf-8"),
        file_name=f"iffs_filtrado_{periodo[0]}_{periodo[1]}.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.caption(
        "Metodología completa, validación y limitaciones documentadas en "
        "`notebooks/fragilidad_fiscal_provincial.ipynb` e `informe/informe_fragilidad_fiscal.pdf`. "
        f"Varianza explicada del PCA (IFFS-C): {meta_iffs['varianza_explicada_pc1']*100:.1f}%."
    )
