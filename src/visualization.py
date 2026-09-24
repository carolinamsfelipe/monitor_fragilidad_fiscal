"""
visualization.py
==================

Funciones de graficación reutilizables para el notebook, el dashboard de
Streamlit y el informe en PDF. Todas las figuras se guardan automáticamente
en outputs/figures/ cuando se les pasa `guardar=True`.

Convención: todo gráfico incluye título, unidades/escala, período cubierto y
fuente en el pie -- requisito de control de calidad del proyecto.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
PALETA_ALERTA = {"Bajo": "#2E7D32", "Medio": "#F9A825", "Alto": "#EF6C00", "Crítico": "#C62828", "Sin datos": "#9E9E9E"}

FUENTE_TEXTO = "Fuente: DNAP, ejecución presupuestaria provincial y BCRA (procesado por el autor)."


def _guardar(fig, nombre: str, guardar: bool) -> None:
    if guardar:
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")


def grafico_evolucion_iffs(panel_iffs: pd.DataFrame, provincias: list[str], guardar: bool = False):
    fig, ax = plt.subplots(figsize=(9, 5))
    subset = panel_iffs[panel_iffs["provincia"].isin(provincias)]
    for prov, g in subset.groupby("provincia"):
        ax.plot(g["anio"], g["IFFS_B"], marker="o", markersize=3, label=prov, linewidth=1.6)
    ax.axvspan(2020, 2021, color="grey", alpha=0.15, label="Shock COVID-19 (2020-2021)")
    ax.set_title("Evolución del IFFS-B (2005-2025)")
    ax.set_xlabel("Año")
    ax.set_ylabel("IFFS-B (percentil 0-100, mayor = más frágil)")
    ax.legend(fontsize=8, ncol=2)
    ax.text(0.0, -0.18, FUENTE_TEXTO, transform=ax.transAxes, fontsize=7, color="grey")
    _guardar(fig, "evolucion_iffs_provincias", guardar)
    return fig


def grafico_ranking_anio(panel_iffs: pd.DataFrame, anio: int, top_n: int = 24, guardar: bool = False):
    df = panel_iffs[panel_iffs["anio"] == anio].dropna(subset=["IFFS_B"]).sort_values("IFFS_B", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 7))
    colores = df["nivel_alerta"].map(PALETA_ALERTA).fillna("#9E9E9E") if "nivel_alerta" in df else "#1f77b4"
    ax.barh(df["provincia"], df["IFFS_B"], color=colores)
    ax.invert_yaxis()
    ax.set_title(f"Ranking de fragilidad fiscal (IFFS-B) — {anio}")
    ax.set_xlabel("IFFS-B (percentil 0-100)")
    ax.text(0.0, -0.08, FUENTE_TEXTO, transform=ax.transAxes, fontsize=7, color="grey")
    _guardar(fig, f"ranking_iffs_{anio}", guardar)
    return fig


def heatmap_provincia_anio(panel_iffs: pd.DataFrame, columna: str = "IFFS_B", guardar: bool = False):
    pivot = panel_iffs.pivot_table(index="provincia", columns="anio", values=columna)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, cmap="RdYlGn_r", ax=ax, cbar_kws={"label": columna}, linewidths=0.3)
    ax.set_title(f"Mapa de calor de fragilidad fiscal ({columna}) por provincia y año")
    ax.set_xlabel("Año")
    ax.set_ylabel("Provincia")
    fig.text(0.01, -0.02, FUENTE_TEXTO, fontsize=7, color="grey")
    _guardar(fig, f"heatmap_{columna}", guardar)
    return fig


def grafico_descomposicion(contribucion_df: pd.DataFrame, provincia: str, anio: int, guardar: bool = False):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(contribucion_df["dimension"], contribucion_df["percentil"], color="#546E7A")
    ax.set_xlim(0, 100)
    ax.set_title(f"Descomposición del IFFS-B — {provincia} ({anio})")
    ax.set_xlabel("Percentil (0-100)")
    ax.text(0.0, -0.2, FUENTE_TEXTO, transform=ax.transAxes, fontsize=7, color="grey")
    _guardar(fig, f"descomposicion_{provincia}_{anio}".replace(" ", "_"), guardar)
    return fig


def grafico_comparacion_especificaciones(panel_iffs: pd.DataFrame, guardar: bool = False):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(panel_iffs["IFFS_A"], panel_iffs["IFFS_C"], alpha=0.4, s=12)
    axes[0].set_xlabel("IFFS-A (parsimoniosa)")
    axes[0].set_ylabel("IFFS-C (PCA)")
    axes[0].set_title("IFFS-A vs. IFFS-C")

    axes[1].scatter(panel_iffs["IFFS_B"], panel_iffs["IFFS_C"], alpha=0.4, s=12, color="#EF6C00")
    axes[1].set_xlabel("IFFS-B (núcleo, 4 dimensiones)")
    axes[1].set_ylabel("IFFS-C (PCA)")
    axes[1].set_title("IFFS-B vs. IFFS-C")

    fig.suptitle("Sensibilidad del índice a la especificación metodológica")
    fig.text(0.01, -0.02, FUENTE_TEXTO, fontsize=7, color="grey")
    _guardar(fig, "sensibilidad_especificaciones", guardar)
    return fig


def grafico_backtest_resumen(backtest_df: pd.DataFrame, guardar: bool = False):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot = backtest_df.pivot(index="horizonte", columns="estrategia", values="spearman_iffs")
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Backtest leave-one-out: correlación (Spearman) IFFS vs. cambio futuro")
    ax.set_xlabel("Horizonte (años)")
    ax.set_ylabel("Correlación de Spearman")
    ax.text(0.0, -0.22, FUENTE_TEXTO, transform=ax.transAxes, fontsize=7, color="grey")
    _guardar(fig, "backtest_resumen", guardar)
    return fig
