"""
visualization.py
==================

Gráficos estáticos (matplotlib) del IFFS v2, compartidos por el notebook, el generador de resultados, el informe y la presentación.
Convención: título que dice qué se ve, unidades/escala, período y fuente en el pie. Color por rol: un tono por subíndice (orden fijo),
tinta neutra para el compuesto, un solo tono para magnitud.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import indicators as ind
from .formato import es

FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COLOR_DIM = {"solvencia": "#2a78d6", "liquidez": "#eb6834", "estructural": "#1baf7a", "coyuntural": "#4a3aa7"}
AZUL, ROJO = "#2a78d6", "#d03b3b"
FUENTE = "Fuente: DNAP (ejecución, deuda y servicio), BCRA; elaboración propia. IFFS v2: percentil dentro de cada año, 0-100 (mayor = más frágil en relación con los pares)."

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "font.family": "DejaVu Sans",
})


_DECIMAL = re.compile(r"^[−-]?\d+\.\d+$")


def _coma_decimal(fig):
    """Textos y ejes del gráfico con coma decimal (castellano)."""
    fig.canvas.draw()
    for ax in fig.axes:
        for eje, lim in ((ax.xaxis, ax.get_xlim), (ax.yaxis, ax.get_ylim)):
            etiquetas = [t.get_text() for t in eje.get_majorticklabels()]
            if any(_DECIMAL.match(e) for e in etiquetas):
                l0 = lim()
                eje.set_ticks(eje.get_majorticklocs())
                eje.set_ticklabels([es(e) for e in etiquetas])
                (ax.set_xlim if eje is ax.xaxis else ax.set_ylim)(l0)
    for t in fig.findobj(matplotlib.text.Text):
        txt = t.get_text()
        if txt and "." in txt and not t.get_text().startswith("$"):
            t.set_text(es(txt))


def _cierre(fig, nombre: str | None, fuente: bool = True):
    if fuente:
        fig.text(0.01, 0.005, FUENTE, fontsize=6.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if fuente else 0, 1, 1))
    _coma_decimal(fig)
    if nombre:
        fig.savefig(FIGURES_DIR / f"{nombre}.png", dpi=170, bbox_inches="tight", facecolor="white")
    return fig


def fig_cobertura(panel: pd.DataFrame, nombre: str | None = "cobertura_compuesto"):
    """Grilla provincia x año: dónde hay compuesto (gris claro = faltante y por qué)."""
    ok = panel.pivot(index="provincia", columns="anio", values="IFFS").notna().astype(int)
    orden = ok.sum(axis=1).sort_values().index.tolist()[:0] + sorted(ok.index)
    ok = ok.loc[orden]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.imshow(ok.to_numpy(), cmap=matplotlib.colors.ListedColormap(["#c3c2b7", "#2a78d6"]), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(ok.shape[1]))
    ax.set_xticklabels(ok.columns, rotation=90)
    ax.set_yticks(range(ok.shape[0]))
    ax.set_yticklabels(ok.index)
    ax.grid(False)
    n, tot = int(ok.to_numpy().sum()), ok.size
    ax.set_title(f"Cobertura del IFFS compuesto: {n} de {tot} provincia-año ({100*n/tot:.1f}%)")
    i = list(ok.index).index("La Pampa")
    ax.text(5.5, i, "stock de deuda no confiable (2005-2017)", ha="center", va="center", fontsize=7.5, color=INK)
    ax.text(ok.shape[1] - 1, i + 0.9, "2025 sin ejecución", ha="right", va="top", fontsize=6.5, color=INK2)
    ax.set_xlabel("Año   (azul = calculado · gris = faltante, sin imputar)")
    return _cierre(fig, nombre)


def fig_ranking(panel: pd.DataFrame, anio: int, nombre: str | None = None):
    """Ranking del compuesto con los cuatro subíndices como puntos (perfil de cada provincia)."""
    d = panel[panel["anio"] == anio].dropna(subset=["IFFS"]).sort_values("IFFS")
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    y = np.arange(len(d))
    ax.barh(y, d["IFFS"], color="#d9d8d1", height=0.62, label="IFFS compuesto (barra)")
    for dim, marc in zip(ind.DIMS, ["o", "s", "^", "D"]):
        ax.scatter(d[f"S_{dim}"], y, s=26, marker=marc, color=COLOR_DIM[dim], label=ind.NOMBRE_DIM[dim] + (" (fuera del compuesto)" if dim not in ind.DIMS_NUCLEO else ""), zorder=3,
                   edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(d["provincia"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Posición relativa 0-100 (percentil entre las jurisdicciones ese año)")
    ax.set_title(f"IFFS {anio}: compuesto y subíndices por jurisdicción")
    ax.legend(loc="lower right", fontsize=7.5, ncol=1)
    ax.grid(axis="y", visible=False)
    return _cierre(fig, nombre or f"ranking_iffs_{anio}")


def fig_deriva(panel: pd.DataFrame, nombre: str | None = "deriva_normalizacion"):
    """Por qué se abandona la ventana base fija: la mediana de Deuda/Ingresos cae dentro de la propia ventana."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    med = panel.groupby("anio")["deuda_sobre_ingresos"].median()
    ax = axes[0]
    ax.axvspan(2005, 2012, color="#e8eefb", label="Ventana base 2005-2012")
    ax.plot(med.index, med.values, color=AZUL, lw=2)
    ax.set_yscale("log")
    ax.set_title("Mediana de Deuda / Ingresos (escala log)")
    ax.set_xticks(range(2005, 2026, 4))
    ax.set_xlabel("Año")
    ax.set_ylabel("Veces los ingresos corrientes")
    ax.legend(fontsize=7.5)
    ax = axes[1]
    en_base = panel["anio"].between(2005, 2012)
    leg = ind._percentil_ecdf(panel["deuda_sobre_ingresos"], panel.loc[en_base, "deuda_sobre_ingresos"])
    m_leg = leg.groupby(panel["anio"]).median()
    m_ca = panel.groupby("anio")["S_solvencia"].median()
    ax.plot(m_leg.index, m_leg.values, color=ROJO, lw=2, label="Base fija 2005-2012 (v1)")
    ax.plot(m_ca.index, m_ca.values, color=INK, lw=2, label="Corte anual (v2)")
    ax.set_ylim(0, 100)
    ax.set_title("Percentil mediano de solvencia por año")
    ax.set_xticks(range(2005, 2026, 4))
    ax.set_xlabel("Año")
    ax.set_ylabel("Percentil 0-100")
    ax.legend(fontsize=7.5, loc="lower left")
    return _cierre(fig, nombre)


def fig_robustez(tabla: pd.DataFrame, nombre: str | None = "robustez_especificaciones"):
    """Correlación de Spearman con la especificación principal + cambio medio de posición (en puestos)."""
    t = tabla.iloc[1:].copy().sort_values("spearman_vs_principal")
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    y = np.arange(len(t))
    ax.hlines(y, 0.8, t["spearman_vs_principal"], color="#c3c2b7", lw=1)
    ax.scatter(t["spearman_vs_principal"], y, s=50 + 55 * t["cambio_rango_medio"], color=AZUL, zorder=3, edgecolor="white")
    for yi, (s, c) in enumerate(zip(t["spearman_vs_principal"], t["cambio_rango_medio"])):
        ax.text(s + 0.006, yi, f"ρ={s:.2f} · Δpuesto {c:.1f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(t["especificacion"], fontsize=7.5)
    ax.set_xlim(0.8, 1.06)
    ax.set_xlabel("Correlación de Spearman con el IFFS principal (todas las provincia-año)   ·   tamaño del punto = cambio medio de puesto")
    ax.set_title("Robustez del ordenamiento a cada decisión de diseño (una a la vez)")
    ax.grid(axis="y", visible=False)
    return _cierre(fig, nombre)


def fig_auc_por_anio(pt_iffs: pd.DataFrame, pt_bonos: pd.DataFrame, pt_legado: pd.DataFrame | None = None, nombre: str | None = "auc_por_anio"):
    """Placebo temporal: AUC del mismo grupo de 12 provincias contra las demás, año por año (incluye antes y después del evento)."""
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.axvspan(2019.5, 2021.5, color="#efeee9", label="Bloque de eventos 2020-21")
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.text(2005.1, 0.505, "azar (0,5)", fontsize=7, color=MUTED, va="bottom")
    ax.fill_between(pt_iffs["anio"], pt_iffs["ic95_inf"], pt_iffs["ic95_sup"], color=AZUL, alpha=0.15, lw=0)
    ax.plot(pt_iffs["anio"], pt_iffs["auc"], color=AZUL, lw=2.2, marker="o", ms=3.5, label="IFFS compuesto (IC 95%)")
    ax.plot(pt_bonos["anio"], pt_bonos["auc"], color=ROJO, lw=1.8, ls="--", marker="s", ms=3, label="Benchmark: % de bonos en la deuda")
    if pt_legado is not None:
        ax.plot(pt_legado["anio"], pt_legado["auc"], color=MUTED, lw=1.4, label="IFFS v1 (base fija 2005-12)")
    ax.set_ylim(0.3, 1.02)
    ax.set_xlim(2004.5, 2025.5)
    ax.set_xticks(range(2005, 2026, 2))
    ax.set_xlabel("Año en que se mide el score")
    ax.set_ylabel("AUC: 12 provincias con evento vs. resto")
    ax.set_title("AUC año por año: provincias con evento 2020-21 vs. el resto")
    ax.legend(loc="lower right", fontsize=7.5)
    return _cierre(fig, nombre)


def fig_estudio_eventos(es: pd.DataFrame, nombre: str | None = "estudio_de_eventos"):
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.fill_between(es["tau"], es["media_control"], es["media_evento"], color=AZUL, alpha=0.08, lw=0)
    ax.plot(es["tau"], es["media_evento"], color=ROJO, lw=2.2, marker="o", ms=4, label="Provincias con evento 2020-21 (n=12)")
    ax.plot(es["tau"], es["media_control"], color=INK2, lw=2.2, marker="s", ms=4, label="Resto (n=11-12)")
    ax.axvline(0, color=MUTED, lw=1, ls=":")
    ax.set_ylim(28, 76)
    ax.text(0.15, 73, "año ancla (convención: 2020)", fontsize=7, color=MUTED, va="center")
    ax.set_xlabel("Años respecto del ancla (τ)")
    ax.set_ylabel("IFFS compuesto medio (0-100)")
    ax.set_title("Estudio de eventos: IFFS medio de las provincias con evento vs. el resto")
    ax.legend(fontsize=7.5, loc="center")
    return _cierre(fig, nombre)


def fig_heatmap(panel: pd.DataFrame, columna: str = "IFFS", nombre: str | None = "heatmap_IFFS"):
    piv = panel.pivot_table(index="provincia", columns="anio", values=columna)
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    im = ax.imshow(piv.to_numpy(), cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=90)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=f"{columna} (0-100)", shrink=0.8)
    ax.set_title(f"Mapa de calor: {columna} por jurisdicción y año (celdas en blanco = sin dato)")
    return _cierre(fig, nombre)


def fig_evolucion(panel: pd.DataFrame, provincias: list[str], nombre: str | None = "evolucion_iffs_provincias"):
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.axvspan(2019.5, 2021.5, color="#efeee9", label="Bloque de eventos 2020-21")
    for prov, g in panel[panel["provincia"].isin(provincias)].groupby("provincia"):
        ax.plot(g["anio"], g["IFFS"], marker="o", ms=3, lw=1.8, label=prov)
    ax.set_xlabel("Año")
    ax.set_ylabel("IFFS compuesto (0-100)")
    ax.set_title("Evolución del IFFS compuesto en provincias seleccionadas")
    ax.legend(fontsize=7.5, ncol=2)
    return _cierre(fig, nombre)


def fig_subindices_provincia(panel: pd.DataFrame, provincia: str, nombre: str | None = None):
    g = panel[panel["provincia"] == provincia].sort_values("anio")
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for dim in ind.DIMS:
        ax.plot(g["anio"], g[f"S_{dim}"], color=COLOR_DIM[dim], lw=1.8, label=ind.NOMBRE_DIM[dim])
    ax.plot(g["anio"], g["IFFS"], color=INK, lw=2.6, label="Compuesto")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Año")
    ax.set_ylabel("0-100")
    ax.set_title(f"{provincia}: subíndices y compuesto")
    ax.legend(fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _cierre(fig, nombre)
