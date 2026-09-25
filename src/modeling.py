"""
modeling.py
============

Base de eventos de estrés y demostración PROGRAMÁTICA de por qué no se entrena un modelo predictivo.

Contexto (ver `src/validation.py` y el notebook, Sección 13): 13 eventos sobre ~504 provincia-año (2,6%); 12 de ellos en el
mismo shock 2020-21 y con año ancla convencional. Por eso:

- NO se reporta accuracy (con tasa base de 2,6%, "nunca hay evento" acierta ~97%).
- NO se hace un split aleatorio fila a fila, NO se usa SMOTE ni se optimizan pesos o umbrales.
- NO se afirma que el índice "predice" defaults: las pruebas viables (discriminación, pretendencia) están en `validation.py`
  y las que no lo son, con su razón, en `validation.pruebas_no_viables`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def construir_target_estres(panel: pd.DataFrame, eventos: pd.DataFrame) -> pd.DataFrame:
    """Agrega `estres` = 1 si (provincia, año) coincide con un evento de la base (el año del evento es el del archivo)."""
    panel = panel.copy()
    eventos_set = set(zip(eventos["provincia"], eventos["anio_evento"]))
    panel["estres"] = [1 if (p, a) in eventos_set else 0 for p, a in zip(panel["provincia"], panel["anio"])]
    return panel


def intento_de_split_temporal(panel_con_target: pd.DataFrame, anio_corte: int = 2019) -> dict:
    """
    Comprueba si un split temporal estándar (entrenar hasta `anio_corte`, evaluar después) es viable. Devuelve los conteos que
    demuestran que no lo es: sin eventos positivos en el entrenamiento no hay nada que estimar.
    """
    tr = panel_con_target[panel_con_target["anio"] <= anio_corte]
    te = panel_con_target[panel_con_target["anio"] > anio_corte]
    return {
        "anio_corte": anio_corte,
        "obs_entrenamiento": int(len(tr)), "eventos_entrenamiento": int(tr["estres"].sum()),
        "obs_evaluacion": int(len(te)), "eventos_evaluacion": int(te["estres"].sum()),
        "viable": bool(tr["estres"].sum() >= 10 and te["estres"].sum() >= 5),
        "eventos_por_anio": panel_con_target.groupby("anio")["estres"].sum().loc[lambda s: s > 0].to_dict(),
    }
