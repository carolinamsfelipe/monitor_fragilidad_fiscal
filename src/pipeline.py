"""
pipeline.py
============

Punto de entrada ÚNICO para construir el panel del IFFS v2. Lo usan el notebook, el dashboard, los scripts de resultados y
los tests, de modo que todos trabajan sobre exactamente los mismos números.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from . import analytics as an
from . import data_processing as dp
from . import indicators as ind

ROOT = Path(__file__).resolve().parent.parent


def huella_del_proyecto() -> str:
    """MD5 del código (`src/*.py`) y de los datos (`data/raw`). Cambia si cambia cualquiera de los dos: sirve para detectar
    tablas/figuras precalculadas que ya no corresponden al código actual."""
    h = hashlib.md5()
    for carpeta, patron in [("src", "*.py"), ("data/raw", "**/*")]:
        for f in sorted((ROOT / carpeta).glob(patron)):
            if f.is_file():
                h.update(f.relative_to(ROOT).as_posix().encode())
                h.update(f.read_bytes())
    return h.hexdigest()


def construir_todo(la_pampa: str = "excluir", spec: ind.Especificacion = ind.ESPEC_PRINCIPAL) -> dict:
    """
    Devuelve un dict con: `panel` (subíndices, compuesto, legado, posición, perfiles, cambios, anomalías), `avisos` (validación de
    datos), `eventos`, `meta_legado`, `info_perfiles` y `backtest`. `la_pampa` = tratamiento del stock dudoso (ver data_processing).
    """
    panel = dp.construir_panel_maestro(la_pampa=la_pampa)
    avisos = dp.validar_panel_maestro(panel)
    p = ind.construir_indice(panel, spec)
    p, meta_legado = ind.construir_iffs_legado(p)
    p, info_perfiles = an.construir_panel_analitico(p)
    return {
        "panel": p, "avisos": avisos, "eventos": dp.load_eventos_estres(), "meta_legado": meta_legado,
        "info_perfiles": info_perfiles, "backtest": an.backtest_proyeccion(p), "la_pampa": la_pampa,
    }
