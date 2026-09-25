"""
formato.py
===========

Formato numérico en castellano para los textos generados automáticamente: coma decimal y punto como separador de miles.
Se aplica solo a TEXTO (no a tablas ni a datos): `es("AUC 0.93 en 1.000 sorteos")` -> "AUC 0,93 en 1.000 sorteos".

Regla: un número con punto y exactamente 3 decimales cuya parte entera no es 0 se interpreta como separador de miles (2.000, 555.836) y se
conserva; cualquier otro número con punto es un decimal y pasa a coma.
"""

from __future__ import annotations

import re

_NUM = re.compile(r"(?<![\d.,])(\d+)\.(\d+)(?![\d.])")


def es(texto):
    if not isinstance(texto, str):
        return texto

    def _f(m: re.Match) -> str:
        ent, dec = m.groups()
        if len(dec) == 3 and ent != "0":
            return m.group(0)
        return f"{ent},{dec}"

    return _NUM.sub(_f, texto)
