# ESTADO del proyecto IFFS (para continuar en otro chat)

_Última actualización: 2026-09-25. ✅ = ejecutado y verificado en esta sesión · ⏳ = pendiente._

## Objetivo
Llevar el Índice de Fragilidad Fiscal Subnacional (IFFS, 24 jurisdicciones, 2005-2025) a nivel universitario/semiprofesional: metodología defendible, análisis reproducible y los mismos números en
notebook, dashboard, README, informe PDF (≤ 3 págs.) y presentación (7-9 diapositivas). Reglas del usuario: no inventar datos; no optimizar pesos/umbrales contra los 13 eventos; el score es **relativo**,
no una probabilidad de default; separar evidencia / interpretación / supuesto / decisión de diseño / limitación.

## Decisiones (no reabrir sin motivo)
- **CABA (servicio):** hoja `Millones` de la DNAP, sin `TOTAL`, con control automático del cociente 1.000. ✅ vigente y testeado.
- **La Pampa stock 2005-2017:** sin confirmación oficial → **faltante** en el análisis principal (`la_pampa="excluir"`); ÷1.000 solo como sensibilidad (`"sensibilidad_miles"`, SUPUESTO). ⚠️ Se **revirtió** la corrección ÷1.000 que estaba publicada (dashboard viejo y PDF viejo).
- **IFFS v2:** 4 subíndices (solvencia, liquidez, estructural, coyuntural) + compuesto = media de solvencia, liquidez, estructural. El resultado fiscal queda afuera por regla ex ante (persistencia de rangos 0,48 < 0,85).
- **Normalización:** percentil dentro de cada año (v1 = base fija 2005-2012, conservada como `LEG_A/B/C` y alternativa). **Posición** = cuartil del compuesto por año (reemplaza cortes 33/66/85).
- **Ventana:** 1 año (regla ex ante; rectifica mi intuición inicial de 3 años). **Pesos:** iguales, sin optimizar.
- **Validación** separada en descripción / discriminación / anticipación. Los AUC se calcularon DESPUÉS de fijar la especificación; se reportan las 18 especificaciones (`T19`).
- 13 eventos: 12 anclados en 2020 por convención propia (11 con fecha ambigua en la fuente), 1 en 2025. Se reporta ancla 2020 y 2021.

## Resultados verificados (ejecutados en esta sesión)
- `python -m pytest -q` → **17 passed**. `python auditoria/fase1_auditoria.py` → 14 controles de integridad OK. `python scripts/generar_resultados.py` (~1 min, semilla 42) → OK.
- Notebook regenerado y **ejecutado de punta a punta sin errores** (`scripts/construir_notebook.py --ejecutar`, 55 celdas).
- Dashboard: corrió en el navegador integrado (sin excepciones; 9 pestañas, escenario de sensibilidad probado) y con `streamlit.testing.AppTest` en un venv limpio con **solo** `requirements.txt`.
- `python scripts/verificar_consistencia.py` → cifras de README, METODOLOGIA, notebook, PDF y PPTX coinciden con `outputs/resultados_clave.json`.
- PDF: 3 páginas (revisado visualmente página por página). PPTX: 9 diapositivas con notas (revisado visualmente en el visor de Google Drive; chequeo de desborde de texto).
- Cifras clave: compuesto en 490/504 provincia-año (97,2%). AUC 2019 = 0,93 [0,81–1,00]; benchmark % de bonos 0,87 [0,68–1,00]; el grupo ya se separaba en 2012 (AUC 0,87) y sigue en 2025 (0,98) → rasgo persistente, no anticipación. Pretendencia T-5→T-1: +5,2 pts [−2,6; +13,5], p=0,28. Robustez de diseño: ρ de Spearman 0,88–0,99.

## Archivos (2026-09-25)
- **Código:** `src/{data_processing,indicators,validation,pipeline,analytics,modeling,visualization}.py`, `app.py`, `scripts/{generar_resultados,construir_notebook,generar_entregables,verificar_consistencia}.py`, `auditoria/fase1_auditoria.py`, `tests/test_iffs.py`.
- **Documentos:** `README.md`, `ESTADO.md`, `docs/METODOLOGIA.md`, `docs/FASE1_AUDITORIA.md` (con banner de actualización), `notebooks/fragilidad_fiscal_provincial.ipynb`, `informe/informe_fragilidad_fiscal.pdf`, `presentacion/presentacion_fragilidad_fiscal.pptx`.
- **Salidas:** `outputs/resultados_clave.json`, `outputs/tables/T*.csv`, `outputs/figures/*.png`, `outputs/auditoria/*.csv`, `data/processed/panel_iffs_scores.csv`. Versión anterior archivada en `outputs/_historico_v1/` (no borrada).
- Orden de regeneración: `generar_resultados` → `construir_notebook --ejecutar` → `generar_entregables` → `verificar_consistencia` (+ `pytest`).

## Limitaciones vigentes
- La Pampa 2005-2017 sin confirmar (y 2025 sin ejecución): 14 de 504 observaciones sin compuesto.
- 13 eventos, 12 en el mismo shock, 11 con año convencional → sin validación predictiva; discriminación confundida con acceso al mercado (% de bonos); entre emisoras solo 3 controles.
- Score relativo: no muestra movimientos comunes (se publica el nivel común). Stock/flujo sobrestima Deuda/Ingresos con inflación alta. Faltan deuda flotante, garantías, vencimientos, moneda (11/24).
- Saltos de servicio: Santa Cruz 2024-25 y San Luis 2014/2022 analizados con identidad stock-flujo (ver `docs/METODOLOGIA.md`); queda La Pampa 2024 sin verificación. Los paneles base son derivados (RAW→panel no reproducible desde el repo).

## Actualización de producto (2026-09-25)
- ✅ `app.py`: se agregó Comparar (selección de Año y pocas jurisdicciones, subíndices + ratios originales); mapa de calor queda como exploración desplegable; las tablas de pantalla usan coma decimal y `Año`; explicación breve sobre posiciones relativas, faltantes y análisis automatizado.
- ✅ Se agregó Fuentes y guía, con enlaces DNAP, FMI y OCDE; se aclara que la bibliografía orienta el diseño, no valida el IFFS para Argentina. El análisis automático existente sí incluye ML descriptivo (k-means e Isolation Forest) y reglas; no es IA generativa ni servicio externo.
- ✅ Se regeneró `outputs/` (`huella deabd9a28e1abd8ea5f49a7c971c24c2`). La advertencia técnica de desincronización ahora es discreta, y la validación/robustez se oculta si la huella no coincide.
- ✅ `README.md` explica por qué se conservan las vistas y cómo usar la comparación.
- ✅ Verificado: `AppTest` (11 pestañas, 0 excepciones), `scripts/verificar_consistencia.py` (README, metodología, notebook, informe de 3 páginas y presentación de 9 diapositivas consistentes), `pytest` (19 pasaron).
- ⏳ Estos cambios están en la carpeta de trabajo, no en el sitio publicado: esta carpeta no contiene `.git` ni un remoto configurado, así que no fue posible enviar los archivos a GitHub desde aquí.

## Actualización 2026-09-26
- ✅ Coma decimal unificada: `print`/tablas del notebook, celdas de texto y tablas del dashboard, ejes y textos de las figuras matplotlib, ejes/hover de Plotly. Se corrigió además un defecto: las tablas del dashboard mostraban el año como "2.020" (separador de miles).
- ✅ Saltos de servicio: ya estaban resueltos en Fase 2 (identidad stock-flujo); solo La Pampa 2024 (stock 500, fuente oficial lo informa así) es inverificable y no cambia puestos.
- ✅ Regenerado todo (huella 139aef89…); 19 tests OK, notebook sin errores, consistencia OK, AppTest 0 excepciones.
- ✅ Carpeta lista para subir al repo: `~/Desktop/subir_a_github/` (src/, app.py, requirements.txt, .streamlit/, data/raw/, outputs/tables/, outputs/resultados_clave.json). NO subida.

- ✅ 2026-09-26: `docs/PEDIDO_DNAP_email.md` (borrador, sin enviar) y `docs/carga_manual/carga_manual_deuda_provincial.xlsx` (plantilla: moneda 2014-2025 × 24 provincias con lo ya cargado en gris; hojas Vencimientos y Flotante vacías). La moneda existente cubre solo 11 provincias y 2014-2020.

## Pendientes
1. ⏳ **Publicar el dashboard:** subir la versión de esta carpeta al repositorio GitHub conectado a Streamlit Community Cloud; el repositorio no está disponible en esta copia local.
2. ⏳ Confirmar con la DNAP las unidades del stock de La Pampa 2005-2017 (si se confirma, cambiar el tratamiento principal y volver a correr todo).
3. ⏳ Incorporar deuda flotante, garantías, vencimientos y composición por moneda con fuentes comparables; sumar eventos futuros para validación fuera de muestra.
