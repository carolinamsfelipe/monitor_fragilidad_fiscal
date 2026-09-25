# Monitor de Fragilidad Fiscal Provincial — IFFS v2

Índice de Fragilidad Fiscal Subnacional para las 24 jurisdicciones argentinas (23 provincias + CABA), 2005-2025.
Segundo Parcial — Programación para Economistas (UADE).

> **Qué es:** una herramienta de apoyo al análisis que ordena a las jurisdicciones por su fragilidad fiscal **relativa** (percentil dentro de cada año),
> con cuatro subíndices y un compuesto. **Qué no es:** una probabilidad de default, un sistema de alerta calibrado ni un modelo predictivo.
> Con 13 eventos (12 en el mismo shock) los datos no permiten validar capacidad predictiva; lo que se muestra es descripción y discriminación exploratoria.

Estado del proyecto para continuar en otro chat: [`ESTADO.md`](ESTADO.md) · Metodología completa: [`docs/METODOLOGIA.md`](docs/METODOLOGIA.md) · Auditoría de datos: [`docs/FASE1_AUDITORIA.md`](docs/FASE1_AUDITORIA.md).

## 1. Resultados principales (generados desde `outputs/resultados_clave.json`)

<!-- RESULTADOS:INICIO -->
| Qué | Resultado | Tipo |
|---|---|---|
| Cobertura del compuesto | **490 de 504** provincia-año (97,2%); faltan La Pampa 2005-2017 (stock no confiable) y 2025 (sin ejecución) | Evidencia |
| Diseño | 4 subíndices; compuesto = solvencia + liquidez + estructural. El resultado fiscal queda afuera (persistencia de rangos 0,48 < 0,85) | Evidencia + diseño |
| Normalización | Percentil dentro de cada año. La mediana de Deuda/Ingresos cae de 1,14 (2005) a 0,31 (2012) y 0,15 (2024): la base fija 2005-2012 no es estacionaria | Evidencia |
| Robustez | ρ de Spearman con el principal entre 0,88 y 0,99 frente a normalización, ventanas, pesos y agregación; con 2.000 pesos aleatorios, mediana 0,94 | Evidencia |
| Discriminación (2019) | AUC **0,93** [IC 95%: 0,81–1,00] para separar a las 12 provincias con evento 2020-21 de las otras 12; el azar supera 0,70 solo el 5% de las veces | Evidencia |
| …pero | El % de bonos en la deuda logra AUC 0,87 [0,68–1,00]. El grupo ya se separaba en 2012 (AUC 0,87) y sigue separado en 2025 (AUC 0,98) | Evidencia → interpretación: rasgo persistente, no anticipación |
| Anticipación | Cambio del compuesto T−5→T−1: +5,2 puntos [−2,6; +13,5] entre provincias con evento y resto (p = 0,28): sin evidencia de deterioro previo distinto | Evidencia + limitación (potencia baja) |
| Predicción | No viable: 0 eventos antes de 2020, 13 en total, 12 simultáneos, 11 con año ancla convencional | Limitación |
<!-- RESULTADOS:FIN -->

## 2. Estructura del proyecto

```
parcial2_monitor_fragilidad_fiscal/
├── README.md · ESTADO.md
├── app.py                      dashboard Streamlit (lee outputs/ y src/)
├── requirements.txt            solo el dashboard (lo lee Streamlit Cloud)
├── requirements-notebook.txt   notebook + tests + informe/presentación
├── src/
│   ├── data_processing.py      carga, validación, tratamientos (CABA, La Pampa), panel maestro 24x21
│   ├── indicators.py           IFFS v2: subíndices, normalizaciones, agregaciones, IFFS v1 (legado)
│   ├── validation.py           criterios de diseño, robustez, discriminación, anticipación, pruebas no viables
│   ├── analytics.py            perfiles (k-means), anomalías, backtest, simulador, análisis automático
│   ├── modeling.py             base de eventos y demostración de por qué no hay modelo predictivo
│   ├── pipeline.py             punto de entrada único (huella de código+datos)
│   └── visualization.py        gráficos
├── scripts/
│   ├── generar_resultados.py   FUENTE ÚNICA de cifras (tablas, figuras, resultados_clave.json)
│   ├── construir_notebook.py   genera y ejecuta el notebook
│   └── generar_entregables.py  README (bloque de resultados), informe PDF y presentación PPTX
├── auditoria/                  fase1_auditoria.py (datos) y fase2_servicio_y_unidades.py (identidad stock-flujo); escriben outputs/auditoria/
├── tests/test_iffs.py          17 pruebas automáticas
├── notebooks/fragilidad_fiscal_provincial.ipynb
├── informe/informe_fragilidad_fiscal.pdf         (3 páginas)
├── presentacion/presentacion_fragilidad_fiscal.pptx   (9 diapositivas con notas)
├── data/raw/                   paneles base y fuentes DNAP (CABA, La Pampa); data/processed/ panel con el índice
└── outputs/                    tables/ · figures/ · auditoria/ · resultados_clave.json · _historico_v1/ (versión anterior, archivada)
```

## 3. Cómo reproducir

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-notebook.txt
python -m pytest -q                        # 17 pruebas
python auditoria/fase1_auditoria.py        # auditoría de datos
python scripts/generar_resultados.py       # tablas, figuras y cifras (~1 min, semilla 42)
python scripts/construir_notebook.py --ejecutar   # notebook ejecutado de punta a punta
python scripts/generar_entregables.py      # README (bloque de resultados), informe PDF y presentación
streamlit run app.py                       # dashboard en http://localhost:8501
```

Todas las rutas son relativas a la raíz del proyecto. El dashboard avisa si `outputs/` no corresponde al código y datos actuales (huella MD5).

### Publicar el dashboard (Streamlit Community Cloud)
Subir la carpeta a un repositorio de GitHub (raíz = esta carpeta; `app.py` y `requirements.txt` en la raíz), incluyendo `outputs/tables/`, `outputs/resultados_clave.json`, `data/raw/` y `src/`
(el dashboard lee las tablas de validación de `outputs/`). En https://share.streamlit.io elegir el repo, rama `main`, archivo `app.py`, Python 3.12 o 3.13.

## 4. Datos y tratamientos

| Panel | Fuente | Cobertura |
|---|---|---|
| Ejecución presupuestaria | DNAP | 2005-2025, 24 jurisdicciones |
| Stock y servicio de deuda | DNAP | 1996/2005-2025 |
| Composición por moneda | Bazza, Tedesca & Mahieu (2025) | 11-13 de 24, 1996-2002 y 2014-2020 |
| Capacidad de respuesta | BCRA | 2005-2025 |
| Eventos de default/reestructuración | DNAP | 13 eventos (12 anclados en 2020 por convención; 1 en 2025) |

- **CABA (evidencia):** el servicio de deuda estaba multiplicado por 1.000 por un error de unidades; se lee la hoja `Millones` de la DNAP y el código falla si el cociente con la hoja en miles no es exactamente 1.000.
- **La Pampa 2005-2017 (decisión de diseño):** el stock puede tener un error de unidades y **no hay confirmación oficial** → se trata como **faltante** en el análisis principal (el original se conserva en `deuda_stock_total_fuente`). La corrección ÷1.000 existe solo como sensibilidad rotulada como supuesto (`la_pampa="sensibilidad_miles"`, y opción en el dashboard).
- Los paneles de `data/raw/base_investigacion/` son **derivados** de las fuentes oficiales; los Excel originales de la DNAP no están en el repositorio (salvo CABA y La Pampa).

## 5. Metodología en una página (detalle y evidencia en `docs/METODOLOGIA.md`)

| Decisión | Elección | Por qué (sin mirar eventos) |
|---|---|---|
| Índice único vs. subíndices | 4 subíndices + compuesto de 3 (solvencia, liquidez, estructural) | Dinámicas distintas (ICC y persistencia), redundancia parcial, implicancias de política distintas |
| Resultado fiscal | Subíndice coyuntural, fuera del compuesto | Regla ex ante: persistencia de rangos < 0,85 |
| Normalización | Percentil dentro de cada año | La base fija 2005-2012 no es estacionaria; sin información futura; inmune a outliers |
| Ventanas | 1 año (regla ex ante); 3 y 5 años como sensibilidad | Persistencia de rangos ≥ 0,85 ya a 1 año |
| Pesos | Iguales; sensibilidad 50/25/25 y 2.000 vectores aleatorios | Sin optimización |
| Posición | Cuartil del compuesto dentro de cada año | Reemplaza cortes 33/66/85 arbitrarios |

Validación separada en **descripción / discriminación / anticipación**; las pruebas no viables (con su razón cuantificada) están en `outputs/tables/T13_pruebas_no_viables.csv`.

## 6. Limitaciones

- **La Pampa 2005-2017** sin confirmación oficial (y 2025 sin ejecución): 14 de 504 observaciones sin compuesto.
- **Score relativo:** no distingue "todas mejoran" de "nadie cambia" (se muestra el nivel común aparte). No es probabilidad de default; no hay umbrales de alerta calibrados.
- **13 eventos**, 12 en el mismo shock, 11 con año ancla convencional: no hay validación predictiva; la discriminación está confundida con el acceso al mercado (% de bonos en la deuda).
- **Stock/flujo:** Deuda a 31/12 sobre ingresos del año sobrestima la razón con inflación alta.
- **Faltan** deuda flotante, garantías, vencimientos futuros y moneda de la deuda (solo 11 de 24 provincias).
- **Saltos de servicio resueltos** con la identidad stock-flujo (`auditoria/fase2_servicio_y_unidades.py`): Santa Cruz 2024-25 y San Luis son amortizaciones reales; solo La Pampa 2024 (stock = 500) no es verificable y no cambia ningún puesto.
- Una alta proporción de gasto en personal puede reflejar provisión de servicios y no rigidez.

## 7. Autoría y reproducibilidad
Semilla fija (42) en todos los procesos aleatorios; rutas relativas; `pytest` verifica datos, construcción del índice, reglas de diseño y consistencia entre `outputs/` y el código.
