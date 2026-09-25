# Monitor de Fragilidad Fiscal Provincial

Sistema de evaluación y alerta temprana para las 24 jurisdicciones argentinas
(23 provincias + CABA). Segundo Parcial — Programación para Economistas.

> Herramienta de **apoyo al análisis económico**, no un mecanismo de
> predicción de default. Cada decisión metodológica (qué variables entran al
> índice, cómo se normalizan, qué umbrales alertan, por qué no hay un modelo
> predictivo formal) está documentada explícitamente en el notebook.

---

## 1. Estructura del proyecto

```
parcial2_monitor_fragilidad_fiscal/
├── README.md                          <- este archivo
├── requirements.txt                   <- solo lo que necesita el dashboard (lo lee Streamlit Cloud)
├── requirements-notebook.txt          <- notebook + regenerar informe/presentación
├── .streamlit/config.toml
├── app.py                             <- dashboard interactivo (Streamlit)
├── notebooks/
│   └── fragilidad_fiscal_provincial.ipynb   <- desarrollo completo
├── data/
│   ├── raw/base_investigacion/        <- paneles base ya auditados (ver Sección 3)
│   └── processed/                     <- panel con el IFFS calculado (generado por el notebook)
├── src/
│   ├── data_processing.py             <- carga, validación, panel maestro
│   ├── indicators.py                  <- construcción del índice IFFS
│   ├── modeling.py                    <- alertas por reglas + modelo exploratorio
│   ├── analytics.py                   <- IA sin costo: perfiles, anomalías, proyección, simulador, análisis automático
│   └── visualization.py               <- gráficos reutilizables
├── outputs/
│   ├── figures/                       <- figuras generadas automáticamente
│   └── tables/                        <- tablas generadas automáticamente
├── informe/
│   └── informe_fragilidad_fiscal.pdf  <- informe (2 páginas, máx. 3)
└── presentacion/
    └── presentacion_fragilidad_fiscal.pptx  <- 10 diapositivas + notas del orador
```

## 2. Cómo ejecutar

### Requisitos

- Python 3.10+
- (Recomendado) un entorno virtual limpio

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-notebook.txt   # notebook + dashboard + informe/pptx
# (solo el dashboard: pip install -r requirements.txt)
```

### Notebook

```bash
cd notebooks
jupyter notebook fragilidad_fiscal_provincial.ipynb
```

También puede abrirse directamente en Google Colab subiendo la carpeta del
proyecto (o montando Google Drive) y ajustando `PROJECT_ROOT` si la
estructura de carpetas no se conserva igual.

El notebook se ejecuta de principio a fin sin errores (verificado con
`nbclient`) y guarda automáticamente sus figuras en `outputs/figures/`, sus
tablas en `outputs/tables/`, y el panel final con el índice calculado en
`data/processed/panel_iffs_scores.csv`.

### Dashboard

```bash
streamlit run app.py
```

Abre en `http://localhost:8501`. Permite elegir provincia y período, comparar
provincias, ver el ranking/mapa de calor de fragilidad, descomponer el índice
por dimensión con una explicación automática en lenguaje claro, y descargar
la tabla filtrada en CSV.

## 2b. Publicar el dashboard en Streamlit Community Cloud

1. Subir la carpeta del proyecto a un repositorio de GitHub (raíz del repo = esta carpeta;
   `app.py` y `requirements.txt` deben quedar en la raíz). Incluir `data/raw/base_investigacion/`
   (pesa ~3 MB); `.venv/` queda afuera por el `.gitignore`.
2. En https://share.streamlit.io → **Create app** → elegir el repo, rama `main` y
   *Main file path* = `app.py`.
3. En *Advanced settings* elegir Python 3.12 o 3.13 y **Deploy**.

El dashboard lee sus datos con rutas relativas al repo, no necesita secretos ni claves de API.

## 3. Origen de los datos

Los paneles de `data/raw/base_investigacion/` fueron procesados y auditados
en un proyecto de investigación previo (Taller de Indicadores Económicos,
2026) contra fuentes oficiales:

| Archivo | Fuente | Cobertura |
|---|---|---|
| `panel_fiscal_provincial.csv`, `panel_fiscal_benchmarks.csv` | DNAP (ejecución presupuestaria) | 2005-2025, 24 jurisdicciones |
| `deuda_provincial_panel_1996_2025.csv` | DNAP (stock de deuda) | 1996-2025, 24 jurisdicciones |
| `servicio_deuda_provincial_panel_2005_2025.csv` | DNAP (servicio de deuda) | 2005-2025, 24 jurisdicciones (CABA corregida: hoja `Millones`) |
| `deuda_ratios_provincia_anio.csv` | Cálculo propio sobre los anteriores | 2005-2025 |
| `deuda_moneda_panel.csv` | Bazza, Tedesca & Mahieu (2025) | 11-13 de 24 provincias, 1996-2002 y 2014-2020 |
| `capacidad_respuesta_panel.csv` | BCRA (depósitos/préstamos) | 2005-2025, 24 jurisdicciones |
| `eventos_estres_provincial.csv` | DNAP (reestructuración de títulos) — **construido para este proyecto**, no existía como archivo tabular en la investigación original | 13 eventos, 2005-2025 |

Este proyecto **no vuelve a parsear** los Excel crudos de la DNAP/BCRA (ese
trabajo, con verificación contra el consolidado oficial, <0.4% de diferencia,
ya está hecho) — reutiliza esos paneles como insumo validado y **reconstruye
desde cero, con código propio** (`src/indicators.py`), la construcción del
índice de fragilidad fiscal.

## 4. Metodología del índice (IFFS)

Ver el detalle completo, con fundamento económico de cada variable, en la
Sección 11 del notebook. En resumen:

- 4 dimensiones núcleo, igual ponderación: Solvencia (Deuda/Ingresos),
  Liquidez (Servicio/Ingresos), Rigidez del gasto (Personal/Gasto total),
  Desempeño fiscal (−Resultado financiero/Ingresos).
- Normalización por percentil ECDF sobre una ventana base **fija** 2005-2012.
- 3 especificaciones (IFFS-A, B, C) para exponer la sensibilidad metodológica.
- Sistema de alerta por reglas transparentes (umbrales de percentil, no una
  calibración estadística contra el outcome real).

## 5. Modelo y eventos raros

La base de eventos tiene 13 observaciones positivas sobre ~504 posibles
(2.6%), 12 concentradas en el bienio 2020-2021. Por eso **no** se reporta
accuracy como métrica principal, no se hace split aleatorio ni SMOTE, y los
resultados se presentan como exploratorios (ver Secciones 12-13 del
notebook). El hallazgo más sólido es descriptivo: en 2019, las provincias que
luego reestructuraron ya tenían deuda y servicio de deuda sobre ingresos
claramente más altos que el resto (AUC descriptiva ≈ 0.97).

## 5b. Capa de IA sin costo (Sección 15b del notebook)

Machine learning clásico (scikit-learn), sin APIs ni claves ni costo, reproducible con semilla fija:

- **Perfiles fiscales (k-means)**, ajustados solo con la ventana 2005-2012 (silhouette ≈ 0.31: estructura moderada).
- **Anomalías (Isolation Forest)**: descriptivo, no predictivo.
- **Proyección por tendencia + backtest**: en el backtest a 1 año **no supera** a repetir el último valor (se informa tal cual).
- **Simulador de escenarios**: caída de ingresos, depreciación y suba del servicio (sensibilidad mecánica, no pronóstico).
- **Alertas y cambios** interanuales, con aviso cuando casi todas las provincias se mueven juntas.
- **Análisis exhaustivo automático**: motor de reglas que redacta solo cifras calculadas (no es IA generativa).

## 6. Tabla de control — criterios de evaluación

| Criterio (peso) | Cómo se satisface | Dónde |
|---|---|---|
| **Claridad del planteo del problema (20%)** | Problema, objetivo, usuarios y clasificación del producto explícitos, con motivación cuantificada (12/24 provincias reestructuraron en 2020-21) | Notebook Secc. 1-5; Informe p.1; Presentación diap. 1-3 |
| **Calidad de la implementación en Python (40%)** | Código modular (`src/`), validación de esquema y tipos, manejo de faltantes/división por cero, funciones documentadas, semillas fijas, sin *data leakage*, notebook ejecuta sin errores de punta a punta, dashboard funcional probado | `src/*.py`; notebook (0 errores en ejecución completa); `app.py` (probado con servidor local) |
| **Uso adecuado de librerías y técnicas del curso (20%)** | pandas/numpy para procesamiento, matplotlib/seaborn/plotly para visualización, scikit-learn (PCA, regresión logística penalizada, validación cruzada agrupada), scipy.stats (Mann-Whitney, tamaño de efecto), Streamlit para el prototipo | `src/indicators.py`, `src/modeling.py`, `app.py` |
| **Claridad y solidez de la presentación oral (20%)** | 10 diapositivas con notas del orador, cobertura de problema/datos/lógica/resultados/limitaciones, gráficos legibles con fuente y período | `presentacion/presentacion_fragilidad_fiscal.pptx` |

## 7. Limitaciones y pendientes (resumen — detalle completo en el notebook, Sección 17)

- **La Pampa:** el stock 2005-2017 estaba en miles aunque la fuente declara millones; se corrige (÷1.000) con evidencia sólida pero **sin confirmación oficial de la DNAP**.
- **Exposición cambiaria por provincia**: sin fuente primaria descargable (solo consolidado nacional).
- **Composición de deuda por moneda**: cobertura de solo 11-13 de 24 provincias.
- Sin base pública gratuita de calificaciones crediticias provinciales.
- Umbrales de alerta: convención expositiva, no calibración estadística.
- Modelo predictivo: no viable con split temporal estándar dada la escasez y concentración de eventos (ver Sección 5).
- No se implementó geopandas/mapa geográfico real (no hay shapefiles de provincias disponibles en el entorno) — se usa un mapa de calor (heatmap) como sustituto funcional.

## 8. Autoría y reproducibilidad

Todas las rutas del código son relativas a la raíz del proyecto (`Path(__file__).resolve().parent`).
Semilla aleatoria fija (`RANDOM_SEED = 42`) en todos los pasos con componente
aleatorio (PCA, split de validación cruzada, regresión logística).
