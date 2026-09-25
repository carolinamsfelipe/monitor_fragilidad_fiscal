# FASE 1 — Auditoría integral del IFFS

> **ACTUALIZACIÓN 2026-09-25 — el tratamiento de La Pampa cambió.** Este documento (escrito el 24-25/09) describía la corrección ÷1.000 del stock de La Pampa 2005-2017 como "CORREGIDO".
> Sin confirmación oficial de la DNAP esa corrección es un **supuesto**, no un hecho: **el análisis principal ahora trata esos años como faltantes** (el valor original se conserva en
> `deuda_stock_total_fuente`) y la corrección ÷1.000 queda solo como escenario de sensibilidad (`la_pampa="sensibilidad_miles"`). Donde este documento dice "corregido" para La Pampa, leer "revertido a faltante".
> La evidencia (La Pampa como millones sería 6x las otras 23 juntas en 1996; salto errático 2015-2018) se mantiene; también la corrección de CABA, que es verificable (cociente exacto 1.000 entre hojas) y sigue vigente.
> Consecuencias de ese cambio, el rediseño posterior y la validación: `docs/METODOLOGIA.md`. Las cifras de AUC de este documento corresponden al IFFS v1 y están superadas por `outputs/resultados_clave.json`.

Reproducible con `python auditoria/fase1_auditoria.py` (tablas en `outputs/auditoria/`).
Etiquetas usadas: **[EMPÍRICO]** medido en los datos · **[ECONÓMICO]** fundamento teórico · **[ESTADÍSTICO]** propiedad estadística ·
**[DISEÑO]** decisión de construcción · **[SUPUESTO]** no verificado · **[LIMITACIÓN]** no resoluble con los datos actuales · **[HEURÍSTICA]**.

---

## 0. Resumen ejecutivo: qué encontré

| # | Hallazgo | Gravedad | Estado |
|---|---|---|---|
| H1 | **El "servicio de CABA no confiable" es un error de unidades del pipeline**, no de la fuente. El archivo tiene una hoja `CABA` (en *miles*) y una hoja `Millones`; el script leyó `sheet_name=0` (miles). | **Crítica** | **CORREGIDO** (2026-09-24) |
| H2 | **El "outlier" de La Pampa (165,7x) parece un error de unidades**, no un fenómeno económico. Cargada como millones, La Pampa en 1996 sería **6 veces la deuda de las otras 23 provincias juntas**. | **Crítica** | **REVERTIDO A FALTANTE** (2026-09-25): sin confirmación oficial no se corrige; ÷1.000 solo como sensibilidad. Hipótesis "todo en miles" descartada con la foto nacional de mar-2026; el año exacto del cambio **sin confirmación oficial** |
| H3 | **La escala 2005-2012 no es estacionaria**: la mediana de Deuda/Ingresos cae de 1,09 (2005) a 0,29 (2012) y 0,15 (2024). En 2025 el 83% de las provincias queda bajo el percentil 33 y ninguna sobre el 66. La dimensión de solvencia pierde poder discriminante desde 2013 **por construcción**. | **Alta** | Confirmado |
| H4 | **Desajuste stock/flujo** en Deuda/Ingresos: stock a 31/12 sobre flujo anual. Con inflación alta sobrestima la razón (proxy: ×1,6 en 2023, ×1,9 en 2024). | Alta | Confirmado en su dirección; magnitud es un proxy |
| H5 | El resultado fiscal es **78% transitorio** (ICC 0,22) y poco persistente (AR1 0,36): es la variable más ruidosa del índice. | Media | Confirmado |
| H6 | Deuda y servicio están correlacionados **ρ = 0,80** (0,89 entre provincias, 0,71 intra-provincia): fuertemente relacionados, no idénticos. | Media | **Corrijo mi corrección**: medí 0,70 con los dos errores de datos sin arreglar; con datos corregidos es 0,80 |
| H7 | Denominadores mixtos (resultado / ingresos *totales*; deuda y servicio / ingresos *corrientes*). Inconsistente en la forma, **empíricamente inocuo** (ρ de rangos ≈ 1,00). | Baja | No urgente |
| H8 | Efecto de valuación cambiaria sobre Deuda/Ingresos: coef. 0,47, EE 0,38, p = 0,22 (n = 77). **No concluyente.** | — | Limitación de potencia |

**Consecuencia para lo ya publicado:** el dashboard, el notebook, el informe y el README afirman que (i) CABA no tiene servicio confiable y
(ii) La Pampa es un "outlier extremo" que estira la escala. Ambas afirmaciones descansan en un error de datos y deben corregirse en la Fase 4.

---

## 1. Datos

### H1 · CABA — servicio de deuda  [EMPÍRICO]
- Fuente: `servicios_provincia_de_ciudad_de_buenos_aires_iv_trim_25_1.xlsx`, hojas `CABA` (encabezado "En **Miles** de Pesos") y `Millones` ("En Millones de Pesos").
- El cociente hoja-CABA / hoja-Millones es **exactamente 1.000 en todos los años** (`A3_caba_servicio_corregido.csv`).
- `scripts/11_servicio_deuda_provincial_panel.py:62` usa `sheet_name=0`, es decir, la hoja en miles, y la trata como millones. Los otros 23 archivos tienen una sola hoja `Millones`.
- **Corrección de mi propia cifra:** en la primera versión de esta auditoría sumé también la columna `TOTAL (5)` de la hoja, duplicando el servicio (mediana 10%, máx 30%, y afirmé que CABA estaba "en el tramo alto"). Detectado al validar contra el panel existente: el panel actual dividido por las 8 categorías sin TOTAL da **exactamente 1.000 en todos los años**.
- Servicio/Ingresos **correcto**: mín 0,023 · mediana 0,051 · máx 0,153. Las otras 23 provincias: p5 0,003 · p50 0,037 · p95 0,140 · máx 0,253. CABA es apenas superior a la mediana y dentro del rango del resto.
- **Implementado:** `data_processing.cargar_servicio_caba_millones()` lee la hoja `Millones`, excluye `TOTAL` y **falla si el cociente con la hoja en miles no es exactamente 1.000**.
- **Consecuencias medidas:** IFFS-A, B y C pasan a 503 observaciones (CABA ya no es "provisional"). Los AUC de 2019 pasan de A 1,00 / B 0,95 / C 1,00 (23 provincias) a **A 0,94 / B 0,93 / C 0,94 (24 provincias)**; el AUC de Servicio/Ingresos baja de 0,98 a 0,91, porque CABA tuvo un servicio alto en 2019 (12,8% de sus ingresos) y **no** reestructuró (falso positivo). (Medido solo tras corregir CABA; el resultado final con ambas correcciones está en H2 y en el notebook: correlaciones A–B 0,79, A–C 0,99, B–C 0,77; cargas PCA deuda 0,69, servicio 0,70, rigidez 0,12, resultado −0,15; varianza explicada 47%.)

### H2 · La Pampa — stock de deuda  [EMPÍRICO + SUPUESTO]
- Evidencia (`A1`, `A2`, `C2`):
  - Deuda/Ingresos = 165,7x (2005) → 23,4x (2012) → 5,4x (2015) → **26,3x (2016)** → 1,18x (2017) → **0,036x (2018)** → ~0 después.
  - Cargada como millones, La Pampa = 6,1x la suma de las otras 23 en 1996, 2,7x en 2005, 1,0x en 2012. Imposible.
  - Único caso del panel con un salto de escala de stock > 10x respecto del comportamiento mediano (2017 y 2018).
- Los 24 archivos de stock dicen "En Millones" y "In Thousands" a la vez (en el de La Pampa el bloque en inglés incluso dice "SALTA": parece una plantilla heredada), así que **el encabezado no permite decidir**. La propia auditoría original documenta que la DNAP cambió de unidad ("miles hasta 2013, millones desde 2014") en sus consolidados.
- **Hipótesis [SUPUESTO]:** el archivo de La Pampa está en miles hasta 2017 y en millones desde 2018. Bajo esa hipótesis Deuda/Ingresos pasa de 0,166 (2005) a 0,0012 (2017): un desendeudamiento suave, plausible para una provincia con muy baja deuda.
- **Resolución de la ambigüedad 2018+ [EMPÍRICO]:** había dos hipótesis compatibles: (a) 2018+ en millones, o (b) todo en miles. La foto nacional `stock_deuda_provincial_mar2026` (Drive) **está en millones** (total 32,6 millones de M$ vs. 33,8 millones de M$ del panel 2025, dentro de 4%) y trae a La Pampa en **500**, igual que el panel 2024. Con eso **(b) queda descartada** y 2018+ está en millones.
- **Lo que sigue sin confirmarse oficialmente:** que el cambio ocurra exactamente entre 2017 y 2018 y que el valor 2016 de Gobierno Nacional (536.440, miles) sea correcto. Una respuesta de un asistente de IA que "confirmó" la hipótesis **no cuenta como evidencia**: reprodujo mis propias cifras, citó como fuente un sitio de radio y se contradijo (dijo que hasta 2017 era en miles y a la vez que el GN de 2017 estaba en millones). La confirmación válida es la de la DNAP o la cuenta de inversión de La Pampa.
- **Implementado:** `data_processing._corregir_stock_la_pampa()` divide por 1.000 el stock 2005-2017 y deja la columna `stock_corregido_unidades`; la app muestra el aviso de corrección. Reversible con una constante.
- **Efecto medido:** el techo de la escala baja de 165,7x a 2,14x; los percentiles del resto de las provincias cambian **3,4 puntos en promedio (máx 4,2)** con la corrección. La Pampa pasa de ser el máximo de la escala a un IFFS-B estable en torno a 19-21 (baja fragilidad), coherente con su perfil desendeudado.

### Otros saltos de escala pendientes de revisión  [LIMITACIÓN]
Servicio: La Pampa 2024 (1.140 → 85), Santa Cruz 2025 (38.658 → 2.636), San Luis 2014 (4,4 → 0,35) y 2022 (507 → 69). Pueden ser amortizaciones *bullet* reales o problemas de unidad. **No los corregí ni los descarté.**

### H3 · Ventana base no estacionaria  [EMPÍRICO]
| Año | Mediana Deuda/Ingresos | Percentil mediano | % bajo P33 | % sobre P66 |
|---|---|---|---|---|
| 2005 | 1,09 | 84,6 | 12,5 | 62,5 |
| 2012 | 0,29 | 34,4 | 50,0 | 0,0 |
| 2019 | 0,38 | 43,8 | 45,8 | 12,5 |
| 2024 | 0,15 | 19,8 | 83,3 | 0,0 |
| 2025 | 0,16 | 19,8 | 82,6 | 0,0 |

*(datos con CABA y La Pampa corregidas)*

**[ESTADÍSTICO]** La ventana 2005-2012 incluye el desendeudamiento posterior a la crisis de 2001-2002 (la razón mediana cae de 1,09 a 0,29 *dentro* de la propia ventana). Normalizar contra ella fija como "referencia" un régimen que ya no existe.
**[ECONÓMICO]** Parte de la caída posterior es licuación inflacionaria de deuda en pesos, no mejora de solvencia real; el percentil no distingue ambas cosas.

### H4 · Desajuste stock/flujo  [SUPUESTO sobre la magnitud]
Deuda a 31/12 (precios de fin de año) sobre ingresos del año (precios medios). Con inflación π, la razón queda sobrestimada por ~(1 + π/2). Usando como *proxy* el crecimiento nominal mediano de ingresos (**no es inflación medida**): factor 1,22 (2019), 1,63 (2023), 1,91 (2024), 1,22 (2025). Comparar con una base 2005-2012 de inflación ~25% mezcla un efecto contable con uno real. (`G1_desajuste_stock_flujo.csv`)

### H7 · Denominadores  [EMPÍRICO]
Recursos de capital: mediana 4,3% de los ingresos totales (p95 15%). Resultado/ingresos totales vs. corrientes: diferencia media 6% del valor absoluto, ρ de rangos ≈ 1,00. Personal/Gasto total vs. Personal/Ingresos: ρ = 0,90.
**Conclusión honesta:** lo señalé como posible falla y **no tiene efecto material**. Se unifica por consistencia, no por urgencia.

### H8 · Valuación cambiaria  [ESTADÍSTICO]
Regresión Δlog(Deuda/Ingresos) sobre (participación USD(t−1) × Δlog TC) con efectos fijos de provincia y año, EE agrupados: coef. 0,47, EE 0,38, p = 0,215, n = 77 (11 provincias × 2015-2020). El signo es el esperado, pero **con 11 provincias y 6 años no puede afirmarse ni descartarse** un traspaso 1 a 1.

### Faltantes y ceros  [EMPÍRICO]
- Servicio de CABA: 100% faltante (recuperable, H1). Composición por moneda: 11 de 24 provincias y 7 de 21 años. Resto: 100% de cobertura.
- 1 observación con ingresos corrientes = 0 (La Pampa 2025): dato no reportado codificado como cero. **[DISEÑO]** debe tratarse como NA.
- **MNAR:** los faltantes no son aleatorios. CABA falta por un problema de procesamiento; la moneda falta por diseño de la fuente secundaria (solo provincias con títulos en el paper de Bazza et al.). No es imputable con un modelo.

---

## 2. Variables

Persistencia y componente estructural (`B1_propiedades_variables.csv`). ICC = parte de la varianza que es diferencia **permanente entre provincias**.

| Variable | Mediana | Asimetría (sin La Pampa) | ICC | AR1 intra | Lectura |
|---|---|---|---|---|---|
| Deuda / Ingresos | 0,30 | 1,7 (antes de corregir La Pampa: 10,6) | 0,32 | 0,90 | Tendencia lenta y **común**; casi no es diferencia permanente entre provincias |
| Servicio / Ingresos | 0,038 | 1,2 | 0,28 | 0,81 | Persistente, "lumpy" por amortizaciones |
| Personal / Gasto total | 0,45 | 0,0 | 0,64 | 0,79 | Predominantemente estructural |
| −Resultado / Ingresos | 0,004 | −0,1 | **0,22** | **0,36** | **Coyuntural y ruidoso** |
| Origen nacional / Ingresos | 0,59 | −0,6 | **0,97** | 0,65 | Rasgo institucional casi fijo |
| IFFS-B | 47,5 | −0,1 | 0,63 | 0,65 | ~63% estructural |

**Implicancia [ESTADÍSTICO]:** la distinción "estructural vs. coyuntural" **es medible** y hoy el IFFS-B las mezcla. La dependencia de transferencias es casi puramente estructural: no puede servir para detectar deterioro temporal, sí para caracterizar la provincia.

Redundancia (datos corregidos): Spearman deuda-servicio **0,80** (pooled), **0,89** (entre provincias), 0,71 (intra-provincia). Están **fuertemente relacionadas, no son idénticas**: entre provincias casi ordenan igual, y en el tiempo comparten ~50% de rango. (Con los datos sin corregir medí 0,70; ese número estaba deprimido por los dos errores.) La correlación 0,99 entre IFFS-A e IFFS-C se debe a que el PCA asigna pesos casi idénticos a ambas (0,69 y 0,70).

### Ficha resumida por variable
| Variable | Qué mide | Dirección | Riesgos principales |
|---|---|---|---|
| Deuda/Ingresos | Solvencia (stock vs. capacidad de pago) | ↑ más frágil | Stock a 31/12 vs. flujo anual (H4); excluye deuda flotante **[LIMITACIÓN]**; incluye deuda en USD valuada al TC de cierre; error de unidades en La Pampa |
| Servicio/Ingresos | Presión de caja (amortización + interés) | ↑ | Mezcla interés (costo) con amortización (perfil de vencimientos); *bullets*; incluye refinanciaciones |
| Personal/Gasto total | Rigidez del gasto | ↑ (**ambigua [ECONÓMICO]**) | Alto peso del personal puede reflejar provisión de servicios (educación, salud, seguridad) y no rigidez; no es endógena a la fragilidad pero sí a la escala de la provincia |
| −Resultado/Ingresos totales | Desempeño fiscal | ↑ déficit | Volátil (ICC 0,22); sensible a ingresos extraordinarios y a la contabilización de capital; manipulable por diferimiento de pagos (deuda flotante) |
| Origen nacional/Ingresos | Dependencia | **Ambigua** | Alta dependencia puede estabilizar (coparticipación) o exponer (discrecionales); no se separa automáticas de discrecionales con los datos actuales |

---

## 3. Código

| Severidad | Hallazgo |
|---|---|
| **Alta** | `data_processing.load_deuda_ratios` anula el servicio cuando `servicio_confiable=False`. Oculta el error H1 en vez de exponerlo. |
| **Alta** | El pipeline de origen (`scripts/10`, `11`) lee `sheet_name=0` sin validar unidades ni el nombre de la hoja. Sin chequeo de unidades por serie. |
| Media | Ingresos corrientes = 0 (La Pampa 2025) entra al panel como dato válido. |
| Media | `descomponer_contribucion` muestra **percentiles**, no contribuciones al puntaje; presentarlo como "descomposición" es engañoso. La contribución real de cada dimensión en IFFS-B es P/4. |
| Media | Sin pruebas automáticas (`tests/` no existe). |
| Media | Los percentiles de la ventana base se calculan *dentro de muestra* (los años base se ordenan contra sí mismos): el puntaje medio en 2005-2012 es ≈ 50 por construcción. |
| Baja | Código muerto: `modeling.regla_base`, `clasificar_alerta`, `cargar_datos()` devuelve `panel_target` que la app no usa; `statsmodels` estaba en requirements sin uso en `src/`. |
| Baja | `_pca_primer_componente` anota `-> pd.Series` pero devuelve una tupla; `np.random.seed(42)` global al importar (efecto colateral). |
| Baja | En el simulador, el shock de ingresos mezcla `ingresos_corrientes` con un resultado sobre ingresos *totales*. Inocuo por H7. |
| **Reproducibilidad** | `data/raw/base_investigacion/` **no es crudo**: son paneles derivados. Los scripts 20-21 de la investigación usan rutas absolutas y `/tmp`. La cadena RAW → panel no es reproducible desde este repo. |

---

## 4. Lo que se puede y no se puede incorporar (Secciones 18-20 del brief)

| Componente | Estado | Nota |
|---|---|---|
| Deuda por acreedor (bonos, bancos, Nación, organismos, FFDP/FFFIR) | **Disponible** 1996-2025, 24 prov. | Permite % bonos, % Nación, % organismos. **No** da moneda ni plazos |
| Amortización e interés por separado | **Disponible** 2005-2025, por acreedor | Separa costo de la deuda de perfil de vencimientos (*realizado*, no futuro) |
| Vencimientos futuros, *duration*, tasas por instrumento, garantías, deuda flotante, deuda intra-sector | **No disponible** | No se inventan. Solo consolidado nacional para tasas |
| Moneda de la deuda | 11 de 24 provincias, 1996-2002 y 2014-2020 | Fuente secundaria; insuficiente para un índice de 24 |
| Ingresos de origen nacional | Disponible; **sin** separar automáticas de discrecionales | La separación existe en una Google Sheet (`ron_automaticas_vs_otras`) que no puedo leer desde aquí; requiere exportar a CSV |
| Capacidad de respuesta (liquidez, flexibilidad) | Disponible 2005-2025 (BCRA) | No incluida en el IFFS actual |
| Ratings, PBG, regalías, IPC INDEC | Existen en la carpeta de investigación (algunas como Google Sheets) | Sin incorporar |

---

## 5. Base de eventos  [SUPUESTO]

12 de los 13 eventos figuran en la fuente como "2020-2021" sin año por provincia. **Los anclé en 2020 por convención mía**; no es un dato. Todo análisis de *timing* (event study, rezagos T−k) hereda ese supuesto y **debe reportarse con esa salvedad** o con ambos anclajes (2020 y 2021) como sensibilidad. Los 13 eventos son, además, esencialmente **un** shock común (COVID) observado en 12 unidades, no 12 observaciones independientes.

---

## 6. Qué estaba equivocado en lo que dije o publiqué antes

1. "CABA no tiene servicio confiable" → **falso**: error de unidades del pipeline (H1).
2. "La Pampa es un outlier extremo (165,7x) que estira la escala" → **error de datos**, no un fenómeno económico (H2). El percentil 100 no representa "un episodio", representa un dato mal cargado.
3. "Deuda y servicio miden casi lo mismo" → con datos corregidos ρ = 0,80 (0,89 entre provincias). Primero lo "corregí" a 0,70 con datos aún erróneos; la formulación original ("muy relacionadas") era la razonable (H6).
4. "Los datos de 2025 pueden ser preliminares" → **no verificado**; los ingresos 2025 crecen de forma coherente con un año completo. Retiro la insinuación.
5. El "AUC 1,00" de IFFS-A/C se calculó sobre 23 provincias (12 vs. 11): CABA quedó fuera **por el error de unidades H1**. Con CABA incluida el AUC es 0,94 (medido); la supuesta ventaja de A sobre B era un artefacto.
6. En mi primera cifra corregida de CABA dupliqué el servicio (ver H1). Corregido.

---

## 7. Supuestos que podrían estar equivocados

- Que el cambio de unidad de La Pampa ocurre entre 2017 y 2018 (H2): **no verificado**.
- Que el proxy de inflación (crecimiento mediano de ingresos) aproxima la inflación (H4).
- Que la Nota de eventos "2020-2021" pueda anclarse en 2020 (sección 5).
- Que el servicio de CABA en millones sea comparable con el de las demás (bullets).
- Que el resto de los archivos de la DNAP no tenga errores de unidad parciales que el filtro de 10x no detecta (un error de 3x-9x pasaría inadvertido).

## 8. Pendiente de la auditoría (no hecho en esta fase)
- Búsqueda bibliográfica y de fuentes oficiales (Sección 31).
- Confirmar unidades de La Pampa contra una fuente primaria y revisar los otros saltos de servicio.
- Revisión de la cadena RAW → panel (requiere los `.xls` originales en el repo).

---

## 9. Fase 2 (2026-09-25) — saltos de servicio y evidencia interna sobre las unidades
`python auditoria/fase2_servicio_y_unidades.py` → `outputs/auditoria/I1-I3`. Dos pruebas que no necesitan fuentes externas: **identidad stock-flujo** (la amortización pagada debe ser del orden de la caída del stock) y **tasa de interés implícita** (intereses / stock del año previo).
- **[EVIDENCIA]** En las otras 23 jurisdicciones (460 observaciones) amortización/stock previo va de 0,016 (p5) a 0,33 (p95) y la tasa implícita de 0,3% a 21%: **ninguna** observación es compatible con un error x1.000. También valida la corrección de CABA (en el panel de servicio original CABA daba cocientes de 50 a 600).
- **Santa Cruz 2024-25:** el Fondo Fiduciario de Desarrollo Provincial pasa de 18.922 a 0 en 2024 y la amortización de ese año (36.472) lo cancela: es un *bullet* real; el 2025 es su consecuencia. **Resuelto.**
- **San Luis 2014 y 2022:** 2019-2022 la amortización a Nación coincide con la caída del stock con diferencia 0,000 (no hay unidad distinta); el "salto" 2014 es sobre una base de 0,35 M$ (0,004% de los ingresos). **Resuelto.**
- **La Pampa 2024:** stock = 500 (cifra redonda) con una caída de 1.742 y amortización de 82: no cierra. **No verificable**, pero con el stock 2023 (2.242) La Pampa sigue en el puesto 24 de 24 y ningún otro puesto cambia.
- **La Pampa 2005-2017 [EVIDENCIA interna, NO confirmación oficial]:** con el stock en miles la tasa implícita es 4-7% (2006-2015; mediana 4,6%); con el stock como viene, 0,005%. Los stocks 2016 (555.836) y 2017 (33.436) son inconsistentes con los intereses de 2017 (108) y 2018 (161) bajo **cualquier** unidad. Refuerza la hipótesis de miles hasta 2015, pero **el tratamiento principal (faltante) no cambia** hasta contar con la confirmación de la DNAP.
