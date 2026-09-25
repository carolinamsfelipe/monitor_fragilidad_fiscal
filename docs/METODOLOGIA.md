# Metodología del IFFS v2

Etiquetas: **[EVIDENCIA]** calculado en los datos · **[ECONÓMICO]** interpretación económica · **[SUPUESTO]** no verificado ·
**[DISEÑO]** decisión de construcción · **[LIMITACIÓN]** lo que no se puede resolver con los datos actuales.
Las cifras exactas están en `outputs/resultados_clave.json` y `outputs/tables/` (las genera `scripts/generar_resultados.py`); este documento las
cita solo cuando son necesarias y siempre coinciden con esa fuente.

## 1. Qué es y qué NO es
- **[DISEÑO]** Un conjunto de **cuatro subíndices** (solvencia, liquidez, estructural, coyuntural) y un **compuesto** de tres de ellos, para las 24 jurisdicciones, 2005-2025.
- **[DISEÑO]** Todos son **posiciones relativas** (percentil 0-100 entre las jurisdicciones de un mismo año). **No son probabilidades de default**, no miden solvencia absoluta y no tienen umbral de "alerta" calibrado.
- **[LIMITACIÓN]** No se afirma capacidad predictiva: hay 13 eventos y 12 son el mismo shock (ver §7).

## 2. Datos y tratamientos (herencia de la auditoría Fase 1)
| Tema | Tratamiento | Tipo |
|---|---|---|
| CABA, servicio de deuda | Se lee la hoja `Millones` de la DNAP, sin la columna `TOTAL`; el código **falla** si el cociente con la hoja en miles no es exactamente 1.000. | [EVIDENCIA] error de unidades del pipeline original, corregido |
| La Pampa, stock 2005-2017 | Posible error de unidades **sin confirmación oficial**. Se trata como **faltante** en el análisis principal; el valor original se conserva en `deuda_stock_total_fuente` y se marca en `stock_estado`. La corrección ÷1.000 existe solo como **escenario de sensibilidad** (`la_pampa="sensibilidad_miles"`). | [DISEÑO] + [SUPUESTO] (la hipótesis de miles) |
| Ejecución presupuestaria = 0 (La Pampa 2025) | Dato no reportado → faltante. | [DISEÑO] |
| Grilla | Panel completo 24 × 21 = 504 filas; los faltantes son explícitos, ninguno se imputa. | [DISEÑO] |
| Saltos de servicio (Santa Cruz 2024-25, San Luis 2014 y 2022, La Pampa 2024) | **Resueltos con la identidad stock-flujo (Fase 2, `auditoria/fase2_servicio_y_unidades.py`)**: Santa Cruz 2024 es la cancelación real del FFDP (stock 18.922 → 0; amortización 36.472) y 2025 su consecuencia; en San Luis la amortización coincide *exactamente* con la caída del stock en 2019-2022 y el "salto" de 2014 es una variación sobre una base ínfima (0,35 M$); ninguna de las 460 observaciones de las otras 23 jurisdicciones es compatible con un error de unidades. La Pampa 2024 (stock = 500, cifra redonda, sin cierre con la amortización) no es verificable, pero **no cambia ningún puesto**. Se conservan todos los datos. | [EVIDENCIA] + [LIMITACIÓN] (La Pampa 2024) |
| Evidencia interna sobre La Pampa 2005-2017 (Fase 2) | Los intereses pagados implican una tasa de ~4-7% anual sobre el stock **solo si el stock está en miles** (con el stock "como viene", 0,005%; el rango del resto de las provincias es 0,3%-21%). Vale para 2005-2016; los stocks 2016-2017 (555.836 y 33.436) son inconsistentes con los intereses de 2017-2018 bajo **cualquier** unidad. Es evidencia interna a favor de la hipótesis de miles, **no una confirmación oficial**: el tratamiento principal no cambia. | [EVIDENCIA] + [SUPUESTO] |
| Eventos | 13; 12 con año ancla 2020 (convención propia para 11 de ellos, cuya fuente dice "2020-2021"); 1 en 2025 (La Rioja). Se reportan ambos anclajes (2020 y 2021). | [SUPUESTO] |

## 3. ¿Un índice único o subíndices? (decisión: subíndices + compuesto parcial)
Criterios **sin usar eventos** (`validation.py`: `propiedades_variables`, `redundancia_subindices`, `pca_subindices`, `criterio_ventanas_y_composicion`):
1. **[EVIDENCIA]** Las variables tienen dinámicas muy distintas: la rigidez del gasto es en gran parte un rasgo permanente (ICC 0,64); el resultado fiscal es 78% transitorio (ICC 0,22; AR(1) intra-provincia 0,36). Promediarlas mezcla una señal lenta con ruido.
2. **[EVIDENCIA]** No son intercambiables: solvencia y liquidez se parecen mucho entre provincias (ρ ≈ 0,9) pero menos en el tiempo (ρ ≈ 0,7); la rigidez se correlaciona *negativamente* con solvencia y liquidez dentro de cada provincia. Un solo factor explica ~54% de la varianza de los cuatro subíndices: un número resume, pero pierde información.
3. **[ECONÓMICO]** Cada dimensión implica una política distinta (reperfilar deuda ≠ ajustar gasto ≠ cubrir un déficit transitorio).
4. **Regla ex ante** (umbral 0,85 de persistencia de rangos t→t+1, fijado antes de mirar eventos): entra al compuesto una dimensión cuya forma anual ya es persistente. Resultado **[EVIDENCIA]**: solvencia, liquidez y estructural entran; el resultado fiscal (persistencia 0,48 a 1 año) **no**, y se reporta como subíndice coyuntural aparte.
- **[DISEÑO]** Compuesto = media simple de solvencia, liquidez y estructural. El compuesto con las 4 dimensiones se evalúa como alternativa.
- **[ECONÓMICO]** *Ambigüedad*: una alta proporción de gasto en personal puede reflejar provisión de servicios (educación, salud, seguridad) y no rigidez. Por eso se evalúa un compuesto sin esa dimensión.
- Se evaluó y **descartó** usar los depósitos del BCRA como "liquidez fiscal": son depósitos del sistema bancario de cada provincia, no caja del Tesoro provincial **[LIMITACIÓN]**.

## 4. Normalización
Alternativas evaluadas (todas en `indicators.py`): percentil dentro de cada año (**principal**), percentil contra ventana base fija (2005-2012 = IFFS v1; y 2005-2019), ventana móvil sin mirar adelante (5 y 8 años), percentil "pooled", z-score con base 2005-2012 y z-score robusto (mediana/MAD, recorte ±3).
Criterios (sin eventos) y **[EVIDENCIA]**:
1. **Estacionariedad de la escala**: la mediana de Deuda/Ingresos cae de 1,14 (2005) a 0,31 (2012) y 0,15 (2024). Contra una base fija, el percentil mediano del compuesto deriva ~29 puntos a lo largo de los años; con el corte anual, ~8.
2. **Sin información futura**: el corte anual solo usa a los pares del mismo año; la base fija 2005-2012 usa datos "futuros" para los años previos.
3. **Robustez a errores de unidades**: un error como el de La Pampa sin tratar cambia el subíndice de solvencia del resto en ~2 puntos con percentiles y en ~22 con z-score.
4. **Interpretación**: el corte anual responde "¿qué tan frágil es respecto de sus pares ese año?", que es lo que un monitor comparativo puede afirmar.
- **[LIMITACIÓN]** Si todas las provincias mejoran o empeoran juntas, el score no lo muestra. Se compensa mostrando el **nivel común** (mediana de cada razón por año) junto al score.
- **[DISEÑO]** La "posición" es el **cuartil del compuesto dentro de cada año** (siempre hay 6 jurisdicciones en el cuartil superior). Reemplaza a los cortes 33/66/85 del v1, que eran arbitrarios y estaban anclados a 2005-2012.

## 5. Ventanas (suavizado)
- **[DISEÑO/EVIDENCIA]** Regla ex ante: la ventana más corta de {1, 3, 5} años con persistencia de rangos ≥ 0,85. Resultado: 1 año para solvencia, liquidez y estructural.
- **Rectificación registrada:** mi intuición previa era suavizar el servicio con media móvil de 3 años (por los *bullets*). La regla aplicada literalmente no lo respalda (la persistencia anual ya es 0,89), así que el principal **no suaviza**; la media móvil de 3 y 5 años queda como sensibilidad.
- **[LIMITACIÓN]** Sin suavizar, un vencimiento único (bullet) sube la liquidez un año.

## 6. Pesos, agregación y valores extremos
- **[DISEÑO]** Pesos iguales entre las tres dimensiones del compuesto. No se optimizan.
- Sensibilidad: pesos 50/25/25 en cada dimensión; 2.000 vectores de pesos aleatorios uniformes (Dirichlet); agregación geométrica, cuadrática y "eslabón débil" (máximo).
- **Valores extremos**: el percentil (ECDF/rango) es inmune por construcción a la magnitud de un outlier; los z-scores no (ver §4.3). No se recorta ninguna serie del principal.

## 7. Validación: descripción, discriminación, anticipación
| Pregunta | Prueba | Qué NO permite concluir |
|---|---|---|
| **Descripción** | cobertura, distribución, nivel común, cambios de ranking entre especificaciones, perfiles (k-means), anomalías | causalidad o riesgo |
| **Discriminación** | AUC por rangos en un año de referencia (2019 y T−1…T−5), IC 95% bootstrap por provincia, prueba de permutación, sorteo de grupos al azar, benchmark de "% de bonos en la deuda", AUC solo entre emisoras de bonos | que el score *anticipe*: un nivel alto puede ser un rasgo permanente del grupo |
| **Anticipación** | pretendencia (cambio T−5→T−1, evento vs. resto), estudio de eventos τ = −8…+3, placebo temporal (AUC del mismo grupo en todos los años) | efectos escalonados: hay una sola fecha común |
- **Pruebas no viables** (con la razón calculada, `outputs/tables/T13_pruebas_no_viables.csv`): modelo supervisado con corte temporal (0 eventos antes de 2020), probabilidad calibrada, optimización de umbrales/pesos, modelos de duración, estudio de eventos escalonado, validación fuera de muestra con La Rioja 2025 (1 caso), comparación con calificaciones (sin datos).
- **Comparaciones múltiples**: se reportan todas las medidas y especificaciones; no se corrigen p-valores porque nada se selecciona a partir de ellos. Todo es exploratorio.

## 8. Registro de decisiones (orden cronológico)
1. La Pampa: se **revierte** la corrección ÷1.000 que estaba publicada, por falta de confirmación oficial → faltante + sensibilidad.
2. Normalización: se cambia la base fija 2005-2012 por corte anual **antes** de calcular ningún AUC (criterios §4).
3. Composición: la regla de persistencia excluye al resultado fiscal del compuesto (§3).
4. Suavizado: la regla contradice la intuición inicial (3 años) y prevalece la regla (§5).
5. Los AUC de eventos se calcularon **después** de fijar la especificación principal; se reportan las 18 especificaciones (`T19_...csv`), incluidas las que dan menor AUC.

## 9. Limitaciones que persisten
- La Pampa 2005-2017 sin confirmar (y 2025 sin ejecución): 14 de 504 observaciones sin compuesto.
- Un score relativo no distingue "todas mejoran" de "nadie cambia" (mitigado con el nivel común).
- Stock a 31/12 sobre flujo anual: la razón Deuda/Ingresos queda sobrestimada con inflación alta (proxy, no medida).
- Deuda flotante, garantías, vencimientos futuros y moneda de la deuda (solo 11 de 24 provincias) no están en el índice.
- Solo La Pampa 2024 (stock 500) sigue sin verificarse; no cambia ningún puesto.
- Los paneles de `data/raw/base_investigacion/` son derivados; los originales de la DNAP no están en el repositorio (salvo CABA y La Pampa).
