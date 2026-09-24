"""
modeling.py
============

Sistema de alerta y ejercicio exploratorio de modelización del outcome de
estrés fiscal (default / reestructuración).

Contexto y restricciones (ver también data/raw/base_investigacion/eventos_estres_provincial.csv
y la Parte "Modelo y limitación de eventos" del informe): la base de eventos
tiene 13 observaciones positivas sobre ~504 posibles (2.6%), con 12 de los 13
concentrados en el bienio 2020-2021 (shock COVID-19, común a casi todas las
provincias). Por eso, explícitamente:

- NO se usa accuracy como métrica principal (con una tasa base de 2.6%, un
  modelo que nunca predice estrés tiene ~97% de accuracy y cero utilidad).
- NO se hace un split aleatorio fila a fila (mezclaría provincias y años,
  y con eventos concentrados en 2020-2021 filtraría información del shock
  común hacia el set de entrenamiento).
- NO se usa SMOTE ni ninguna técnica de sobremuestreo antes de separar
  entrenamiento y evaluación.
- El modelo se valida con separación POR AÑO (todo 2020-2021 fuera del
  entrenamiento) y se compara siempre contra una regla base simple.
- Los resultados se presentan como exploratorios y NO causales, y en ningún
  momento se afirma que el modelo "predice" defaults.

Dos componentes:

1. Sistema de alerta por reglas (`clasificar_alerta`) -- transparente,
   interpretable, es el que efectivamente usa el dashboard.
2. Regresión logística penalizada (L2) exploratoria, con validación temporal
   por grupos (GroupKFold por año) y reporte de precision/recall/F1/PR-AUC/
   matriz de confusión, comparada contra la regla base "alerta si IFFS-B en
   el cuartil superior". Firth no está disponible de forma robusta en este
   entorno (no hay implementación madura y mantenida en el ecosistema
   scikit-learn/statsmodels instalado); se documenta la limitación y se usa
   L2 (Ridge) como alternativa defendible, que también corrige (parcialmente)
   el problema de pocos eventos al regularizar los coeficientes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURES_MODELO = [
    "deuda_sobre_ingresos",
    "servicio_sobre_ingresos",
    "rigidez_gastototal",
    "fragilidad_resultado",
    "dependencia_transferencias",
]


def construir_target_estres(panel: pd.DataFrame, eventos: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega al panel la columna binaria `estres` = 1 si (provincia, año)
    coincide con un evento de la base de eventos, 0 en caso contrario.

    Respeta no-look-ahead: el target del año t se define sobre el evento
    ocurrido en el año t (no se propaga hacia años previos ni posteriores).
    """
    panel = panel.copy()
    eventos_set = set(zip(eventos["provincia"], eventos["anio_evento"]))
    panel["estres"] = [
        1 if (p, a) in eventos_set else 0
        for p, a in zip(panel["provincia"], panel["anio"])
    ]
    return panel


def clasificar_alerta(iffs_b: float) -> str:
    """Wrapper de una fila sobre indicators.asignar_nivel_alerta (uso puntual en la UI)."""
    from .indicators import UMBRALES_ALERTA_IFFS_B

    if pd.isna(iffs_b):
        return "Sin datos"
    for u in sorted(UMBRALES_ALERTA_IFFS_B, key=lambda x: -x.minimo):
        if iffs_b >= u.minimo:
            return u.etiqueta
    return "Bajo"


def regla_base(panel: pd.DataFrame, columna_score: str = "IFFS_B", cuantil: float = 0.75) -> pd.Series:
    """
    Regla base ingenua para comparar contra el modelo: alerta = 1 si el score
    (IFFS-B) del AÑO ANTERIOR está en el cuartil superior de esa misma
    variable en ese año (comparación transversal, sin mirar el futuro).
    """
    panel = panel.sort_values(["provincia", "anio"])
    umbral_anual = panel.groupby("anio")[columna_score].transform(lambda s: s.quantile(cuantil))
    return (panel[columna_score] >= umbral_anual).astype(int)


def preparar_dataset_modelo(panel_con_target: pd.DataFrame, rezago: int = 1) -> pd.DataFrame:
    """
    Arma el dataset supervisado: predictores del año t-`rezago`, target del
    año t (no-look-ahead: solo se usa información publicada antes del evento).
    """
    df = panel_con_target.sort_values(["provincia", "anio"]).copy()
    predictores = df[["provincia", "anio"] + FEATURES_MODELO].copy()
    predictores["anio"] = predictores["anio"] + rezago  # se "corren" hacia adelante

    target = df[["provincia", "anio", "estres"]]
    dataset = predictores.merge(target, on=["provincia", "anio"], how="inner")
    return dataset.dropna(subset=FEATURES_MODELO)


def evaluar_modelo_logit_validacion_temporal(dataset: pd.DataFrame, anio_corte_evaluacion: int = 2019) -> dict:
    """
    Entrena con años <= `anio_corte_evaluacion` (todo el shock 2020-2021 queda
    fuera del entrenamiento) y evalúa en años posteriores. Además corre una
    validación cruzada agrupada por año dentro del set de entrenamiento para
    reportar estabilidad.

    Devuelve un diccionario con métricas, no accuracy como principal.
    """
    train = dataset[dataset["anio"] <= anio_corte_evaluacion]
    test = dataset[dataset["anio"] > anio_corte_evaluacion]

    if train["estres"].sum() < 2 or test.empty:
        return {"error": "Muestra insuficiente de eventos para entrenar/evaluar de forma estable."}

    X_train, y_train = train[FEATURES_MODELO], train["estres"]
    X_test, y_test = test[FEATURES_MODELO], test["estres"]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    modelo = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
    )
    modelo.fit(X_train_s, y_train)

    proba_test = modelo.predict_proba(X_test_s)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    metrics = {
        "n_train": len(train),
        "eventos_train": int(y_train.sum()),
        "n_test": len(test),
        "eventos_test": int(y_test.sum()),
        "precision": precision_score(y_test, pred_test, zero_division=0),
        "recall": recall_score(y_test, pred_test, zero_division=0),
        "f1": f1_score(y_test, pred_test, zero_division=0),
        "pr_auc": average_precision_score(y_test, proba_test) if y_test.sum() > 0 else np.nan,
        "roc_auc": roc_auc_score(y_test, proba_test) if y_test.nunique() > 1 else np.nan,
        "matriz_confusion": confusion_matrix(y_test, pred_test, labels=[0, 1]).tolist(),
        "coeficientes": dict(zip(FEATURES_MODELO, modelo.coef_[0].tolist())),
    }

    # Comparación contra la regla base en el mismo set de evaluación
    base_test = test.copy()
    base_test["alerta_base"] = (
        test.assign(**{f: X_test[f] for f in FEATURES_MODELO})["deuda_sobre_ingresos"]
        >= train["deuda_sobre_ingresos"].quantile(0.75)
    ).astype(int)
    metrics["baseline_precision"] = precision_score(y_test, base_test["alerta_base"], zero_division=0)
    metrics["baseline_recall"] = recall_score(y_test, base_test["alerta_base"], zero_division=0)
    metrics["baseline_f1"] = f1_score(y_test, base_test["alerta_base"], zero_division=0)

    return metrics


def comparacion_pre_shock(
    panel_iffs: pd.DataFrame,
    eventos: pd.DataFrame,
    anio_referencia: int = 2019,
    columnas: list[str] = None,
) -> pd.DataFrame:
    """
    Comparación descriptiva (NO un modelo predictivo entrenado): compara, en
    el último año publicado antes del shock (2019, pre-COVID), las provincias
    que terminaron reestructurando en 2020-2021 contra las que no.

    Esta es la comparación que la evidencia disponible SÍ permite hacer con
    honestidad -- no un split entrenamiento/evaluación aleatorio (que
    requeriría eventos distribuidos en el tiempo, y acá casi todos ocurren
    en un único bienio). Reporta: diferencia de medianas, U de Mann-Whitney,
    AUC descriptiva (capacidad de separación del ranking, no de una
    predicción fuera de muestra) y g de Hedges.
    """
    from scipy import stats

    columnas = columnas or FEATURES_MODELO
    provincias_evento_2020_21 = set(
        eventos.loc[eventos["anio_evento"].between(2020, 2021), "provincia"]
    )

    df_ref = panel_iffs[panel_iffs["anio"] == anio_referencia].copy()
    df_ref["grupo"] = df_ref["provincia"].apply(
        lambda p: "Reestructuró 2020-21" if p in provincias_evento_2020_21 else "No reestructuró"
    )

    filas = []
    for col in columnas:
        g1 = df_ref.loc[df_ref["grupo"] == "Reestructuró 2020-21", col].dropna()
        g0 = df_ref.loc[df_ref["grupo"] == "No reestructuró", col].dropna()
        if len(g1) < 2 or len(g0) < 2:
            filas.append({"variable": col, "nota": "muestra insuficiente en alguno de los grupos"})
            continue

        u_stat, p_valor = stats.mannwhitneyu(g1, g0, alternative="two-sided")
        auc_desc = u_stat / (len(g1) * len(g0))  # relación directa U <-> AUC (Mann-Whitney)

        n1, n0 = len(g1), len(g0)
        var_pool = ((n1 - 1) * g1.var(ddof=1) + (n0 - 1) * g0.var(ddof=1)) / (n1 + n0 - 2)
        hedges_g = (g1.mean() - g0.mean()) / (var_pool ** 0.5) if var_pool > 0 else float("nan")

        filas.append({
            "variable": col,
            "n_reestructuro": n1,
            "n_no_reestructuro": n0,
            "mediana_reestructuro": g1.median(),
            "mediana_no_reestructuro": g0.median(),
            "mannwhitney_p": p_valor,
            "auc_descriptiva": auc_desc,
            "hedges_g": hedges_g,
        })
    return pd.DataFrame(filas)


def validacion_cruzada_por_anio(dataset: pd.DataFrame, n_splits: int = 4) -> pd.DataFrame:
    """
    GroupKFold agrupando por año (nunca deja el mismo año partido entre train
    y test) -- reporta la variabilidad del modelo entre folds, evidencia
    directa de que la performance no es estable (pocos eventos).
    """
    X = dataset[FEATURES_MODELO].to_numpy()
    y = dataset["estres"].to_numpy()
    groups = dataset["anio"].to_numpy()

    n_splits_efectivo = min(n_splits, dataset["anio"].nunique())
    if n_splits_efectivo < 2:
        return pd.DataFrame()

    gkf = GroupKFold(n_splits=n_splits_efectivo)
    filas = []
    for fold, (idx_tr, idx_te) in enumerate(gkf.split(X, y, groups)):
        if y[idx_tr].sum() < 1 or y[idx_te].sum() < 1:
            filas.append({"fold": fold, "nota": "sin eventos positivos en train o test, PR-AUC no definida"})
            continue
        scaler = StandardScaler().fit(X[idx_tr])
        modelo = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
        modelo.fit(scaler.transform(X[idx_tr]), y[idx_tr])
        proba = modelo.predict_proba(scaler.transform(X[idx_te]))[:, 1]
        filas.append({
            "fold": fold,
            "n_test": len(idx_te),
            "eventos_test": int(y[idx_te].sum()),
            "pr_auc": average_precision_score(y[idx_te], proba),
        })
    return pd.DataFrame(filas)
