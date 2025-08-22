# baseline_linear.py
# ------------------------------------------------------------
# Línea base: regresión lineal (OLS / Ridge / Lasso) con split
# aleatorio en entrenamiento/validación y diagnóstico básico.
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from utils.model_prep import ensure_outdir
from utils.evaluation import evaluate_regression, plot_diagnostics


# ------------------------ Configuración ------------------------

@dataclass
class BaselineConfig:
    data_path: str                          # Ruta al CSV de entrada
    target_col: str                         # Columna objetivo (p.ej. 'biomasa')
    feature_cols: Optional[List[str]] = None# Si None, usa todas menos target
    val_size: float = 0.2                   # Tamaño del conjunto de validación
    random_state: int = 42                  # Semilla reproducible
    model_type: str = "ridge"               # {"ols","ridge","lasso"}
    # Hiperparámetros para Ridge/Lasso (puedes ajustar)
    alpha: float = 1.0
    # Si quieres guardar artefactos:
    outdir: str = "./artifacts_linear_baseline"


def build_preprocessor(num_features: List[str]) -> ColumnTransformer:
    """
    Preprocesamiento para features numéricas:
    - Imputación por mediana (evita fuga si más adelante haces CV)
    - Estandarización (útil para Ridge/Lasso; OLS no lo requiere, pero no le perjudica)
    """
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    pre = ColumnTransformer(
        transformers=[("num", num_pipe, num_features)],
        remainder="drop"
    )
    return pre

def build_model(model_type: str, alpha: float):
    """Devuelve el estimador de regresión seleccionado."""
    model_type = model_type.lower()
    if model_type == "ols":
        return LinearRegression()
    if model_type == "ridge":
        return Ridge(alpha=alpha, random_state=None)
    if model_type == "lasso":
        return Lasso(alpha=alpha, random_state=None, max_iter=10000)
    raise ValueError("model_type debe ser {'ols','ridge','lasso'}")

def build_pipeline(pre: ColumnTransformer, model) -> Pipeline:
    """Crea el Pipeline sklearn: preprocesamiento + modelo."""
    return Pipeline(steps=[("pre", pre), ("model", model)])

def run_baseline_from_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    target: Optional[str] = None,
    num_features: Optional[List[str]] = None,
    model_type: str = "ridge",
    alpha: float = 1.0,
    outdir: str = "./artifacts_linear_baseline"
) -> Dict[str, Dict[str, float]]:
    """
    Ejecuta la línea base de regresión lineal/Ridge/Lasso usando conjuntos
    ya preparados de entrenamiento, validación y prueba.

    Parámetros:
    -----------
    X_train, y_train, X_val, y_val, X_test, y_test : pd.DataFrame / pd.Series
        Conjuntos de datos ya divididos (listas de características y variable objetivo).
    num_features : list
        Lista con nombres de columnas numéricas.
    model_type : str
        'ols' | 'ridge' | 'lasso'.
    alpha : float
        Parámetro de regularización para Ridge/Lasso.
    outdir : str
        Carpeta donde se guardan métricas, coeficientes y gráficos.

    Retorna:
    --------
    metrics : dict
        Métricas por split: train, val, test.
    """

    outdir = ensure_outdir(outdir)

    # 1) Preprocesador
    pre = build_preprocessor(num_features)

    # 2) Modelo
    model = build_model(model_type, alpha)

    # 3) Pipeline
    pipe = build_pipeline(pre, model)

    # 4) Entrenamiento
    pipe.fit(X_train, y_train)

    # 5) Predicciones
    yhat_train = pipe.predict(X_train)
    yhat_val   = pipe.predict(X_val)

    # 6) Métricas
    metrics = {
        "train": evaluate_regression(y_train.values, yhat_train),
        "val":   evaluate_regression(y_val.values,   yhat_val),
    }

    # 7) Gráficos diagnósticos
    plot_diagnostics(y_train.values, yhat_train, outdir, f'train set - {model_type}', target)
    plot_diagnostics(y_val.values,   yhat_val,   outdir, f'val set - {model_type}', target)

    # 8) Coeficientes
    try:
        coefs = pipe.named_steps["model"].coef_
        coef_df = pd.DataFrame({
            "feature": num_features,
            "coef": coefs
        }).sort_values("coef", key=np.abs, ascending=False)
        coef_df.to_csv(outdir / f"coeficientes_{target} {model_type}.csv", index=False)
    except Exception as e:
        with open(outdir / f"coeficientes_{target} {model_type}.txt", "w") as f:
            f.write(f"No fue posible exportar coeficientes: {repr(e)}\n")

    # 9) Guardar métricas
    pd.DataFrame(metrics).to_csv(outdir / f"metrics_{target} {model_type}.csv")

    return metrics



# ------------------------ Ejecución de ejemplo ------------------------

if __name__ == "__main__":
    # Ajusta estos valores a tu caso:
    cfg = BaselineConfig(
        data_path="datos_sar_vegetacion.csv",  # <-- tu CSV
        target_col="biomasa",                  # <-- tu objetivo (p.ej., 'LAI', 'CWC', 'biomasa')
        target="biomasa",
        X_train=None,
        y_train=None,
        X_val=None,
        y_val=None,
        feature_cols=None,                     # usa todas menos la y (o pasa lista explícita)
        num_features=None,
        val_size=0.2,
        random_state=42,
        model_type="ridge",                    # 'ols' | 'ridge' | 'lasso'
        alpha=1.0,
        outdir="./artifacts_linear_baseline"
    )
    m = run_baseline_from_splits(cfg)
    print("Métricas (RMSE, MAE, R2, rRMSE%) por split:")
    for split, d in m.items():
        print(split, d)

