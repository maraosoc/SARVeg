# baseline_lightgbm.py
# ------------------------------------------------------------
# LightGBM Regressor con optimización de hiperparámetros
# Usando conjuntos de entrenamiento y validación ya listos.
# ------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.model_prep import ensure_outdir
from utils.evaluation import evaluate_regression, plot_diagnostics

import lightgbm as lgb

# ------------------------ Configuración ------------------------
@dataclass
class LGBMConfig:
    X_train: pd.DataFrame  # Conjunto de entrenamiento
    y_train: pd.Series      # Objetivo de entrenamiento
    X_val: pd.DataFrame      # Conjunto de validación
    y_val: pd.Series        # Objetivo de validación
    target_col: str                         # Columna objetivo (p.ej. 'biomasa')
    param_dist: Optional[Dict[str, list]] = None  # Parámetros de búsqueda para RandomizedSearchCV
    num_features: List[str] = None  # Características numéricas
    outdir: str = "./artifacts_lgbm"  # Directorio de salida


def build_preprocessor(num_features: List[str]) -> ColumnTransformer:
    """Imputación + escalado para todas las variables numéricas."""
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))#,
        #("scaler",  StandardScaler())
    ])
    pre = ColumnTransformer(
        transformers=[("num", num_pipe, num_features)],
        remainder="drop"
    )
    return pre


# ------------------------ LightGBM con optimización de hiperparámetros ------------------------

def optimize_lgbm_hyperparameters(
    pre: ColumnTransformer,
    X_train: pd.DataFrame, y_train: pd.Series,
    param_dist: Dict[str, list]=None,
    n_iter_search: int = 100,
    random_state: int = 42
) -> RandomizedSearchCV:
    """
    Optimiza hiperparámetros de LightGBM usando RandomizedSearchCV con parámetros
    personalizados proporcionados por el usuario.

    Parámetros:
    -----------
    pre : ColumnTransformer
        Preprocesador (ajustado en los datos de entrenamiento).
    X_train : pd.DataFrame
        Conjunto de entrenamiento.
    y_train : pd.Series
        Etiquetas para el conjunto de entrenamiento.
    param_dist : dict
        Diccionario de parámetros de búsqueda. Claves: nombres de hiperparámetros, 
        valores: lista de valores a probar.
    n_iter_search : int
        Número de combinaciones aleatorias a probar (por defecto 100).
    random_state : int
        Semilla para reproducibilidad (por defecto 42).

    Retorna:
    --------
    random_search : RandomizedSearchCV
        Objeto con la búsqueda aleatoria realizada, incluyendo el mejor modelo encontrado.
    """
    # 1) Ajustar preprocesador
    pre_fit = pre.fit(X_train, y_train)
    Xtr = pre_fit.transform(X_train)

    # 2) Definir el modelo
    lgbm = lgb.LGBMRegressor(random_state=random_state)

    # 3) Definir el espacio de búsqueda
    if param_dist is None:
        param_dist = {
            "n_estimators": np.arange(100, 1500, 100),
            "max_depth": [None, 10, 20, 30, 40, 50],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_samples": [5, 10, 20]
            }

    # 3) Realizar búsqueda aleatoria con los parámetros proporcionados
    random_search = RandomizedSearchCV(
        lgbm, param_distributions=param_dist,
        n_iter=n_iter_search, cv=5, n_jobs=-1, verbose=2, random_state=random_state
    )

    # 4) Entrenar el modelo con la búsqueda aleatoria
    random_search.fit(Xtr, y_train)
    
    return random_search


def run_lgbm_from_splits(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    target_col: str,
    param_dist: Optional[Dict[str, list]],
    num_features: List[str],
    outdir: str = "./artifacts_lgbm"
) -> Dict[str, Dict[str, float]]:
    """
    Entrena un modelo LightGBM sobre TRAIN y evalúa sobre VALID.
    Incluye optimización de hiperparámetros usando RandomizedSearchCV.
    """
    outdir = ensure_outdir(outdir)

    # 1) Preprocesador (igual que antes)
    pre = build_preprocessor(num_features)

    # 2) Optimización de hiperparámetros (RandomizedSearchCV)
    random_search = optimize_lgbm_hyperparameters(
        pre=pre, X_train=X_train, y_train=y_train, param_dist=param_dist, n_iter_search=100
    )

    # 3) Mejor modelo
    lgbm_best = random_search.best_estimator_

    # 4) Predicciones en TRAIN y VALID
    yhat_train = lgbm_best.predict(X_train)
    yhat_val   = lgbm_best.predict(X_val)

    # 5) Métricas
    metrics = {
        "train": evaluate_regression(y_train.values, yhat_train),
        "val":   evaluate_regression(y_val.values,   yhat_val)
    }
    pd.DataFrame(metrics).to_csv(outdir / f"metrics_{target_col} LightGBM.csv")

    # 6) Gráficos diagnósticos (idénticos)
    plot_diagnostics(y_train.values, yhat_train, outdir, f"train set - LGBM",  target_col)
    plot_diagnostics(y_val.values,   yhat_val,   outdir, f"val set - LGBM", target_col)

    # 7) Importancia de características
    feature_importances = lgbm_best.feature_importances_
    feature_df = pd.DataFrame({
        "feature": num_features,
        "importance": feature_importances
    }).sort_values("importance", ascending=False)
    feature_df.to_csv(outdir / f"lgbm_feature_importance_{target_col}.csv", index=False)

    # 8) Guardar el modelo entrenado
    import joblib
    dir_pkl = pathlib.Path(outdir) / "pkl"
    dir_pkl.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgbm_best, dir_pkl / f"lgbm_best_model_{target_col}.pkl")

    return metrics


# ------------------------ Ejemplo de uso ------------------------
if __name__ == "__main__":
    # Diccionario de parámetros para LightGBM
    cfg = LGBMConfig(
        X_train=None,
        y_train=None,
        X_val=None,
        y_val=None,
        target_col="biomass",  # Cambia según tu caso
        param_dist=None,  # Puedes definir tu rejilla de hiperparámetros
        num_features=None,  # Lista de características numéricas
        outdir="./artifacts_rf_example"  # Directorio de salida
    )
    # Llama a la función principal
    m = run_lgbm_from_splits(cfg)
    print("Métricas (RMSE, MAE, R2, rRMSE%) por split:")
    for split_name, metrics in m.items():
        print(split_name, metrics)

