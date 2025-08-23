# baseline_random_forest.py
# ------------------------------------------------------------
# Random Forest Regressor con optimización de hiperparámetros
# Usando conjuntos de entrenamiento y validación ya listos.
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from utils.model_prep import ensure_outdir
from utils.evaluation import evaluate_regression, plot_diagnostics

# ------------------------ Configuración ------------------------
@dataclass
class RFConfig:
    X_train: pd.DataFrame  # Conjunto de entrenamiento
    y_train: pd.Series      # Objetivo de entrenamiento
    X_val: pd.DataFrame      # Conjunto de validación
    y_val: pd.Series        # Objetivo de validación
    target_col: str                         # Columna objetivo (p.ej. 'biomasa')
    param_dist: Optional[Dict[str, list]] = None  # Parámetros de búsqueda para RandomizedSearchCV
    num_features: List[str] = None  # Características numéricas
    outdir: str = "./artifacts_rf"  # Directorio de salida para arte

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

# ------------------------ Random Forest con optimización de hiperparámetros ------------------------

def optimize_rf_hyperparameters(
    pre: ColumnTransformer,
    X_train: pd.DataFrame, y_train: pd.Series,
    param_dist: Dict[str, list]= None,
    n_iter_search: int = 100,
    random_state: int = 42
) -> RandomizedSearchCV:
    """
    Optimiza hiperparámetros de RandomForest usando RandomizedSearchCV.
    """
    # 1) Preprocesamiento
    pre_fit = pre.fit(X_train, y_train)
    Xtr = pre_fit.transform(X_train)

    # 2) Definir el modelo
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)

    # 3) Definir el espacio de búsqueda
    if param_dist is None:
        param_dist = {
            "n_estimators": np.arange(100, 1500, 100),
            "max_depth": [None] + list(np.arange(10, 50, 5)),
            "min_samples_split": np.arange(2, 21, 2),
            "min_samples_leaf": np.arange(1, 21, 2),
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": [True, False]
        }

    # 4) Realizar búsqueda aleatoria
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist,
        n_iter=n_iter_search, cv=5, n_jobs=-1, verbose=2, random_state=random_state
    )

    # 5) Entrenar el modelo con la búsqueda aleatoria
    random_search.fit(Xtr, y_train)
    return random_search

def run_rf_from_splits(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    target_col: str,
    param_dist: Optional[Dict[str, list]],
    num_features: List[str],
    outdir: str = "./artifacts_rf"
) -> Dict[str, Dict[str, float]]:
    """
    Entrena un modelo Random Forest sobre TRAIN y evalúa sobre VALID.
    Incluye optimización de hiperparámetros usando RandomizedSearchCV.
    """
    outdir = ensure_outdir(outdir)

    # 1) Preprocesador (igual que antes)
    pre = build_preprocessor(num_features)

    # 2) Optimización de hiperparámetros (RandomizedSearchCV)
    random_search = optimize_rf_hyperparameters(pre, X_train, y_train, param_dist, n_iter_search=100)

    # 3) Selección del mejor modelo
    rf_best = random_search.best_estimator_

    # 4) Predicciones en TRAIN y VALID
    yhat_train = rf_best.predict(X_train)
    yhat_val   = rf_best.predict(X_val)

    # 5) Métricas
    metrics = {
        "train": evaluate_regression(y_train.values, yhat_train),
        "val":   evaluate_regression(y_val.values,   yhat_val)
    }
    pd.DataFrame(metrics).to_csv(outdir / f"metrics_{target_col} RF.csv")

    # 6) Gráficos diagnósticos
    plot_diagnostics(y_train.values, yhat_train, outdir, f"train set - RF", target_col)
    plot_diagnostics(y_val.values,   yhat_val,   outdir, f"val set - RF", target_col)

    # 7) Importancia de características
    feature_importances = rf_best.feature_importances_
    feature_df = pd.DataFrame({
        "feature": num_features,
        "importance": feature_importances
    }).sort_values("importance", ascending=False)
    feature_df.to_csv(outdir / f"rf_feature_importance_{target_col}.csv", index=False)

    # 8) Guardar el modelo entrenado
    import joblib
    dir_pkl = pathlib.Path(outdir) / "pkl"
    dir_pkl.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_best, dir_pkl / f"rf_best_model_{target_col}.pkl")

    return metrics


# ------------------------ Ejemplo de uso ------------------------
if __name__ == "__main__":
    # Ejemplo (debes preparar estos objetos fuera):
    # X_train, y_train, X_val, y_val, num_features = ...
    # Aquí se muestra cómo llamar a la función
    cfg = RFConfig(
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
    m = run_rf_from_splits(cfg)
    print("Métricas (RMSE, MAE, R2, rRMSE%) por split:")
    for split_name, metrics in m.items():
        print(split_name, metrics)
