# baseline_random_forest.py
# ------------------------------------------------------------
# Random Forest Regressor con optimización de hiperparámetros
# Usando conjuntos de entrenamiento y validación ya listos.
# ------------------------------------------------------------

from __future__ import annotations
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------ Utilidades compartidas ------------------------

def ensure_outdir(path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Métricas estándar (coherentes con la línea base)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    rrmse = float(100.0 * rmse / np.mean(np.abs(y_true)))
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "rRMSE(%)": rrmse}

def plot_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, outdir: pathlib.Path, split_name: str):
    """Gráficos: predicho-vs-observado y residuales."""
    # Scatter predicho vs observado
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7, color='darkslategrey')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims)
    plt.plot(lims, lims, color='darkslategrey')  # línea identidad
    plt.xlabel("Observed")
    plt.ylabel("Estimated")
    plt.title(f"Estimated vs Observed ({split_name})")
    # Agregar leyenda con valor de R2
    plt.legend(handles=[], title=f"$R^2$: {r2_score(y_true, y_pred):.2f}", loc="upper left", frameon=True, facecolor='white', edgecolor='black')
    plt.tight_layout()
    plt.savefig(outdir / f"scatter_{split_name}.png", dpi=200)
    plt.close()

    # Residuales
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.7, color='darkslategrey')
    plt.axhline(0, linestyle="--", color='darkslategrey')
    plt.xlabel("Estimated")
    plt.ylabel("Residual (y - ŷ)")
    plt.title(f"Residuals Diagnostic ({split_name})")
    plt.tight_layout()
    plt.savefig(outdir / f"residuals_{split_name}.png", dpi=200)
    plt.close()

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
    pd.DataFrame(metrics).to_csv(outdir / f"rf_metricas_{target_col}.csv")

    # 6) Gráficos diagnósticos
    plot_diagnostics(y_train.values, yhat_train, outdir, f"train {target_col} RF")
    plot_diagnostics(y_val.values,   yhat_val,   outdir, f"val {target_col} RF")

    # 7) Importancia de características
    feature_importances = rf_best.feature_importances_
    feature_df = pd.DataFrame({
        "feature": num_features,
        "importance": feature_importances
    }).sort_values("importance", ascending=False)
    feature_df.to_csv(outdir / f"rf_feature_importance_{target_col}.csv", index=False)

    # 8) Guardar el modelo entrenado
    import joblib
    joblib.dump(rf_best, outdir / f"rf_best_model_{target_col}.pkl")

    return metrics


# ------------------------ Ejemplo de uso ------------------------
if __name__ == "__main__":
    # Ejemplo (debes preparar estos objetos fuera):
    # X_train, y_train, X_val, y_val, num_features = ...
    # Aquí se muestra cómo llamar a la función
    pass

from baseline_linear import select_xy, random_splits, load_dataset
data_path = r'C:\Users\Usuario\OneDrive\20211021_Int_Datafusion\20240118_MSc_MR_Datafusion\Processing\Sentinel\notebooks\outputs\selected_variables_for_modeling.csv'
out_dir = r'C:\Users\Usuario\OneDrive\20211021_Int_Datafusion\20240118_MSc_MR_Datafusion\Processing\Sentinel\notebooks\outputs\artifacts_random_forest_baseline'
df = load_dataset(data_path)
df.head()

# Definir las columnas de características y objetivo
target_col = 'VWC'
feature_cols = ['Sigma0_RATIO_VH_VV', 'Gamma0_RATIO_VH_VV', 'PC4', 'Alpha']

X, y = select_xy(df, target_col, feature_cols)
num_features = list(X.columns)

# Dividir en conjuntos de prueba y validacion
splits = random_splits(X, y, val_size=0.2, random_state=42)
(X_train, y_train) = splits["train"]
(X_val,   y_val)   = splits["val"]

# Diccionario de parámetros de búsqueda (el usuario puede personalizarlo)
param_dist = {
    "n_estimators": np.arange(100, 1500, 100),  # Puedes probar desde 100 hasta 1500 árboles
    "max_depth": [None, 10, 20, 30, 40, 50],  # Profundidad del árbol
    "min_samples_split": np.arange(2, 21, 2),  # Mínimo de muestras para dividir un nodo
    "min_samples_leaf": np.arange(1, 21, 2),  # Mínimo de muestras para ser hoja
    "max_features": ["auto", "sqrt", "log2"],  # Número de features a probar
    "bootstrap": [True, False]  # Usar muestras con reemplazo o no
}

# Llamar la funcion principal
metrics_rf = run_rf_from_splits(
    X_train, y_train, X_val, y_val,
    target_col,
    param_dist,
    num_features=num_features,
    outdir=out_dir
)
