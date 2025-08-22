# baseline_lightgbm.py
# ------------------------------------------------------------
# LightGBM Regressor con optimización de hiperparámetros
# Usando conjuntos de entrenamiento y validación ya listos.
# ------------------------------------------------------------

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

import lightgbm as lgb

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
    pd.DataFrame(metrics).to_csv(outdir / f"lgbm_metricas_{target_col}.csv")

    # 6) Gráficos diagnósticos (idénticos)
    plot_diagnostics(y_train.values, yhat_train, outdir, f"train {target_col} LGBM")
    plot_diagnostics(y_val.values,   yhat_val,   outdir, f"val {target_col} LGBM")

    # 7) Importancia de características
    feature_importances = lgbm_best.feature_importances_
    feature_df = pd.DataFrame({
        "feature": num_features,
        "importance": feature_importances
    }).sort_values("importance", ascending=False)
    feature_df.to_csv(outdir / f"lgbm_feature_importance_{target_col}.csv", index=False)

    # 8) Guardar el modelo entrenado
    import joblib
    joblib.dump(lgbm_best, outdir / f"lgbm_best_model_{target_col}.pkl")

    return metrics


# ------------------------ Ejemplo de uso ------------------------
if __name__ == "__main__":
    # Diccionario de parámetros para LightGBM
    pass

from baseline_linear import select_xy, random_splits, load_dataset
data_path = r'C:\Users\Usuario\OneDrive\20211021_Int_Datafusion\20240118_MSc_MR_Datafusion\Processing\Sentinel\notebooks\outputs\selected_variables_for_modeling.csv'
out_dir = r'C:\Users\Usuario\OneDrive\20211021_Int_Datafusion\20240118_MSc_MR_Datafusion\Processing\Sentinel\notebooks\outputs\artifacts_lightgbm_baseline'
df = load_dataset(data_path)
df.head()

# Definir las columnas de características y objetivo
target_col = ['VWC', 'total_biomass', 'IBA']
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
for target in target_col:
    metrics_rf = run_lgbm_from_splits(
        X_train, y_train[target], X_val, y_val[target],
        target,
        param_dist,
        num_features=num_features,
        outdir=out_dir
    )