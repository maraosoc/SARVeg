from __future__ import annotations
# baseline_nadaraya_watson.py
# ------------------------------------------------------------
# Nadaraya–Watson multivariado (statsmodels.KernelReg) con:
# - Preprocesamiento (imputación + escalado) en Pipeline
# - Optimización de ancho de banda h en validación (grid)
# - Métricas y gráficos similares a la regresión lineal base
# - Reutiliza splits externos: X_train/X_val/X_test, y_*
# ------------------------------------------------------------
from dataclasses import dataclass

import pathlib
from typing import List, Dict, Tuple, Iterable, Optional

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

from statsmodels.nonparametric.kernel_regression import KernelReg

from utils.model_prep import ensure_outdir
from utils.evaluation import evaluate_regression, plot_diagnostics, plot_risk_curve

# ------------------------ Configuración ---------------------
@dataclass
class NWConfig:
    target_col: str                         # Columna objetivo (p.ej. 'biomasa')
    feature_cols: Optional[List[str]] = None# Si None, usa todas menos target
    random_state: int = 42                  # Semilla reproducible
    # Rejilla de bandwidth h (si None, se usa np.logspace(-2, 1, 30))
    bandwidth_grid: Optional[Iterable[float]] = None
    reg_type: str = "lc"                    # "lc" local constant, "ll" local linear
    # Si quieres guardar artefactos:
    outdir: str = "./artifacts_nadaraya-watson"


def build_preprocessor(num_features: List[str]) -> ColumnTransformer:
    """Imputación + escalado para todas las variables numéricas."""
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    pre = ColumnTransformer(
        transformers=[("num", num_pipe, num_features)],
        remainder="drop"
    )
    return pre


# ------------------------ Estimador Nadaraya-Watson (envoltura sklearn) ------------------------

class NWRegressor(BaseEstimator, RegressorMixin):
    """
    Envoltura sklearn para statsmodels.KernelReg (Nadaraya–Watson).
    - Kernel gaussiano, regresión local constante (default en KernelReg).
    - Soporta bandwidth escalar 'h' (aplicado a todas las features tras el escalado).
    - var_type se infiere como 'c' * n_features (todas continuas).
    """
    def __init__(self, bandwidth: float = 1.0, reg_type: str = "lc"):
        self.bandwidth = float(bandwidth)
        self.reg_type = reg_type  # "lc" local constant, "ll" local linear
        self._model = None
        self._n_features = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        self._n_features = X.shape[1]
        var_type = "c" * self._n_features
        # statsmodels permite un array de bandwidth; repetimos el escalar
        bw = np.repeat(self.bandwidth, self._n_features)
        # ckertype='gaussian' por defecto en KernelReg; reg_type: 'lc'/'ll'
        self._model = KernelReg(endog=y, exog=X, var_type=var_type, bw=bw, reg_type=self.reg_type)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("NWRegressor no está ajustado. Llama a fit() primero.")
        mean, _mfx = self._model.fit(np.asarray(X))
        return mean


# ------------------------ Optimización de h en validación ------------------------

def evaluate_bandwidth_on_val(
    pre: ColumnTransformer,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame,   y_val: pd.Series,
    h: float,
    reg_type: str = "lc"
) -> float:
    """
    Ajusta NW con bandwidth=h en TRAIN y devuelve RMSE en VALID.
    El preprocesamiento se ajusta sólo con TRAIN (como en el resto de modelos).
    """
    # Ajustar preprocesador en train y transformar
    pre_fit = pre.fit(X_train, y_train)
    Xtr = pre_fit.transform(X_train)
    Xva = pre_fit.transform(X_val)

    model = NWRegressor(bandwidth=h, reg_type=reg_type)
    model.fit(Xtr, y_train.values)
    yhat_val = model.predict(Xva)
    rmse = np.sqrt(mean_squared_error(y_val.values, yhat_val)) 
    return float(rmse)


# ------------------------ Funciones auxiliares para Leave-One-Out ------------------------

def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    """
    Distancias cuadradas por pares (n x n) sin bucles.
    X debe estar estandarizado (tras el preprocesamiento).
    """
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j
    sq_norms = np.sum(X**2, axis=1, keepdims=True)       # (n, 1)
    d2 = sq_norms + sq_norms.T - 2.0 * (X @ X.T)         # (n, n)
    # Evitar pequeñas negativas por redondeo numérico
    np.maximum(d2, 0.0, out=d2)
    return d2

def _loo_rmse_gaussian(X: np.ndarray, y: np.ndarray, h: float) -> float:
    """
    RMSE Leave-One-Out para Nadaraya–Watson (kernel gaussiano, lc).
    Pred_i^{(-i)} = sum_{j!=i} K_ij y_j / sum_{j!=i} K_ij
    con K_ij = exp(-0.5 * ||x_i - x_j||^2 / h^2)
    """
    if h <= 0:
        return np.inf
    d2 = _pairwise_sq_dists(X)                            # (n, n)
    K = np.exp(-0.5 * d2 / (h * h))                      # (n, n)
    np.fill_diagonal(K, 0.0)                             # excluir i en LOO
    denom = K.sum(axis=1) + 1e-12                        # estabilidad
    num = K @ y
    y_hat_loo = num / denom
    rmse = float(np.sqrt(np.mean((y - y_hat_loo) ** 2)))
    return rmse

def optimize_bandwidth_loo(
    pre,                        # ColumnTransformer ya definido (no ajustado)
    X_train_df, y_train_ser,    # DataFrames/Series originales (no transformados)
    bandwidth_grid,             # iterable de floats (p.ej., np.logspace)
    out_csv_path=None
):
    """
    Selecciona h por LOO RMSE en TRAIN (tras ajustar y aplicar el preprocesamiento).
    Devuelve (best_h, df_risk) donde df_risk tiene columnas: ['h','RMSE_LOO'].
    """
    # Ajustar preprocesamiento SOLO con TRAIN y transformar
    pre_fit = pre.fit(X_train_df, y_train_ser)
    Xtr = pre_fit.transform(X_train_df)                   # numpy (n, d)
    ytr = y_train_ser.values.astype(float).ravel()

    records = []
    for h in bandwidth_grid:
        rmse = _loo_rmse_gaussian(Xtr, ytr, float(h))
        records.append({"h": float(h), "RMSE_LOO": rmse})

    df_risk = pd.DataFrame(records).sort_values("h").reset_index(drop=True)
    best_row = df_risk.loc[df_risk["RMSE_LOO"].idxmin()]
    best_h = float(best_row["h"])

    if out_csv_path is not None:
        df_risk.to_csv(out_csv_path, index=False)

    return best_h, df_risk


# ------------------------ Runner principal (desde splits) ------------------------

def run_nw_from_splits(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: Optional[pd.DataFrame]=None, y_val: Optional[pd.Series]=None,
    target_col: Optional[str] = None,
    num_features:  Optional[List[str]] = None,
    bandwidth_grid: Optional[np.ndarray] = None,
    reg_type: str = "lc",
    outdir: str = "./artifacts_nw"
) -> Dict[str, Dict[str, float]]:
    """
    Ejecuta Nadaraya-Watson con splits ya preparados de entrenamiento y validación.
    """
    outdir = ensure_outdir(outdir)

    if bandwidth_grid is None:
        # Rejilla por defecto razonable tras StandardScaler
        bandwidth_grid = np.logspace(-2, 1, 30)

    # 1) Preprocesador 
    pre = build_preprocessor(num_features)

    # 2) Optimización LOO en TRAIN
    best_h, df_risk = optimize_bandwidth_loo(
        pre=pre,
        X_train_df=X_train,
        y_train_ser=y_train,
        bandwidth_grid=bandwidth_grid,
        out_csv_path=outdir / f"nw_risk_grid_{target_col}.csv"
    )
    with open(outdir / f"nw_best_h_{target_col}.txt", "w") as f:
        f.write(f"best_h={best_h}\n")

    # Gráfico de riesgo vs h (usa la columna 'RMSE_LOO')
    plot_risk_curve(df_risk, best_h, target_col, outdir)

    # 3) Reajustar el preprocesador con TRAIN y transformar todo
    pre_fit = pre.fit(X_train, y_train)
    Xtr = pre_fit.transform(X_train)
    Xva = pre_fit.transform(X_val)

    # 4) Entrenar modelo final con statsmodels
    model = NWRegressor(bandwidth=best_h, reg_type=reg_type)
    model.fit(Xtr, y_train.values)

    # 5) Predicciones y métricas
    yhat_train = model.predict(Xtr)
    yhat_val   = model.predict(Xva)

    metrics = {
        "train": evaluate_regression(y_train.values, yhat_train),
        "val":   evaluate_regression(y_val.values,   yhat_val),
    }
    pd.DataFrame(metrics).to_csv(outdir / f"metrics_{target_col} NW.csv")

    # 6) Gráficos diagnósticos (idénticos)
    plot_diagnostics(y_train.values, yhat_train, outdir, f"train set - NW", target_col)
    plot_diagnostics(y_val.values,   yhat_val,   outdir, f"val set - NW", target_col)

    # 7) Guardar el modelo entrenado
    import joblib
    dir_pkl = pathlib.Path(outdir) / "pkl"
    dir_pkl.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dir_pkl / f"nw_model_{target_col}.pkl")

    return metrics



# ------------------------ Ejemplo de uso ------------------------
if __name__ == "__main__":
    # Ejemplo (debes preparar estos objetos fuera, reusando los splits del proyecto):
    # X_train, y_train, X_val, y_val, X_test, y_test, num_features = ...
    # Aquí se muestra cómo llamar a la función con una rejilla de h personalizada.
    cfg = NWConfig(      
        target_col="biomass", 
        bandwidth_grid=np.logspace(-2, 1, 30), # Rejilla de h
        reg_type="lc",                         # Tipo de regresión local (constante o lineal
        outdir="./artifacts_nadaraya-watson"   # Directorio de salida
    )
    m = run_nw_from_splits(cfg)
    print("Métricas (RMSE, MAE, R2, rRMSE%) por split:")
    for split_name, metrics in m.items():
        print(split_name, metrics)

