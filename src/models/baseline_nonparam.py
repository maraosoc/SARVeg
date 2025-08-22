# baseline_nadaraya_watson.py
# ------------------------------------------------------------
# Nadaraya–Watson multivariado (statsmodels.KernelReg) con:
# - Preprocesamiento (imputación + escalado) en Pipeline
# - Optimización de ancho de banda h en validación (grid)
# - Métricas y gráficos similares a la regresión lineal base
# - Reutiliza splits externos: X_train/X_val/X_test, y_*
# ------------------------------------------------------------

from __future__ import annotations
import pathlib
from typing import List, Dict, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from statsmodels.nonparametric.kernel_regression import KernelReg

# Gráficos más estéticos
plt.style.use("seaborn-v0_8-whitegrid")
# Tamaño y tipo de letra
plt.rcParams.update({"font.size": 12})
plt.rcParams.update({"font.family": 'serif'})
# Cambiar paleta de colores
plt.set_cmap("Paired")


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
    plt.legend(handles=[], title=f"R2: {r2_score(y_true, y_pred):.2f}", loc="upper left", frameon=True, facecolor='white', edgecolor='black')
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
    rmse = np.sqrt(mean_squared_error(y_val.values, yhat_val)) # tambien puede ser RMSE agregando np.sqrt
    return float(rmse)


def plot_risk_curve(df_risk: pd.DataFrame, best_h: float, target_col: str, outdir: pathlib.Path):
    """Gráfico RMSE_LOO vs h (escala log en h)."""
    plt.figure()
    plt.plot(df_risk["h"], df_risk["RMSE_LOO"], lw=1.8, color='darkslategrey')
    plt.axvline(best_h, linestyle="--", color='darkslategrey')
    # Agregar una leyenda con el mejor h
    plt.legend(handles=[], title=f"Best h: {best_h:.2e}", loc="center left", frameon=True, facecolor='white', edgecolor='black')
    plt.xscale("log")
    plt.xlabel("h (Bandwidth)")
    plt.ylabel("Mean error (Risk)")
    plt.title(f"Estimated risk vs h ({target_col})")
    plt.tight_layout()
    plt.savefig(outdir / f"nw_risk_vs_h_{target_col}.png", dpi=200)
    plt.close()


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
    X_train, y_train,
    X_val,   y_val,
    target_col,
    num_features,
    bandwidth_grid=None,
    reg_type="lc",
    outdir="./artifacts_nw"
):
    outdir = ensure_outdir(outdir)

    if bandwidth_grid is None:
        # Rejilla por defecto razonable tras StandardScaler
        bandwidth_grid = np.logspace(-2, 1, 30)

    # 1) Preprocesador (igual que antes)
    pre = build_preprocessor(num_features)

    # 2) **Optimización LOO en TRAIN**  ← NUEVO
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

    # 4) Entrenar modelo final con statsmodels (misma API que antes)
    model = NWRegressor(bandwidth=best_h, reg_type=reg_type)
    model.fit(Xtr, y_train.values)

    # 5) Predicciones y métricas como antes
    yhat_train = model.predict(Xtr)
    yhat_val   = model.predict(Xva)

    metrics = {
        "train": evaluate_regression(y_train.values, yhat_train),
        "val":   evaluate_regression(y_val.values,   yhat_val),
    }
    pd.DataFrame(metrics).to_csv(outdir / f"nw_metrics_{target_col}.csv")

    # 6) Gráficos diagnósticos (idénticos)
    plot_diagnostics(y_train.values, yhat_train, outdir, f"train {target_col} NW")
    plot_diagnostics(y_val.values,   yhat_val,   outdir, f"val {target_col} NW")

    return metrics



# ------------------------ Ejemplo de uso ------------------------
if __name__ == "__main__":
    # Ejemplo (debes preparar estos objetos fuera, reusando los splits del proyecto):
    # X_train, y_train, X_val, y_val, X_test, y_test, num_features = ...
    # Aquí se muestra cómo llamar a la función con una rejilla de h personalizada.
    pass
"""
from baseline_linear import select_xy, random_splits, load_dataset
data_path = r'C:\Users\Usuario\OneDrive\20211021_Int_Datafusion\20240118_MSc_MR_Datafusion\Processing\Sentinel\notebooks\outputs\selected_variables_for_modeling.csv'
out_dir = r'C:\Users\Usuario\OneDrive\20211021_Int_Datafusion\20240118_MSc_MR_Datafusion\Processing\Sentinel\notebooks\outputs\artifacts_nadaraya_watson_baseline'
df = load_dataset(data_path)
df.head()

# Definir las columnas de características y objetivo
target_col = 'total_biomass'
feature_cols = ['Sigma0_RATIO_VH_VV', 'Gamma0_RATIO_VH_VV', 'PC4', 'Alpha']

X, y = select_xy(df, target_col, feature_cols)
num_features = list(X.columns)

# Dividir en conjuntos de prueba y validacion
splits = random_splits(X, y, val_size=0.2, random_state=42)
(X_train, y_train) = splits["train"]
(X_val,   y_val)   = splits["val"]

# Definir el ancho de banda
bandwidth_grid = np.logspace(-3, 1, 50) # para total_biomass
# Ejecutar
metrics = run_nw_from_splits(
    X_train, y_train,
    X_val, y_val,
    target_col=target_col,
    num_features=num_features,
    bandwidth_grid=bandwidth_grid,
    reg_type="lc",
    outdir=out_dir
)
"""