# baseline_linear.py
# ------------------------------------------------------------
# Línea base: regresión lineal (OLS / Ridge / Lasso) con split
# aleatorio en entrenamiento/validación y diagnóstico básico.
# ------------------------------------------------------------

from __future__ import annotations
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gráficos más estéticos
plt.style.use("seaborn-v0_8-whitegrid")
# Aumentar el tamaño de la letra
plt.rcParams.update({"font.size": 12})
plt.rcParams.update({"font.family": 'serif'})
# Cambiar paleta de colores
plt.set_cmap("Paired")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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

# ------------------------ Utilidades ------------------------

def ensure_outdir(path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_dataset(path: str) -> pd.DataFrame:
    """Carga un CSV a DataFrame, infiere NA y limpia columnas de espacio."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def select_xy(
    df: pd.DataFrame,
    target_col: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa X e y. Si feature_cols es None, usa todas salvo la y."""
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y

def random_splits(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float,
    random_state: int
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Genera splits aleatorios: train/val/test. Primero separa test,
    luego sobre train crea validación. Reproducible por semilla.
    """
    # Validación es una fracción del train_full
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        random_state=random_state
    )
    return {
        "train": (X_train, y_train),
        "val":   (X_val,   y_val),
    }

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

def evaluate_regression(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Calcula métricas estándar."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    # rRMSE (%) relativo a la media de la verdad terreno
    rrmse = float(100.0 * rmse / np.mean(np.abs(y_true)))
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "rRMSE(%)": rrmse}

def plot_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, outdir: pathlib.Path, split_name: str):
    """Gráficos sencillos: y_pred vs y_true y residuales."""
    # y_pred vs y_true
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7, color='darkslategrey')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
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

def run_baseline_from_splits(
    X_train, y_train,
    X_val, y_val,
    target,
    num_features,
    model_type="ridge",
    alpha=1.0,
    outdir="./artifacts_linear_baseline"
):
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
    plot_diagnostics(y_train.values, yhat_train, outdir, f'train {target} {model_type}')
    plot_diagnostics(y_val.values,   yhat_val,   outdir, f'val {target} {model_type}')

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
    m = run_baseline(cfg)
    print("Métricas (RMSE, MAE, R2, rRMSE%) por split:")
    for split, d in m.items():
        print(split, d)

