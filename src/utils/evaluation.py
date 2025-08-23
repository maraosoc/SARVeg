import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Optional, Dict, Tuple
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
# Gráficos más estéticos
plt.style.use("seaborn-v0_8-whitegrid")
# Cambiar estilo de la letra
plt.rcParams.update({"font.family": 'serif'})
plt.rcParams.update({'font.size': 12})
# Cambiar paleta de colores
plt.set_cmap("Paired")

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

def plot_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, outdir: pathlib.Path, split_name: str, target: str):
    pdf_dir = Path(outdir) / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir = Path(outdir) / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    """Gráficos sencillos: y_pred vs y_true y residuales."""
    # y_pred vs y_true
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7, color='darkslategrey')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, color='darkslategrey')  # línea identidad
    plt.xlabel(f"Observed {target}")
    plt.ylabel(f"Estimated {target}")
    plt.title(f"Estimated vs Observed ({split_name})")
    # Agregar leyenda con valor de R2
    plt.legend(handles=[], title=f"$R^2$: {r2_score(y_true, y_pred):.2f}", loc="upper left", frameon=True, facecolor='white', edgecolor='black')
    plt.tight_layout()
    plt.savefig(pdf_dir / f"scatter_{target} {split_name}.pdf", format='pdf')
    plt.savefig(png_dir / f"scatter_{target} {split_name}.png", format='png', dpi=200)
    plt.close()

    # Residuales
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.7, color='darkslategrey')
    plt.axhline(0, linestyle="--", color='darkslategrey')
    plt.xlabel(f"Estimated {target}")
    plt.ylabel(f"Residual (y - ŷ) {target}")
    plt.title(f"Residuals Diagnostic ({split_name})")
    plt.tight_layout()
    plt.savefig(png_dir / f"residuals_{target} {split_name}.png", format='png', dpi=200)
    plt.close()

# Grafica de riesgo para NW
def plot_risk_curve(df_risk: pd.DataFrame, best_h: float, target_col: str, outdir: pathlib.Path):
    """Gráfico RMSE_LOO vs h (escala log en h)."""
    png_dir = Path(outdir) / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
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
    plt.savefig(png_dir / f"nw_risk_vs_h_{target_col}.png", dpi=200)
    plt.close()
