# ------------------------ Utilidades ------------------------
import pathlib
from typing import List, Optional, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

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