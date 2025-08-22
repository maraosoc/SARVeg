from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
# Gráficos más estéticos
plt.style.use("seaborn-v0_8-whitegrid")
# Cambiar estilo de la letra
plt.rcParams.update({"font.family": 'serif'})
plt.rcParams.update({'font.size': 12})
# Cambiar paleta de colores
plt.set_cmap("Paired")
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns


def run_pca(X: np.ndarray, sar_cols: list, n_components: int = None):
    """
    Ejecuta PCA sobre las variables SAR y retorna el modelo PCA, las cargas (loadings) y los datos transformados.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    loadings = pd.DataFrame(pca.components_.T, index=sar_cols)
    return pca, loadings, X_pca

def plot_pca_variance(pca: PCA, figsize=(8, 5)):
    """
    Grafica la varianza explicada acumulada por los componentes principales de un modelo PCA.
    """
    explained_var = pca.explained_variance_ratio_
    plt.figure(figsize=figsize)
    plt.plot(np.cumsum(explained_var)*100, marker='o')
    plt.xlabel('Number of principal components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title('PCA: Explained Variance by Principal Components')
    plt.grid(True)
    plt.show()

# Función para Random Forest Feature Importance
def rf_feature_importance(X: np.ndarray, y: np.ndarray, sar_cols: list, target_cols: list, n_estimators: int = 500, random_state: int = 42):
    """
    Calcula la importancia de las variables SAR usando Random Forest para cada variable de pasto.
    Grafica la importancia relativa.
    """
    importance_dict = {}
    for i, target in enumerate(target_cols):
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X, y[:, i])
        importance = pd.Series(rf.feature_importances_, index=sar_cols).sort_values(ascending=False)
        importance_dict[target] = importance

        # Gráfico
        plt.figure(figsize=(10,5))
        sns.barplot(x=importance.values, y=importance.index)
        plt.title(f'RF importance for {target}')
        plt.xlabel('Relative Importance')
        plt.ylabel('SAR Variable')
        plt.show()
    
    return importance_dict