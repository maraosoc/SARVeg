from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def pca_analysis(X: np.ndarray, sar_cols: list, n_components: int = None):
    """
    Realiza PCA sobre las variables SAR y grafica la varianza explicada.
    Retorna el PCA y las cargas (loadings) para identificar variables más relevantes.
    """
    # Estandarizar las variables SAR
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Varianza explicada
    explained_var = pca.explained_variance_ratio_
    plt.figure(figsize=(8,5))
    plt.plot(np.cumsum(explained_var)*100, marker='o')
    plt.xlabel('Number of principal components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title('PCA: Explained Variance by Principal Components')
    plt.grid(True)
    plt.show()

    # Cargas de variables (loadings)
    loadings = pd.DataFrame(pca.components_.T, index=sar_cols)
    print("Cargas de PCA (loadings) por variable SAR:")
    
    return pca, loadings, X_pca

