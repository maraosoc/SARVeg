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
from pathlib import Path


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
def rf_feature_importance(X: np.ndarray, y: np.ndarray, sar_cols: list, target_cols: list, n_estimators: int = 500, random_state: int = 42, figsize=(10, 5), save_path=None, format='png', resolution=200):
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
        plt.figure(figsize=figsize)
        sns.barplot(x=importance.values, y=importance.index, color='darkslategrey')
        plt.title(f'RF importance for {target}', fontsize=16)
        plt.xlabel('Relative Importance', fontsize=14)
        plt.ylabel('SAR Variable', fontsize=14)
        plt.xticks(fontsize=12)
        #plt.savefig(f'{save_path}/{target}_RF_importance.{format}', bbox_inches='tight', format=format)
        if save_path:
            if format == 'png':
                plt.savefig(f'{save_path}/RF_importance_{target}.{format}', bbox_inches='tight', format=format, dpi=resolution)
                plt.show()
            else:
                plt.savefig(f'{save_path}/RF_importance_{target}.{format}', bbox_inches='tight', format=format)
                plt.show()
        else:
            plt.show()
    
    return importance_dict

# Funcion para graficar feature importance en un solo grafico de barras
def plot_total_rf_importances(rf_importances: dict, 
                        palette: dict = None, 
                        figsize: tuple = (9,10), 
                        title: str = 'SAR importance per grass target',
                        xlabel: str = 'Relative importance',
                        ylabel: str = 'SAR variable',
                        legend_title: str = 'Grass target',
                        save_path: Path = None):
    """
    Grafica las importancias de variables SAR por cada variable de pasto (Random Forest).

    Args:
        rf_importances: diccionario de diccionarios {variable_vegetal: {variable_SAR: importancia}}.
        palette: diccionario con colores para cada variable de pasto.
        figsize: tupla para tamaño de figura (ancho, alto).
        title: título del gráfico.
        xlabel: etiqueta eje X.
        ylabel: etiqueta eje Y.
        legend_title: título de la leyenda.
        save_path: Path o str donde guardar el PDF, opcional.
    """
    # Convertir a DataFrame largo
    rows = []
    for veg_var, sar_dict in rf_importances.items():
        for sar_var, imp in sar_dict.items():
            rows.append({'VariableVegetal': veg_var, 'VariableSAR': sar_var, 'Importancia': imp})
    df_rf = pd.DataFrame(rows)
    
    # Crear el gráfico
    plt.figure(figsize=figsize)
    sns.barplot(x='Importancia', y='VariableSAR', hue='VariableVegetal', data=df_rf, palette=palette)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title=legend_title, loc="lower right", frameon=True, facecolor='white', edgecolor='black')
    plt.tight_layout()
    
    # Guardar PDF si se indica
    if save_path:
        if isinstance(save_path, (str, Path)):
            save_path = Path(save_path)
            plt.savefig(save_path)
        else:
            raise TypeError("save_path must be a str or pathlib.Path object")
    
    plt.show()