from scipy.stats import spearmanr, pearsonr
import pandas as pd
import matplotlib.pyplot as plt
# Gráficos más estéticos
plt.style.use("seaborn-v0_8-whitegrid")
# Cambiar estilo de la letra
plt.rcParams.update({"font.family": 'serif'})
plt.rcParams.update({'font.size': 12})
# Cambiar paleta de colores
plt.set_cmap("Paired")
import seaborn as sns
## Función para calcular correlaciones de Pearson y Spearman
def calculate_correlations(df: pd.DataFrame, target_cols: list, sar_cols: list):
    """
    Calcula las correlaciones de Pearson y Spearman entre las variables de SAR y pastos.
    """
    pearson_corr = pd.DataFrame(index=target_cols, columns=sar_cols)
    spearman_corr = pd.DataFrame(index=target_cols, columns=sar_cols)
    
    # Calcular correlaciones para cada combinación
    for target in target_cols:
        for sar in sar_cols:
            pearson_corr.loc[target, sar] = pearsonr(df[target], df[sar])[0]
            spearman_corr.loc[target, sar] = spearmanr(df[target], df[sar])[0]
    
    return pearson_corr, spearman_corr

## Función para graficar las correlaciones en gráficos de barras
def plot_correlation_bars(corr_df: pd.DataFrame, title: str, figsize=(14, 8)):
    """
    Genera gráficos de barras para mostrar la correlación entre variables.
    """
    corr_df = corr_df.astype(float)  # Asegurarse que los valores sean numéricos
    corr_df.plot(kind='bar', figsize=figsize, legend=True, colormap='Paired', width=0.8)
    plt.legend(title='SAR Variables', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.ylabel('Correlation coefficient')
    plt.xticks(rotation=45)
    plt.show()

## Función para graficar matrices de correlación
def plot_correlation_matrix(corr_df: pd.DataFrame, method='pearson', figsize=(10, 10), save_path=None, format='png', resolution=200):
    """
    Grafica la matriz de correlación como un mapa de calor.
    Si save_path se especifica, guarda la figura en ese archivo.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(corr_df.astype(float), annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, fmt=".2f", cbar=True)
    plt.title(f'{method.capitalize()} correlation matrix')
    if save_path:
        if format == 'png':
            plt.savefig(f'{save_path}/{method}_correlation_matrix.{format}', bbox_inches='tight', format=format, dpi=resolution)
            plt.show()
        else:
            plt.savefig(f'{save_path}/{method}_correlation_matrix.{format}', bbox_inches='tight', format=format)
            plt.show()
    else:
        plt.show()

def corr_selection(df_corr, umbral=0.4):
    """
    Selecciona variables SAR con correlación absoluta > umbral
    para todas las variables de pasto (filas).
    
    Parámetros:
        df_corr: DataFrame de correlaciones (filas = pasto, columnas = SAR)
        umbral: valor mínimo de correlación absoluta
        
    Retorna:
        Lista de variables SAR que cumplen el criterio en al menos una variable de pasto.
    """
    # Para cada columna SAR, verifica si alguna correlación en las filas es > umbral
    sar_seleccionadas = [
        col for col in df_corr.columns
        if (df_corr[col].abs() > umbral).any()
    ]
    return sar_seleccionadas