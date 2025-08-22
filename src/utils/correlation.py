from scipy.stats import spearmanr, pearsonr
import pandas as pd
import matplotlib.pyplot as plt
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
def plot_correlation_matrix(corr_df: pd.DataFrame, method='pearson', figsize=(10, 10)):
    """
    Grafica la matriz de correlación como un mapa de calor.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(corr_df.astype(float), annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, fmt=".2f", cbar=True)
    plt.title(f'{method.capitalize()} correlation matrix')
    plt.show()

