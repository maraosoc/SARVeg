import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Gráficos más estéticos
plt.style.use("seaborn-v0_8-whitegrid")
# Cambiar estilo de la letra
plt.rcParams.update({"font.family": 'serif'})
plt.rcParams.update({'font.size': 12})
# Cambiar paleta de colores
plt.set_cmap("Paired")

from sklearn.decomposition import PCA
import pandas as pd

## Función para inferir tipo de cobertura basado en id_point
def infer_coverage_type(id_point: str) -> str:
    """
    Inferir tipo de cobertura basado en el identificador del punto.
    'G' para pastos, 'Y' para cultivo joven, 'O' para cultivo maduro.
    """
    if id_point.startswith('G'):
        return 'Grass'
    elif id_point.startswith('Y'):
        return 'Young crop'
    elif id_point.startswith('O'):
        return 'Old crop'
    else:
        return 'Unknown'

## Función para análisis de variabilidad por cobertura y fecha
def stats_variability(df: pd.DataFrame, coverage_col: str, date_col: str, target_cols: list):
    """
    Análisis de variabilidad por tipo de cobertura y por fecha.
    Calcula estadísticas descriptivas por cobertura y por fecha para las variables de interés.
    """
    # Incluir la columna de tipo de cobertura
    df['Tipo de cobertura'] = df[coverage_col].apply(infer_coverage_type)
    
    # Estadísticas por tipo de cobertura
    print("\nEstadísticas descriptivas por cobertura:")
    stats_coverage = df.groupby('Tipo de cobertura')[target_cols].describe()
    print(stats_coverage)
    
    # Estadísticas por fecha
    print("\nEstadísticas descriptivas por fecha:")
    stats_date = df.groupby(date_col)[target_cols].describe()
    print(stats_date)
    
    return stats_coverage, stats_date

## Función para análisis de variabilidad por tipo de cobertura (boxplot)
def boxplot_variability_by_coverage(df: pd.DataFrame, coverage_col: str, target_cols: list, figsize=(10, 6)):
    """
    Análisis de variabilidad por tipo de cobertura.
    Crea boxplots para cada variable de interés, agrupado por tipo de cobertura.
    """
    # Incluir la columna de tipo de cobertura
    df['Cover Type'] = df[coverage_col].apply(infer_coverage_type)
    
    # Generar boxplots por cada variable
    for target in target_cols:
        plt.figure(figsize=figsize)
        sns.boxplot(x='Cover Type', y=target, data=df, color='darkslategrey')
        plt.title(f'Boxplot of {target} by cover type')
        plt.xlabel('Cover Type')
        plt.ylabel(target)
        plt.show()

## Función para análisis de variabilidad por fecha (líneas de tiempo por mes, con opción por tipo de cobertura)
def lineplot_variability_by_date(df: pd.DataFrame, date_col: str, target_cols: list, coverage_col: str = None, figsize=(10, 6)):
    """
    Análisis de variabilidad por fecha (mes), con gráficos de líneas.
    Si `coverage_col` se pasa, genera líneas por tipo de cobertura (3 líneas).
    """
    # Asegurarse de que la columna de fecha esté en formato datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    
    # Agrupar por mes, extrayendo el primer día de cada mes
    df['Mes'] = df[date_col].dt.to_period('M').dt.start_time
    
    # Si se quiere desglosar por tipo de cobertura, añadir columna de cobertura
    if coverage_col:
        df['Tipo de cobertura'] = df[coverage_col].apply(infer_coverage_type)
    
    # Generar gráficos de líneas para cada variable
    for target in target_cols:
        plt.figure(figsize=figsize)
        
        if coverage_col:
            # Si hay cobertura, graficar líneas separadas por tipo de cobertura
            sns.lineplot(x='Mes', y=target, data=df, hue='Tipo de cobertura', marker='o')
            plt.legend(title='Cover Type', loc='center left', frameon=True, facecolor='white', edgecolor='black')
            plt.title(f'Variability of {target} by cover type and date')
        else:
            # Si no hay cobertura, graficar una única línea (promediando todas las coberturas)
            sns.lineplot(x='Mes', y=target, data=df, marker='o', color='darkslategrey')
            plt.title(f'Variability of {target} by date (averaged)')

        plt.xlabel('Date')
        plt.ylabel(target)
        plt.xticks(rotation=45)
        plt.tight_layout()  # Para evitar solapamiento de texto
        plt.show()

def scatter_grid(df, veg_vars, sar_vars, coverage_col=None, figsize=(20,15)):
    """
    Crea un grid de scatterplots: filas = variables vegetales, columnas = variables SAR.
    Opcional: colorea los puntos por tipo de cobertura.
    
    Args:
        df: DataFrame con datos.
        veg_vars: lista de variables vegetales (filas).
        sar_vars: lista de variables SAR seleccionadas (columnas).
        coverage_col: columna que indica tipo de cobertura (opcional).
        figsize: tamaño de la figura.
    """
    n_rows = len(veg_vars)
    n_cols = len(sar_vars)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    for i, veg in enumerate(veg_vars):
        for j, sar in enumerate(sar_vars):
            ax = axes[i, j]
            
            if coverage_col:
                df['Tipo de cobertura'] = df[coverage_col].apply(infer_coverage_type)
                sns.scatterplot(
                    x=sar, y=veg, hue='Tipo de cobertura', 
                    data=df, ax=ax, palette="Set1", s=50, alpha=0.7, legend=True
                )
                #plt.legend(title='Cover Type', bbox_to_anchor=(1.15, 1))
            else:
                sns.scatterplot(
                    x=sar, y=veg, data=df, ax=ax, color="b", s=50, alpha=0.7
                )
            
            if i == 0:
                ax.set_title(sar)
            if j == 0:
                ax.set_ylabel(veg)
            else:
                ax.set_ylabel("")
            
            ax.set_xlabel("")
    
    # Ajustar leyenda si hay cobertura
    if coverage_col:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))

    plt.tight_layout()
    plt.show()