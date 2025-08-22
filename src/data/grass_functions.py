# -*- coding: utf-8 -*-
import re
import pandas as pd
from typing import List, Optional, Tuple, Dict

def _normalize_date_tag(s: str) -> str:
    """Quita separadores no alfanuméricos para facilitar el parseo del sufijo de fecha."""
    if s is None:
        return ""
    return re.sub(r"[^0-9A-Za-z]", "", str(s)).strip()

def _parse_date_tag_to_date(s: str) -> Optional[pd.Timestamp]:
    """
    Interpreta un sufijo de fecha muy flexible y devuelve un pd.Timestamp (fecha) si es posible.
    Soporta formas como:
        19Feb2025, 19_Feb_2025, 2025-02-19, 20250219, 08Abr2025, 8Apr2025, 08042025
    Incluye abreviaturas EN/ES para los meses.
    """
    if not s or (isinstance(s, float) and pd.isna(s)):
        return None

    z = _normalize_date_tag(s)

    # Mapeo de meses (EN/ES, 3 letras) -> número
    month_map = {
        # English
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        # Español
        "ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AGO": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DIC": 12,
    }

    # Caso 1: contiene letras (e.g. 8Apr2025, 08Abr2025, Apr82025)
    if re.search(r"[A-Za-z]", z):
        # Extrae el token de letras (el mes) y los grupos numéricos
        m = re.search(r"[A-Za-z]+", z)
        nums = re.findall(r"\d+", z)
        if m and nums:
            mon_token = m.group(0).upper()[:3]  # usar primeras 3 letras
            if mon_token in month_map:
                try:
                    # Heurística: si la cadena comienza con dígitos, asuma orden D-M-YYYY
                    if re.match(r"^\d", z):
                        day = int(nums[0])
                        year = int(nums[-1]) if len(nums) > 1 else pd.NaT
                        month = month_map[mon_token]
                    else:
                        # Si empieza con letras (p. ej., Apr82025): M-D-YYYY
                        month = month_map[mon_token]
                        day = int(nums[0])
                        year = int(nums[-1]) if len(nums) > 1 else pd.NaT
                    return pd.Timestamp(year=year, month=month, day=day).normalize()
                except Exception:
                    return None
        return None

    # Caso 2: sólo dígitos
    # Soporta: YYYYMMDD (20250408) o DDMMYYYY (08042025)
    if z.isdigit():
        try:
            if len(z) == 8:
                if z.startswith(("19", "20")):  # asuma YYYYMMDD para años 19xx o 20xx
                    year, month, day = int(z[0:4]), int(z[4:6]), int(z[6:8])
                else:
                    # Asuma DDMMYYYY
                    day, month, year = int(z[0:2]), int(z[2:4]), int(z[4:8])
                return pd.Timestamp(year=year, month=month, day=day).normalize()
        except Exception:
            return None

    return None

def _tidy_sar_by_point_date(df_sar: pd.DataFrame,
                            sar_point_col: str) -> pd.DataFrame:
    """
    Reestructura df_sar desde ancho (una columna por variable-fecha) a largo y
    luego pivotea por (punto, fecha) -> columnas = variables SAR base.
    """
    # Identificar columnas de valores (todas menos el ID de punto)
    value_cols = [c for c in df_sar.columns if c != sar_point_col]
    if len(value_cols) == 0:
        raise ValueError("No se encontraron columnas de valores en df_sar.")

    long_df = df_sar.melt(id_vars=[sar_point_col], value_vars=value_cols,
                          var_name="var_full", value_name="val")

    # Separar en base y etiqueta de fecha asumiendo último '_' como separador
    # (soporta bases con underscores, ej. Sigma0_RATIO_VH_VV_19Feb2025)
    split_idx = long_df["var_full"].str.rfind("_")
    long_df["var_base"] = long_df.apply(
        lambda r: r["var_full"][:split_idx[r.name]] if split_idx[r.name] != -1 else r["var_full"],
        axis=1
    )
    long_df["date_tag"] = long_df.apply(
        lambda r: r["var_full"][split_idx[r.name]+1:] if split_idx[r.name] != -1 else "",
        axis=1
    )

    # Parsear etiquetas de fecha a fecha calendario
    unique_tags = long_df["date_tag"].unique().tolist()
    tag_to_date: Dict[str, Optional[pd.Timestamp]] = {t: _parse_date_tag_to_date(t) for t in unique_tags}

    long_df["date"] = long_df["date_tag"].map(tag_to_date)
    long_df["date"] = pd.to_datetime(long_df["date_tag"], errors="coerce").dt.normalize()

    # Eliminar filas con fecha no interpretable (no deberían unirse)
    long_df = long_df[long_df["date"].notna()].copy()

    # Pivotear: (punto, fecha) x var_base
    tidy = (long_df
            .pivot_table(index=[sar_point_col, "date"],
                         columns="var_base",
                         values="val",
                         aggfunc="first")
            .reset_index())

    # Asegurar orden natural de columnas
    tidy.columns.name = None
    tidy["date"] = pd.to_datetime(tidy["date"]).dt.normalize()
    return tidy

def align_grass_with_sar(df_grass: pd.DataFrame,
                         df_sar: pd.DataFrame,
                         grass_point_col: Optional[str] = None,
                         grass_datetime_col: Optional[str] = None,
                         sar_point_col: Optional[str] = None,
                         keep_only_vars: Optional[List[str]] = None,
                         nearest_day_tolerance: Optional[pd.Timedelta] = pd.Timedelta(days=2)
                         ) -> Tuple[pd.DataFrame, dict]:
    """
    Alinea datos de pasto (df_grass) con SAR (df_sar) por (punto, fecha), tomando
    para cada registro de pasto únicamente el valor SAR correspondiente a la MISMA fecha,
    y los almacena en columnas sin sufijos de fecha.

    Parámetros
    ----------
    df_grass : DataFrame
        Hoja/tabla de pastos. Debe contener identificador de punto y fecha-hora.
    df_sar : DataFrame
        Hoja/tabla SAR con una fila por punto y columnas con patrón <var_base>_<date_tag>.
        Ejemplos de <date_tag>: 19Feb2025, 2025-02-19, 08Abr2025, etc.
    grass_point_col : str
        Nombre de la columna de punto en df_grass (p.ej., 'Punto' o 'Point').
    grass_datetime_col : str
        Nombre de la columna de fecha-hora en df_grass.
    sar_point_col : str
        Nombre de la columna de punto en df_sar.
    keep_only_vars : list[str] or None
        Si se provee, se filtrarán sólo estas variables SAR base en la salida.
        Por ejemplo: ['Sigma0_VH','Sigma0_VV','Gamma0_VH','Gamma0_VV',
                      'Sigma0_RATIO_VH_VV','Gamma0_RATIO_VH_VV',
                      'Sigma0_SUM_VHVV','Gamma0_SUM_VHVV',
                      'Entropy','Anisotropy','Alpha','dpRVI']

    Retorna
    -------
    DataFrame
        df_SARgrass: df_grass + columnas SAR seleccionadas para la fecha correspondiente,
        con columnas SAR sin sufijo de fecha.
    """
    def detect_point_col(cols: List[str]) -> Optional[str]:
        for c in cols:
            if c.strip().lower() in ["punto", "point", "punto_muestreo", "id_point"]:
                return c
        for c in cols:
            if "punto" in c.lower() or "point" in c.lower():
                return c
        return cols[0]
    def detect_datetime_col(df: pd.DataFrame) -> Optional[str]:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return c
        for c in df.columns:
            if any(k in c.strip().lower() for k in ["fechahora","datetime","date_time","fecha_hora","fecha","date"]):
                try:
                    pd.to_datetime(df[c])
                    return c
                except Exception:
                    continue
        return None

    if grass_point_col is None: grass_point_col = detect_point_col(df_grass.columns.tolist())
    if grass_datetime_col is None: grass_datetime_col = detect_datetime_col(df_grass)
    if sar_point_col is None: sar_point_col = detect_point_col(df_sar.columns.tolist())

    dbg = {"point_col_grass": grass_point_col, "datetime_col_grass": grass_datetime_col,
           "point_col_sar": sar_point_col}


    # Copias para no modificar originales
    g = df_grass.copy()
    s = df_sar.copy()

    g[grass_point_col] = g[grass_point_col].astype(str).str.strip().str.upper()
    s[sar_point_col] = s[sar_point_col].astype(str).str.strip().str.upper()
    g["_Datetime_norm"] = pd.to_datetime(g[grass_datetime_col], errors="coerce").dt.normalize()


    # Reestructurar SAR a (punto, fecha) -> vars base
    s_tidy = _tidy_sar_by_point_date(s, sar_point_col=sar_point_col)
    s_tidy[sar_point_col] = s_tidy[sar_point_col].astype(str).str.strip().str.upper()

    # Empate por (punto, fecha)
    merged = pd.merge(
        g,
        s_tidy,
        how="left",
        left_on=[grass_point_col, "_Datetime_norm"],
        right_on=[sar_point_col, "date"])
    
    unmatched_exact = merged["date"].isna().sum()
    filled = 0

    if unmatched_exact > 0 and nearest_day_tolerance is not None:
        s_idx = s_tidy.set_index([sar_point_col, "date"])
        dates_by_point = {p: [pd.Timestamp(d) for d in sorted(sub["date"].unique())] for p, sub in s_tidy.groupby(sar_point_col)}
        for i in merged.index[merged["date"].isna()]:
            p = merged.at[i, grass_point_col]
            dn = pd.Timestamp(merged.at[i, "_Datetime_norm"]) if pd.notna(merged.at[i, "_Datetime_norm"]) else None
            if p in dates_by_point and dn is not None:
                diffs = [(abs(d - dn), d) for d in dates_by_point[p]]
                if diffs:
                    mindiff, dsel = min(diffs, key=lambda x: x[0])
                    if mindiff <= nearest_day_tolerance:
                        vals = s_idx.loc[(p, dsel)]
                        for col in s_tidy.columns:
                            if col not in [sar_point_col, "date"] and col in vals.index:
                                merged.at[i, col] = vals[col]
                        merged.at[i, "date"] = dsel
                        filled += 1

    dbg["unmatched_exact"] = int(unmatched_exact)
    dbg["filled_with_nearest"] = int(filled)
    
    # No eliminar la columna de puntos si el nombre coincide
    drop_cols = []
    if sar_point_col in merged.columns and sar_point_col != grass_point_col:
        drop_cols.append(sar_point_col)
    if "_Datetime_norm" in merged.columns:
        drop_cols.append("_Datetime_norm")
    merged.drop(columns=drop_cols, inplace=True, errors="ignore")



    # Limpiar columnas auxiliares
    #merged.drop(columns=[c for c in ["date", sar_point_col] if c in merged.columns], inplace=True)
    #merged.drop(columns=["_date"], inplace=True, errors='ignore')
    

    # Filtrar sólo variables SAR deseadas si se especifica
    if keep_only_vars is not None:
        keep_cols = [c for c in keep_only_vars if c in merged.columns]
        grass_cols = [c for c in df_grass.columns if c in merged.columns]
        merged = merged[grass_cols + keep_cols]

    cols = [c for c in [grass_point_col, grass_datetime_col] if c in merged.columns] + \
           [c for c in df_grass.columns if c not in [grass_point_col, grass_datetime_col] and c in merged.columns] + \
           [c for c in merged.columns if c not in df_grass.columns]
    merged = merged.reindex(columns=cols)
    merged.rename(columns={"date": "SarDate"}, inplace=True)

    return merged, dbg

# Sugerencia de lista de variables base a conservar
DEFAULT_SAR_VARS = [
    "Sigma0_VH", "Sigma0_VV",
    "Gamma0_VH", "Gamma0_VV",
    "Sigma0_RATIO_VH_VV", "Gamma0_RATIO_VH_VV",
    "Sigma0_SUM_VHVV", "Gamma0_SUM_VHVV",
    "Entropy", "Anisotropy", "Alpha",
    "dpRVI"
]


""""
Para cada registro de GrassBiomass (pasto), toma la fecha (ignorando la hora) y el punto.

Busca en SARValues la columna de cada variable SAR cuyo sufijo de fecha coincide con esa fecha.

Devuelve un df_SARgrass que conserva todas las columnas de pasto y agrega una única columna por variable SAR (sin sufijo de fecha): Sigma0_VH, Sigma0_VV, Gamma0_VH, Gamma0_VV, Sigma0_RATIO_VH_VV, Gamma0_RATIO_VH_VV, Sigma0_SUM_VHVV, Gamma0_SUM_VHVV, Entropy, Anisotropy, Alpha, dpRVI.

Detalles técnicos clave:

La función es robusta a sufijos de fecha en columnas SAR como 19Feb2025, 19_Feb_2025, 2025-02-19, 20250219, 08Abr2025, 8Apr2025, 08042025, etc. Soporta abreviaturas EN/ES (Apr/Abr).

Reestructura SAR de ancho→largo, parsea el sufijo a fecha calendario y luego pivot a (Punto, Fecha) para unir con pastos.

"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA

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
        return 'Mature crop'
    else:
        return 'Unknown'

## Función para análisis de variabilidad por cobertura y fecha
def analyze_variability(df: pd.DataFrame, coverage_col: str, date_col: str, target_cols: list):
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
def analyze_variability_by_coverage(df: pd.DataFrame, coverage_col: str, target_cols: list):
    """
    Análisis de variabilidad por tipo de cobertura.
    Crea boxplots para cada variable de interés, agrupado por tipo de cobertura.
    """
    # Incluir la columna de tipo de cobertura
    df['Tipo de cobertura'] = df[coverage_col].apply(infer_coverage_type)
    
    # Generar boxplots por cada variable
    for target in target_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Tipo de cobertura', y=target, data=df)
        plt.title(f'Boxplot of {target} by cover type')
        plt.xlabel('Cover Type')
        plt.ylabel(target)
        plt.show()

## Función para análisis de variabilidad por fecha (líneas de tiempo por mes, con opción por tipo de cobertura)
def analyze_variability_by_date(df: pd.DataFrame, date_col: str, target_cols: list, coverage_col: str = None):
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
        plt.figure(figsize=(10, 6))
        
        if coverage_col:
            # Si hay cobertura, graficar líneas separadas por tipo de cobertura
            sns.lineplot(x='Mes', y=target, data=df, hue='Tipo de cobertura', marker='o')
            plt.legend(title='Cover Type')
            plt.title(f'Variability of {target} by cover type and date')
        else:
            # Si no hay cobertura, graficar una única línea (promediando todas las coberturas)
            sns.lineplot(x='Mes', y=target, data=df, marker='o')
            plt.title(f'Variability of {target} by date (averaged)')

        plt.xlabel('Date')
        plt.ylabel(target)
        plt.xticks(rotation=45)
        plt.tight_layout()  # Para evitar solapamiento de texto
        plt.show()

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

## Función para graficar pairplots
def plot_correlation_matrix(df: pd.DataFrame, cols: list):
    """
    Grafica un pairplot para explorar visualmente las relaciones entre variables.
    """
    sns.pairplot(df[cols], height=2.5, kind="scatter")
    plt.show()
