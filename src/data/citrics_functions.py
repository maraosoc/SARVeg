# citrics_functions.py

from click import Tuple
import pandas as pd
import numpy as np
import re

from typing import Optional, Tuple, List, Dict

# --------------------
# Función para inferir tipo de cobertura (opcional)
# --------------------
def infer_coverage_type(id_point: str) -> str:
    """
    Inferir tipo de cobertura basado en el identificador del punto.
    'Y' para cultivo joven, 'O' para cultivo maduro.
    """
    if id_point.startswith('Y'):
        return 'Young crop'
    elif id_point.startswith('O'):
        return 'Mature crop'
    else:
        return 'Unknown'

# --------------------
# Función para parsear sufijo de fecha en columnas SAR
# --------------------
def parse_sar_date_tag(date_tag: str) -> pd.Timestamp:
    """
    Convierte el sufijo de fecha de la columna SAR a pd.Timestamp.
    """
    if pd.isna(date_tag):
        return None
    # Quitar caracteres no alfanuméricos
    date_tag = re.sub(r"[^0-9A-Za-z]", "", str(date_tag)).strip()
    # Intentar parse ISO o formatos DDMMYYYY / YYYYMMDD
    try:
        return pd.to_datetime(date_tag, errors='coerce').normalize()
    except:
        return None

# --------------------
# Función para reestructurar SAR a formato tidy
# --------------------
def tidy_sar(df_sar: pd.DataFrame, sar_point_col: str) -> pd.DataFrame:
    """
    Convierte el DataFrame SAR de formato ancho a formato tidy (long -> pivot).
    Cada fila será un punto y fecha, con columnas para cada variable SAR.
    """
    # Melt
    value_cols = [c for c in df_sar.columns if c != sar_point_col]
    long_df = df_sar.melt(id_vars=[sar_point_col], value_vars=value_cols,
                          var_name='var_full', value_name='val')
    
    # Separar nombre base y fecha
    split_pos = long_df['var_full'].str.rfind('_')
    long_df['var_base'] = long_df.apply(lambda r: r['var_full'][:split_pos[r.name]] if split_pos[r.name]!=-1 else r['var_full'], axis=1)
    long_df['date'] = long_df.apply(lambda r: parse_sar_date_tag(r['var_full'][split_pos[r.name]+1:]) if split_pos[r.name]!=-1 else None, axis=1)
    
    # Filtrar fechas nulas
    long_df = long_df[long_df['date'].notna()].copy()
    long_df = long_df.dropna(subset=['val'])
    
    # Pivot para volver a formato ancho
    tidy = long_df.pivot_table(index=[sar_point_col, 'date'],
                               columns='var_base', values='val', aggfunc='first').reset_index()
    tidy.columns.name = None
    return tidy

# --------------------
# Función principal: alinear cítricos con SAR
# --------------------
def align_citric_with_sar(df_citric: pd.DataFrame, df_sar: pd.DataFrame, 
                          citric_point_col: str = 'id_point', citric_datetime_col: str = 'datetime', 
                          sar_point_col: str = 'id_point', sar_filter_prefixes: list = ['Y','O'],
                          nearest_day_tolerance: pd.Timedelta = pd.Timedelta(days=2)) -> Tuple[pd.DataFrame, dict]:
    """
    Alinea los datos de vegetación de cítricos con los valores SAR correspondientes
    a la fecha de medición. Retorna un DataFrame con todas las columnas originales
    de cítricos y las columnas SAR correspondientes a cada punto y fecha.
    
    Parámetros:
        df_citric: DataFrame con los datos de vegetación de cítricos.
        df_sar: DataFrame con los datos SAR.
        citric_point_col: columna de identificador de punto en df_citric.
        citric_datetime_col: columna datetime en df_citric.
        sar_point_col: columna de identificador de punto en df_sar.
        sar_filter_prefixes: lista de prefijos de parcelas a incluir (ej. ['Y','O']).
    """
    # Filtrar SAR por parcelas de interés
    df_sar = df_sar[df_sar[sar_point_col].str.startswith(tuple(sar_filter_prefixes))].copy()

    dbg = {"point_col_grass": citric_point_col, "datetime_col_grass": citric_datetime_col,
            "point_col_sar": sar_point_col}

    # Normalizar identificadores
    df_citric[citric_point_col] = df_citric[citric_point_col].astype(str).str.strip().str.upper()
    df_sar[sar_point_col] = df_sar[sar_point_col].astype(str).str.strip().str.upper()
    
    # Crear columna de fecha normalizada
    df_citric['_Datetime_norm'] = pd.to_datetime(df_citric[citric_datetime_col], errors='coerce').dt.normalize()
    
    # Preparar SAR tidy
    s_tidy = tidy_sar(df_sar, sar_point_col)
    
    # Merge inicial exacto
    merged = pd.merge(df_citric, s_tidy, how='left', left_on=[citric_point_col, '_Datetime_norm'], right_on=[sar_point_col, 'date'])
    
    # Evitar sufijos x/y de pandas
    if 'date_x' in merged.columns:
        merged.rename(columns={'date_x': 'date'}, inplace=True)
    if 'date_y' in merged.columns:
        merged.drop(columns=['date_y'], inplace=True)
    
    # Validación
    if 'date' not in merged.columns:
        raise KeyError("Column 'date' missing after merge. Check SAR date parsing.")
    
    # Ajustar fechas con tolerancia
    # Rellenar valores con fecha más cercana si nearest_day_tolerance > 0
    unmatched_exact = merged["Alpha"].isna().sum()
    filled = 0

    if unmatched_exact > 0 and nearest_day_tolerance is not None:
        s_idx = s_tidy.set_index([sar_point_col, "date"])
        dates_by_point = {p: [pd.Timestamp(d) for d in sorted(sub["date"].unique())] for p, sub in s_tidy.groupby(sar_point_col)}
        for i in merged.index[merged["Alpha"].isna()]:
            p = merged.at[i, citric_point_col]
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

    # Limpiar columnas auxiliares
    drop_cols = []
    # No eliminar la columna de puntos si el nombre coincide
    if sar_point_col in merged.columns and sar_point_col != citric_point_col:
        drop_cols.append(sar_point_col)
    if "_Datetime_norm" in merged.columns:
        drop_cols.append("_Datetime_norm")
    merged.drop(columns=drop_cols, inplace=True, errors="ignore")

    return merged, dbg
# --------------------
