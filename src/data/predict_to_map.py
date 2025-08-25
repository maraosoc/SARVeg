import os
import re
import glob
import joblib
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import Window

# ============ CONFIGURACIÓN CLAVE (EDITA ESTO A TU CASO) ============

# 1) Mapeos de bandas -> metadatos (fecha, polarización/param)
#    Las claves son índices 1-based de banda en el TIFF (Banda 01 -> 1, etc).
#    Valor: dict con campos estandarizados que usaremos para agrupar por fecha/variable.
#    Ejemplo de intensidad: {'product':'Sigma0','pol':'VV','date':'2025-02-19'}
#    Ejemplo de dpRVI:      {'product':'dpRVI','date':'2025-02-19'}
#    Ejemplo de H-a-alpha:  {'product':'Halpha','param':'Entropy','date':'2025-02-19'}

# --- EJEMPLO (sustituye con tus mapeos reales por archivo) ---
# Para flexibilidad, el código permite pasar estos mapas como argumento de la función.
# Aquí solo dejo la forma esperada del objeto:
# band_maps = {
#   'intensity': { 'G1.tif': {1:{'product':'Sigma0','pol':'VH','date':'2025-02-19'}, ...}, ... },
#   'dpRVI':     { 'G1.tif': {1:{'product':'dpRVI','date':'2025-02-19'}, ...}, ... },
#   'H-a-alpha': { 'G1.tif': {1:{'product':'Halpha','param':'Entropy','date':'2025-02-19'}, ...}, ... },
# }

# 2) Patrón para identificar la parcela desde el nombre de archivo
PARCEL_REGEX = re.compile(r'(G1|G2|Y1|Y2|O1|O2)', re.IGNORECASE)

# 3) Cómo encontrar el .pkl del mejor modelo de la variable objetivo
def find_model_pkl(models_dir, target_name):
    """
    Busca un archivo .pkl cuyo nombre contenga el nombre de la variable objetivo.
    Prioriza coincidencia exacta '<target_name>.pkl'; si no, el primero que contenga el texto.
    """
    exact = os.path.join(models_dir, f'{target_name}.pkl')
    if os.path.isfile(exact):
        return exact
    cands = glob.glob(os.path.join(models_dir, '*.pkl'))
    cands = [p for p in cands if target_name.lower() in os.path.basename(p).lower()]
    if not cands:
        raise FileNotFoundError(f'No .pkl found for target "{target_name}" in {models_dir}')
    return cands[0]

# ===================== UTILIDADES DE LECTURA ========================

def list_parcel_rasters(root_folder):
    """
    Recorre las subcarpetas esperadas y devuelve un dict:
    {
      'G1': {'intensity': '...tif', 'dpRVI': '...tif', 'H-a-alpha': '...tif'},
      ...
    }
    Permite múltiples archivos por producto/parcela: si hay varios, toma el primero;
    ajusta si necesitas apilar varios.
    """
    subfolders = {
        'intensity': os.path.join(root_folder, 'backscatter_intensities'),
        'dpRVI': os.path.join(root_folder, 'dpRVI'),
        'H-a-alpha': os.path.join(root_folder, 'H-a-alpha'),
    }
    parcels = {}
    for prod, folder in subfolders.items():
        for tif in glob.glob(os.path.join(folder, '*.tif')):
            m = PARCEL_REGEX.search(os.path.basename(tif))
            if not m:
                continue
            parc = m.group(1).upper()
            parcels.setdefault(parc, {})
            # Si hay varios por producto, puedes guardar lista; aquí guardamos único.
            parcels[parc][prod] = tif
    return parcels

def read_all_bands(path):
    """
    Lee todas las bandas (como float32), la transform y bounds.
    Retorna: arr (bands, rows, cols), profile (rasterio profile), extent [left,right,bottom,top]
    """
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)  # shape: (count, H, W)
        b = src.bounds
        extent = [b.left, b.right, b.bottom, b.top]
        profile = src.profile
    return arr, profile, extent

# ================== CONSTRUCCIÓN DE FEATURES ========================

def compute_temporal_mean_intensity(int_arr, int_meta_map, want):
    """
    int_arr: (B, H, W) intensidades
    int_meta_map: dict {band_index (1-based): {'product','pol','date'}}
                  product ∈ {'Sigma0','Gamma0'}; pol ∈ {'VV','VH'}
    want: nombre de variable base a construir: 'Sigma0_VV', 'Sigma0_VH', 'Gamma0_VV', 'Gamma0_VH'

    Retorna (H,W) con el promedio temporal de esa variable.
    """
    prod, pol = want.split('_')
    layers = []
    for bidx, meta in int_meta_map.items():
        if meta.get('product') == prod and meta.get('pol') == pol:
            layers.append(int_arr[bidx-1])
    if not layers:
        raise ValueError(f'No intensity layers found for {want}')
    stack = np.stack(layers, axis=0)  # (T,H,W)
    return np.nanmean(stack, axis=0)

def compute_temporal_sum_ratio(int_arr, int_meta_map, prod, mode):
    """
    Suma y ratio VH/VV por fecha -> luego promedio temporal.
    mode: 'SUM_VHVV' o 'RATIO_VH_VV'
    prod: 'Sigma0' o 'Gamma0'
    """
    # Agrupa por fecha y arma pares (VH, VV)
    by_date = {}
    for bidx, meta in int_meta_map.items():
        if meta.get('product') != prod:
            continue
        date = meta.get('date')
        pol  = meta.get('pol')
        by_date.setdefault(date, {})
        by_date[date][pol] = int_arr[bidx-1]

    per_date = []
    for date, pols in by_date.items():
        if 'VH' not in pols or 'VV' not in pols:
            continue
        vh = pols['VH']
        vv = pols['VV']
        if mode == 'SUM_VHVV':
            per_date.append(vh + vv)
        elif mode == 'RATIO_VH_VV':
            with np.errstate(divide='ignore', invalid='ignore'):
                per_date.append(vh / np.where(vv == 0, np.nan, vv))
        else:
            raise ValueError('mode must be SUM_VHVV or RATIO_VH_VV')
    if not per_date:
        raise ValueError(f'No complete VH/VV pairs for {prod} to compute {mode}')
    stack = np.stack(per_date, axis=0)
    return np.nanmean(stack, axis=0)  # (H,W)

def compute_temporal_mean_dprvi(dprvi_arr, dprvi_meta_map):
    """
    Promedio temporal del índice dpRVI (monobanda por fecha).
    """
    layers = []
    for bidx, meta in dprvi_meta_map.items():
        if meta.get('product','dpRVI').lower() == 'dprvi':
            layers.append(dprvi_arr[bidx-1])
    if not layers:
        raise ValueError('No dpRVI layers found')
    stack = np.stack(layers, axis=0)
    return np.nanmean(stack, axis=0)

def compute_temporal_mean_halpa(halpha_arr, halpha_meta_map, param):
    """
    param ∈ {'Entropy','Anisotropy','Alpha'}
    """
    layers = []
    for bidx, meta in halpha_meta_map.items():
        if meta.get('param') == param:
            layers.append(halpha_arr[bidx-1])
    if not layers:
        raise ValueError(f'No H-a-alpha layers found for {param}')
    stack = np.stack(layers, axis=0)
    return np.nanmean(stack, axis=0)

def build_feature_cube_for_parcel(paths_by_product, band_maps_for_parcel, feature_list):
    """
    paths_by_product: {'intensity': tif, 'dpRVI': tif, 'H-a-alpha': tif}
    band_maps_for_parcel: {'intensity': {...}, 'dpRVI': {...}, 'H-a-alpha': {...}}
        donde cada {...} es el map: {band_index: meta}
    feature_list: lista (en orden) de nombres de feature "base".
      Soporta:
        Intensidades: 'Sigma0_VV','Sigma0_VH','Gamma0_VV','Gamma0_VH'
        Derivadas:    'SUM_VHVV','RATIO_VH_VV' (se calcularán para cada producto; ver abajo)
        dpRVI:        'dpRVI'
        H-a-alpha:    'Entropy','Anisotropy','Alpha'
      Nota sobre SUM/RATIO: por defecto se calcula con 'Sigma0' (más usado).
      Si quieres con 'Gamma0', incluye 'Gamma0:SUM_VHVV' / 'Gamma0:RATIO_VH_VV' en la lista.

    Retorna:
      feature_stack: (N_features, H, W)
      profile, extent
    """
    # Lee lo disponible
    int_arr = prof_i = ext_i = None
    dprvi_arr = prof_d = ext_d = None
    halpha_arr = prof_h = ext_h = None

    if 'intensity' in paths_by_product:
        int_arr, prof_i, ext_i = read_all_bands(paths_by_product['intensity'])
    if 'dpRVI' in paths_by_product:
        dprvi_arr, prof_d, ext_d = read_all_bands(paths_by_product['dpRVI'])
    if 'H-a-alpha' in paths_by_product:
        halpha_arr, prof_h, ext_h = read_all_bands(paths_by_product['H-a-alpha'])

    # Usamos el profile/ext de la prioridad: intensity > dpRVI > H-a-alpha
    profile = prof_i or prof_d or prof_h
    extent  = ext_i or ext_d or ext_h

    # Mapeos
    int_map = band_maps_for_parcel.get('intensity', {})
    dprvi_map = band_maps_for_parcel.get('dpRVI', {})
    halpha_map = band_maps_for_parcel.get('H-a-alpha', {})

    layers = []
    for feat in feature_list:
        # Caso SUM/RATIO con Gamma0 explícito
        m = re.match(r'^(Gamma0|Sigma0):(SUM_VHVV|RATIO_VH_VV)$', feat)
        if m:
            prod = m.group(1)
            mode = m.group(2)
            if int_arr is None:
                raise ValueError('Intensity stack not provided but SUM/RATIO requested.')
            layers.append( compute_temporal_sum_ratio(int_arr, int_map, prod, mode) )
            continue

        # SUM/RATIO por defecto (Sigma0)
        if feat in ('SUM_VHVV','RATIO_VH_VV'):
            if int_arr is None:
                raise ValueError('Intensity stack not provided but SUM/RATIO requested.')
            layers.append( compute_temporal_sum_ratio(int_arr, int_map, 'Sigma0', feat) )
            continue

        # Intensidades base
        if feat in ('Sigma0_VV','Sigma0_VH','Gamma0_VV','Gamma0_VH'):
            if int_arr is None:
                raise ValueError(f'Intensity stack not provided but "{feat}" requested.')
            layers.append( compute_temporal_mean_intensity(int_arr, int_map, feat) )
            continue

        # dpRVI
        if feat.lower() == 'dprvi':
            if dprvi_arr is None:
                raise ValueError('dpRVI stack not provided but "dpRVI" requested.')
            layers.append( compute_temporal_mean_dprvi(dprvi_arr, dprvi_map) )
            continue

        # H-a-alpha
        if feat in ('Entropy','Anisotropy','Alpha'):
            if halpha_arr is None:
                raise ValueError(f'H-a-alpha stack not provided but "{feat}" requested.')
            layers.append( compute_temporal_mean_halpa(halpha_arr, halpha_map, feat) )
            continue

        raise ValueError(f'Unknown feature name: {feat}')

    feature_stack = np.stack(layers, axis=0)  # (F,H,W)
    return feature_stack, profile, extent

# =================== PREDICCIÓN Y GRAFICADO =========================

def predict_maps_for_all_parcels(
    sar_root_folder,
    models_dir,
    target_variable_name,
    feature_list,
    band_maps,
    output_figs_dir='figs_predictions',
    cmap='viridis'
):
    """
    - sar_root_folder: carpeta raíz con subcarpetas 'backscatter_intensities','dpRVI','H-a-alpha'
    - models_dir: carpeta con los .pkl de modelos (uno por variable objetivo)
    - target_variable_name: string para localizar el .pkl (p.ej., 'Soil_Moisture' o 'Biomass')
    - feature_list: lista de features SAR en el orden de entrenamiento (ver notas en build_feature_cube_for_parcel)
    - band_maps: dict de dicts con mapeos de bandas por archivo (ver formato en comentarios de CONFIGURACIÓN)
    - output_figs_dir: carpeta para guardar las figuras
    - cmap: mapa de color para las figuras

    Efecto:
      - Para cada parcela detectada, construye las features (promedios temporales + SUM/Ratio si procede),
        arma la tabla (Npix, Nfeat), predice con el .pkl y reconstruye el mapa.
      - Grafica y guarda PNG por parcela. Devuelve un dict con {'G1': pred_arr, ...}.
    """
    os.makedirs(output_figs_dir, exist_ok=True)

    # Carga el mejor modelo
    model_path = find_model_pkl(models_dir, target_variable_name)
    model = joblib.load(model_path)

    # Rasters por parcela
    parcel_paths = list_parcel_rasters(sar_root_folder)
    if not parcel_paths:
        raise RuntimeError('No parcel rasters found. Check folder structure and filenames.')

    predictions = {}

    for parcel, paths_by_product in parcel_paths.items():
        # band_maps puede tener claves por nombre de archivo; resolvemos el map para cada producto
        parcel_maps = {}
        for prod, tif_path in paths_by_product.items():
            fname = os.path.basename(tif_path)
            # si no existe mapeo específico por archivo, intentamos mapeo por "plantilla" (misma clave para todos)
            # prioridades: (1) exacto por filename, (2) único map para el producto, (3) vacío
            prod_maps = band_maps.get(prod, {})
            if fname in prod_maps:
                parcel_maps[prod] = prod_maps[fname]
            elif all(isinstance(k, int) for k in prod_maps.keys()):
                # El usuario pasó un único mapa por producto (igual para todos los archivos)
                parcel_maps[prod] = prod_maps
            else:
                parcel_maps[prod] = {}

        # 1) Construir cubo de features (F,H,W)
        feat_cube, profile, extent = build_feature_cube_for_parcel(
            paths_by_product=paths_by_product,
            band_maps_for_parcel=parcel_maps,
            feature_list=feature_list
        )
        F, H, W = feat_cube.shape

        # 2) Reestructurar a tabla (Npix, Nfeat) respetando el orden de feature_list
        X = feat_cube.reshape(F, -1).T  # (H*W, F)

        # 3) Predicción
        y_pred = model.predict(X)  # (H*W,)
        pred_img = y_pred.reshape(H, W)
        predictions[parcel] = pred_img

        # 4) Graficar
        left, right, bottom, top = extent
        plt.figure(figsize=(6, 6))
        im = plt.imshow(pred_img, extent=[left, right, bottom, top], origin='upper', cmap=cmap)
        cbar = plt.colorbar(im)
        cbar.set_label(f'{target_variable_name}')
        plt.title(f'Prediction: {target_variable_name} — Parcel {parcel}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        out_png = os.path.join(output_figs_dir, f'{target_variable_name}_{parcel}.png')
        plt.savefig(out_png, dpi=200)
        plt.close()

    return predictions
