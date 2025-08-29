import os
import re
import glob
import joblib
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# --------- utilidades ----------
def _read_stack(path):
    src = rasterio.open(path)
    arr = src.read().astype(np.float32)  # (B,H,W)
    # máscara válida por banda y global
    mask = np.ones(arr.shape[1:], dtype=bool)
    if src.nodata is not None:
        mask &= ~np.all(arr == src.nodata, axis=0)
    mask &= np.all(np.isfinite(arr), axis=0)
    return src, arr, mask

def _temporal_mean(layers):
    if len(layers) == 0:
        return None
    if len(layers) == 1:
        return layers[0]
    st = np.stack(layers, axis=0)  # (T,H,W)
    with np.errstate(invalid='ignore'):
        m = np.nanmean(st, axis=0)
    return m.astype(np.float32)

# --------- parser de nombres de banda de intensidad ----------
_date_pat = r'(\d{2}[A-Za-z]{3}\d{4}|\d{4}[-_]\d{2}[-_]\d{2}|\d{8})'
def _parse_intensity_name(name):
    """
    Espera nombres tipo:
      'Sigma0_IW1_VH_mst_19Feb2025'  o  'Gamma0_IW1_VV_slv4_08Apr2025'
    Retorna dict {'prod': 'Sigma0'|'Gamma0', 'pol': 'VH'|'VV', 'date': '...'} o None si no matchea.
    """
    m = re.search(r'(Sigma0|Gamma0).*?(VH|VV).*?' + _date_pat, name, flags=re.IGNORECASE)
    if not m:
        return None
    return {'prod': m.group(1), 'pol': m.group(2).upper(), 'date': m.group(3)}

# --------- núcleo simplificado ----------
def predict_canopy_from_ratios(
    target_variable: str,
    intensity_raster_path: str,
    intensity_band_dict: dict,
    pkl_folder: str,
    output_tif: str = 'prediction_canopy.tif',
    fig_title: str = None,
    cmap: str = 'viridis',
    nodata_value: float = -9999.0
):
    """
    Construye SOLO las features:
      - Sigma0_RATIO_VH_VV = mean_t(Sigma0_VH) / mean_t(Sigma0_VV)
      - Gamma0_RATIO_VH_VV = mean_t(Gamma0_VH) / mean_t(Gamma0_VV)
    y predice un mapa con el mejor modelo cuyo filename contiene `target_variable`.

    Parámetros:
      target_variable       : nombre de la variable de vegetación (se buscará un .pkl que lo contenga).
      intensity_raster_path : ruta al GeoTIFF multibanda de intensidades.
      intensity_band_dict   : {'Banda 01': 'Sigma0_IW1_VH_mst_19Feb2025', ...}
      pkl_folder            : carpeta con los .pkl entrenados (Pipeline recomendado).
      output_tif            : nombre del GeoTIFF de salida.
      fig_title             : título del plot.
      cmap                  : colormap matplotlib.
      nodata_value          : valor nodata para el GeoTIFF.

    Retorna:
      dict con 'prediction' (2D np.ndarray), 'profile' (rasterio profile), 'figure' (matplotlib fig), 'model_path' (str)
    """
    # 1) Abrir stack
    if not os.path.exists(intensity_raster_path):
        raise FileNotFoundError(intensity_raster_path)
    src, arr, mask = _read_stack(intensity_raster_path)
    H, W = src.height, src.width
    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs

    # 2) Separar capas por producto/pol y promediar fechas
    sig_vh_list, sig_vv_list = [], []
    gam_vh_list, gam_vv_list = [], []

    for b in range(arr.shape[0]):
        bname = intensity_band_dict.get(f'Banda {b+1:02d}', None)
        if not bname:
            continue
        meta = _parse_intensity_name(bname)
        if not meta:
            continue

        layer = arr[b, :, :].astype(np.float32)
        layer = np.where(np.isfinite(layer), layer, np.nan)

        if meta['prod'].lower() == 'sigma0':
            if meta['pol'] == 'VH':
                sig_vh_list.append(layer)
            elif meta['pol'] == 'VV':
                sig_vv_list.append(layer)
        elif meta['prod'].lower() == 'gamma0':
            if meta['pol'] == 'VH':
                gam_vh_list.append(layer)
            elif meta['pol'] == 'VV':
                gam_vv_list.append(layer)

    mean_sig_vh = _temporal_mean(sig_vh_list)
    mean_sig_vv = _temporal_mean(sig_vv_list)
    mean_gam_vh = _temporal_mean(gam_vh_list)
    mean_gam_vv = _temporal_mean(gam_vv_list)

    if any(x is None for x in [mean_sig_vh, mean_sig_vv, mean_gam_vh, mean_gam_vv]):
        src.close()
        raise ValueError("Faltan bandas para calcular los ratios (revisa Sigma0/Gamma0 y VH/VV en el diccionario).")

    # 3) Ratios (VH/VV) con control numérico
    with np.errstate(divide='ignore', invalid='ignore'):
        feat_sigma_ratio = mean_sig_vh / mean_sig_vv
        feat_gamma_ratio = mean_gam_vh / mean_gam_vv

    # 4) Máscara válida (datos finitos + máscara global del stack)
    feat_stack = np.stack([feat_sigma_ratio, feat_gamma_ratio], axis=0)  # (2,H,W)
    finite_mask = np.all(np.isfinite(feat_stack), axis=0) & mask
    if finite_mask.sum() == 0:
        src.close()
        raise RuntimeError("No hay píxeles válidos para predecir.")

    # 5) Tabla X (Npix × 2) en el orden requerido
    X = feat_stack[:, finite_mask].T  # columnas: [Sigma0_RATIO_VH_VV, Gamma0_RATIO_VH_VV]

    # 6) Localizar modelo .pkl: nombre que contenga target_variable (case-insensitive)
    pattern = os.path.join(pkl_folder, "*.pkl")
    candidates = [p for p in glob.glob(pattern) if re.search(re.escape(target_variable), os.path.basename(p), flags=re.IGNORECASE)]
    if not candidates:
        src.close()
        raise FileNotFoundError(f"No .pkl found in '{pkl_folder}' containing '{target_variable}' in the filename.")
    # si hay varios, toma el más reciente por mtime
    model_path = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)[0]
    model = joblib.load(model_path)

    # 7) Predicción
    y_pred = model.predict(X).astype(np.float32)
    pred_map = np.full((H, W), np.nan, dtype=np.float32)
    pred_map[finite_mask] = y_pred

    # 8) Guardar GeoTIFF
    out_profile = profile.copy()
    out_profile.update(dtype='float32', count=1, nodata=nodata_value, compress='deflate', predictor=3)
    out_data = np.where(np.isfinite(pred_map), pred_map, nodata_value).astype(np.float32)
    with rasterio.open(output_tif, 'w', **out_profile) as dst:
        dst.write(out_data, 1)

    # 9) Gráfica con ejes en CRS nativo + colorbar
    left, bottom, right, top = rasterio.transform.array_bounds(H, W, transform)
    fig = plt.figure(figsize=(7.5, 6))
    ax = plt.gca()
    im = ax.imshow(pred_map, extent=[left, right, bottom, top], origin='upper', cmap=cmap)
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(f'Predicted {target_variable} [kg/$m^3$]')
    ax.set_xlabel(f'Longitude ({crs})')
    ax.set_ylabel(f'Latitude ({crs})')
    # redondear ejes
    ax.set_xticklabels(np.round(ax.get_xticks(), 3), rotation=45)
    ax.set_yticklabels(np.round(ax.get_yticks(), 3))
    ax.set_title(fig_title or f'{target_variable} prediction')
    plt.tight_layout()
    plt.savefig(output_tif.replace('.tif', '.pdf'), format='pdf')

    src.close()
    return {'prediction': pred_map, 'profile': out_profile, 'figure': fig, 'model_path': model_path}

# Diccionario de nombres de banda
intensity_band_dict = {
        'Banda 01': 'Sigma0_VH_19Feb2025',
        'Banda 02': 'Sigma0_VV_19Feb2025',
        'Banda 03': 'Gamma0_VH_19Feb2025',
        'Banda 04': 'Gamma0_VV_19Feb2025',
        'Banda 05': 'Sigma0_VH_15Mar2025',
        'Banda 06': 'Sigma0_VV_15Mar2025',
        'Banda 07': 'Gamma0_VH_15Mar2025',
        'Banda 08': 'Gamma0_VV_15Mar2025',
        'Banda 09': 'Sigma0_VH_08Apr2025',
        'Banda 10': 'Sigma0_VV_08Apr2025',
        'Banda 11': 'Gamma0_VH_08Apr2025',
        'Banda 12': 'Gamma0_VV_08Apr2025',
        'Banda 13': 'Sigma0_VH_26May2025',
        'Banda 14': 'Sigma0_VV_26May2025',
        'Banda 15': 'Gamma0_VH_26May2025',
        'Banda 16': 'Gamma0_VV_26May2025',
        'Banda 17': 'Sigma0_VH_19Jun2025',
        'Banda 18': 'Sigma0_VV_19Jun2025',
        'Banda 19': 'Gamma0_VH_19Jun2025',
        'Banda 20': 'Gamma0_VV_19Jun2025',
    }

raster_folder = r'C:\Users\mramoso\Documents\SARVeg\data\raw\SAR\intensities.tif'
pkl_folder = r'C:\Users\mramoso\Documents\SARVeg\results\canopy\artifacts_lgbm\pkl'
out_tif = r'C:\Users\mramoso\Documents\SARVeg\results\maps\Predicted_biomass_LGBM.tif'
res = predict_canopy_from_ratios(
    target_variable='biomass',                 # texto que aparece en el nombre del .pkl
    intensity_raster_path=raster_folder,
    intensity_band_dict=intensity_band_dict,
    pkl_folder=pkl_folder,                      # carpeta con los .pkl
    output_tif=out_tif,
    fig_title='Biomass prediction map using LGBM'
)

# res['figure'] es la figura; 'Predicted_biomass_LGBM.tif' se abre en QGIS.
