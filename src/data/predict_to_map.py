import os
import re
import joblib
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import matplotlib.pyplot as plt

# -----------------------------
# Helpers: parsing & band maps
# -----------------------------

_date_pat = r'(\d{2}[A-Za-z]{3}\d{4}|\d{4}[-_]\d{2}[-_]\d{2}|\d{8})'

def _parse_intensity_name(name):
    """
    Parse intensity band name like:
      'Sigma0_IW1_VH_mst_19Feb2025' or 'Gamma0_IW1_VV_slv4_08Apr2025'
    Returns dict with keys: prod ('Sigma0'|'Gamma0'), pol ('VH'|'VV'), date (string)
    """
    m = re.search(r'(Sigma0|Gamma0).*?(VH|VV).*?' + _date_pat, name, flags=re.IGNORECASE)
    if not m:
        return None
    prod = m.group(1)
    pol  = m.group(2).upper()
    date = m.group(3)
    return {'prod': prod, 'pol': pol, 'date': date}

def _parse_haalpha_name(name):
    """
    Parse Ha-Alpha band name like:
      'Entropy_mst_19Feb2025', 'Anisotropy_slv8_26May2025', 'Alpha_slv12_19Jun2025'
    Returns dict with keys: param ('Entropy'|'Anisotropy'|'Alpha'), date (string)
    """
    m = re.search(r'(Entropy|Anisotropy|Alpha).*?' + _date_pat, name, flags=re.IGNORECASE)
    if not m:
        return None
    param = m.group(1).capitalize()
    date  = m.group(2)
    return {'param': param, 'date': date}

def _parse_dprvi_name(name):
    """
    Parse dpRVI band name like 'dpRVI_19Feb2025' (recommended) or any name containing dpRVI + date.
    """
    m = re.search(r'(dpRVI).*?' + _date_pat, name, flags=re.IGNORECASE)
    if not m:
        # fallback: allow bare dpRVI with implicit date unknown
        if re.search('dprvi', name, flags=re.IGNORECASE):
            return {'date': 'NA'}
        return None
    date = m.group(2)
    return {'date': date}

def _read_stack(raster_path):
    """Open raster and read all bands to a float32 array with mask; returns (arr, profile, mask)."""
    src = rasterio.open(raster_path)
    arr = src.read()  # (bands, rows, cols)
    arr = arr.astype(np.float32, copy=False)
    # Build mask: True where data is valid
    mask = np.ones(arr.shape[1:], dtype=bool)
    if src.nodata is not None:
        mask &= ~np.all(arr == src.nodata, axis=0)
    # Also mask NaNs
    mask &= np.all(np.isfinite(arr), axis=0)
    return src, arr, mask

def _temporal_mean(stack_list):
    """
    Given a list of 2D arrays (same shape), compute per-pixel mean ignoring NaN.
    Returns 2D float32.
    """
    if len(stack_list) == 1:
        return stack_list[0]
    stack = np.stack(stack_list, axis=0)  # (T, H, W)
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(stack, axis=0)
    return mean.astype(np.float32)

# --------------------------------------
# Core: build features & predict a map
# --------------------------------------

def predict_vegetation_map(
    target_variable: str,
    feature_list: list,
    raster_folder: str,
    pkl_folder: str,
    output_tif: str = 'prediction.tif',
    # band dictionaries: {'Banda 01': 'Sigma0_IW1_VH_mst_19Feb2025', ...}
    intensity_band_dict: dict = None,
    dprvi_band_dict: dict = None,
    haalpha_band_dict: dict = None,
    # file names inside raster_folder
    intensity_filename: str = 'batch_intensity.tif',
    dprvi_filename: str = 'batch_dprvi.tif',
    haalpha_filename: str = 'batch_haalpha.tif',
    # plotting
    fig_title: str = None,
    cmap: str = 'viridis',
    nodata_value: float = -9999.0
):
    """
    Build pixel-wise features (temporal means) from SAR stacks (intensity/dpRVI/Ha-Alpha),
    optionally compute SUM_VHVV and/or RATIO_VH_VV when requested as target,
    load best model (.pkl), predict, write GeoTIFF, and plot with coordinate axes + colorbar.

    Parameters
    ----------
    target_variable : str
        Vegetation variable to predict. Also used to select model file: looks for '{target_variable}.pkl' in pkl_folder.
        Special targets: 'SUM_VHVV' or 'RATIO_VH_VV' trigger on-the-fly computation from intensities if needed.
    feature_list : list[str]
        Ordered list of SAR features used to train the model. Allowed tokens (case-insensitive):
          'Sigma0_VV','Sigma0_VH','Gamma0_VV','Gamma0_VH','dpRVI','Entropy','Anisotropy','Alpha'
        The function will temporal-average across available dates for each feature.
    raster_folder : str
        Folder containing the three stacks: intensities, dprvi, haalpha.
    pkl_folder : str
        Folder containing the best model .pkl for the chosen target (expects '{target_variable}.pkl').
    output_tif : str
        Path to write the predicted GeoTIFF.
    *_band_dict : dict
        Mapping from 'Banda XX' -> semantic name (as exported from SNAP) for each stack type.
    *_filename : str
        File names of the three stacks inside raster_folder.
    fig_title : str
        Title for the plot (optional).
    cmap : str
        Matplotlib colormap for the map.
    nodata_value : float
        Nodata value to write in the output raster.

    Returns
    -------
    dict with:
        'prediction': 2D ndarray (float32),
        'profile': rasterio profile used for writing,
        'figure': matplotlib figure handle,
        'model_path': str
    """
    # ---------- open stacks ----------
    path_int = os.path.join(raster_folder, intensity_filename)
    path_dpr = os.path.join(raster_folder, dprvi_filename)
    path_haa = os.path.join(raster_folder, haalpha_filename)

    src_int, arr_int, mask_int = (None, None, None)
    src_dpr, arr_dpr, mask_dpr = (None, None, None)
    src_haa, arr_haa, mask_haa = (None, None, None)

    if os.path.exists(path_int):
        src_int, arr_int, mask_int = _read_stack(path_int)
    if os.path.exists(path_dpr):
        src_dpr, arr_dpr, mask_dpr = _read_stack(path_dpr)
    if os.path.exists(path_haa):
        src_haa, arr_haa, mask_haa = _read_stack(path_haa)

    # Check a reference profile (priority: intensities, else dprvi, else haalpha)
    ref_src = src_int or src_dpr or src_haa
    if ref_src is None:
        raise FileNotFoundError("No raster stacks found in 'raster_folder'.")

    H, W = ref_src.height, ref_src.width
    profile = ref_src.profile.copy()
    transform = ref_src.transform
    crs = ref_src.crs

    # Global valid mask: only predict where all required inputs exist
    global_mask = np.ones((H, W), dtype=bool)
    for m in [mask_int, mask_dpr, mask_haa]:
        if m is not None:
            global_mask &= m

    # ---------- build per-feature temporal means ----------
    # We will fill a dict 'feature_maps' with 2D arrays (H, W), one per feature name in feature_list (case preserved order).

    feature_maps = {}

    # Helper: get per-date layers for intensity and compute temporal mean for a requested combo (e.g., 'Sigma0_VH')
    def _mean_intensity(prod: str, pol: str):
        if src_int is None or intensity_band_dict is None:
            return None
        per_date = []
        for b in range(arr_int.shape[0]):
            bname = intensity_band_dict.get(f'Banda {b+1:02d}', None)
            if not bname:
                continue
            meta = _parse_intensity_name(bname)
            if not meta:
                continue
            if meta['prod'].lower() == prod.lower() and meta['pol'].upper() == pol.upper():
                layer = arr_int[b, :, :].astype(np.float32)
                layer = np.where(np.isfinite(layer), layer, np.nan)
                per_date.append(layer)
        if len(per_date) == 0:
            return None
        return _temporal_mean(per_date)

    # Helper: compute SUM_VHVV or RATIO_VH_VV from intensity per date, then temporal mean
    def _mean_cross_metric(prod: str, metric: str):
        if src_int is None or intensity_band_dict is None:
            return None
        vh_dates = {}
        vv_dates = {}
        for b in range(arr_int.shape[0]):
            bname = intensity_band_dict.get(f'Banda {b+1:02d}', None)
            if not bname:
                continue
            meta = _parse_intensity_name(bname)
            if not meta or meta['prod'].lower() != prod.lower():
                continue
            if meta['pol'].upper() == 'VH':
                vh_dates[meta['date']] = arr_int[b, :, :].astype(np.float32)
            elif meta['pol'].upper() == 'VV':
                vv_dates[meta['date']] = arr_int[b, :, :].astype(np.float32)

        # intersect dates present in both VH & VV
        common_dates = sorted(set(vh_dates.keys()) & set(vv_dates.keys()))
        per_date = []
        for d in common_dates:
            vh = vh_dates[d]
            vv = vv_dates[d]
            vh = np.where(np.isfinite(vh), vh, np.nan)
            vv = np.where(np.isfinite(vv), vv, np.nan)
            if metric == 'sum_vhvv':
                val = vh + vv
            elif metric == 'ratio_vh_vv':
                with np.errstate(divide='ignore', invalid='ignore'):
                    val = vh / vv
            else:
                raise ValueError("Unknown cross metric")
            per_date.append(val)
        if len(per_date) == 0:
            return None
        return _temporal_mean(per_date)

    # Helper: dpRVI mean
    def _mean_dprvi():
        if src_dpr is None or dprvi_band_dict is None:
            return None
        layers = []
        for b in range(arr_dpr.shape[0]):
            bname = dprvi_band_dict.get(f'Banda {b+1:02d}', None)
            if not bname:
                continue
            meta = _parse_dprvi_name(bname) or {'date': 'NA'}
            # single-band per date; include all
            layer = arr_dpr[b, :, :].astype(np.float32)
            layers.append(np.where(np.isfinite(layer), layer, np.nan))
        if len(layers) == 0:
            return None
        return _temporal_mean(layers)

    # Helper: Ha-Alpha mean per parameter
    def _mean_haa(param: str):
        if src_haa is None or haalpha_band_dict is None:
            return None
        layers = []
        for b in range(arr_haa.shape[0]):
            bname = haalpha_band_dict.get(f'Banda {b+1:02d}', None)
            if not bname:
                continue
            meta = _parse_haalpha_name(bname)
            if not meta:
                continue
            if meta['param'].lower() == param.lower():
                layer = arr_haa[b, :, :].astype(np.float32)
                layers.append(np.where(np.isfinite(layer), layer, np.nan))
        if len(layers) == 0:
            return None
        return _temporal_mean(layers)
    #transform to lower case the feature names
    feature_list = [f.lower() for f in feature_list]
    # Build features in the exact order given
    for feat in feature_list:
        key = feat.strip()
        kLOW = key.lower()

        fmap = None
        if kLOW == 'dprvi':
            fmap = _mean_dprvi()
        elif kLOW in ('entropy', 'anisotropy', 'alpha'):
            fmap = _mean_haa(key)
        elif kLOW in ('sigma0_vv','sigma0_vh','gamma0_vv','gamma0_vh'):
            prod, pol = key.split('_')
            fmap = _mean_intensity(prod, pol)
        elif kLOW in ('sigma0_sum_vhvv','sigma0_ratio_vh_vv', 'gamma0_sum_vhvv', 'gamma0_ratio_vh_vv'):
            # Detecta el product y metrica según el token solicitado
            prod, met = key.split('_')
            fmap = _mean_cross_metric(prod.capitalize(), met.upper())
        else:
            raise ValueError(f"Unknown feature token: {key}")

        if fmap is None:
            raise ValueError(f"Feature '{key}' could not be built from the provided stacks/dictionaries.")
        feature_maps[key] = fmap

    # Stack features → (Npix, Nfeat)
    feat_order = feature_list[:]  # preserve order
    stack_feats = np.stack([feature_maps[k] for k in feat_order], axis=0)  # (F, H, W)

    # Mask invalids across all selected features + global stack mask
    finite_mask = np.all(np.isfinite(stack_feats), axis=0) & global_mask
    N_valid = finite_mask.sum()
    if N_valid == 0:
        raise RuntimeError("No valid pixels after masking. Check nodata/overlaps.")

    X = stack_feats[:, finite_mask].T  # (Npix_valid, F)

    # ---------------------------------------
    # Load model & SPECIAL TARGETS handling
    # ---------------------------------------
    model_path = os.path.join(pkl_folder, f"{target_variable}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(X).astype(np.float32)
    pred_map = np.full((H, W), np.nan, dtype=np.float32)
    pred_map[finite_mask] = y_pred

    # ---------------------------------------
    # Write GeoTIFF
    # ---------------------------------------
    out_profile = profile.copy()
    out_profile.update(
        dtype='float32',
        count=1,
        nodata=nodata_value,
        compress='deflate',
        predictor=3
    )
    # Replace NaN with nodata_value
    out_data = np.where(np.isfinite(pred_map), pred_map, nodata_value).astype(np.float32)
    with rasterio.open(output_tif, 'w', **out_profile) as dst:
        dst.write(out_data, 1)

    # ---------------------------------------
    # Plot with coordinate axes + colorbar
    # ---------------------------------------
    left, bottom, right, top = rasterio.transform.array_bounds(H, W, transform)
    fig = plt.figure(figsize=(7.5, 7))
    ax = plt.gca()
    im = ax.imshow(
        pred_map,
        extent=[left, right, bottom, top],
        origin='upper',
        cmap=cmap
    )
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(f'Predicted {target_variable}')
    ax.set_xlabel(f'X ({crs})')
    ax.set_ylabel(f'Y ({crs})')
    ax.set_title(fig_title or f'Predicted {target_variable}')
    plt.tight_layout()

    # Close datasets
    for s in [src_int, src_dpr, src_haa]:
        try:
            if s is not None: s.close()
        except:
            pass

    return {
        'prediction': pred_map,
        'profile': out_profile,
        'figure': fig,
        'model_path': model_path
    }

if __name__ == "__main__":
    print("Iniciando script de predicción de mapas...")
    # Definir las configuracones
    raster_folder = r'C:\Users\mramoso\Documents\SARVeg\data\raw\SAR'
    pkl_folder = r'C:\Users\mramoso\Documents\SARVeg\results\canopy\artifacts_rf\pkl'
    out_tif = r'C:\Users\mramoso\Documents\SARVeg\results\maps\Predicted_tree_height.tif'  # Añadido r para raw string
    from utils.model_prep import ensure_outdir
    ensure_outdir(os.path.dirname(out_tif))
    print(f"Configuración completada. Salida: {out_tif}")
    print("Ejecutando predicción...")
    # Diccionarios de los nombres de las bandas
    dprvi_band_dict = {
        'Banda 01': 'dpRVI_19Feb2025',
        'Banda 02': 'dpRVI_15Mar2025',
        'Banda 03': 'dpRVI_08Apr2025',
        'Banda 04': 'dpRVI_26May2025',
        'Banda 05': 'dpRVI_19Jun2025'
    }

    haalpha_band_dict = {
        'Banda 01': 'Entropy_19Feb2025',
        'Banda 02': 'Anisotropy_19Feb2025',
        'Banda 03': 'Alpha_19Feb2025',
        'Banda 04': 'Entropy_15Mar2025',
        'Banda 05': 'Anisotropy_15Mar2025',
        'Banda 06': 'Alpha_15Mar2025',
        'Banda 07': 'Entropy_08Apr2025',
        'Banda 08': 'Anisotropy_08Apr2025',
        'Banda 09': 'Alpha_08Apr2025',
        'Banda 10': 'Entropy_26May2025',
        'Banda 11': 'Anisotropy_26May2025',
        'Banda 12': 'Alpha_26May2025',
        'Banda 13': 'Entropy_19Jun2025',
        'Banda 14': 'Anisotropy_19Jun2025',
        'Banda 15': 'Alpha_19Jun2025'
    }

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
    m = predict_vegetation_map(
        target_variable = 'tree_height',
        feature_list = ['Sigma0_RATIO_VH_VV', 'Gamma0_RATIO_VH_VV'],  # Solo usar features disponibles por ahora
        raster_folder = raster_folder,
        pkl_folder = pkl_folder,
        output_tif = out_tif,  # Usar la variable con ruta completa
        # band dictionaries: {'Banda 01': 'Sigma0_IW1_VH_mst_19Feb2025', ...}
        intensity_band_dict = intensity_band_dict,
        dprvi_band_dict = dprvi_band_dict,
        haalpha_band_dict = haalpha_band_dict,
        # file names inside raster_folder
        intensity_filename = 'intensities.tif',
        dprvi_filename = 'dprvi.tif',
        haalpha_filename = 'haalpha.tif',
        # plotting
        fig_title = None,
        cmap = 'viridis',
        nodata_value = -9999.0
    )
    
    print("Script execution completed!")
    print(f"Output file should be at: {out_tif}")
    print(f"Output file exists: {os.path.exists(out_tif)}")
    if os.path.exists(out_tif):
        print(f"File size: {os.path.getsize(out_tif)} bytes")
    else:
        print("Checking for any errors in the directory...")
        import glob
        maps_dir = os.path.dirname(out_tif)
        print(f"Files in maps directory: {os.listdir(maps_dir)}")
        print(f"Looking for any .tif files: {glob.glob(os.path.join(maps_dir, '*.tif'))}")