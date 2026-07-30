"""Pangu-Weather 6h: 96-init campaign v2. Uses WB2's precomputed day-of-year x hour-of-day climatology
(precise) instead of our earlier rough monthly approximation. Computes RMSE/ACC plus additional cheap
diagnostics (bias, variance ratio, raw correlation with climatology, zonal power spectrum) -- all derived
from fields we already have in memory, no extra model runs needed. Also CACHES raw prediction fields
(all 13 levels) to results/campaign_pangu6_predictions/, so future metric/climatology changes don't
require rerunning the (expensive, GPU-bound) forecast again.
"""
import os
import sys
import glob
import time
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import onnxruntime as ort

_nvidia_lib_dirs = glob.glob(os.path.join(sys.prefix, 'lib', 'python3.10', 'site-packages', 'nvidia', '*', 'lib'))
os.environ['LD_LIBRARY_PATH'] = ':'.join(_nvidia_lib_dirs) + ':' + os.environ.get('LD_LIBRARY_PATH', '')

PRESSURE_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
CLIM_LEVELS = [500, 850, 1000]  # only these 3 have downloaded WB2 climatology
SURFACE_VARS = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
UPPER_VARS = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
N_STEPS = 8  # 8 x 6h = 48h

RESULTS_DIR = os.path.expanduser('~/weather-interpretability/results/campaign_pangu6')
PRED_DIR = os.path.expanduser('~/weather-interpretability/results/campaign_pangu6_predictions')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'


def log(msg):
    print(f'[pangu6] {msg}', flush=True)


log('Opening local 2024-2025 ERA5 (surface + upper)...')
ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None).sel(level=PRESSURE_LEVELS_HPA)

log('Opening WB2 climatology (day-of-year x hour, 1990-2019)...')
clim_surf_ds = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_surface.zarr', chunks=None)
clim_upper_ds = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_upper_500_850_1000.zarr', chunks=None)


def get_climatology(time_val):
    """Fetch (surface_dict, upper_dict) climatology fields for a given datetime, matched by day-of-year+hour."""
    ts = pd.Timestamp(time_val)
    doy = min(ts.dayofyear, 366)
    hour = (ts.hour // 6) * 6
    surf_snap = clim_surf_ds.sel(dayofyear=doy, hour=hour)
    upper_snap = clim_upper_ds.sel(dayofyear=doy, hour=hour)
    surf_clim = {v: surf_snap[v].values for v in SURFACE_VARS}
    upper_clim = {v: upper_snap[v].values for v in UPPER_VARS}  # (level=3, lat, lon), level order = CLIM_LEVELS
    return surf_clim, upper_clim


log('Loading model...')
session = ort.InferenceSession('/home/irina/weather-interpretability/model_weights/pangu_weather_6.onnx',
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_names = [i.name for i in session.get_inputs()]
output_names = [o.name for o in session.get_outputs()]


def run_model(input_surface, input_upper):
    name_map = {}
    for name in input_names:
        name_map[name] = input_surface if 'surface' in name.lower() else input_upper
    raw = session.run(None, name_map)
    by_name = dict(zip(output_names, raw))
    out_surf = next(v for k, v in by_name.items() if 'surface' in k.lower())
    out_upp = next(v for k, v in by_name.items() if 'surface' not in k.lower())
    return out_surf, out_upp


def build_tensors(time_val):
    snap_surf = ds_surf.sel(time=time_val)
    snap_upper = ds_upper.sel(time=time_val)
    surf = np.stack([snap_surf[v].values for v in SURFACE_VARS], axis=0).astype(np.float32)
    upp = np.stack([snap_upper[v].values for v in UPPER_VARS], axis=0).astype(np.float32)
    return surf, upp


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def bias(a, b):
    return float(np.mean(a - b))


def variance_ratio(a, b):
    stdb = np.std(b)
    return float(np.std(a) / stdb) if stdb > 1e-9 else float('nan')


def corr_raw(a, b):
    """Plain (non-anomaly) correlation between two raw fields -- cosine similarity of flattened arrays."""
    a, b = a.ravel(), b.ravel()
    num = np.sum(a * b)
    den = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    return float(num / den) if den > 1e-9 else float('nan')


def acc(pred, truth, clim):
    return corr_raw(pred - clim, truth - clim)


def pearson_corr(a, b):
    """Proper Pearson correlation (each field's own spatial mean subtracted first). Needed for
    field-vs-climatology comparisons -- corr_raw (cosine similarity, no mean subtraction) trivially
    returns ~1.0 for any two fields sharing a huge common baseline (e.g. MSLP ~101000 Pa, geopotential,
    absolute temperature in Kelvin), since that shared offset dominates the dot product regardless of
    whether the actual weather pattern is unusual. Pearson removes that baseline, testing pattern
    similarity instead of raw-magnitude similarity."""
    a, b = a.ravel(), b.ravel()
    a = a - a.mean()
    b = b - b.mean()
    return corr_raw(a, b)


def zonal_power_spectrum(field, lat_weights):
    """1D zonal (longitude) power spectrum, latitude-weighted average. Returns power array indexed by
    zonal wavenumber (0..n_lon//2). Cheap diagnostic for spectral blurring, standard in AI-NWP papers."""
    fft = np.fft.rfft(field, axis=-1)
    power = np.abs(fft) ** 2
    return np.average(power, axis=0, weights=lat_weights)


init_dates = []
for year in [2024, 2025]:
    for month in range(1, 13):
        for day in [1, 9, 17, 25]:
            init_dates.append(np.datetime64(f'{year}-{month:02d}-{day:02d}T00:00'))
log(f'{len(init_dates)} init dates queued')

lat_weights = np.cos(np.deg2rad(ds_surf.latitude.values))

t_start = time.time()
for idx, init_time in enumerate(init_dates):
    out_path = os.path.join(RESULTS_DIR, f'{str(init_time)}.pkl'.replace(':', ''))
    pred_path = os.path.join(PRED_DIR, f'{str(init_time)}.npz'.replace(':', ''))
    if os.path.exists(out_path) and os.path.exists(pred_path):
        continue

    try:
        input_surf, input_upp = build_tensors(init_time)
    except KeyError:
        log(f'  [{idx+1}/{len(init_dates)}] {init_time} not in dataset, skip')
        continue

    init_surf_clim, init_upp_clim = get_climatology(init_time)
    input_corr_clim = {}
    for i, v in enumerate(SURFACE_VARS):
        input_corr_clim[v] = pearson_corr(input_surf[i], init_surf_clim[v])
    for i, v in enumerate(UPPER_VARS):
        for li, lvl in enumerate(CLIM_LEVELS):
            src_li = PRESSURE_LEVELS_HPA.index(lvl)
            input_corr_clim[f'{v}@{lvl}hPa'] = pearson_corr(input_upp[i, src_li], init_upp_clim[v][li])

    surf, upp = input_surf.copy(), input_upp.copy()
    metrics = {'init_time': str(init_time), 'lead_hours': [], 'rmse': [], 'bias': [], 'variance_ratio': [],
               'acc': [], 'corr_pred_clim': [], 'corr_truth_clim': [], 'input_corr_clim': input_corr_clim,
               'spectral': {'pred': [], 'truth': []}}
    pred_cache = {}

    for step in range(1, N_STEPS + 1):
        surf, upp = run_model(surf, upp)
        lead_hours = step * 6
        valid_time = init_time + np.timedelta64(lead_hours, 'h')

        try:
            truth_surf_ds = ds_surf.sel(time=valid_time)
            truth_upp_ds = ds_upper.sel(time=valid_time)
        except KeyError:
            break

        vc_surf, vc_upp = get_climatology(valid_time)

        step_rmse, step_bias, step_var, step_acc = {}, {}, {}, {}
        step_corr_pred, step_corr_truth = {}, {}
        spectral_pred, spectral_truth = {}, {}

        for i, v in enumerate(SURFACE_VARS):
            truth = truth_surf_ds[v].values
            step_rmse[v] = rmse(surf[i], truth)
            step_bias[v] = bias(surf[i], truth)
            step_var[v] = variance_ratio(surf[i], truth)
            step_acc[v] = acc(surf[i], truth, vc_surf[v])
            step_corr_pred[v] = pearson_corr(surf[i], vc_surf[v])
            step_corr_truth[v] = pearson_corr(truth, vc_surf[v])
            spectral_pred[v] = zonal_power_spectrum(surf[i], lat_weights)
            spectral_truth[v] = zonal_power_spectrum(truth, lat_weights)

        for i, v in enumerate(UPPER_VARS):
            for lvl_idx, lvl in enumerate(PRESSURE_LEVELS_HPA):
                truth = truth_upp_ds[v].sel(level=lvl).values
                key = f'{v}@{lvl}hPa'
                step_rmse[key] = rmse(upp[i, lvl_idx], truth)
                step_bias[key] = bias(upp[i, lvl_idx], truth)
                step_var[key] = variance_ratio(upp[i, lvl_idx], truth)
                spectral_pred[key] = zonal_power_spectrum(upp[i, lvl_idx], lat_weights)
                spectral_truth[key] = zonal_power_spectrum(truth, lat_weights)
                if lvl in CLIM_LEVELS:
                    li = CLIM_LEVELS.index(lvl)
                    step_acc[key] = acc(upp[i, lvl_idx], truth, vc_upp[v][li])
                    step_corr_pred[key] = pearson_corr(upp[i, lvl_idx], vc_upp[v][li])
                    step_corr_truth[key] = pearson_corr(truth, vc_upp[v][li])

        metrics['lead_hours'].append(lead_hours)
        metrics['rmse'].append(step_rmse)
        metrics['bias'].append(step_bias)
        metrics['variance_ratio'].append(step_var)
        metrics['acc'].append(step_acc)
        metrics['corr_pred_clim'].append(step_corr_pred)
        metrics['corr_truth_clim'].append(step_corr_truth)
        metrics['spectral']['pred'].append(spectral_pred)
        metrics['spectral']['truth'].append(spectral_truth)

        pred_cache[f'surf_lead{lead_hours}'] = surf.copy()
        pred_cache[f'upp_lead{lead_hours}'] = upp.copy()

    with open(out_path, 'wb') as f:
        pickle.dump(metrics, f)
    np.savez_compressed(pred_path, **pred_cache)

    if (idx + 1) % 8 == 0:
        log(f'  [{idx+1}/{len(init_dates)}] done, elapsed {time.time()-t_start:.0f}s')

log(f'All done in {time.time()-t_start:.0f}s')
