"""Aurora: 96-init campaign v2. WB2 precise climatology (day-of-year x hour) + extra diagnostics
(bias, variance ratio, raw corr-with-climatology, zonal power spectrum) + caches raw prediction fields
(all 13 levels) so future metric changes don't need rerunning the model.
"""
import os
import sys
import glob
import time
import pickle
import numpy as np
import pandas as pd

_nvidia_lib_dirs = glob.glob(os.path.join(sys.prefix, 'lib', 'python3.10', 'site-packages', 'nvidia', '*', 'lib'))
os.environ['LD_LIBRARY_PATH'] = ':'.join(_nvidia_lib_dirs) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import xarray as xr
from aurora import Aurora, Batch, Metadata
from aurora.rollout import _advance_batch

PRESSURE_LEVELS_HPA = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
CLIM_LEVELS = [500, 850, 1000]
SURFACE_VARS = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
UPPER_VARS = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
AURORA_SURF_KEYS = {'mean_sea_level_pressure': 'msl', '10m_u_component_of_wind': '10u',
                    '10m_v_component_of_wind': '10v', '2m_temperature': '2t'}
AURORA_UPPER_KEYS = {'geopotential': 'z', 'specific_humidity': 'q', 'temperature': 't',
                     'u_component_of_wind': 'u', 'v_component_of_wind': 'v'}
N_STEPS = 8

RESULTS_DIR = os.path.expanduser('~/weather-interpretability/results/campaign/campaign_aurora')
PRED_DIR = os.path.expanduser('~/weather-interpretability/results/campaign/campaign_aurora_predictions')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'


def log(msg):
    print(f'[aurora] {msg}', flush=True)


def crop_to_720(field):
    return field[..., :-1, :] if field.shape[-2] == 721 else field


log('Opening local 2024-2025 ERA5...')
ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None).sel(level=PRESSURE_LEVELS_HPA)

log('Opening WB2 climatology...')
clim_surf_ds = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_surface.zarr', chunks=None)
clim_upper_ds = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_upper_500_850_1000.zarr', chunks=None)


def get_climatology(time_val):
    ts = pd.Timestamp(time_val)
    doy = min(ts.dayofyear, 366)
    hour = (ts.hour // 6) * 6
    surf_snap = clim_surf_ds.sel(dayofyear=doy, hour=hour)
    upper_snap = clim_upper_ds.sel(dayofyear=doy, hour=hour)
    surf_clim = {v: crop_to_720(surf_snap[v].values) for v in SURFACE_VARS}
    upper_clim = {v: crop_to_720(upper_snap[v].values) for v in UPPER_VARS}
    return surf_clim, upper_clim


log('Loading static vars...')
static_path = glob.glob(os.path.expanduser(
    '~/.cache/huggingface/hub/models--microsoft--aurora/snapshots/*/aurora-0.25-static.pickle'))[0]
with open(static_path, 'rb') as f:
    static_raw = pickle.load(f)
static_vars = {k: torch.from_numpy(v).float() for k, v in static_raw.items()}
lat_full = torch.from_numpy(ds_surf.latitude.values).float()
lon_full = torch.from_numpy(ds_surf.longitude.values).float()

log('Loading model...')
model = Aurora(use_lora=False)
model.load_checkpoint('microsoft/aurora', 'aurora-0.25-pretrained.ckpt')
model.eval()
model = model.to('cuda')


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def bias(a, b):
    return float(np.mean(a - b))


def variance_ratio(a, b):
    stdb = np.std(b)
    return float(np.std(a) / stdb) if stdb > 1e-9 else float('nan')


def corr_raw(a, b):
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
    fft = np.fft.rfft(field, axis=-1)
    power = np.abs(fft) ** 2
    return np.average(power, axis=0, weights=lat_weights)


init_dates = []
for year in [2024, 2025]:
    for month in range(1, 13):
        for day in [1, 9, 17, 25]:
            init_dates.append(np.datetime64(f'{year}-{month:02d}-{day:02d}T00:00'))
log(f'{len(init_dates)} init dates queued')

lat_weights_720 = np.cos(np.deg2rad(ds_surf.latitude.values[:-1]))

t_start = time.time()
for idx, init_time in enumerate(init_dates):
    out_path = os.path.join(RESULTS_DIR, f'{str(init_time)}.pkl'.replace(':', ''))
    pred_path = os.path.join(PRED_DIR, f'{str(init_time)}.npz'.replace(':', ''))
    if os.path.exists(out_path) and os.path.exists(pred_path):
        continue

    prev_time = init_time - np.timedelta64(6, 'h')
    try:
        snap_prev_surf = ds_surf.sel(time=prev_time)
        snap_now_surf = ds_surf.sel(time=init_time)
        snap_prev_upp = ds_upper.sel(time=prev_time)
        snap_now_upp = ds_upper.sel(time=init_time)
    except KeyError:
        log(f'  [{idx+1}/{len(init_dates)}] {init_time} not available, skip')
        continue

    surf_vars = {AURORA_SURF_KEYS[v]: torch.from_numpy(
        np.stack([snap_prev_surf[v].values, snap_now_surf[v].values])[None]).float() for v in SURFACE_VARS}
    atmos_vars = {AURORA_UPPER_KEYS[v]: torch.from_numpy(
        np.stack([snap_prev_upp[v].values, snap_now_upp[v].values])[None]).float() for v in UPPER_VARS}

    batch = Batch(
        surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars,
        metadata=Metadata(lat=lat_full, lon=lon_full,
                           time=(init_time.astype('datetime64[s]').tolist(),),
                           atmos_levels=tuple(PRESSURE_LEVELS_HPA)),
    )
    batch = batch.crop(model.patch_size)
    batch = batch.to('cuda')

    init_surf_clim, init_upp_clim = get_climatology(init_time)
    input_corr_clim = {}
    for v in SURFACE_VARS:
        input_corr_clim[v] = pearson_corr(crop_to_720(snap_now_surf[v].values), init_surf_clim[v])
    for v in UPPER_VARS:
        for lvl in CLIM_LEVELS:
            li_clim = CLIM_LEVELS.index(lvl)
            input_corr_clim[f'{v}@{lvl}hPa'] = pearson_corr(
                crop_to_720(snap_now_upp[v].sel(level=lvl).values), init_upp_clim[v][li_clim])

    metrics = {'init_time': str(init_time), 'lead_hours': [], 'rmse': [], 'bias': [], 'variance_ratio': [],
               'acc': [], 'corr_pred_clim': [], 'corr_truth_clim': [], 'input_corr_clim': input_corr_clim,
               'spectral': {'pred': [], 'truth': []}}
    pred_cache = {}

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for step in range(1, N_STEPS + 1):
            pred = model.forward(batch)
            lead_hours = step * 6
            valid_time = init_time + np.timedelta64(lead_hours, 'h')

            try:
                truth_surf_ds = ds_surf.sel(time=valid_time)
                truth_upp_ds = ds_upper.sel(time=valid_time)
            except KeyError:
                batch = _advance_batch(batch, pred)
                continue

            vc_surf, vc_upp = get_climatology(valid_time)

            pred_surf_np = {AURORA_SURF_KEYS[v]: pred.surf_vars[AURORA_SURF_KEYS[v]][0, 0].cpu().numpy()
                            for v in SURFACE_VARS}
            pred_upp_np = {AURORA_UPPER_KEYS[v]: pred.atmos_vars[AURORA_UPPER_KEYS[v]][0, 0].cpu().numpy()
                           for v in UPPER_VARS}

            step_rmse, step_bias, step_var, step_acc = {}, {}, {}, {}
            step_corr_pred, step_corr_truth = {}, {}
            spectral_pred, spectral_truth = {}, {}

            for v in SURFACE_VARS:
                truth = crop_to_720(truth_surf_ds[v].values)
                p = pred_surf_np[AURORA_SURF_KEYS[v]]
                step_rmse[v] = rmse(p, truth)
                step_bias[v] = bias(p, truth)
                step_var[v] = variance_ratio(p, truth)
                step_acc[v] = acc(p, truth, vc_surf[v])
                step_corr_pred[v] = pearson_corr(p, vc_surf[v])
                step_corr_truth[v] = pearson_corr(truth, vc_surf[v])
                spectral_pred[v] = zonal_power_spectrum(p, lat_weights_720)
                spectral_truth[v] = zonal_power_spectrum(truth, lat_weights_720)

            for v in UPPER_VARS:
                for lvl_idx, lvl in enumerate(PRESSURE_LEVELS_HPA):
                    truth = crop_to_720(truth_upp_ds[v].sel(level=lvl).values)
                    p = pred_upp_np[AURORA_UPPER_KEYS[v]][lvl_idx]
                    key = f'{v}@{lvl}hPa'
                    step_rmse[key] = rmse(p, truth)
                    step_bias[key] = bias(p, truth)
                    step_var[key] = variance_ratio(p, truth)
                    spectral_pred[key] = zonal_power_spectrum(p, lat_weights_720)
                    spectral_truth[key] = zonal_power_spectrum(truth, lat_weights_720)
                    if lvl in CLIM_LEVELS:
                        li = CLIM_LEVELS.index(lvl)
                        step_acc[key] = acc(p, truth, vc_upp[v][li])
                        step_corr_pred[key] = pearson_corr(p, vc_upp[v][li])
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

            pred_cache[f'surf_lead{lead_hours}'] = np.stack([pred_surf_np[AURORA_SURF_KEYS[v]] for v in SURFACE_VARS])
            pred_cache[f'upp_lead{lead_hours}'] = np.stack([pred_upp_np[AURORA_UPPER_KEYS[v]] for v in UPPER_VARS])

            batch = _advance_batch(batch, pred)
            del pred
            torch.cuda.empty_cache()

    with open(out_path, 'wb') as f:
        pickle.dump(metrics, f)
    np.savez_compressed(pred_path, **pred_cache)

    if (idx + 1) % 8 == 0:
        log(f'  [{idx+1}/{len(init_dates)}] done, elapsed {time.time()-t_start:.0f}s')

log(f'All done in {time.time()-t_start:.0f}s')
