"""Stormer: 96-init campaign v2. WB2 precise climatology (regridded to Stormer's 128x256) + extra
diagnostics (bias, variance ratio, raw corr-with-climatology, zonal power spectrum) + caches raw
prediction fields so future metric changes don't need rerunning the model.
Note: Stormer's public checkpoint is 1.40625deg (128x256), coarser than Pangu/Aurora's native 0.25deg --
its RMSE/ACC/spectrum are not directly comparable pixel-for-pixel without accounting for resolution.
"""
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torchvision.transforms import transforms

sys.path.insert(0, os.path.expanduser('~/stormer'))
from stormer.models.hub.stormer import Stormer
from stormer.models.iterative_module import GlobalForecastIterativeModule

PRESSURE_LEVELS_HPA = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
CLIM_LEVELS = [500, 850, 1000]
SURFACE_MAP = {'2m_temperature': '2m_temperature', '10m_u_component_of_wind': '10m_u_component_of_wind',
               '10m_v_component_of_wind': '10m_v_component_of_wind', 'mean_sea_level_pressure': 'mean_sea_level_pressure'}
UPPER_VARS_LONG = ['geopotential', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity']
VARIABLES = list(SURFACE_MAP.keys()) + [f'{v}_{l}' for v in UPPER_VARS_LONG for l in PRESSURE_LEVELS_HPA]

TARGET_LAT = np.linspace(-90 + 1.40625 / 2, 90 - 1.40625 / 2, 128)
TARGET_LON = np.arange(0, 360, 1.40625)

N_STEPS = 8
RESULTS_DIR = os.path.expanduser('~/weather-interpretability/results/campaign/campaign_stormer_test_v2')
PRED_DIR = os.path.expanduser('~/weather-interpretability/results/campaign/campaign_stormer_test_v2_predictions')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'


def log(msg):
    print(f'[stormer] {msg}', flush=True)


def regrid(da):
    return da.interp(latitude=TARGET_LAT, longitude=TARGET_LON)


log('Opening local 2024-2025 ERA5...')
ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None).sel(level=PRESSURE_LEVELS_HPA)

log('Opening + regridding WB2 climatology to 128x256...')
clim_surf_ds_native = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_surface.zarr', chunks=None)
clim_upper_ds_native = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_upper_500_850_1000.zarr', chunks=None)


def get_climatology(time_val):
    """Fetch + regrid WB2 climatology for one (dayofyear, hour) to Stormer's 128x256 grid.
    Regridded on demand (cheap: single time slice x few vars) rather than regridding the whole
    366 x 4 climatology upfront (which would be ~40x more interpolation work than needed)."""
    ts = pd.Timestamp(time_val)
    doy = min(ts.dayofyear, 366)
    hour = (ts.hour // 6) * 6
    surf_snap = regrid(clim_surf_ds_native.sel(dayofyear=doy, hour=hour))
    upper_snap = regrid(clim_upper_ds_native.sel(dayofyear=doy, hour=hour))
    surf_clim = {v: surf_snap[v].values for v in SURFACE_MAP}
    upper_clim = {v: upper_snap[v].values for v in UPPER_VARS_LONG}  # (level=3, lat, lon)
    return surf_clim, upper_clim


log('Loading normalization constants...')
norm_dir = os.path.expanduser('~/stormer/normalization_constants')
normalize_mean = dict(np.load(os.path.join(norm_dir, 'normalize_mean.npz')))
normalize_mean = np.concatenate([normalize_mean[v] for v in VARIABLES], axis=0)
normalize_std = dict(np.load(os.path.join(norm_dir, 'normalize_std.npz')))
normalize_std = np.concatenate([normalize_std[v] for v in VARIABLES], axis=0)
inp_transform = transforms.Normalize(normalize_mean, normalize_std)

log('Loading model...')
net = Stormer(in_img_size=[128, 256], variables=VARIABLES, patch_size=2, hidden_size=1024, depth=24, num_heads=16, mlp_ratio=4)
pretrained_path = 'https://huggingface.co/tungnd/stormer/resolve/main/stormer_1.40625_patch_size_2.ckpt'
model = GlobalForecastIterativeModule(net, pretrained_path=pretrained_path)
model.eval()
model = model.to('cuda')

out_transforms = {}
for interval in [6, 12, 24]:
    diff_std = dict(np.load(os.path.join(norm_dir, f'normalize_diff_std_{interval}.npz')))
    diff_std = np.concatenate([diff_std[v] for v in VARIABLES], axis=0)
    out_transforms[interval] = transforms.Normalize(np.zeros_like(diff_std), diff_std)
model.set_transforms(inp_transform, out_transforms)


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


def build_input(time_val):
    surf_snap = regrid(ds_surf.sel(time=time_val))
    upper_snap = regrid(ds_upper.sel(time=time_val))
    channels = [surf_snap[v].values.astype(np.float32) for v in SURFACE_MAP]
    for v in UPPER_VARS_LONG:
        for lvl in PRESSURE_LEVELS_HPA:
            channels.append(upper_snap[v].sel(level=lvl).values.astype(np.float32))
    return np.stack(channels, axis=0)


init_dates = []
for year in [2024]:
    for month in [1]:
        for day in [9, 17]:
            init_dates.append(np.datetime64(f'{year}-{month:02d}-{day:02d}T00:00'))
log(f'{len(init_dates)} init dates queued')

lat_weights = np.cos(np.deg2rad(TARGET_LAT))

t_start = time.time()
for idx, init_time in enumerate(init_dates):
    out_path = os.path.join(RESULTS_DIR, f'{str(init_time)}.pkl'.replace(':', ''))
    pred_path = os.path.join(PRED_DIR, f'{str(init_time)}.npz'.replace(':', ''))
    if os.path.exists(out_path) and os.path.exists(pred_path):
        continue

    try:
        inp = build_input(init_time)
    except KeyError:
        log(f'  [{idx+1}/{len(init_dates)}] {init_time} not available, skip')
        continue
    if np.isnan(inp).any():
        log(f'  [{idx+1}/{len(init_dates)}] NaN after regrid, skip')
        continue

    init_surf_clim, init_upp_clim = get_climatology(init_time)
    input_corr_clim = {}
    for i, v in enumerate(SURFACE_MAP):
        input_corr_clim[v] = pearson_corr(inp[i], init_surf_clim[v])
    offset = len(SURFACE_MAP)
    for vi, v in enumerate(UPPER_VARS_LONG):
        for lvl in CLIM_LEVELS:
            li_src = PRESSURE_LEVELS_HPA.index(lvl)
            li_clim = CLIM_LEVELS.index(lvl)
            ch = offset + vi * len(PRESSURE_LEVELS_HPA) + li_src
            input_corr_clim[f'{v}@{lvl}hPa'] = pearson_corr(inp[ch], init_upp_clim[v][li_clim])

    inp_tensor = torch.from_numpy(inp).unsqueeze(0).to('cuda')
    inp_tensor = inp_transform(inp_tensor)

    metrics = {'init_time': str(init_time), 'lead_hours': [], 'rmse': [], 'bias': [], 'variance_ratio': [],
               'acc': [], 'corr_pred_clim': [], 'corr_truth_clim': [], 'input_corr_clim': input_corr_clim,
               'spectral': {'pred': [], 'truth': []}}
    pred_cache = {}

    with torch.no_grad():
        for step in range(1, N_STEPS + 1):
            pred_norm = model.forward_validation(inp_tensor, VARIABLES, 6, step)
            pred_physical = model.reverse_inp_transform(pred_norm).squeeze(0).cpu().numpy()

            lead_hours = step * 6
            valid_time = init_time + np.timedelta64(lead_hours, 'h')
            try:
                truth = build_input(valid_time)
            except KeyError:
                continue

            vc_surf, vc_upp = get_climatology(valid_time)

            step_rmse, step_bias, step_var, step_acc = {}, {}, {}, {}
            step_corr_pred, step_corr_truth = {}, {}
            spectral_pred, spectral_truth = {}, {}

            for i, v in enumerate(SURFACE_MAP):
                p, t = pred_physical[i], truth[i]
                step_rmse[v] = rmse(p, t)
                step_bias[v] = bias(p, t)
                step_var[v] = variance_ratio(p, t)
                step_acc[v] = acc(p, t, vc_surf[v])
                step_corr_pred[v] = pearson_corr(p, vc_surf[v])
                step_corr_truth[v] = pearson_corr(t, vc_surf[v])
                spectral_pred[v] = zonal_power_spectrum(p, lat_weights)
                spectral_truth[v] = zonal_power_spectrum(t, lat_weights)

            for vi, v in enumerate(UPPER_VARS_LONG):
                for li, lvl in enumerate(PRESSURE_LEVELS_HPA):
                    ch = offset + vi * len(PRESSURE_LEVELS_HPA) + li
                    key = f'{v}@{lvl}hPa'
                    p, t = pred_physical[ch], truth[ch]
                    step_rmse[key] = rmse(p, t)
                    step_bias[key] = bias(p, t)
                    step_var[key] = variance_ratio(p, t)
                    spectral_pred[key] = zonal_power_spectrum(p, lat_weights)
                    spectral_truth[key] = zonal_power_spectrum(t, lat_weights)
                    if lvl in CLIM_LEVELS:
                        li_clim = CLIM_LEVELS.index(lvl)
                        step_acc[key] = acc(p, t, vc_upp[v][li_clim])
                        step_corr_pred[key] = pearson_corr(p, vc_upp[v][li_clim])
                        step_corr_truth[key] = pearson_corr(t, vc_upp[v][li_clim])

            metrics['lead_hours'].append(lead_hours)
            metrics['rmse'].append(step_rmse)
            metrics['bias'].append(step_bias)
            metrics['variance_ratio'].append(step_var)
            metrics['acc'].append(step_acc)
            metrics['corr_pred_clim'].append(step_corr_pred)
            metrics['corr_truth_clim'].append(step_corr_truth)
            metrics['spectral']['pred'].append(spectral_pred)
            metrics['spectral']['truth'].append(spectral_truth)

            pred_cache[f'lead{lead_hours}'] = pred_physical.copy()

    with open(out_path, 'wb') as f:
        pickle.dump(metrics, f)
    np.savez_compressed(pred_path, **pred_cache)

    if (idx + 1) % 8 == 0:
        log(f'  [{idx+1}/{len(init_dates)}] done, elapsed {time.time()-t_start:.0f}s')

log(f'All done in {time.time()-t_start:.0f}s')
