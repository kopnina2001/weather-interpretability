"""Stormer: 96-init skill campaign. Native call to forward_validation() handles the full N-step
rollout in one call (unlike Pangu/Aurora where we loop manually).
Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 run_campaign_stormer.py
Saves compact per-init metrics (RMSE + ACC per variable/level/lead time) to results/, no full fields.
Note: Stormer's public checkpoint is 1.40625deg (128x256), coarser than Pangu/Aurora's native 0.25deg
(721x1440) -- all data (input, ground truth, climatology) is bilinearly regridded to Stormer's grid for
this campaign, so its RMSE/ACC numbers are NOT directly comparable pixel-for-pixel to Pangu/Aurora's
without accounting for the resolution difference (a coarser grid smooths out small-scale error, which
tends to inflate skill scores relative to a native-resolution model).
"""
import os
import sys
import time
import pickle
import numpy as np
import xarray as xr
import torch
from torchvision.transforms import transforms

sys.path.insert(0, os.path.expanduser('~/stormer'))
from stormer.models.hub.stormer import Stormer
from stormer.models.iterative_module import GlobalForecastIterativeModule

PRESSURE_LEVELS_HPA = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SURFACE_MAP = {'2m_temperature': '2m_temperature', '10m_u_component_of_wind': '10m_u_component_of_wind',
               '10m_v_component_of_wind': '10m_v_component_of_wind', 'mean_sea_level_pressure': 'mean_sea_level_pressure'}
UPPER_VARS_LONG = ['geopotential', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity']
VARIABLES = list(SURFACE_MAP.keys()) + [f'{v}_{l}' for v in UPPER_VARS_LONG for l in PRESSURE_LEVELS_HPA]

# Channel order used when monthly_climatology.pkl was built (compute_monthly_climatology.py) -- deliberately
# NOT the same order as SURFACE_MAP/UPPER_VARS_LONG above (which is Stormer's own required input order).
CLIM_SURFACE_VARS = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
CLIM_UPPER_VARS = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']

TARGET_LAT = np.linspace(-90 + 1.40625 / 2, 90 - 1.40625 / 2, 128)
TARGET_LON = np.arange(0, 360, 1.40625)

N_STEPS = 8  # 8 x 6h = 48h
RESULTS_DIR = os.path.expanduser('~/weather-interpretability/results/campaign_stormer_test_fixed')
os.makedirs(RESULTS_DIR, exist_ok=True)
DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'


def log(msg):
    print(f'[stormer] {msg}', flush=True)


def regrid(da):
    return da.interp(latitude=TARGET_LAT, longitude=TARGET_LON)


log('Loading climatology (regridding to 128x256)...')
with open('/srv/exw/runs/irina_weather_interpretability/monthly_climatology.pkl', 'rb') as f:
    clim_data = pickle.load(f)
climatology_raw = clim_data['climatology']
clim_lat = xr.DataArray(np.linspace(90, -90, 721), dims='latitude')  # native ARCO orientation
clim_lon = xr.DataArray(np.arange(0, 360, 0.25), dims='longitude')

climatology_regridded = {}
for month in [1]:
    surf_da = xr.DataArray(climatology_raw[month]['surface'], dims=['var', 'latitude', 'longitude'],
                            coords={'latitude': clim_lat, 'longitude': clim_lon})
    upp_da = xr.DataArray(climatology_raw[month]['upper'], dims=['var', 'level', 'latitude', 'longitude'],
                           coords={'latitude': clim_lat, 'longitude': clim_lon, 'level': PRESSURE_LEVELS_HPA})
    climatology_regridded[month] = {
        'surface': surf_da.interp(latitude=TARGET_LAT, longitude=TARGET_LON).values,
        'upper': upp_da.interp(latitude=TARGET_LAT, longitude=TARGET_LON).values,
    }
log('Climatology regridded.')

log('Opening local 2024-2025 ERA5...')
ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None).sel(level=PRESSURE_LEVELS_HPA)

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


def acc(pred, truth, clim):
    pred_anom = (pred - clim).ravel()
    truth_anom = (truth - clim).ravel()
    num = np.sum(pred_anom * truth_anom)
    den = np.sqrt(np.sum(pred_anom ** 2) * np.sum(truth_anom ** 2))
    return float(num / den) if den > 1e-9 else float('nan')


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

t_start = time.time()
for idx, init_time in enumerate(init_dates):
    out_path = os.path.join(RESULTS_DIR, f'{str(init_time)}.pkl'.replace(':', ''))
    if os.path.exists(out_path):
        continue

    try:
        inp = build_input(init_time)
    except KeyError:
        log(f'  [{idx+1}/{len(init_dates)}] {init_time} not available, skip')
        continue
    if np.isnan(inp).any():
        log(f'  [{idx+1}/{len(init_dates)}] NaN after regrid, skip')
        continue

    month = (init_time.astype('datetime64[M]').astype(int) % 12) + 1
    # IMPORTANT: climatology_regridded's channel order comes from compute_monthly_climatology.py's own
    # SURFACE_VARS/UPPER_VARS lists, which do NOT match this script's SURFACE_MAP/UPPER_VARS_LONG order
    # (Stormer needs its own specific channel order for the model input, unrelated to how climatology was
    # saved). Indexing clim_surf[i]/clim_upp[vi] positionally silently paired each variable with the WRONG
    # variable's climatology (e.g. MSLP got T2M's climatology) -- look up by NAME instead.
    clim_surf_by_name = {v: climatology_regridded[month]['surface'][i] for i, v in enumerate(CLIM_SURFACE_VARS)}
    clim_upp_by_name = {v: climatology_regridded[month]['upper'][i] for i, v in enumerate(CLIM_UPPER_VARS)}

    inp_tensor = torch.from_numpy(inp).unsqueeze(0).to('cuda')
    inp_tensor = inp_transform(inp_tensor)

    metrics = {'init_time': str(init_time), 'lead_hours': [], 'rmse': [], 'acc': []}

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

            step_rmse = {}
            step_acc = {}
            for i, v in enumerate(SURFACE_MAP):
                step_rmse[v] = rmse(pred_physical[i], truth[i])
                step_acc[v] = acc(pred_physical[i], truth[i], clim_surf_by_name[v])
            offset = len(SURFACE_MAP)
            for vi, v in enumerate(UPPER_VARS_LONG):
                for li, lvl in enumerate(PRESSURE_LEVELS_HPA):
                    ch = offset + vi * len(PRESSURE_LEVELS_HPA) + li
                    key = f'{v}@{lvl}hPa'
                    step_rmse[key] = rmse(pred_physical[ch], truth[ch])
                    step_acc[key] = acc(pred_physical[ch], truth[ch], clim_upp_by_name[v][li])

            metrics['lead_hours'].append(lead_hours)
            metrics['rmse'].append(step_rmse)
            metrics['acc'].append(step_acc)

    with open(out_path, 'wb') as f:
        pickle.dump(metrics, f)

    if (idx + 1) % 8 == 0:
        log(f'  [{idx+1}/{len(init_dates)}] done, elapsed {time.time()-t_start:.0f}s')

log(f'All done in {time.time()-t_start:.0f}s')
