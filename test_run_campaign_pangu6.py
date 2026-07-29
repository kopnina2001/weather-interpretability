"""Pangu-Weather 6h model: 96-init skill campaign. Autoregressive 6h steps, up to 8 steps (48h).
Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 run_campaign_pangu6.py
Saves compact per-init metrics (RMSE + ACC per variable/level/lead time) to results/, no full fields.
"""
import os
import sys
import glob
import time
import pickle
import numpy as np
import xarray as xr
import onnxruntime as ort

_nvidia_lib_dirs = glob.glob(os.path.join(sys.prefix, 'lib', 'python3.10', 'site-packages', 'nvidia', '*', 'lib'))
os.environ['LD_LIBRARY_PATH'] = ':'.join(_nvidia_lib_dirs) + ':' + os.environ.get('LD_LIBRARY_PATH', '')

PRESSURE_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SURFACE_VARS = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
UPPER_VARS = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
N_STEPS = 8  # 8 x 6h = 48h

RESULTS_DIR = os.path.expanduser('~/weather-interpretability/results/campaign_pangu6')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'


def log(msg):
    print(f'[pangu6] {msg}', flush=True)


log('Loading climatology...')
with open('/srv/exw/runs/irina_weather_interpretability/monthly_climatology.pkl', 'rb') as f:
    clim_data = pickle.load(f)
climatology = clim_data['climatology']

log('Opening local 2024-2025 ERA5 (surface + upper)...')
ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None).sel(level=PRESSURE_LEVELS_HPA)

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


def acc(pred, truth, clim):
    pred_anom = (pred - clim).ravel()
    truth_anom = (truth - clim).ravel()
    num = np.sum(pred_anom * truth_anom)
    den = np.sqrt(np.sum(pred_anom ** 2) * np.sum(truth_anom ** 2))
    return float(num / den) if den > 1e-9 else float('nan')


# 96 init dates: 4 per month (1st, 9th, 17th, 25th at 00 UTC) x 24 months (2024-2025)
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
        surf, upp = build_tensors(init_time)
    except KeyError:
        log(f'  [{idx+1}/{len(init_dates)}] {init_time} not in dataset, skip')
        continue

    month = (init_time.astype('datetime64[M]').astype(int) % 12) + 1
    clim_surf = climatology[month]['surface']
    clim_upp = climatology[month]['upper']

    metrics = {'init_time': str(init_time), 'lead_hours': [], 'rmse': [], 'acc': []}

    for step in range(1, N_STEPS + 1):
        surf, upp = run_model(surf, upp)
        lead_hours = step * 6
        valid_time = init_time + np.timedelta64(lead_hours, 'h')

        try:
            truth_surf_ds = ds_surf.sel(time=valid_time)
            truth_upp_ds = ds_upper.sel(time=valid_time)
        except KeyError:
            break

        step_rmse = {}
        step_acc = {}
        for i, v in enumerate(SURFACE_VARS):
            truth = truth_surf_ds[v].values
            step_rmse[v] = rmse(surf[i], truth)
            step_acc[v] = acc(surf[i], truth, clim_surf[i])
        for i, v in enumerate(UPPER_VARS):
            for lvl_idx, lvl in enumerate(PRESSURE_LEVELS_HPA):
                truth = truth_upp_ds[v].sel(level=lvl).values
                key = f'{v}@{lvl}hPa'
                step_rmse[key] = rmse(upp[i, lvl_idx], truth)
                step_acc[key] = acc(upp[i, lvl_idx], truth, clim_upp[i, lvl_idx])

        metrics['lead_hours'].append(lead_hours)
        metrics['rmse'].append(step_rmse)
        metrics['acc'].append(step_acc)

    with open(out_path, 'wb') as f:
        pickle.dump(metrics, f)

    if (idx + 1) % 8 == 0:
        log(f'  [{idx+1}/{len(init_dates)}] done, elapsed {time.time()-t_start:.0f}s')

log(f'All done in {time.time()-t_start:.0f}s')
