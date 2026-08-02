"""Aurora: 96-init skill campaign. Manual 6h-step rollout (bypassing aurora.rollout(), see
run_case_aurora.py for why), up to 8 steps (48h).
Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 run_campaign_aurora.py
Saves compact per-init metrics (RMSE + ACC per variable/level/lead time) to results/, no full fields.
"""
import os
import sys
import glob
import time
import pickle
import numpy as np

_nvidia_lib_dirs = glob.glob(os.path.join(sys.prefix, 'lib', 'python3.10', 'site-packages', 'nvidia', '*', 'lib'))
os.environ['LD_LIBRARY_PATH'] = ':'.join(_nvidia_lib_dirs) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import xarray as xr
from aurora import Aurora, Batch, Metadata
from aurora.rollout import _advance_batch

PRESSURE_LEVELS_HPA = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SURFACE_VARS = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
UPPER_VARS = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
AURORA_SURF_KEYS = {'mean_sea_level_pressure': 'msl', '10m_u_component_of_wind': '10u',
                    '10m_v_component_of_wind': '10v', '2m_temperature': '2t'}
AURORA_UPPER_KEYS = {'geopotential': 'z', 'specific_humidity': 'q', 'temperature': 't',
                     'u_component_of_wind': 'u', 'v_component_of_wind': 'v'}
N_STEPS = 8  # 8 x 6h = 48h

RESULTS_DIR = os.path.expanduser('~/weather-interpretability/results/campaign/campaign_aurora')
os.makedirs(RESULTS_DIR, exist_ok=True)
DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'


def log(msg):
    print(f'[aurora] {msg}', flush=True)


def crop_to_720(field):
    # field[:-1] would slice the FIRST axis, wrong for 3D (level, lat, lon) upper-air arrays --
    # explicitly target the second-to-last axis (latitude) so it works for both 2D and 3D fields.
    return field[..., :-1, :] if field.shape[-2] == 721 else field


log('Loading climatology...')
with open('/srv/exw/runs/irina_weather_interpretability/monthly_climatology.pkl', 'rb') as f:
    clim_data = pickle.load(f)
climatology = clim_data['climatology']

log('Opening local 2024-2025 ERA5...')
ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None).sel(level=PRESSURE_LEVELS_HPA)

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


def run_model(input_names, output_names, surf_t, upp_t):
    pass  # placeholder, unused -- real forward done via model.forward(batch) below


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def acc(pred, truth, clim):
    pred_anom = (pred - clim).ravel()
    truth_anom = (truth - clim).ravel()
    num = np.sum(pred_anom * truth_anom)
    den = np.sqrt(np.sum(pred_anom ** 2) * np.sum(truth_anom ** 2))
    return float(num / den) if den > 1e-9 else float('nan')


init_dates = []
for year in [2024, 2025]:
    for month in range(1, 13):
        for day in [1, 9, 17, 25]:
            init_dates.append(np.datetime64(f'{year}-{month:02d}-{day:02d}T00:00'))
log(f'{len(init_dates)} init dates queued')

t_start = time.time()
for idx, init_time in enumerate(init_dates):
    out_path = os.path.join(RESULTS_DIR, f'{str(init_time)}.pkl'.replace(':', ''))
    if os.path.exists(out_path):
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

    month = (init_time.astype('datetime64[M]').astype(int) % 12) + 1
    clim_surf = {v: crop_to_720(climatology[month]['surface'][i]) for i, v in enumerate(SURFACE_VARS)}
    clim_upp = {v: crop_to_720(climatology[month]['upper'][i]) for i, v in enumerate(UPPER_VARS)}

    metrics = {'init_time': str(init_time), 'lead_hours': [], 'rmse': [], 'acc': []}

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

            pred_surf_np = {AURORA_SURF_KEYS[v]: pred.surf_vars[AURORA_SURF_KEYS[v]][0, 0].cpu().numpy()
                            for v in SURFACE_VARS}
            pred_upp_np = {AURORA_UPPER_KEYS[v]: pred.atmos_vars[AURORA_UPPER_KEYS[v]][0, 0].cpu().numpy()
                           for v in UPPER_VARS}

            step_rmse = {}
            step_acc = {}
            for v in SURFACE_VARS:
                truth = crop_to_720(truth_surf_ds[v].values)
                step_rmse[v] = rmse(pred_surf_np[AURORA_SURF_KEYS[v]], truth)
                step_acc[v] = acc(pred_surf_np[AURORA_SURF_KEYS[v]], truth, clim_surf[v])
            for v in UPPER_VARS:
                for lvl_idx, lvl in enumerate(PRESSURE_LEVELS_HPA):
                    truth = crop_to_720(truth_upp_ds[v].sel(level=lvl).values)
                    key = f'{v}@{lvl}hPa'
                    step_rmse[key] = rmse(pred_upp_np[AURORA_UPPER_KEYS[v]][lvl_idx], truth)
                    step_acc[key] = acc(pred_upp_np[AURORA_UPPER_KEYS[v]][lvl_idx], truth, clim_upp[v][lvl_idx])

            metrics['lead_hours'].append(lead_hours)
            metrics['rmse'].append(step_rmse)
            metrics['acc'].append(step_acc)

            batch = _advance_batch(batch, pred)
            del pred
            torch.cuda.empty_cache()

    with open(out_path, 'wb') as f:
        pickle.dump(metrics, f)

    if (idx + 1) % 8 == 0:
        log(f'  [{idx+1}/{len(init_dates)}] done, elapsed {time.time()-t_start:.0f}s')

log(f'All done in {time.time()-t_start:.0f}s')
