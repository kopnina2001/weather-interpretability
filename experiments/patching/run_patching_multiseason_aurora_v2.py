"""Same as run_patching_multiseason_aurora.py, but takes an explicit comma-separated list of
date LABELS to (re)compute via argv[2], instead of an even chunk split -- used to recompute
only the missing/corrupted dates after the disk-full incident. argv[1] is still the GPU/worker
index (just for logging), argv[2] is the comma-separated label list for this process.
"""
import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import xarray as xr

_nvidia_lib_dirs = glob.glob(os.path.join(sys.prefix, 'lib', 'python3.10', 'site-packages', 'nvidia', '*', 'lib'))
os.environ['LD_LIBRARY_PATH'] = ':'.join(_nvidia_lib_dirs) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from aurora import Aurora, Batch, Metadata
from aurora.rollout import _advance_batch

WORKER_IDX = sys.argv[1]
LABELS_ARG = sys.argv[2].split(',')

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
OUT_DIR = os.path.expanduser('~/weather-interpretability/results/legacy/patching_experiment_multiseason')
os.makedirs(OUT_DIR, exist_ok=True)
N_STEPS = 4
KEEP_STEP_IDX = [0, 3]
LEAD_LABELS = {0: 6, 3: 24}

date_triples = [(y, m, d) for y in (2024, 2025) for m in range(1, 13) for d in (8, 23)]
hours_cycle = [0, 6, 12, 18] * (len(date_triples) // 4)
ALL_DATES = {f'{y}{m:02d}{d:02d}_{h:02d}': np.datetime64(f'{y}-{m:02d}-{d:02d}T{h:02d}:00')
             for (y, m, d), h in zip(date_triples, hours_cycle)}

DATES = {k: ALL_DATES[k] for k in LABELS_ARG}
print(f'Worker {WORKER_IDX}: {len(DATES)} dates: {list(DATES.keys())}', flush=True)

SURFACE_VARS = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
UPPER_VARS = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
AURORA_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
CLIM_LEVELS = [500, 850, 1000]
AURORA_SURF_KEYS = {'mean_sea_level_pressure': 'msl', '10m_u_component_of_wind': '10u',
                    '10m_v_component_of_wind': '10v', '2m_temperature': '2t'}
AURORA_UPPER_KEYS = {'geopotential': 'z', 'specific_humidity': 'q', 'temperature': 't',
                     'u_component_of_wind': 'u', 'v_component_of_wind': 'v'}

COMPONENTS = {
    'MSLP': ('surf', 'mean_sea_level_pressure', None),
    'T2M': ('surf', '2m_temperature', None),
    'Z500': ('upper', 'geopotential', 500),
    'Z850': ('upper', 'geopotential', 850),
    'T850': ('upper', 'temperature', 850),
    'U850': ('upper', 'u_component_of_wind', 850),
    'V850': ('upper', 'v_component_of_wind', 850),
}

print('Loading ERA5 + climatology...', flush=True)
ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None).sel(level=AURORA_LEVELS)
clim_surf_ds = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_surface.zarr', chunks=None)
clim_upper_ds = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_upper_500_850_1000.zarr', chunks=None)

lat_full = ds_surf.latitude.values
lon_full = ds_surf.longitude.values


def get_climatology(time_val):
    ts = pd.Timestamp(time_val)
    doy = min(ts.dayofyear, 366)
    hour = (ts.hour // 6) * 6
    surf_snap = clim_surf_ds.sel(dayofyear=doy, hour=hour)
    upper_snap = clim_upper_ds.sel(dayofyear=doy, hour=hour)
    surf_clim = {v: surf_snap[v].values for v in SURFACE_VARS}
    upper_clim = {(v, lvl): upper_snap[v].sel(level=lvl).values for v in UPPER_VARS for lvl in CLIM_LEVELS}
    return surf_clim, upper_clim


static_path = glob.glob(os.path.expanduser(
    '~/.cache/huggingface/hub/models--microsoft--aurora/snapshots/*/aurora-0.25-static.pickle'))[0]
with open(static_path, 'rb') as f:
    static_raw = pickle.load(f)
static_vars = {k: torch.from_numpy(v).float() for k, v in static_raw.items()}
lat_t = torch.from_numpy(lat_full).float()
lon_t = torch.from_numpy(lon_full).float()

print('Loading Aurora model...', flush=True)
model = Aurora(use_lora=False)
model.load_checkpoint('microsoft/aurora', 'aurora-0.25-pretrained.ckpt')
model.eval()
model = model.to('cuda')


def build_real_tensors(time_val):
    snap_surf = ds_surf.sel(time=time_val)
    snap_upp = ds_upper.sel(time=time_val)
    surf = {v: snap_surf[v].values.astype(np.float32) for v in SURFACE_VARS}
    upper = {v: snap_upp[v].values.astype(np.float32) for v in UPPER_VARS}
    return surf, upper


def apply_patch(surf0, upper0, surf1, upper1, clim0_surf, clim0_upp, clim1_surf, clim1_upp, component):
    kind, var, level = COMPONENTS[component]
    surf0, surf1 = {k: v.copy() for k, v in surf0.items()}, {k: v.copy() for k, v in surf1.items()}
    upper0, upper1 = {k: v.copy() for k, v in upper0.items()}, {k: v.copy() for k, v in upper1.items()}
    if kind == 'surf':
        surf0[var] = clim0_surf[var].astype(np.float32)
        surf1[var] = clim1_surf[var].astype(np.float32)
    else:
        li = AURORA_LEVELS.index(level)
        upper0[var][li] = clim0_upp[(var, level)].astype(np.float32)
        upper1[var][li] = clim1_upp[(var, level)].astype(np.float32)
    return surf0, upper0, surf1, upper1


def run_rollout(surf0, upper0, surf1, upper1, init_time):
    surf_vars = {AURORA_SURF_KEYS[v]: torch.from_numpy(np.stack([surf0[v], surf1[v]])[None]).float()
                 for v in SURFACE_VARS}
    atmos_vars = {AURORA_UPPER_KEYS[v]: torch.from_numpy(np.stack([upper0[v], upper1[v]])[None]).float()
                  for v in UPPER_VARS}
    batch = Batch(surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars,
                  metadata=Metadata(lat=lat_t, lon=lon_t,
                                     time=(init_time.astype('datetime64[s]').tolist(),),
                                     atmos_levels=tuple(AURORA_LEVELS)))
    batch = batch.crop(model.patch_size)
    batch = batch.to('cuda')
    kept = {}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for step in range(N_STEPS):
            pred = model.forward(batch)
            if step in KEEP_STEP_IDX:
                lead = LEAD_LABELS[step]
                fields = {}
                for cname, (kind, var, level) in COMPONENTS.items():
                    if kind == 'surf':
                        fields[cname] = pred.surf_vars[AURORA_SURF_KEYS[var]][0, 0].float().cpu().numpy()
                    else:
                        li = AURORA_LEVELS.index(level)
                        fields[cname] = pred.atmos_vars[AURORA_UPPER_KEYS[var]][0, 0, li].float().cpu().numpy()
                kept[lead] = fields
            batch = _advance_batch(batch, pred)
    return kept


for date_label, init_time in DATES.items():
    print(f'=== date {date_label}: init={init_time} ===', flush=True)
    prev_time = init_time - np.timedelta64(6, 'h')
    surf0, upper0 = build_real_tensors(prev_time)
    surf1, upper1 = build_real_tensors(init_time)
    clim0_surf, clim0_upp = get_climatology(prev_time)
    clim1_surf, clim1_upp = get_climatology(init_time)

    results = {}
    print('  running baseline...', flush=True)
    baseline_kept = run_rollout(surf0, upper0, surf1, upper1, init_time)
    for lead, fields in baseline_kept.items():
        for cname, arr in fields.items():
            results[f'baseline_lead{lead}_{cname}'] = arr

    for component in COMPONENTS:
        print(f'  running patch_{component}...', flush=True)
        p_surf0, p_upper0, p_surf1, p_upper1 = apply_patch(
            surf0, upper0, surf1, upper1, clim0_surf, clim0_upp, clim1_surf, clim1_upp, component)
        patched_kept = run_rollout(p_surf0, p_upper0, p_surf1, p_upper1, init_time)
        for lead, fields in patched_kept.items():
            for cname, arr in fields.items():
                results[f'patch{component}_lead{lead}_{cname}'] = arr

    out_path = os.path.join(OUT_DIR, f'{date_label}_aurora.npz')
    np.savez_compressed(out_path, **results)
    print(f'  saved to {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)', flush=True)

print(f'Worker {WORKER_IDX} all done.', flush=True)
