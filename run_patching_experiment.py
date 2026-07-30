"""Input-level climatological patching experiment for Aurora: for two dates (Bebinca typhoon,
init=2024-09-15; and a quiet control, init=2024-11-15), replace ONE physical component of the
model input (both input timesteps) with its climatological mean, keep everything else real,
run the full 4-step (6h) rollout, and save the output alongside an unpatched baseline (all
real input) run. This is a causal input-ablation test: how much does the forecast change (and
how much does its ACCURACY change vs ground truth) when the model is denied real information
about one physical component.

Components tested: surface pressure (MSLP), wind (10m + upper u/v), mass field (upper
geopotential), upper temperature, and 2m temperature (separately from MSLP).

CAVEAT (documented, not hidden): our downloaded WB2 climatology only covers 3 pressure levels
(500, 850, 1000 hPa) for upper-air variables, out of the 13 levels Aurora uses. So wind/
mass-field/temperature patches are PARTIAL-COLUMN: only those 3 levels are replaced by
climatology; the other 10 levels remain real. Surface variables (MSLP, T2M, 10m wind) have no
such restriction -- full climatological replacement.

Each component is patched in two scopes: 'global' (everywhere) and 'regional' (only inside
CHINA_EXTENT, rest of the globe stays real).
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

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
OUT_DIR = os.path.expanduser('~/weather-interpretability/results/patching_experiment')
os.makedirs(OUT_DIR, exist_ok=True)
CHINA_EXTENT = [95, 130, 15, 45]  # lon_min, lon_max, lat_min, lat_max
N_STEPS = 4

DATES = {'storm': np.datetime64('2024-09-15T00:00'), 'quiet': np.datetime64('2024-11-15T00:00')}

SURFACE_VARS = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
UPPER_VARS = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
AURORA_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
CLIM_LEVELS = [500, 850, 1000]
CLIM_LEVEL_IDX = [AURORA_LEVELS.index(l) for l in CLIM_LEVELS]  # indices into the 13-level axis
AURORA_SURF_KEYS = {'mean_sea_level_pressure': 'msl', '10m_u_component_of_wind': '10u',
                    '10m_v_component_of_wind': '10v', '2m_temperature': '2t'}
AURORA_UPPER_KEYS = {'geopotential': 'z', 'specific_humidity': 'q', 'temperature': 't',
                     'u_component_of_wind': 'u', 'v_component_of_wind': 'v'}

# component -> (surf_vars_to_patch, upper_vars_to_patch)
COMPONENTS = {
    'surface_mslp': (['mean_sea_level_pressure'], []),
    'wind': (['10m_u_component_of_wind', '10m_v_component_of_wind'],
              ['u_component_of_wind', 'v_component_of_wind']),
    'mass_field': ([], ['geopotential']),
    'temperature': ([], ['temperature']),
    't2m': (['2m_temperature'], []),
}
SCOPES = ['global', 'regional']

print('Loading ERA5 + climatology...', flush=True)
ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None).sel(level=AURORA_LEVELS)
clim_surf_ds = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_surface.zarr', chunks=None)
clim_upper_ds = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_upper_500_850_1000.zarr', chunks=None)

lat_full = ds_surf.latitude.values
lon_full = ds_surf.longitude.values
lat_mask = (lat_full >= CHINA_EXTENT[2]) & (lat_full <= CHINA_EXTENT[3])
lon_mask = (lon_full >= CHINA_EXTENT[0]) & (lon_full <= CHINA_EXTENT[1])
region_idx = np.ix_(lat_mask, lon_mask)


def get_climatology(time_val):
    ts = pd.Timestamp(time_val)
    doy = min(ts.dayofyear, 366)
    hour = (ts.hour // 6) * 6
    surf_snap = clim_surf_ds.sel(dayofyear=doy, hour=hour)
    upper_snap = clim_upper_ds.sel(dayofyear=doy, hour=hour)
    surf_clim = {v: surf_snap[v].values for v in SURFACE_VARS}
    upper_clim = {v: upper_snap[v].values for v in UPPER_VARS}  # (level=3, lat, lon)
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
    upper = {v: snap_upp[v].values.astype(np.float32) for v in UPPER_VARS}  # (13, lat, lon)
    return surf, upper


def apply_patch(surf0, upper0, surf1, upper1, clim0_surf, clim0_upp, clim1_surf, clim1_upp,
                 component, scope):
    """Return patched copies of (surf0, upper0, surf1, upper1) -- the two input timesteps --
    with the given component replaced by its climatology, in the given spatial scope."""
    surf_vars_to_patch, upper_vars_to_patch = COMPONENTS[component]
    surf0, surf1 = {k: v.copy() for k, v in surf0.items()}, {k: v.copy() for k, v in surf1.items()}
    upper0, upper1 = {k: v.copy() for k, v in upper0.items()}, {k: v.copy() for k, v in upper1.items()}

    def patch_field(field, clim_field):
        if scope == 'global':
            return clim_field.astype(np.float32)
        else:
            out = field.copy()
            out[region_idx] = clim_field[region_idx]
            return out

    for v in surf_vars_to_patch:
        surf0[v] = patch_field(surf0[v], clim0_surf[v])
        surf1[v] = patch_field(surf1[v], clim1_surf[v])
    for v in upper_vars_to_patch:
        for li_full, li_clim in zip(CLIM_LEVEL_IDX, range(len(CLIM_LEVELS))):
            upper0[v][li_full] = patch_field(upper0[v][li_full], clim0_upp[v][li_clim])
            upper1[v][li_full] = patch_field(upper1[v][li_full], clim1_upp[v][li_clim])
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
    surf_out, upper_out = [], []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for _ in range(N_STEPS):
            pred = model.forward(batch)
            surf_out.append(np.stack([pred.surf_vars[AURORA_SURF_KEYS[v]][0, 0].float().cpu().numpy()
                                       for v in SURFACE_VARS]))
            upper_out.append(np.stack([pred.atmos_vars[AURORA_UPPER_KEYS[v]][0, 0].float().cpu().numpy()
                                        for v in UPPER_VARS]))
            batch = _advance_batch(batch, pred)
    return np.stack(surf_out), np.stack(upper_out)  # (N_STEPS, nvars, [level,] lat, lon)


for date_label, init_time in DATES.items():
    print(f'=== date {date_label}: init={init_time} ===', flush=True)
    prev_time = init_time - np.timedelta64(6, 'h')
    surf0, upper0 = build_real_tensors(prev_time)
    surf1, upper1 = build_real_tensors(init_time)
    clim0_surf, clim0_upp = get_climatology(prev_time)
    clim1_surf, clim1_upp = get_climatology(init_time)

    results = {}
    print('  running baseline (all real)...', flush=True)
    b_surf, b_upper = run_rollout(surf0, upper0, surf1, upper1, init_time)
    results['baseline_surf'] = b_surf
    results['baseline_upper'] = b_upper

    for component in COMPONENTS:
        for scope in SCOPES:
            key = f'{component}_{scope}'
            print(f'  running {key}...', flush=True)
            p_surf0, p_upper0, p_surf1, p_upper1 = apply_patch(
                surf0, upper0, surf1, upper1, clim0_surf, clim0_upp, clim1_surf, clim1_upp,
                component, scope)
            p_surf, p_upper = run_rollout(p_surf0, p_upper0, p_surf1, p_upper1, init_time)
            results[f'{key}_surf'] = p_surf
            results[f'{key}_upper'] = p_upper

    out_path = os.path.join(OUT_DIR, f'{date_label}.npz')
    np.savez_compressed(out_path, **results)
    print(f'  saved to {out_path}', flush=True)

print('All done.', flush=True)
