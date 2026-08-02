"""Input-level climatological patching for Aurora -- full 19x19 design.

Sampling: ONE year (2024), 4 init dates per month (days 4/11/18/25) x 12 months = 48 inits,
init hour cycling 00/06/12/18 evenly (12 of each) so time-of-day is balanced.

Variables (19) -- every field for which we have WB2 climatology, so every one is patchable
with a FULL (not partial-column) climatological replacement:
    surface: MSLP, U10, V10, T2M
    upper @ 1000/850/500 hPa: Z, Q, T, U, V
Each is patched one at a time (global scope), and each is also an output column -> 19x19.

Saved per date: all 19 output fields for baseline + each of the 19 patched runs, at leads
+6h and +24h (one 4-step rollout per run; steps 0 and 3 kept).
    20 runs x 2 leads x 19 fields = 760 arrays/date, ~1.7 GB/date, ~81 GB for 48 dates.

argv[1] = worker index (for logging), argv[2] = comma-separated date labels for this worker.
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

WORKER = sys.argv[1]
DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
OUT_DIR = os.path.expanduser('~/weather-interpretability/results/patching_19var')
os.makedirs(OUT_DIR, exist_ok=True)
N_STEPS = 4
KEEP_STEPS = {0: 6, 3: 24}   # rollout step index -> lead hours

# ---- 48 init dates: 2024 only, days 4/11/18/25, hour cycling 00/06/12/18 ----
_triples = [(2024, m, d) for m in range(1, 13) for d in (4, 11, 18, 25)]
_hours = [0, 6, 12, 18] * (len(_triples) // 4)
ALL_DATES = {f'{y}{m:02d}{d:02d}_{h:02d}': np.datetime64(f'{y}-{m:02d}-{d:02d}T{h:02d}:00')
             for (y, m, d), h in zip(_triples, _hours)}
assert len(ALL_DATES) == 48, len(ALL_DATES)

DATES = {k: ALL_DATES[k] for k in sys.argv[2].split(',')}
print(f'Worker {WORKER}: {len(DATES)} dates -> {list(DATES)}', flush=True)

SURFACE_VARS = ['mean_sea_level_pressure', '10m_u_component_of_wind',
                '10m_v_component_of_wind', '2m_temperature']
UPPER_VARS = ['geopotential', 'specific_humidity', 'temperature',
              'u_component_of_wind', 'v_component_of_wind']
AURORA_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
CLIM_LEVELS = [500, 850, 1000]
SK = {'mean_sea_level_pressure': 'msl', '10m_u_component_of_wind': '10u',
      '10m_v_component_of_wind': '10v', '2m_temperature': '2t'}
UK = {'geopotential': 'z', 'specific_humidity': 'q', 'temperature': 't',
      'u_component_of_wind': 'u', 'v_component_of_wind': 'v'}

# name -> (kind, era5_var_name, level)
VARS: dict = {
    'MSLP': ('surf', 'mean_sea_level_pressure', None),
    'U10': ('surf', '10m_u_component_of_wind', None),
    'V10': ('surf', '10m_v_component_of_wind', None),
    'T2M': ('surf', '2m_temperature', None),
}
for _short, _long in (('Z', 'geopotential'), ('Q', 'specific_humidity'), ('T', 'temperature'),
                      ('U', 'u_component_of_wind'), ('V', 'v_component_of_wind')):
    for _lvl in (1000, 850, 500):
        VARS[f'{_short}{_lvl}'] = ('upper', _long, _lvl)
assert len(VARS) == 19, len(VARS)
NAMES = list(VARS)

print('Opening ERA5 + climatology...', flush=True)
ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None).sel(level=AURORA_LEVELS)
clim_surf = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_surface.zarr', chunks=None)
clim_upper = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_upper_500_850_1000.zarr', chunks=None)

lat_np = ds_surf.latitude.values
lon_np = ds_surf.longitude.values


def get_climatology(t):
    ts = pd.Timestamp(t)
    doy = min(ts.dayofyear, 366)
    hour = (ts.hour // 6) * 6
    s = clim_surf.sel(dayofyear=doy, hour=hour)
    u = clim_upper.sel(dayofyear=doy, hour=hour)
    cs = {v: s[v].values for v in SURFACE_VARS}
    cu = {(v, lvl): u[v].sel(level=lvl).values for v in UPPER_VARS for lvl in CLIM_LEVELS}
    return cs, cu


static_path = glob.glob(os.path.expanduser(
    '~/.cache/huggingface/hub/models--microsoft--aurora/snapshots/*/aurora-0.25-static.pickle'))[0]
with open(static_path, 'rb') as f:
    static_raw = pickle.load(f)
static_vars = {k: torch.from_numpy(v).float() for k, v in static_raw.items()}
lat_t = torch.from_numpy(lat_np).float()
lon_t = torch.from_numpy(lon_np).float()

print('Loading Aurora...', flush=True)
model = Aurora(use_lora=False)
model.load_checkpoint('microsoft/aurora', 'aurora-0.25-pretrained.ckpt')
model.eval()
model = model.to('cuda')


def build_real(t):
    ss, su = ds_surf.sel(time=t), ds_upper.sel(time=t)
    surf = {v: ss[v].values.astype(np.float32) for v in SURFACE_VARS}
    upper = {v: su[v].values.astype(np.float32) for v in UPPER_VARS}   # (13, lat, lon)
    return surf, upper


def apply_patch(s0, u0, s1, u1, cs0, cu0, cs1, cu1, name):
    kind, var, lvl = VARS[name]
    s0, s1 = {k: v.copy() for k, v in s0.items()}, {k: v.copy() for k, v in s1.items()}
    u0, u1 = {k: v.copy() for k, v in u0.items()}, {k: v.copy() for k, v in u1.items()}
    if kind == 'surf':
        s0[var] = cs0[var].astype(np.float32)
        s1[var] = cs1[var].astype(np.float32)
    else:
        li = AURORA_LEVELS.index(lvl)
        u0[var][li] = cu0[(var, lvl)].astype(np.float32)
        u1[var][li] = cu1[(var, lvl)].astype(np.float32)
    return s0, u0, s1, u1


def run_rollout(s0, u0, s1, u1, init_time):
    sv = {SK[v]: torch.from_numpy(np.stack([s0[v], s1[v]])[None]).float() for v in SURFACE_VARS}
    av = {UK[v]: torch.from_numpy(np.stack([u0[v], u1[v]])[None]).float() for v in UPPER_VARS}
    batch = Batch(surf_vars=sv, static_vars=static_vars, atmos_vars=av,
                  metadata=Metadata(lat=lat_t, lon=lon_t,
                                     time=(init_time.astype('datetime64[s]').tolist(),),
                                     atmos_levels=tuple(AURORA_LEVELS)))
    batch = batch.crop(model.patch_size).to('cuda')
    kept = {}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for step in range(N_STEPS):
            pred = model.forward(batch)
            if step in KEEP_STEPS:
                lead = KEEP_STEPS[step]
                f = {}
                for nm, (kind, var, lvl) in VARS.items():
                    if kind == 'surf':
                        f[nm] = pred.surf_vars[SK[var]][0, 0].float().cpu().numpy()
                    else:
                        li = AURORA_LEVELS.index(lvl)
                        f[nm] = pred.atmos_vars[UK[var]][0, 0, li].float().cpu().numpy()
                kept[lead] = f
            batch = _advance_batch(batch, pred)
    return kept


for label, init_time in DATES.items():
    out_path = os.path.join(OUT_DIR, f'{label}_aurora.npz')
    if os.path.exists(out_path):
        print(f'=== {label}: exists, skip', flush=True)
        continue
    print(f'=== {label}: init={init_time} ===', flush=True)
    prev = init_time - np.timedelta64(6, 'h')
    s0, u0 = build_real(prev)
    s1, u1 = build_real(init_time)
    cs0, cu0 = get_climatology(prev)
    cs1, cu1 = get_climatology(init_time)

    res = {}
    print('  baseline...', flush=True)
    for lead, f in run_rollout(s0, u0, s1, u1, init_time).items():
        for nm, arr in f.items():
            res[f'baseline_lead{lead}_{nm}'] = arr

    for name in NAMES:
        print(f'  patch {name}...', flush=True)
        ps0, pu0, ps1, pu1 = apply_patch(s0, u0, s1, u1, cs0, cu0, cs1, cu1, name)
        for lead, f in run_rollout(ps0, pu0, ps1, pu1, init_time).items():
            for nm, arr in f.items():
                res[f'patch{name}_lead{lead}_{nm}'] = arr

    tmp_path = out_path + '.tmp.npz'
    np.savez_compressed(tmp_path, **res)
    os.replace(tmp_path, out_path)   # atomic: a crash mid-write can never leave a half-file
    print(f'  saved {out_path} ({os.path.getsize(out_path)/1e6:.0f} MB, {len(res)} arrays)', flush=True)

print(f'Worker {WORKER} done.', flush=True)
