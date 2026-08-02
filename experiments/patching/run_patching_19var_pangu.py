"""Input-level climatological patching for Pangu-Weather (6h checkpoint) -- exact companion to
run_patching_19var.py (Aurora), same 48 init dates / same 19 variables / same 2 leads, so the
two models' 19x19 matrices are directly comparable.

Pangu-specific details that have caused bugs before, guarded here:
  * pressure levels are DESCENDING [1000...50], the reverse of Aurora's -- levels are looked up
    BY VALUE (`PRESSURE_LEVELS_HPA.index(lvl)`), never by a hardcoded position;
  * Pangu keeps the full 721-latitude grid (Aurora crops the south pole to 720), so truth and
    climatology must NOT be cropped when comparing against Pangu output;
  * the ONNX graph's outputs come back as [upper, surface], so they are matched by NAME
    substring rather than by position.

Unlike Aurora (2 input timesteps), Pangu conditions on a SINGLE snapshot, so the patch is
applied to just that one -- an inherent architecture difference, noted when comparing.

argv[1] = worker index (logging only), argv[2] = comma-separated date labels for this worker.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
import onnxruntime as ort

_nvidia_lib_dirs = glob.glob(os.path.join(sys.prefix, 'lib', 'python3.10', 'site-packages', 'nvidia', '*', 'lib'))
os.environ['LD_LIBRARY_PATH'] = ':'.join(_nvidia_lib_dirs) + ':' + os.environ.get('LD_LIBRARY_PATH', '')

WORKER = sys.argv[1]
DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
OUT_DIR = os.path.expanduser('~/weather-interpretability/results/patching/patching_19var_pangu')
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_PATH = os.path.expanduser('~/weather-interpretability/model_weights/pangu_weather_6.onnx')
N_STEPS = 4
KEEP_STEPS = {0: 6, 3: 24}

# ---- identical 48 init dates as the Aurora run ----
_triples = [(2024, m, d) for m in range(1, 13) for d in (4, 11, 18, 25)]
_hours = [0, 6, 12, 18] * (len(_triples) // 4)
ALL_DATES = {f'{y}{m:02d}{d:02d}_{h:02d}': np.datetime64(f'{y}-{m:02d}-{d:02d}T{h:02d}:00')
             for (y, m, d), h in zip(_triples, _hours)}
assert len(ALL_DATES) == 48

DATES = {k: ALL_DATES[k] for k in sys.argv[2].split(',')}
print(f'Worker {WORKER}: {len(DATES)} dates -> {list(DATES)}', flush=True)

# Pangu's own variable order and its DESCENDING level order
PRESSURE_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SURFACE_VARS = ['mean_sea_level_pressure', '10m_u_component_of_wind',
                '10m_v_component_of_wind', '2m_temperature']
UPPER_VARS = ['geopotential', 'specific_humidity', 'temperature',
              'u_component_of_wind', 'v_component_of_wind']
CLIM_LEVELS = [500, 850, 1000]

VARS: dict = {
    'MSLP': ('surf', 'mean_sea_level_pressure', None),
    'U10': ('surf', '10m_u_component_of_wind', None),
    'V10': ('surf', '10m_v_component_of_wind', None),
    'T2M': ('surf', '2m_temperature', None),
}
for _s, _l in (('Z', 'geopotential'), ('Q', 'specific_humidity'), ('T', 'temperature'),
               ('U', 'u_component_of_wind'), ('V', 'v_component_of_wind')):
    for _lv in (1000, 850, 500):
        VARS[f'{_s}{_lv}'] = ('upper', _l, _lv)
assert len(VARS) == 19
NAMES = list(VARS)

print('Opening ERA5 + climatology...', flush=True)
ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr',
                        chunks=None).sel(level=PRESSURE_LEVELS_HPA)
clim_surf = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_surface.zarr', chunks=None)
clim_upper = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_upper_500_850_1000.zarr', chunks=None)


def get_climatology(t):
    ts = pd.Timestamp(t)
    doy, hour = min(ts.dayofyear, 366), (ts.hour // 6) * 6
    s = clim_surf.sel(dayofyear=doy, hour=hour)
    u = clim_upper.sel(dayofyear=doy, hour=hour)
    cs = {v: s[v].values for v in SURFACE_VARS}
    cu = {(v, lvl): u[v].sel(level=lvl).values for v in UPPER_VARS for lvl in CLIM_LEVELS}
    return cs, cu


print('Loading Pangu-Weather 6h (ONNX)...', flush=True)
session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_names = [i.name for i in session.get_inputs()]
output_names = [o.name for o in session.get_outputs()]


def run_model(surf, upp):
    """One 6h step. Outputs are matched by NAME (graph order is [upper, surface])."""
    feed = {n: (surf if 'surface' in n.lower() else upp) for n in input_names}
    raw = session.run(None, feed)
    by_name = dict(zip(output_names, raw))
    out_surf = next(v for k, v in by_name.items() if 'surface' in k.lower())
    out_upp = next(v for k, v in by_name.items() if 'surface' not in k.lower())
    return out_surf, out_upp


def build_real(t):
    ss, su = ds_surf.sel(time=t), ds_upper.sel(time=t)
    surf = np.stack([ss[v].values for v in SURFACE_VARS], axis=0).astype(np.float32)
    upp = np.stack([su[v].values for v in UPPER_VARS], axis=0).astype(np.float32)
    return surf, upp   # (4,721,1440), (5,13,721,1440)


def apply_patch(surf, upp, cs, cu, name):
    kind, var, lvl = VARS[name]
    surf, upp = surf.copy(), upp.copy()
    if kind == 'surf':
        surf[SURFACE_VARS.index(var)] = cs[var].astype(np.float32)
    else:
        vi = UPPER_VARS.index(var)
        li = PRESSURE_LEVELS_HPA.index(lvl)      # by VALUE, never a hardcoded index
        upp[vi, li] = cu[(var, lvl)].astype(np.float32)
    return surf, upp


def run_rollout(surf, upp):
    kept = {}
    for step in range(N_STEPS):
        surf, upp = run_model(surf, upp)
        if step in KEEP_STEPS:
            lead = KEEP_STEPS[step]
            f = {}
            for nm, (kind, var, lvl) in VARS.items():
                if kind == 'surf':
                    f[nm] = surf[SURFACE_VARS.index(var)].copy()
                else:
                    f[nm] = upp[UPPER_VARS.index(var), PRESSURE_LEVELS_HPA.index(lvl)].copy()
            kept[lead] = f
    return kept


for label, init_time in DATES.items():
    out_path = os.path.join(OUT_DIR, f'{label}_pangu.npz')
    if os.path.exists(out_path):
        print(f'=== {label}: exists, skip', flush=True)
        continue
    print(f'=== {label}: init={init_time} ===', flush=True)
    surf, upp = build_real(init_time)
    cs, cu = get_climatology(init_time)

    res = {}
    print('  baseline...', flush=True)
    for lead, f in run_rollout(surf, upp).items():
        for nm, arr in f.items():
            res[f'baseline_lead{lead}_{nm}'] = arr

    for name in NAMES:
        print(f'  patch {name}...', flush=True)
        ps, pu = apply_patch(surf, upp, cs, cu, name)
        for lead, f in run_rollout(ps, pu).items():
            for nm, arr in f.items():
                res[f'patch{name}_lead{lead}_{nm}'] = arr

    tmp = out_path + '.tmp.npz'
    np.savez_compressed(tmp, **res)
    os.replace(tmp, out_path)
    print(f'  saved {out_path} ({os.path.getsize(out_path)/1e6:.0f} MB, {len(res)} arrays)', flush=True)

print(f'Worker {WORKER} done.', flush=True)
