"""Dose-response experiment: instead of the binary real-vs-climatology switch, blend the two
continuously and measure how the forecast error of the OTHER fields grows with the blend
fraction.

    x_alpha = (1 - alpha) * x_real + alpha * c_clim ,     alpha in [0, 1]

alpha=0 is the clean baseline and alpha=1 the full replacement -- both already computed in
patching_19var*, so only the intermediate alphas are run here.

Patched fields: Z1000, Z850, Z500.  Saves the same 19 output fields at leads +6h/+24h.

argv[1] = model ('aurora' | 'pangu'), argv[2] = worker tag, argv[3] = comma-separated dates.
"""
import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import xarray as xr

_nv = glob.glob(os.path.join(sys.prefix, 'lib', 'python3.10', 'site-packages', 'nvidia', '*', 'lib'))
os.environ['LD_LIBRARY_PATH'] = ':'.join(_nv) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

MODEL = sys.argv[1]
WORKER = sys.argv[2]
DATES_ARG = sys.argv[3].split(',')

ALPHAS = [0.2, 0.4, 0.6, 0.8]        # 0.0 and 1.0 already exist in patching_19var*
PATCH_VARS = ['Z1000']
N_STEPS = 4
KEEP_STEPS = {0: 6, 3: 24}

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
OUT_DIR = os.path.expanduser(f'~/weather-interpretability/results/dose_response/dose_response_z1000_{MODEL}')
os.makedirs(OUT_DIR, exist_ok=True)

_tri = [(2024, m, d) for m in range(1, 13) for d in (4, 11, 18, 25)]
_hrs = [0, 6, 12, 18] * (len(_tri) // 4)
ALL_DATES = {f'{y}{m:02d}{d:02d}_{h:02d}': np.datetime64(f'{y}-{m:02d}-{d:02d}T{h:02d}:00')
             for (y, m, d), h in zip(_tri, _hrs)}
DATES = {k: ALL_DATES[k] for k in DATES_ARG}
print(f'{MODEL} worker {WORKER}: {len(DATES)} дат x {len(PATCH_VARS)} перем. x {len(ALPHAS)} alpha '
      f'= {len(DATES)*len(PATCH_VARS)*len(ALPHAS)} прогонов', flush=True)

SURFACE_VARS = ['mean_sea_level_pressure', '10m_u_component_of_wind',
                '10m_v_component_of_wind', '2m_temperature']
UPPER_VARS = ['geopotential', 'specific_humidity', 'temperature',
              'u_component_of_wind', 'v_component_of_wind']
CLIM_LEVELS = [500, 850, 1000]
# each model keeps its own native level ordering -- looked up BY VALUE everywhere
LEVELS = ([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] if MODEL == 'aurora'
          else [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50])

VARS = {'MSLP': ('surf', 'mean_sea_level_pressure', None),
        'U10': ('surf', '10m_u_component_of_wind', None),
        'V10': ('surf', '10m_v_component_of_wind', None),
        'T2M': ('surf', '2m_temperature', None)}
for _s, _l in (('Z', 'geopotential'), ('Q', 'specific_humidity'), ('T', 'temperature'),
               ('U', 'u_component_of_wind'), ('V', 'v_component_of_wind')):
    for _lv in (1000, 850, 500):
        VARS[f'{_s}{_lv}'] = ('upper', _l, _lv)

print('Открываю ERA5 + климатологию...', flush=True)
ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None).sel(level=LEVELS)
clim_surf = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_surface.zarr', chunks=None)
clim_upper = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_upper_500_850_1000.zarr', chunks=None)
lat_np, lon_np = ds_surf.latitude.values, ds_surf.longitude.values


def get_clim(t):
    ts = pd.Timestamp(t)
    doy, hour = min(ts.dayofyear, 366), (ts.hour // 6) * 6
    u = clim_upper.sel(dayofyear=doy, hour=hour)
    return {(v, lv): u[v].sel(level=lv).values for v in UPPER_VARS for lv in CLIM_LEVELS}


def build_real(t):
    ss, su = ds_surf.sel(time=t), ds_upper.sel(time=t)
    return ({v: ss[v].values.astype(np.float32) for v in SURFACE_VARS},
            {v: su[v].values.astype(np.float32) for v in UPPER_VARS})


# --------------------------------------------------------------------------- model set-up
if MODEL == 'aurora':
    import torch
    from aurora import Aurora, Batch, Metadata
    from aurora.rollout import _advance_batch
    SK = {'mean_sea_level_pressure': 'msl', '10m_u_component_of_wind': '10u',
          '10m_v_component_of_wind': '10v', '2m_temperature': '2t'}
    UK = {'geopotential': 'z', 'specific_humidity': 'q', 'temperature': 't',
          'u_component_of_wind': 'u', 'v_component_of_wind': 'v'}
    _sp = glob.glob(os.path.expanduser(
        '~/.cache/huggingface/hub/models--microsoft--aurora/snapshots/*/aurora-0.25-static.pickle'))[0]
    with open(_sp, 'rb') as f:
        static_vars = {k: torch.from_numpy(v).float() for k, v in pickle.load(f).items()}
    lat_t, lon_t = torch.from_numpy(lat_np).float(), torch.from_numpy(lon_np).float()
    model = Aurora(use_lora=False)
    model.load_checkpoint('microsoft/aurora', 'aurora-0.25-pretrained.ckpt')
    model.eval()
    model = model.to('cuda')

    def rollout(s0, u0, s1, u1, init_time):
        sv = {SK[v]: torch.from_numpy(np.stack([s0[v], s1[v]])[None]).float() for v in SURFACE_VARS}
        av = {UK[v]: torch.from_numpy(np.stack([u0[v], u1[v]])[None]).float() for v in UPPER_VARS}
        b = Batch(surf_vars=sv, static_vars=static_vars, atmos_vars=av,
                  metadata=Metadata(lat=lat_t, lon=lon_t,
                                     time=(init_time.astype('datetime64[s]').tolist(),),
                                     atmos_levels=tuple(LEVELS)))
        b = b.crop(model.patch_size).to('cuda')
        kept = {}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            for step in range(N_STEPS):
                pred = model.forward(b)
                if step in KEEP_STEPS:
                    f = {}
                    for nm, (kind, var, lv) in VARS.items():
                        f[nm] = (pred.surf_vars[SK[var]][0, 0].float().cpu().numpy() if kind == 'surf'
                                 else pred.atmos_vars[UK[var]][0, 0, LEVELS.index(lv)].float().cpu().numpy())
                    kept[KEEP_STEPS[step]] = f
                b = _advance_batch(b, pred)
        return kept
else:
    import onnxruntime as ort
    sess = ort.InferenceSession(
        os.path.expanduser('~/weather-interpretability/model_weights/pangu_weather_6.onnx'),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]

    def rollout(s0, u0, s1, u1, init_time):        # Pangu ignores the t-6h snapshot
        surf = np.stack([s1[v] for v in SURFACE_VARS], axis=0).astype(np.float32)
        upp = np.stack([u1[v] for v in UPPER_VARS], axis=0).astype(np.float32)
        kept = {}
        for step in range(N_STEPS):
            feed = {n: (surf if 'surface' in n.lower() else upp) for n in in_names}
            raw = dict(zip(out_names, sess.run(None, feed)))
            surf = next(v for k, v in raw.items() if 'surface' in k.lower())
            upp = next(v for k, v in raw.items() if 'surface' not in k.lower())
            if step in KEEP_STEPS:
                f = {}
                for nm, (kind, var, lv) in VARS.items():
                    f[nm] = (surf[SURFACE_VARS.index(var)].copy() if kind == 'surf'
                             else upp[UPPER_VARS.index(var), LEVELS.index(lv)].copy())
                kept[KEEP_STEPS[step]] = f
        return kept


def blend(s0, u0, s1, u1, c0, c1, name, alpha):
    """x_alpha = (1-alpha) * real + alpha * climatology, applied to one field only."""
    _, var, lv = VARS[name]
    u0, u1 = {k: v.copy() for k, v in u0.items()}, {k: v.copy() for k, v in u1.items()}
    li = LEVELS.index(lv)
    u0[var][li] = (1 - alpha) * u0[var][li] + alpha * c0[(var, lv)].astype(np.float32)
    u1[var][li] = (1 - alpha) * u1[var][li] + alpha * c1[(var, lv)].astype(np.float32)
    return dict(s0), u0, dict(s1), u1


for label, init_time in DATES.items():
    out_path = os.path.join(OUT_DIR, f'{label}_{MODEL}.npz')
    if os.path.exists(out_path):
        print(f'=== {label}: есть, пропуск', flush=True)
        continue
    print(f'=== {label} ===', flush=True)
    prev = init_time - np.timedelta64(6, 'h')
    s0, u0 = build_real(prev)
    s1, u1 = build_real(init_time)
    c0, c1 = get_clim(prev), get_clim(init_time)

    res = {}
    for name in PATCH_VARS:
        # how big is the perturbation actually, in units of the field's own spread?
        li = LEVELS.index(VARS[name][2])
        d_in = np.abs(c1[(VARS[name][1], VARS[name][2])] - u1[VARS[name][1]][li])
        res[f'inputdiff_{name}'] = np.array([d_in.mean(), d_in.std(), u1[VARS[name][1]][li].std()])
        for a in ALPHAS:
            print(f'  {name} alpha={a}', flush=True)
            ps0, pu0, ps1, pu1 = blend(s0, u0, s1, u1, c0, c1, name, a)
            for lead, f in rollout(ps0, pu0, ps1, pu1, init_time).items():
                for nm, arr in f.items():
                    res[f'{name}_a{a}_lead{lead}_{nm}'] = arr

    tmp = out_path + '.tmp.npz'
    np.savez_compressed(tmp, **res)
    os.replace(tmp, out_path)
    print(f'  сохранено {out_path} ({os.path.getsize(out_path)/1e6:.0f} МБ)', flush=True)

print(f'worker {WORKER} готов.', flush=True)
