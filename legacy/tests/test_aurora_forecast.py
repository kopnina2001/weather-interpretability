import os
import sys
import glob
import pickle
import numpy as np

_nvidia_lib_dirs = glob.glob(os.path.join(sys.prefix, 'lib', 'python3.10', 'site-packages', 'nvidia', '*', 'lib'))
os.environ['LD_LIBRARY_PATH'] = ':'.join(_nvidia_lib_dirs) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import xarray as xr
from aurora import Aurora, Batch, Metadata
from aurora.rollout import _advance_batch  # bypass aurora.rollout()'s preprocessing wrapper (see note below)
from datetime import datetime

PRESSURE_LEVELS_HPA = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SURF_VARS = {'2t': '2m_temperature', '10u': '10m_u_component_of_wind', '10v': '10m_v_component_of_wind', 'msl': 'mean_sea_level_pressure'}
ATMOS_VARS = {'z': 'geopotential', 'u': 'u_component_of_wind', 'v': 'v_component_of_wind', 't': 'temperature', 'q': 'specific_humidity'}

CASE = 'storm_eunice'
CASE_TIME = np.datetime64('2022-02-18T12:00')
PREV_TIME = CASE_TIME - np.timedelta64(6, 'h')

print('Opening ARCO...')
ARCO_URL = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
ds_arco = xr.open_zarr(ARCO_URL, storage_options=dict(token='anon'), chunks=None)
ds_arco = ds_arco.assign_coords(longitude=(ds_arco.longitude % 360)).sortby('longitude')

print('Fetching t-6h and t snapshots...')
snap_prev = ds_arco.sel(time=PREV_TIME, method='nearest')
snap_now = ds_arco.sel(time=CASE_TIME, method='nearest')

surf_vars = {}
for short, long in SURF_VARS.items():
    prev = snap_prev[long].values
    now = snap_now[long].values
    surf_vars[short] = torch.from_numpy(np.stack([prev, now])[None]).float()  # (1, 2, 721, 1440)

atmos_vars = {}
for short, long in ATMOS_VARS.items():
    prev = snap_prev[long].sel(level=PRESSURE_LEVELS_HPA).values
    now = snap_now[long].sel(level=PRESSURE_LEVELS_HPA).values
    atmos_vars[short] = torch.from_numpy(np.stack([prev, now])[None]).float()  # (1, 2, 13, 721, 1440)

print('Loading static vars...')
static_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--microsoft--aurora/snapshots/*/aurora-0.25-static.pickle'))[0]
with open(static_path, 'rb') as f:
    static_raw = pickle.load(f)
static_vars = {k: torch.from_numpy(v).float() for k, v in static_raw.items()}

lat = torch.from_numpy(ds_arco.latitude.values).float()
lon = torch.from_numpy(ds_arco.longitude.values).float()

batch = Batch(
    surf_vars=surf_vars,
    static_vars=static_vars,
    atmos_vars=atmos_vars,
    metadata=Metadata(
        lat=lat, lon=lon,
        time=(CASE_TIME.astype('datetime64[s]').tolist(),),  # single reference time, NOT one per input timestep
        atmos_levels=tuple(PRESSURE_LEVELS_HPA),
    ),
)

print('Loading model...')
model = Aurora(use_lora=False)
model.load_checkpoint('microsoft/aurora', 'aurora-0.25-pretrained.ckpt')
model.eval()
model = model.to('cuda')
batch = batch.crop(model.patch_size)  # model output drops the last (south pole) lat row; crop input to match once,
batch = batch.to('cuda')              # so subsequent _advance_batch concatenations have matching shapes
# fp32 weights kept (lat/lon metadata must stay fp32/64 for numerical stability -- Aurora asserts this),
# but forward compute runs under autocast fp16 below to cut activation memory (V100 has fast fp16 tensor cores).
#
# NOTE: aurora.rollout()'s own preprocessing (batch_transform_hook + type + crop + to(device), run fresh
# every step) pushed memory ~8GB higher than a bare model.forward() call and OOM'd even under autocast.
# A direct model.forward() call (which crops internally anyway, see aurora/model/aurora.py:361) peaked at
# only 19.6GB on the same real data. So we replicate rollout()'s 4-step autoregression manually here,
# calling forward() directly and advancing the batch with rollout's own _advance_batch() helper.

print('Running manual rollout (4 steps x 6h = 24h), bypassing aurora.rollout()...')
preds = []
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
    for i in range(4):
        pred = model.forward(batch)
        preds.append(pred.to('cpu'))
        batch = _advance_batch(batch, pred)
        del pred
        torch.cuda.empty_cache()
        print(f'  step {i+1}/4 done, peak mem so far: {torch.cuda.max_memory_allocated()/1e9:.1f} GB')

print(f'Got {len(preds)} prediction steps')
final = preds[-1]
print('final surf 2t shape:', final.surf_vars['2t'].shape)
print('final valid time:', final.metadata.time)

os.makedirs(os.path.expanduser('~/weather-interpretability/results'), exist_ok=True)
with open(os.path.expanduser(f'~/weather-interpretability/results/{CASE}_aurora_baseline.pkl'), 'wb') as f:
    pickle.dump({
        'surf_vars': {k: v.numpy() for k, v in final.surf_vars.items()},
        'atmos_vars': {k: v.numpy() for k, v in final.atmos_vars.items()},
        'lat': final.metadata.lat.numpy(),
        'lon': final.metadata.lon.numpy(),
        'valid_time': final.metadata.time,
    }, f)

print('Saved. Done.')
