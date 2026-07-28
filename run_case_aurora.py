"""Aurora 24h baseline forecast for one case, runnable standalone (one process per GPU).
Usage: CUDA_VISIBLE_DEVICES=<gpu_id> python3 run_case_aurora.py <case_name>
Saves results/<case_name>_aurora_baseline.pkl
"""
import os
import sys
import glob
import pickle
import numpy as np

case = sys.argv[1]

_nvidia_lib_dirs = glob.glob(os.path.join(sys.prefix, 'lib', 'python3.10', 'site-packages', 'nvidia', '*', 'lib'))
os.environ['LD_LIBRARY_PATH'] = ':'.join(_nvidia_lib_dirs) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import xarray as xr
from aurora import Aurora, Batch, Metadata
from aurora.rollout import _advance_batch

PRESSURE_LEVELS_HPA = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SURF_VARS = {'2t': '2m_temperature', '10u': '10m_u_component_of_wind', '10v': '10m_v_component_of_wind', 'msl': 'mean_sea_level_pressure'}
ATMOS_VARS = {'z': 'geopotential', 'u': 'u_component_of_wind', 'v': 'v_component_of_wind', 't': 'temperature', 'q': 'specific_humidity'}

CASE_DATES = {
    'storm_eunice':   '2022-02-18T12:00',
    'heatwave_block': '2022-07-18T12:00',
    'quiet_baseline': '2021-10-05T12:00',
}


def log(msg):
    print(f'[{case}] {msg}', flush=True)


CASE_TIME = np.datetime64(CASE_DATES[case])
PREV_TIME = CASE_TIME - np.timedelta64(6, 'h')

log('Opening ARCO...')
ARCO_URL = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
ds_arco = xr.open_zarr(ARCO_URL, storage_options=dict(token='anon'), chunks=None)
ds_arco = ds_arco.assign_coords(longitude=(ds_arco.longitude % 360)).sortby('longitude')

log('Fetching t-6h and t snapshots...')
snap_prev = ds_arco.sel(time=PREV_TIME, method='nearest')
snap_now = ds_arco.sel(time=CASE_TIME, method='nearest')

surf_vars = {k: torch.from_numpy(np.stack([snap_prev[v].values, snap_now[v].values])[None]).float()
             for k, v in SURF_VARS.items()}
atmos_vars = {k: torch.from_numpy(np.stack([snap_prev[v].sel(level=PRESSURE_LEVELS_HPA).values,
                                             snap_now[v].sel(level=PRESSURE_LEVELS_HPA).values])[None]).float()
              for k, v in ATMOS_VARS.items()}

log('Loading static vars...')
static_path = glob.glob(os.path.expanduser(
    '~/.cache/huggingface/hub/models--microsoft--aurora/snapshots/*/aurora-0.25-static.pickle'))[0]
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
        time=(CASE_TIME.astype('datetime64[s]').tolist(),),
        atmos_levels=tuple(PRESSURE_LEVELS_HPA),
    ),
)

log('Loading model...')
model = Aurora(use_lora=False)
model.load_checkpoint('microsoft/aurora', 'aurora-0.25-pretrained.ckpt')
model.eval()
model = model.to('cuda')
batch = batch.crop(model.patch_size)
batch = batch.to('cuda')

log('Running manual rollout (4 steps x 6h = 24h)...')
preds = []
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
    for i in range(4):
        pred = model.forward(batch)
        preds.append(pred.to('cpu'))
        batch = _advance_batch(batch, pred)
        del pred
        torch.cuda.empty_cache()
        log(f'  step {i+1}/4 done, peak mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB')

final = preds[-1]
log(f'final valid time: {final.metadata.time}')

os.makedirs(os.path.expanduser('~/weather-interpretability/results'), exist_ok=True)
with open(os.path.expanduser(f'~/weather-interpretability/results/{case}_aurora_baseline.pkl'), 'wb') as f:
    pickle.dump({
        'surf_vars': {k: v.numpy() for k, v in final.surf_vars.items()},
        'atmos_vars': {k: v.numpy() for k, v in final.atmos_vars.items()},
        'lat': final.metadata.lat.numpy(),
        'lon': final.metadata.lon.numpy(),
        'valid_time': final.metadata.time,
    }, f)

log('Saved. Done.')
