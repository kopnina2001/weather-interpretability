"""Per-case occlusion pipeline, runnable standalone so 3 cases can run in parallel on 3 GPUs.
Usage: CUDA_VISIBLE_DEVICES=<gpu_id> python3 run_case_occlusion.py <case_name>
Saves results to results/<case_name>.pkl for the notebook to load and plot.
"""
import os
import sys
import time
import glob
import pickle

case = sys.argv[1]

_nvidia_lib_dirs = glob.glob(os.path.join(sys.prefix, 'lib', 'python3.10', 'site-packages', 'nvidia', '*', 'lib'))
os.environ['LD_LIBRARY_PATH'] = ':'.join(_nvidia_lib_dirs) + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import numpy as np
import xarray as xr
import onnxruntime as ort

PRESSURE_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SURFACE_ORDER = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
UPPER_ORDER = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']

CASE_DATES = {
    'storm_eunice':   '2022-02-18T12:00',
    'heatwave_block': '2022-07-18T12:00',
    'quiet_baseline': '2021-10-05T12:00',
}
CLIMATOLOGY_YEARS_BACK = 8
PATCH_SIZE = 60
STRIDE = 60

SPATIAL_CHANNELS = [
    {'label': 'MSLP', 'channel_type': 'surface', 'channel': 'mean_sea_level_pressure', 'level': None},
    {'label': 'Z@500hPa', 'channel_type': 'upper', 'channel': 'geopotential', 'level': 500},
]
for spec in SPATIAL_CHANNELS:
    if spec['channel_type'] == 'surface':
        spec['var_idx'] = SURFACE_ORDER.index(spec['channel'])
        spec['level_idx'] = None
    else:
        spec['var_idx'] = UPPER_ORDER.index(spec['channel'])
        spec['level_idx'] = PRESSURE_LEVELS_HPA.index(spec['level'])


def to_pangu_lon(ds):
    return ds.assign_coords(longitude=(ds.longitude % 360)).sortby('longitude')


def log(msg):
    print(f'[{case}] {msg}', flush=True)


t_start = time.time()
log('Loading model...')
session = ort.InferenceSession('model_weights/pangu_weather_24.onnx',
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
log(f'Providers: {session.get_providers()}')
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


log('Building input tensors...')
surface_ds = to_pangu_lon(xr.open_zarr(f'data/{case}_surface.zarr'))
upper_ds = to_pangu_lon(xr.open_zarr(f'data/{case}_upper_air.zarr')).sel(level=PRESSURE_LEVELS_HPA)
input_surface = np.stack([surface_ds[v].values for v in SURFACE_ORDER], axis=0).astype(np.float32)
input_upper = np.stack([upper_ds[v].values for v in UPPER_ORDER], axis=0).astype(np.float32)
assert input_surface.shape == (4, 721, 1440)
assert input_upper.shape == (5, 13, 721, 1440)

log('Computing climatology + std...')
ARCO_URL = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
ds_arco = xr.open_zarr(ARCO_URL, storage_options=dict(token='anon'), chunks=None)
ds_arco = ds_arco.assign_coords(longitude=(ds_arco.longitude % 360)).sortby('longitude')

base_time = np.datetime64(CASE_DATES[case])
sample_times = [base_time - np.timedelta64(365 * i, 'D') for i in range(1, CLIMATOLOGY_YEARS_BACK + 1)]
surface_samples = ds_arco[SURFACE_ORDER].sel(time=sample_times, method='nearest').load()
upper_samples = ds_arco[UPPER_ORDER].sel(time=sample_times, method='nearest').sel(level=PRESSURE_LEVELS_HPA).load()

clim_surface = np.stack([surface_samples[v].mean(dim='time').values for v in SURFACE_ORDER], axis=0).astype(np.float32)
clim_upper = np.stack([upper_samples[v].mean(dim='time').values for v in UPPER_ORDER], axis=0).astype(np.float32)
std_surface = {v: float(surface_samples[v].std()) for v in SURFACE_ORDER}
std_upper = {v: float(upper_samples[v].std()) for v in UPPER_ORDER}

log('Baseline forecast...')
baseline = run_model(input_surface, input_upper)


def output_diff_norm(baseline, perturbed):
    base_surf, base_upp = baseline
    pert_surf, pert_upp = perturbed
    total = 0.0
    for i, v in enumerate(SURFACE_ORDER):
        std = std_surface[v] if std_surface[v] > 1e-6 else 1.0
        total += float(np.mean(((pert_surf[i] - base_surf[i]) / std) ** 2))
    for i, v in enumerate(UPPER_ORDER):
        std = std_upper[v] if std_upper[v] > 1e-6 else 1.0
        total += float(np.mean(((pert_upp[i] - base_upp[i]) / std) ** 2))
    return np.sqrt(total)


def perturbed_forward(channel_type, var_idx, level_idx, mode):
    surf = input_surface.copy()
    upp = input_upper.copy()
    if channel_type == 'surface':
        surf[var_idx] = 0.0 if mode == 'zero' else clim_surface[var_idx]
    else:
        upp[var_idx, level_idx] = 0.0 if mode == 'zero' else clim_upper[var_idx, level_idx]
    return run_model(surf, upp)


log('Running global occlusion (69 channels x 2 modes)...')
global_results = {}
for mode in ['zero', 'climatology']:
    results = []
    for i, v in enumerate(SURFACE_ORDER):
        pert = perturbed_forward('surface', i, None, mode)
        results.append({'channel': v, 'level': None, 'score': output_diff_norm(baseline, pert)})
    for i, v in enumerate(UPPER_ORDER):
        for lvl_idx, lvl_hpa in enumerate(PRESSURE_LEVELS_HPA):
            pert = perturbed_forward('upper', i, lvl_idx, mode)
            results.append({'channel': v, 'level': lvl_hpa, 'score': output_diff_norm(baseline, pert)})
    global_results[mode] = results
    log(f'  global occlusion mode={mode} done')

log('Running spatial patch occlusion (MSLP, Z@500hPa)...')
n_lat, n_lon = 721, 1440
spatial_saliency = {}
for spec in SPATIAL_CHANNELS:
    sal = np.zeros((n_lat, n_lon), dtype=np.float32)
    for lat0 in range(0, n_lat, STRIDE):
        lat1 = min(lat0 + PATCH_SIZE, n_lat)
        for lon0 in range(0, n_lon, STRIDE):
            lon1 = min(lon0 + PATCH_SIZE, n_lon)
            surf = input_surface.copy()
            upp = input_upper.copy()
            if spec['channel_type'] == 'surface':
                surf[spec['var_idx'], lat0:lat1, lon0:lon1] = clim_surface[spec['var_idx'], lat0:lat1, lon0:lon1]
            else:
                upp[spec['var_idx'], spec['level_idx'], lat0:lat1, lon0:lon1] = \
                    clim_upper[spec['var_idx'], spec['level_idx'], lat0:lat1, lon0:lon1]
            pert = run_model(surf, upp)
            sal[lat0:lat1, lon0:lon1] = output_diff_norm(baseline, pert)
    spatial_saliency[spec['label']] = sal
    log(f"  spatial occlusion {spec['label']} done")

os.makedirs('results', exist_ok=True)
with open(f'results/{case}.pkl', 'wb') as f:
    pickle.dump({
        'global_results': global_results,
        'spatial_saliency': spatial_saliency,
        'baseline_output': baseline,
        'std_surface': std_surface,
        'std_upper': std_upper,
    }, f)

log(f'Done in {time.time() - t_start:.0f}s')
