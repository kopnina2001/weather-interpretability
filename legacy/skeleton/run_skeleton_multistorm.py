"""Skeleton decomposition (Aurora), same methodology as run_skeleton_case.py (notebook 07),
applied to 5 additional storms (one per year: 2018, 2019, 2021, 2022, 2023) near the China
coast in September -- a robustness/generalization check for the "landmarks cluster on the
storm centre" finding beyond the single Bebinca (2024) case. Data source: the small
era5_multistorm_6h_{surface,upper}.zarr windows downloaded specifically for these storms
(download_era5_september_multiyear.py) -- NOT the main era5_2024_2025 cache.

For each storm, init_time = (reference/landfall date) - 1 day, so the landfall falls at the
+24h lead, matching the Bebinca setup exactly (init=Sept15 -> landfall Sept16 = +24h).
"""
import os
import sys
import glob
import pickle
import numpy as np
import xarray as xr
import scipy.linalg.interpolative as sli

_nvidia_lib_dirs = glob.glob(os.path.join(sys.prefix, 'lib', 'python3.10', 'site-packages', 'nvidia', '*', 'lib'))
os.environ['LD_LIBRARY_PATH'] = ':'.join(_nvidia_lib_dirs) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from aurora import Aurora, Batch, Metadata
from aurora.rollout import _advance_batch

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
N_STEPS = 4
K_LANDMARKS = 256
OUT_DIR = os.path.expanduser('~/weather-interpretability/results/legacy/skeleton_decomposition')
os.makedirs(OUT_DIR, exist_ok=True)

# storm_name -> reference (landfall/peak) date; init_time = ref - 1 day
STORMS = {
    'mangkhut_2018': np.datetime64('2018-09-16'),
    'lingling_2019': np.datetime64('2019-09-05'),
    'chanthu_2021': np.datetime64('2021-09-12'),
    'muifa_2022': np.datetime64('2022-09-13'),
    'haikui_2023': np.datetime64('2023-09-05'),
}

SURFACE_VARS = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
UPPER_VARS = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
AURORA_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
AURORA_SURF_KEYS = {'mean_sea_level_pressure': 'msl', '10m_u_component_of_wind': '10u',
                    '10m_v_component_of_wind': '10v', '2m_temperature': '2t'}
AURORA_UPPER_KEYS = {'geopotential': 'z', 'specific_humidity': 'q', 'temperature': 't',
                     'u_component_of_wind': 'u', 'v_component_of_wind': 'v'}

print('Loading multistorm ERA5 window...', flush=True)
ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_multistorm_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_multistorm_6h_upper.zarr', chunks=None).sel(level=AURORA_LEVELS)

static_path = glob.glob(os.path.expanduser(
    '~/.cache/huggingface/hub/models--microsoft--aurora/snapshots/*/aurora-0.25-static.pickle'))[0]
with open(static_path, 'rb') as f:
    static_raw = pickle.load(f)
static_vars = {k: torch.from_numpy(v).float() for k, v in static_raw.items()}
lat_full = torch.from_numpy(ds_surf.latitude.values).float()
lon_full = torch.from_numpy(ds_surf.longitude.values).float()

print('Loading Aurora model...', flush=True)
model = Aurora(use_lora=False)
model.load_checkpoint('microsoft/aurora', 'aurora-0.25-pretrained.ckpt')
model.eval()
model = model.to('cuda')

hook_handles = []
captured = {}


def make_hook(stage_idx):
    def hook(module, inp, output):
        x = output[0] if isinstance(output, tuple) else output
        captured.setdefault(stage_idx, []).append(x.detach().float().cpu())
    return hook


def skeleton_decompose(X, k):
    X64 = X.astype(np.float64)
    idx_r, proj_r = sli.interp_decomp(X64.T, k)
    row_landmarks = idx_r[:k]
    P_row = sli.reconstruct_interp_matrix(idx_r, proj_r)
    Xhat_row = (P_row.T @ X64[row_landmarks, :])
    row_rel_error = float(np.linalg.norm(Xhat_row - X64) / np.linalg.norm(X64))

    idx_c, proj_c = sli.interp_decomp(X64, k)
    col_landmarks = idx_c[:k]
    P_col = sli.reconstruct_interp_matrix(idx_c, proj_c)
    Xhat_col = X64[:, col_landmarks] @ P_col
    col_rel_error = float(np.linalg.norm(Xhat_col - X64) / np.linalg.norm(X64))

    return {
        'row_landmarks': row_landmarks, 'col_landmarks': col_landmarks,
        'row_rel_error': row_rel_error, 'col_rel_error': col_rel_error,
    }


for storm_name, ref_date in STORMS.items():
    init_time = ref_date - np.timedelta64(1, 'D')
    print(f'=== {storm_name}: init={init_time} (ref={ref_date}) ===', flush=True)

    prev_time = init_time - np.timedelta64(6, 'h')
    snap_prev_surf = ds_surf.sel(time=prev_time)
    snap_now_surf = ds_surf.sel(time=init_time)
    snap_prev_upp = ds_upper.sel(time=prev_time)
    snap_now_upp = ds_upper.sel(time=init_time)

    surf_vars = {AURORA_SURF_KEYS[v]: torch.from_numpy(
        np.stack([snap_prev_surf[v].values, snap_now_surf[v].values])[None]).float() for v in SURFACE_VARS}
    atmos_vars = {AURORA_UPPER_KEYS[v]: torch.from_numpy(
        np.stack([snap_prev_upp[v].values, snap_now_upp[v].values])[None]).float() for v in UPPER_VARS}

    batch = Batch(surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars,
                  metadata=Metadata(lat=lat_full, lon=lon_full,
                                     time=(init_time.astype('datetime64[s]').tolist(),),
                                     atmos_levels=tuple(AURORA_LEVELS)))
    batch = batch.crop(model.patch_size)
    batch = batch.to('cuda')

    captured.clear()
    for h in hook_handles:
        h.remove()
    hook_handles = []
    for i, layer in enumerate(model.backbone.encoder_layers):
        hook_handles.append(layer.register_forward_hook(make_hook(i)))

    patch_res_per_step = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for step in range(N_STEPS):
            H, W = batch.spatial_shape
            patch_res = (model.encoder.latent_levels, H // model.encoder.patch_size, W // model.encoder.patch_size)
            patch_res_per_step.append(patch_res)
            pred = model.forward(batch)
            batch = _advance_batch(batch, pred)

    for h in hook_handles:
        h.remove()

    all_enc_res, _ = model.backbone.get_encoder_specs(patch_res_per_step[0])
    last_stage_idx = max(captured.keys())

    results = {'k': K_LANDMARKS, 'stage_res': all_enc_res, 'init_time': str(init_time), 'stages': {}}
    for stage_idx, tensors in captured.items():
        res_idx = stage_idx if stage_idx == last_stage_idx else stage_idx + 1
        C_dim, H_s, W_s = all_enc_res[res_idx]
        stage_out = []
        for step_idx, x in enumerate(tensors):
            x = x[0]
            L, D = x.shape
            x_grid = x.view(C_dim, H_s, W_s, D).mean(dim=0).numpy()
            X = x_grid.reshape(H_s * W_s, D)
            dec = skeleton_decompose(X, K_LANDMARKS)
            dec['lead_hours'] = (step_idx + 1) * 6
            dec['H'], dec['W'] = H_s, W_s
            stage_out.append(dec)
            print(f'  stage {stage_idx} step +{dec["lead_hours"]}h: '
                  f'row_rel_error={dec["row_rel_error"]:.4f}', flush=True)
        results['stages'][stage_idx] = stage_out

    out_path = os.path.join(OUT_DIR, f'{storm_name}_aurora.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'  saved to {out_path}', flush=True)

print('All done.', flush=True)
