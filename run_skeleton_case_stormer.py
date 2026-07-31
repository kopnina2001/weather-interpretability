"""Skeleton (interpolative) decomposition of Stormer's transformer-block hidden states, for
the Bebinca storm case (init=2024-09-15) and a quiet control (init=2024-11-15) -- the same
methodology as run_skeleton_case.py for Aurora (notebook 07), applied to a much simpler
architecture: Stormer is a flat ViT-style stack of 24 identical transformer blocks at a single
resolution (no U-Net downsampling like Aurora's Swin backbone), operating on a coarse
1.40625-deg (128x256, patchified to 64x128 tokens) grid.

We hook 3 representative blocks (shallow=0, middle=11, deep=23) and capture their token output
(B, L=8192, D=1024) once per autoregressive step -- forward_validation(steps=4, interval=6)
loops internally 4 times (once per 6h step to reach +24h), calling net.forward (and therefore
every block) each time, so the hooks fire 4 times per block automatically.
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

import torch
sys.path.insert(0, os.path.expanduser('~/stormer'))
from stormer.models.hub.stormer import Stormer
from stormer.models.iterative_module import GlobalForecastIterativeModule
from torchvision.transforms import transforms

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
N_STEPS = 4
K_LANDMARKS = 256
BLOCKS_TO_HOOK = {'shallow': 0, 'middle': 11, 'deep': 23}
OUT_DIR = os.path.expanduser('~/weather-interpretability/results/skeleton_decomposition')
os.makedirs(OUT_DIR, exist_ok=True)

DATES = {'storm': np.datetime64('2024-09-15T00:00'), 'quiet': np.datetime64('2024-11-15T00:00')}

AURORA_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SURFACE_MAP_KEYS = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure']
UPPER_VARS_LONG = ['geopotential', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity']
VARIABLES = SURFACE_MAP_KEYS + [f'{v}_{l}' for v in UPPER_VARS_LONG for l in AURORA_LEVELS]

TARGET_LAT = np.linspace(-90 + 1.40625 / 2, 90 - 1.40625 / 2, 128)
TARGET_LON = np.arange(0, 360, 1.40625)
# token grid after patch_size=2: average consecutive pairs of the 128x256 physical grid
TOKEN_LAT = TARGET_LAT.reshape(64, 2).mean(axis=1)
TOKEN_LON = TARGET_LON.reshape(128, 2).mean(axis=1)

print('Loading ERA5...', flush=True)
ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None).sel(level=AURORA_LEVELS)


def regrid(da):
    return da.interp(latitude=TARGET_LAT, longitude=TARGET_LON)


print('Loading Stormer model...', flush=True)
norm_dir = os.path.expanduser('~/stormer/normalization_constants')
normalize_mean = dict(np.load(os.path.join(norm_dir, 'normalize_mean.npz')))
normalize_mean = np.concatenate([normalize_mean[v] for v in VARIABLES], axis=0)
normalize_std = dict(np.load(os.path.join(norm_dir, 'normalize_std.npz')))
normalize_std = np.concatenate([normalize_std[v] for v in VARIABLES], axis=0)
inp_transform = transforms.Normalize(normalize_mean, normalize_std)

net = Stormer(in_img_size=[128, 256], variables=VARIABLES, patch_size=2, hidden_size=1024, depth=24, num_heads=16, mlp_ratio=4)
model = GlobalForecastIterativeModule(net, pretrained_path='https://huggingface.co/tungnd/stormer/resolve/main/stormer_1.40625_patch_size_2.ckpt')
model.eval()
model = model.to('cuda')

out_transforms = {}
for interval in [6, 12, 24]:
    diff_std = dict(np.load(os.path.join(norm_dir, f'normalize_diff_std_{interval}.npz')))
    diff_std = np.concatenate([diff_std[v] for v in VARIABLES], axis=0)
    out_transforms[interval] = transforms.Normalize(np.zeros_like(diff_std), diff_std)
model.set_transforms(inp_transform, out_transforms)


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

    residual_row = (X64 - Xhat_row).astype(np.float32)
    return {
        'row_landmarks': row_landmarks, 'col_landmarks': col_landmarks,
        'row_residual': residual_row, 'row_rel_error': row_rel_error, 'col_rel_error': col_rel_error,
    }


for date_label, init_time in DATES.items():
    print(f'=== {date_label}: init={init_time} ===', flush=True)
    captured = {}
    handles = []

    def make_hook(name):
        def hook(module, inp, output):
            captured.setdefault(name, []).append(output.detach().float().cpu())
        return hook

    for name, block_idx in BLOCKS_TO_HOOK.items():
        handles.append(net.blocks[block_idx].register_forward_hook(make_hook(name)))

    surf_snap = regrid(ds_surf.sel(time=init_time))
    upper_snap = regrid(ds_upper.sel(time=init_time))
    channels = [surf_snap[v].values.astype(np.float32) for v in SURFACE_MAP_KEYS]
    for v in UPPER_VARS_LONG:
        for lvl in AURORA_LEVELS:
            channels.append(upper_snap[v].sel(level=lvl).values.astype(np.float32))
    inp = np.stack(channels, axis=0)
    inp_tensor = torch.from_numpy(inp).unsqueeze(0).to('cuda')
    inp_tensor = inp_transform(inp_tensor)

    with torch.no_grad():
        _ = model.forward_validation(inp_tensor, VARIABLES, 6, N_STEPS)

    for h in handles:
        h.remove()

    results = {'k': K_LANDMARKS, 'blocks': BLOCKS_TO_HOOK, 'stages': {}}
    raw_X = {}
    for name, tensors in captured.items():
        stage_out = []
        for step_idx, x in enumerate(tensors):
            X = x[0].numpy()  # (L=8192, D=1024)
            raw_X[f'{name}_step{step_idx}'] = X.astype(np.float32)
            dec = skeleton_decompose(X, K_LANDMARKS)
            dec['lead_hours'] = (step_idx + 1) * 6
            stage_out.append(dec)
            print(f'  {name} (block {BLOCKS_TO_HOOK[name]}) step +{dec["lead_hours"]}h: '
                  f'row_rel_error={dec["row_rel_error"]:.4f} col_rel_error={dec["col_rel_error"]:.4f}', flush=True)
        results['stages'][name] = stage_out

    with open(os.path.join(OUT_DIR, f'{date_label}_stormer.pkl'), 'wb') as f:
        pickle.dump(results, f)
    np.savez_compressed(os.path.join(OUT_DIR, f'{date_label}_stormer_raw_X.npz'), **raw_X)
    print(f'  saved {date_label}_stormer.pkl / _raw_X.npz', flush=True)

print('All done.', flush=True)
