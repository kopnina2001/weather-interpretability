"""Skeleton (interpolative) decomposition of Aurora backbone hidden states for the Typhoon
Bebinca case (init=2024-09-15, 4 rollout steps of 6h -> valid 2024-09-16..17).

Where in Aurora we hook, precisely:
  Aurora = encoder (Perceiver-style compression of variables/levels into a 2D token grid)
           -> backbone (3D Swin-Transformer U-Net: 4 encoder stages that progressively halve
              H,W via patch merging, then 4 decoder stages that upsample back with skip
              connections -- see aurora/model/swin3d.py: Swin3DTransformerBackbone)
           -> decoder (Perceiver-style expansion back to physical variables/levels)

  The "Advection Heads" paper analyses attention-weight matrices (Q-K routing) inside the
  backbone's window-attention blocks. Here we instead decompose the *residual-stream hidden
  state* itself (what is represented, not how it is routed) at the output of each encoder-path
  stage of the backbone (model.backbone.encoder_layers -- 3 stages for this checkpoint),
  captured via forward hooks. Stage 0 is the finest resolution (closest to raw encoded field),
  the last stage is the bottleneck (coarsest resolution, maximal receptive field / most
  globally-mixed representation).

For each stage and each rollout step, the hidden state is a token matrix of shape
(C*H*W tokens, D channels), where (C,H,W) is given by Swin3DTransformerBackbone.get_encoder_specs
and C is the small Perceiver-compressed pseudo-level dimension (not a physical pressure level;
averaged out here for 2D visualization). We reshape to (H*W, D) and compute two one-sided
interpolative (skeleton) decompositions with scipy.linalg.interpolative.interp_decomp: a row
skeleton (k representative spatial locations that reconstruct every grid point's feature
vector) and a column skeleton (k representative channels that reconstruct every channel's
spatial map).
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
INIT_TIME = np.datetime64('2024-09-15T00:00')
N_STEPS = 4
K_LANDMARKS = 24  # number of landmark rows/columns to keep in the skeleton decomposition
OUT_PATH = os.path.expanduser('~/weather-interpretability/results/skeleton_decomposition/bebinca_aurora.pkl')
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

SURFACE_VARS = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
UPPER_VARS = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
AURORA_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
AURORA_SURF_KEYS = {'mean_sea_level_pressure': 'msl', '10m_u_component_of_wind': '10u',
                    '10m_v_component_of_wind': '10v', '2m_temperature': '2t'}
AURORA_UPPER_KEYS = {'geopotential': 'z', 'specific_humidity': 'q', 'temperature': 't',
                     'u_component_of_wind': 'u', 'v_component_of_wind': 'v'}

print('Loading ERA5...', flush=True)
ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None).sel(level=AURORA_LEVELS)

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

prev_time = INIT_TIME - np.timedelta64(6, 'h')
snap_prev_surf = ds_surf.sel(time=prev_time)
snap_now_surf = ds_surf.sel(time=INIT_TIME)
snap_prev_upp = ds_upper.sel(time=prev_time)
snap_now_upp = ds_upper.sel(time=INIT_TIME)

surf_vars = {AURORA_SURF_KEYS[v]: torch.from_numpy(
    np.stack([snap_prev_surf[v].values, snap_now_surf[v].values])[None]).float() for v in SURFACE_VARS}
atmos_vars = {AURORA_UPPER_KEYS[v]: torch.from_numpy(
    np.stack([snap_prev_upp[v].values, snap_now_upp[v].values])[None]).float() for v in UPPER_VARS}

batch = Batch(surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars,
              metadata=Metadata(lat=lat_full, lon=lon_full,
                                 time=(INIT_TIME.astype('datetime64[s]').tolist(),),
                                 atmos_levels=tuple(AURORA_LEVELS)))
batch = batch.crop(model.patch_size)
batch = batch.to('cuda')

# ---- register forward hooks on the 4 backbone encoder-path stages ----
captured = {}  # (stage_idx) -> list of (B, L, D) tensors, one appended per rollout step
hook_handles = []


def make_hook(stage_idx):
    def hook(module, inp, output):
        x = output[0] if isinstance(output, tuple) else output
        captured.setdefault(stage_idx, []).append(x.detach().float().cpu())
    return hook


for i, layer in enumerate(model.backbone.encoder_layers):
    hook_handles.append(layer.register_forward_hook(make_hook(i)))

patch_res_per_step = []

print('Running Aurora rollout with hooks...', flush=True)
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
print('Per-stage (levels, H, W) resolution:', all_enc_res, flush=True)


def skeleton_decompose(X, k):
    """One-sided interpolative (skeleton) decompositions of X (N_spatial x D_channels):
      - row skeleton: k representative spatial locations (rows) s.t. every row of X is a
        linear combination of those k landmark rows -- X ~ P_row @ X[row_landmarks, :].
      - column skeleton: k representative channels (columns) s.t. every column of X is a
        linear combination of those k landmark channels -- X ~ X[:, col_landmarks] @ P_col.
    Both use scipy's rank-revealing QR-based interp_decomp/reconstruct, which is numerically
    stable (no separate matrix inversion of a possibly near-singular intersection submatrix,
    unlike naive CUR)."""
    X64 = X.astype(np.float64)

    idx_r, proj_r = sli.interp_decomp(X64.T, k)  # selects k columns of X.T == k rows of X
    row_landmarks = idx_r[:k]
    P_row = sli.reconstruct_interp_matrix(idx_r, proj_r)  # (k, N_spatial)
    Xhat_row = (P_row.T @ X64[row_landmarks, :])
    row_rel_error = float(np.linalg.norm(Xhat_row - X64) / np.linalg.norm(X64))

    idx_c, proj_c = sli.interp_decomp(X64, k)  # selects k columns of X
    col_landmarks = idx_c[:k]
    P_col = sli.reconstruct_interp_matrix(idx_c, proj_c)  # (k, D_channels)
    Xhat_col = X64[:, col_landmarks] @ P_col
    col_rel_error = float(np.linalg.norm(Xhat_col - X64) / np.linalg.norm(X64))

    residual_row = (X64 - Xhat_row).astype(np.float32)
    return {
        'row_landmarks': row_landmarks,
        'col_landmarks': col_landmarks,
        'row_reconstruction': Xhat_row.astype(np.float32),
        'row_residual': residual_row,
        'row_rel_error': row_rel_error,
        'col_rel_error': col_rel_error,
    }


print('Running skeleton decomposition per stage/step...', flush=True)
last_stage_idx = max(captured.keys())
results = {'k': K_LANDMARKS, 'stage_res': all_enc_res, 'stages': {}}
for stage_idx, tensors in captured.items():
    # Stages with a downsample emit tokens at the *next* stage's input resolution (they
    # merge patches at the end of the block); the last (bottleneck) stage has no downsample,
    # so its output resolution equals its own input resolution.
    res_idx = stage_idx if stage_idx == last_stage_idx else stage_idx + 1
    C_dim, H_s, W_s = all_enc_res[res_idx]
    stage_out = []
    for step_idx, x in enumerate(tensors):
        x = x[0]  # drop batch dim -> (L, D)
        L, D = x.shape
        assert L == C_dim * H_s * W_s, f'stage {stage_idx} step {step_idx}: L={L} != {C_dim}*{H_s}*{W_s}'
        # C_dim is the small Perceiver-compressed pseudo-level dimension (not a physical
        # pressure level); we average over it to get a single 2D (H_s, W_s, D) map for
        # visualization purposes.
        x_grid = x.view(C_dim, H_s, W_s, D).mean(dim=0).numpy()
        X = x_grid.reshape(H_s * W_s, D)  # (H_s*W_s, D), row-major over (H_s, W_s)
        dec = skeleton_decompose(X, K_LANDMARKS)
        dec['lead_hours'] = (step_idx + 1) * 6
        dec['H'], dec['W'] = H_s, W_s
        stage_out.append(dec)
        print(f'  stage {stage_idx} (H={H_s},W={W_s},D={D}) step +{dec["lead_hours"]}h: '
              f'row_rel_error={dec["row_rel_error"]:.4f} col_rel_error={dec["col_rel_error"]:.4f}', flush=True)
    results['stages'][stage_idx] = stage_out

with open(OUT_PATH, 'wb') as f:
    pickle.dump(results, f)
print(f'Saved to {OUT_PATH}', flush=True)
