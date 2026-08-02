"""Skeleton (interpolative) decomposition of Aurora's WINDOW ATTENTION matrices, for the same
Typhoon Bebinca case. This complements run_skeleton_case.py (which decomposed the *hidden
state*) by looking at *how information is routed* -- the same object the Advection Heads paper
studies -- but using skeleton decomposition instead of SVD.

What we capture: Aurora computes window attention via a fused kernel
(F.scaled_dot_product_attention) that never materialises the (query x key) probability matrix.
We temporarily monkeypatch WindowAttention.forward on the hooked instances to compute the
identical q,k,v projections and an explicit eager softmax(q @ k^T / sqrt(head_dim)) attention
matrix -- capturing it -- then finish the same computation (attn @ v, proj) so model outputs
during the rollout are unaffected (this is the standard "eager attention for interpretability"
trick).

Scope for this first pass: block index 0 (non-shifted -- shift_size=(0,0,0) by construction
for even block indices) of stage 0 and stage 1 (stage 2's resolution, 45x90, does not divide
evenly by the window size 6x12, which would need padding-index bookkeeping we skip here).
For a non-shifted block, window w=(c_idx, h_idx, w_idx) tiles the stage's own INPUT (C,H,W)
grid directly (no roll), so window token (c,h,w)-within-window maps straightforwardly back to
absolute (row, col) = (h_idx*ws_H + h, w_idx*ws_W + w).

Row skeleton on the (144, 144) query x key attention matrix (per head): k representative QUERY
positions -- i.e. a small set of window positions whose "where do I look" attention pattern is
enough to reconstruct every other position's pattern (linear combination). If most of the
window looks in a locally-uniform (wind-following/advective) direction, few landmarks needed;
sharp direction changes (e.g. curvature around a vortex) need more landmarks concentrated there.

Column skeleton: k representative KEY positions -- targets that many queries' attention rows
can be reconstructed from. If unrelated queries converge on the *same* few key landmarks, that
is a "hub" pattern, different from (spatially-varying, per-query) advection.
"""
import os
import sys
import glob
import math
import pickle
import numpy as np
import xarray as xr
import scipy.linalg.interpolative as sli

_nvidia_lib_dirs = glob.glob(os.path.join(sys.prefix, 'lib', 'python3.10', 'site-packages', 'nvidia', '*', 'lib'))
os.environ['LD_LIBRARY_PATH'] = ':'.join(_nvidia_lib_dirs) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn.functional as F
from einops import rearrange
from aurora import Aurora, Batch, Metadata
from aurora.rollout import _advance_batch

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
INIT_TIME = np.datetime64('2024-11-15T00:00')  # quiet control: no typhoon near China this date
N_STEPS = 4
K_LANDMARKS = 16  # out of 144 window tokens
TYPHOON_LAT, TYPHOON_LON = 31.0, 121.5  # SAME window box as the Bebinca case, for comparison
STAGES_TO_ANALYZE = [0, 1]  # skip stage2: 45x90 doesn't divide evenly by window (6,12)
OUT_PATH = os.path.expanduser('~/weather-interpretability/results/legacy/skeleton_decomposition/quiet_attention.pkl')
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
lat_full = ds_surf.latitude.values
lon_full = ds_surf.longitude.values
lat_crop = lat_full[:720]
lon_crop = lon_full

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
              metadata=Metadata(lat=lat_t, lon=lon_t,
                                 time=(INIT_TIME.astype('datetime64[s]').tolist(),),
                                 atmos_levels=tuple(AURORA_LEVELS)))
batch = batch.crop(model.patch_size)
batch = batch.to('cuda')

# ---- monkeypatch WindowAttention.forward on the target blocks to expose the attention matrix ----
captured_attn = {}  # (stage_idx, step_idx) -> tensor (nW*B, heads, 144, 144)


def make_eager_forward(stage_idx):
    def eager_forward(self, x, mask=None, rollout_step=0):
        qkv = self.qkv(x) + self.lora_qkv(x, rollout_step)
        qkv = rearrange(qkv, "B N (qkv H D) -> qkv B H N D", H=self.num_heads, qkv=3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            mask_ = mask.unsqueeze(1).unsqueeze(0)
            B = q.shape[0] // mask_.shape[1]
            mask_ = mask_.repeat(B, 1, 1, 1, 1).reshape(-1, *mask_.shape[2:])
            attn = attn + mask_
        attn = attn.softmax(dim=-1)
        captured_attn.setdefault(stage_idx, []).append(attn.detach().float().cpu())
        out = attn @ v
        out = rearrange(out, "B H N D -> B N (H D)")
        out = self.proj(out) + self.lora_proj(out, rollout_step)
        out = self.proj_drop(out)
        return out
    return eager_forward


import types
patched_modules = []
for stage_idx in STAGES_TO_ANALYZE:
    block0 = model.backbone.encoder_layers[stage_idx].blocks[0]
    assert all(s == 0 for s in block0.shift_size), f'stage {stage_idx} block0 unexpectedly shifted'
    attn_module = block0.attn
    attn_module.forward = types.MethodType(make_eager_forward(stage_idx), attn_module)
    patched_modules.append(attn_module)

patch_res_per_step = []
print('Running Aurora rollout with attention capture...', flush=True)
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
    for step in range(N_STEPS):
        H, W = batch.spatial_shape
        patch_res = (model.encoder.latent_levels, H // model.encoder.patch_size, W // model.encoder.patch_size)
        patch_res_per_step.append(patch_res)
        pred = model.forward(batch)
        batch = _advance_batch(batch, pred)

all_enc_res, _ = model.backbone.get_encoder_specs(patch_res_per_step[0])
print('Per-stage (levels, H, W) *input* resolution:', all_enc_res, flush=True)
ws = model.backbone.window_size
print('window_size:', ws, flush=True)


def find_window_index(stage_idx, target_lat, target_lon):
    """Locate the (c_idx, h_idx, w_idx) window containing (target_lat, target_lon) for the
    stage's own input resolution (block0 is non-shifted, so windows tile the grid directly),
    plus the absolute (row, col) offset of that window's top-left corner."""
    _, H_s, W_s = all_enc_res[stage_idx]
    factor_h = len(lat_crop) // H_s
    factor_w = len(lon_crop) // W_s
    lat_s = lat_crop[::factor_h][:H_s]
    lon_s = lon_crop[::factor_w][:W_s]
    row = int(np.argmin(np.abs(lat_s - target_lat)))
    col = int(np.argmin(np.abs(lon_s - target_lon)))
    h_idx = row // ws[1]
    w_idx = col // ws[2]
    c_idx = 0  # only 1 c-window group of interest (C dim is tiny, level-like, not spatial)
    row0 = h_idx * ws[1]
    col0 = w_idx * ws[2]
    return c_idx, h_idx, w_idx, row0, col0, lat_s, lon_s


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


results = {'k': K_LANDMARKS, 'window_size': ws, 'stage_res': all_enc_res, 'stages': {}}
for stage_idx in STAGES_TO_ANALYZE:
    c_idx, h_idx, w_idx, row0, col0, lat_s, lon_s = find_window_index(stage_idx, TYPHOON_LAT, TYPHOON_LON)
    _, H_s, W_s = all_enc_res[stage_idx]
    nW_h = H_s // ws[1]
    nW_w = W_s // ws[2]
    window_flat_idx = c_idx * nW_h * nW_w + h_idx * nW_w + w_idx
    print(f'stage {stage_idx}: window containing ({TYPHOON_LAT},{TYPHOON_LON}) '
          f'-> (h_idx={h_idx}, w_idx={w_idx}), flat_idx={window_flat_idx}, '
          f'lat range [{lat_s[row0]:.2f},{lat_s[min(row0+ws[1]-1, len(lat_s)-1)]:.2f}], '
          f'lon range [{lon_s[col0]:.2f},{lon_s[min(col0+ws[2]-1, len(lon_s)-1)]:.2f}]', flush=True)

    stage_out = {'window_flat_idx': window_flat_idx, 'row0': row0, 'col0': col0,
                 'lat_s': lat_s, 'lon_s': lon_s, 'steps': []}
    for step_idx, attn in enumerate(captured_attn[stage_idx]):
        # attn: (nW*B, heads, 144, 144); B=1 here.
        window_attn = attn[window_flat_idx]  # (heads, 144, 144)
        n_heads = window_attn.shape[0]
        head_decs = []
        for h in range(n_heads):
            A = window_attn[h].numpy()
            dec = skeleton_decompose(A, K_LANDMARKS)
            head_decs.append(dec)
        stage_out['steps'].append({'lead_hours': (step_idx + 1) * 6, 'heads': head_decs})
        mean_row_err = np.mean([d['row_rel_error'] for d in head_decs])
        mean_col_err = np.mean([d['col_rel_error'] for d in head_decs])
        print(f'  stage {stage_idx} step +{(step_idx+1)*6}h: heads={n_heads} '
              f'mean_row_err={mean_row_err:.4f} mean_col_err={mean_col_err:.4f}', flush=True)
    results['stages'][stage_idx] = stage_out

with open(OUT_PATH, 'wb') as f:
    pickle.dump(results, f)
print(f'Saved to {OUT_PATH}', flush=True)
