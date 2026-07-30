"""Task 2: capture per-head OUTPUT representations (attn @ v, before the final head-merge and
proj) from Aurora's window attention (block 0, non-shifted, stages 0 and 1), for both the
Bebinca storm case and the quiet control, so we can compute CKA (Centered Kernel Alignment)
similarity between heads -- a standard way to test whether different heads learn redundant
representations (a form of matrix-based interpretability distinct from the skeleton
decomposition already done: here we compare *whole heads* to each other, not decompose one
matrix into landmarks).

We reuse the same monkeypatch-eager-attention trick as run_attention_skeleton.py, but this
time save the per-head output for the SAME window used before (covering the Bebinca landfall
area, ~31N 121.5E) instead of immediately decomposing anything.
"""
import os
import sys
import glob
import pickle
import numpy as np
import xarray as xr

_nvidia_lib_dirs = glob.glob(os.path.join(sys.prefix, 'lib', 'python3.10', 'site-packages', 'nvidia', '*', 'lib'))
os.environ['LD_LIBRARY_PATH'] = ':'.join(_nvidia_lib_dirs) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from einops import rearrange
from aurora import Aurora, Batch, Metadata
from aurora.rollout import _advance_batch

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
N_STEPS = 4
TYPHOON_LAT, TYPHOON_LON = 31.0, 121.5
STAGES_TO_ANALYZE = [0, 1]
OUT_DIR = os.path.expanduser('~/weather-interpretability/results/tensor_decomposition')
os.makedirs(OUT_DIR, exist_ok=True)

DATES = {'storm': np.datetime64('2024-09-15T00:00'), 'quiet': np.datetime64('2024-11-15T00:00')}

SURFACE_VARS = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
UPPER_VARS = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
AURORA_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
AURORA_SURF_KEYS = {'mean_sea_level_pressure': 'msl', '10m_u_component_of_wind': '10u',
                    '10m_v_component_of_wind': '10v', '2m_temperature': '2t'}
AURORA_UPPER_KEYS = {'geopotential': 'z', 'specific_humidity': 'q', 'temperature': 't',
                     'u_component_of_wind': 'u', 'v_component_of_wind': 'v'}

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

ws = model.backbone.window_size


def build_tensors(time_val):
    snap_surf = ds_surf.sel(time=time_val)
    snap_upp = ds_upper.sel(time=time_val)
    surf = {v: snap_surf[v].values.astype(np.float32) for v in SURFACE_VARS}
    upper = {v: snap_upp[v].values.astype(np.float32) for v in UPPER_VARS}
    return surf, upper


def find_window_index(all_enc_res, stage_idx, target_lat, target_lon):
    _, H_s, W_s = all_enc_res[stage_idx]
    factor_h = len(lat_crop) // H_s
    factor_w = len(lon_crop) // W_s
    lat_s = lat_crop[::factor_h][:H_s]
    lon_s = lon_crop[::factor_w][:W_s]
    row = int(np.argmin(np.abs(lat_s - target_lat)))
    col = int(np.argmin(np.abs(lon_s - target_lon)))
    h_idx, w_idx = row // ws[1], col // ws[2]
    nW_h, nW_w = H_s // ws[1], W_s // ws[2]
    return h_idx * nW_w + w_idx


for date_label, init_time in DATES.items():
    print(f'=== {date_label} ===', flush=True)
    captured_head_out = {}

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
            head_out = attn @ v  # (nW*B, heads, 144, head_dim) -- per-head output, pre-merge
            captured_head_out.setdefault(stage_idx, []).append(head_out.detach().float().cpu())
            out = rearrange(head_out, "B H N D -> B N (H D)")
            out = self.proj(out) + self.lora_proj(out, rollout_step)
            out = self.proj_drop(out)
            return out
        return eager_forward

    import types
    handles = []
    for stage_idx in STAGES_TO_ANALYZE:
        block0 = model.backbone.encoder_layers[stage_idx].blocks[0]
        attn_module = block0.attn
        attn_module.forward = types.MethodType(make_eager_forward(stage_idx), attn_module)

    prev_time = init_time - np.timedelta64(6, 'h')
    surf0, upper0 = build_tensors(prev_time)
    surf1, upper1 = build_tensors(init_time)
    surf_vars = {AURORA_SURF_KEYS[v]: torch.from_numpy(np.stack([surf0[v], surf1[v]])[None]).float()
                 for v in SURFACE_VARS}
    atmos_vars = {AURORA_UPPER_KEYS[v]: torch.from_numpy(np.stack([upper0[v], upper1[v]])[None]).float()
                  for v in UPPER_VARS}
    batch = Batch(surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars,
                  metadata=Metadata(lat=lat_t, lon=lon_t,
                                     time=(init_time.astype('datetime64[s]').tolist(),),
                                     atmos_levels=tuple(AURORA_LEVELS)))
    batch = batch.crop(model.patch_size)
    batch = batch.to('cuda')

    patch_res_per_step = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for step in range(N_STEPS):
            H, W = batch.spatial_shape
            patch_res = (model.encoder.latent_levels, H // model.encoder.patch_size, W // model.encoder.patch_size)
            patch_res_per_step.append(patch_res)
            pred = model.forward(batch)
            batch = _advance_batch(batch, pred)

    all_enc_res, _ = model.backbone.get_encoder_specs(patch_res_per_step[0])

    save_dict = {}
    for stage_idx in STAGES_TO_ANALYZE:
        window_flat_idx = find_window_index(all_enc_res, stage_idx, TYPHOON_LAT, TYPHOON_LON)
        for step_idx, head_out in enumerate(captured_head_out[stage_idx]):
            # head_out: (nW*B, heads, 144, head_dim) -- select our window
            arr = head_out[window_flat_idx].numpy()  # (heads, 144, head_dim)
            save_dict[f'stage{stage_idx}_step{step_idx}'] = arr

    out_path = os.path.join(OUT_DIR, f'{date_label}_head_outputs.npz')
    np.savez_compressed(out_path, **save_dict)
    print(f'  saved to {out_path}', flush=True)

print('All done.', flush=True)
