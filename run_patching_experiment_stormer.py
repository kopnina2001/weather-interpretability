"""Input-level climatological patching experiment for Stormer: companion to
run_patching_experiment.py (Aurora), adapted to Stormer's channel-stacked flat-tensor interface
and its coarser 1.40625deg (128x256) regridded resolution -- Stormer's public checkpoint's native
grid, not ERA5's native 0.25deg. For two dates (Bebinca typhoon, init=2024-09-15; and a quiet
control, init=2024-11-15), replace ONE physical component of the model input with its
climatological mean, keep everything else real, run the 4-step (6h) forecast, and save the
output alongside an unpatched baseline (all real input) run.

Unlike Aurora (which conditions on two input timesteps and is rolled out autoregressively step by
step), Stormer conditions on a SINGLE input snapshot and predicts each lead time directly via
model.forward_validation(inp_tensor, VARIABLES, 6, step) -- see run_campaign_stormer_v2.py.

CAVEAT (same as the Aurora version): our downloaded WB2 climatology only covers 3 pressure levels
(500, 850, 1000 hPa) for upper-air variables, out of the 13 Stormer uses. So wind/mass-field/
temperature patches are PARTIAL-COLUMN: only those 3 levels are replaced by climatology, the other
10 remain real. Surface variables (MSLP, T2M, 10m wind) have no such restriction.

Each component is patched in two scopes: 'global' (everywhere) and 'regional' (only inside
CHINA_EXTENT, rest of the globe stays real).
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torchvision.transforms import transforms

sys.path.insert(0, os.path.expanduser('~/stormer'))
from stormer.models.hub.stormer import Stormer
from stormer.models.iterative_module import GlobalForecastIterativeModule

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
OUT_DIR = os.path.expanduser('~/weather-interpretability/results/patching_experiment_stormer')
os.makedirs(OUT_DIR, exist_ok=True)
CHINA_EXTENT = [95, 130, 15, 45]  # lon_min, lon_max, lat_min, lat_max
N_STEPS = 4

DATES = {'storm': np.datetime64('2024-09-15T00:00'), 'quiet': np.datetime64('2024-11-15T00:00')}

PRESSURE_LEVELS_HPA = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
CLIM_LEVELS = [500, 850, 1000]
SURFACE_MAP = {'mean_sea_level_pressure': 'mean_sea_level_pressure',
               '10m_u_component_of_wind': '10m_u_component_of_wind',
               '10m_v_component_of_wind': '10m_v_component_of_wind',
               '2m_temperature': '2m_temperature'}
UPPER_VARS_LONG = ['geopotential', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity']
VARIABLES = list(SURFACE_MAP.keys()) + [f'{v}_{l}' for v in UPPER_VARS_LONG for l in PRESSURE_LEVELS_HPA]

TARGET_LAT = np.linspace(-90 + 1.40625 / 2, 90 - 1.40625 / 2, 128)
TARGET_LON = np.arange(0, 360, 1.40625)

# component -> (surf_vars_to_patch, upper_vars_to_patch)
COMPONENTS = {
    'surface_mslp': (['mean_sea_level_pressure'], []),
    'wind': (['10m_u_component_of_wind', '10m_v_component_of_wind'],
              ['u_component_of_wind', 'v_component_of_wind']),
    'mass_field': ([], ['geopotential']),
    'temperature': ([], ['temperature']),
    't2m': (['2m_temperature'], []),
}
SCOPES = ['global', 'regional']


def regrid(da):
    return da.interp(latitude=TARGET_LAT, longitude=TARGET_LON)


print('Loading ERA5 + climatology...', flush=True)
ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None).sel(level=PRESSURE_LEVELS_HPA)
clim_surf_ds = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_surface.zarr', chunks=None)
clim_upper_ds = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_upper_500_850_1000.zarr', chunks=None)

lat_mask = (TARGET_LAT >= CHINA_EXTENT[2]) & (TARGET_LAT <= CHINA_EXTENT[3])
lon_mask = (TARGET_LON >= CHINA_EXTENT[0]) & (TARGET_LON <= CHINA_EXTENT[1])
region_idx = np.ix_(lat_mask, lon_mask)


def get_climatology(time_val):
    ts = pd.Timestamp(time_val)
    doy = min(ts.dayofyear, 366)
    hour = (ts.hour // 6) * 6
    surf_snap = regrid(clim_surf_ds.sel(dayofyear=doy, hour=hour))
    upper_snap = regrid(clim_upper_ds.sel(dayofyear=doy, hour=hour))
    surf_clim = {v: surf_snap[v].values for v in SURFACE_MAP}
    upper_clim = {v: upper_snap[v].values for v in UPPER_VARS_LONG}  # (level=3, lat, lon)
    return surf_clim, upper_clim


def build_real_tensors(time_val):
    surf_snap = regrid(ds_surf.sel(time=time_val))
    upper_snap = regrid(ds_upper.sel(time=time_val))
    surf = {v: surf_snap[v].values.astype(np.float32) for v in SURFACE_MAP}
    upper = {v: upper_snap[v].values.astype(np.float32) for v in UPPER_VARS_LONG}  # (13, lat, lon)
    return surf, upper


print('Loading normalization constants...', flush=True)
norm_dir = os.path.expanduser('~/stormer/normalization_constants')
normalize_mean = dict(np.load(os.path.join(norm_dir, 'normalize_mean.npz')))
normalize_mean = np.concatenate([normalize_mean[v] for v in VARIABLES], axis=0)
normalize_std = dict(np.load(os.path.join(norm_dir, 'normalize_std.npz')))
normalize_std = np.concatenate([normalize_std[v] for v in VARIABLES], axis=0)
inp_transform = transforms.Normalize(normalize_mean, normalize_std)

print('Loading Stormer model...', flush=True)
net = Stormer(in_img_size=[128, 256], variables=VARIABLES, patch_size=2, hidden_size=1024,
              depth=24, num_heads=16, mlp_ratio=4)
pretrained_path = 'https://huggingface.co/tungnd/stormer/resolve/main/stormer_1.40625_patch_size_2.ckpt'
model = GlobalForecastIterativeModule(net, pretrained_path=pretrained_path)
model.eval()
model = model.to('cuda')

out_transforms = {}
for interval in [6, 12, 24]:
    diff_std = dict(np.load(os.path.join(norm_dir, f'normalize_diff_std_{interval}.npz')))
    diff_std = np.concatenate([diff_std[v] for v in VARIABLES], axis=0)
    out_transforms[interval] = transforms.Normalize(np.zeros_like(diff_std), diff_std)
model.set_transforms(inp_transform, out_transforms)


def apply_patch(surf, upper, clim_surf, clim_upp, component, scope):
    """Return patched copies of (surf, upper) -- the single input snapshot -- with the given
    component replaced by its climatology, in the given spatial scope."""
    surf_vars_to_patch, upper_vars_to_patch = COMPONENTS[component]
    surf = {k: v.copy() for k, v in surf.items()}
    upper = {k: v.copy() for k, v in upper.items()}

    def patch_field(field, clim_field):
        if scope == 'global':
            return clim_field.astype(np.float32)
        out = field.copy()
        out[region_idx] = clim_field[region_idx]
        return out

    for v in surf_vars_to_patch:
        surf[v] = patch_field(surf[v], clim_surf[v])
    for v in upper_vars_to_patch:
        for li_full, lvl in enumerate(PRESSURE_LEVELS_HPA):
            if lvl in CLIM_LEVELS:
                li_clim = CLIM_LEVELS.index(lvl)
                upper[v][li_full] = patch_field(upper[v][li_full], clim_upp[v][li_clim])
    return surf, upper


def build_input_tensor(surf, upper):
    channels = [surf[v] for v in SURFACE_MAP]
    for v in UPPER_VARS_LONG:
        for li in range(len(PRESSURE_LEVELS_HPA)):
            channels.append(upper[v][li])
    return np.stack(channels, axis=0)


def run_rollout(surf, upper):
    inp = build_input_tensor(surf, upper)
    inp_tensor = torch.from_numpy(inp).unsqueeze(0).to('cuda')
    inp_tensor = inp_transform(inp_tensor)
    surf_out, upper_out = [], []
    n_surf = len(SURFACE_MAP)
    n_lvl = len(PRESSURE_LEVELS_HPA)
    with torch.no_grad():
        for step in range(1, N_STEPS + 1):
            pred_norm = model.forward_validation(inp_tensor, VARIABLES, 6, step)
            pred_physical = model.reverse_inp_transform(pred_norm).squeeze(0).cpu().numpy()
            surf_out.append(pred_physical[:n_surf].copy())
            upper_out.append(pred_physical[n_surf:].reshape(len(UPPER_VARS_LONG), n_lvl,
                                                              *pred_physical.shape[1:]).copy())
    return np.stack(surf_out), np.stack(upper_out)  # (N_STEPS, nvars, [level,] lat, lon)


for date_label, init_time in DATES.items():
    print(f'=== date {date_label}: init={init_time} ===', flush=True)
    surf, upper = build_real_tensors(init_time)
    clim_surf, clim_upp = get_climatology(init_time)

    results = {}
    print('  running baseline (all real)...', flush=True)
    b_surf, b_upper = run_rollout(surf, upper)
    results['baseline_surf'] = b_surf
    results['baseline_upper'] = b_upper

    for component in COMPONENTS:
        for scope in SCOPES:
            key = f'{component}_{scope}'
            print(f'  running {key}...', flush=True)
            p_surf, p_upper = apply_patch(surf, upper, clim_surf, clim_upp, component, scope)
            p_surf_out, p_upper_out = run_rollout(p_surf, p_upper)
            results[f'{key}_surf'] = p_surf_out
            results[f'{key}_upper'] = p_upper_out

    out_path = os.path.join(OUT_DIR, f'{date_label}.npz')
    np.savez_compressed(out_path, **results)
    print(f'  saved to {out_path}', flush=True)

print('All done.', flush=True)
