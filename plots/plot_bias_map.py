"""Bias localisation map: where does replacing ONE input field with climatology make the
forecast worse?

For a chosen date and lead:
    E_base(s)  = |y_base(s)  - y_truth(s)|      per-pixel absolute error, clean run
    E_patch(s) = |y_patch(s) - y_truth(s)|      per-pixel absolute error, patched run
    D(s)       = E_patch(s) - E_base(s)         > 0  -> the patch made it worse here

Usage: plot_bias_map.py MODEL DATE LEAD PATCHED [TARGETS...]
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

MODEL = sys.argv[1]
DATE = sys.argv[2]
LEAD = int(sys.argv[3])
PATCHED = sys.argv[4]
TARGETS = sys.argv[5:] or ['Q1000', 'T1000', 'U1000', 'V1000']

CROP = 720 if MODEL == 'aurora' else None
DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
RES_DIR = os.path.expanduser('~/weather-interpretability/results/patching/patching_19var'
                             + ('' if MODEL == 'aurora' else f'_{MODEL}'))
FIG_DIR = os.path.expanduser('~/weather-interpretability/figures/bias_maps')
os.makedirs(FIG_DIR, exist_ok=True)

VARS = {
    'MSLP': ('surf', 'mean_sea_level_pressure', None, 'Па'),
    'U10': ('surf', '10m_u_component_of_wind', None, 'м/с'),
    'V10': ('surf', '10m_v_component_of_wind', None, 'м/с'),
    'T2M': ('surf', '2m_temperature', None, 'К'),
}
for _s, _l, _u in (('Z', 'geopotential', 'м²/с²'), ('Q', 'specific_humidity', 'кг/кг'),
                   ('T', 'temperature', 'К'), ('U', 'u_component_of_wind', 'м/с'),
                   ('V', 'v_component_of_wind', 'м/с')):
    for _lv in (1000, 850, 500):
        VARS[f'{_s}{_lv}'] = ('upper', _l, _lv, _u)

init = np.datetime64(f'{DATE[:4]}-{DATE[4:6]}-{DATE[6:8]}T{DATE[9:11]}:00')
valid = init + np.timedelta64(LEAD, 'h')
ts = pd.Timestamp(valid)

ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None)
lat = ds_surf.latitude.values[:CROP]
lon = ds_surf.longitude.values
W = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
W = W / W.sum()

d = np.load(os.path.join(RES_DIR, f'{DATE}_{MODEL}.npz'))
print(f'{MODEL} | init={init} | valid={valid} | подмена: {PATCHED}', flush=True)

# Robinson axes keep a fixed aspect, so tight_layout cannot be trusted here -- lay the
# panels out explicitly with generous vertical spacing instead.
fig, axes = plt.subplots(2, 2, figsize=(17, 10.5),
                         subplot_kw={'projection': ccrs.Robinson(central_longitude=0)})
fig.subplots_adjust(left=0.02, right=0.97, top=0.88, bottom=0.03, wspace=0.08, hspace=0.22)
for ax, tgt in zip(axes.ravel(), TARGETS):
    kind, var, lvl, unit = VARS[tgt]
    truth = (ds_surf.sel(time=valid)[var].values[:CROP] if kind == 'surf'
             else ds_upper.sel(time=valid, level=lvl)[var].values[:CROP])
    base = d[f'baseline_lead{LEAD}_{tgt}']
    patch = d[f'patch{PATCHED}_lead{LEAD}_{tgt}']

    e_base = np.abs(base - truth)
    e_patch = np.abs(patch - truth)
    delta = e_patch - e_base

    v = np.percentile(np.abs(delta), 99)      # robust symmetric scale
    im = ax.pcolormesh(lon, lat, delta, transform=ccrs.PlateCarree(),
                       cmap='RdBu_r', vmin=-v, vmax=v, shading='auto', rasterized=True)
    ax.coastlines(linewidth=0.4, color='0.25')
    ax.set_global()
    cb = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.72, pad=0.02)
    cb.set_label(f'Δ|ошибка|, {unit}', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    worse = float((W * (delta > 0)).sum()) * 100          # area fraction, %
    mean_d = float((W * delta).sum())
    rel = float((W * e_patch).sum()) / float((W * e_base).sum())
    ax.set_title(f'{tgt}:  |ош. с подменой| − |ош. без|\n'
                 f'хуже на {worse:.0f}% площади,  ⟨Δ⟩={mean_d:+.3g} {unit},  '
                 f'ошибка ×{rel:.2f}', fontsize=9.5, pad=8)

fig.suptitle(f'{MODEL.capitalize()}: локализация ущерба от подмены {PATCHED} на климатологию\n'
             f'инициализация {init}, лид +{LEAD}ч   '
             f'(красное — подмена ухудшила прогноз здесь)', fontsize=12, y=0.97)
out = f'{FIG_DIR}/biasmap_{MODEL}_{DATE}_lead{LEAD}_patch{PATCHED}.png'
plt.savefig(out, dpi=150)
print('saved', out, flush=True)
