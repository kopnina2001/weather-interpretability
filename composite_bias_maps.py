"""Composite bias maps: per-pixel RMSE over N dates, baseline vs blended-Z1000, one figure
per blend level alpha.

For every grid point s independently (this is a TEMPORAL rmse across dates, not a spatial
one, so no cos(phi) weighting enters -- nothing is aggregated over the globe here):

    RMSE_a(s) = sqrt( (1/N) * sum_d ( yhat_{a,d}(s) - y_d(s) )^2 )
    D_a(s)    = RMSE_a(s) - RMSE_0(s)          > 0  ->  the blend makes this pixel worse

alpha = 0 is the clean baseline, so D_0 == 0 identically -- drawn anyway as a sanity check
that the pipeline is wired correctly.

All alphas share one colour scale per field (taken from alpha=1), so the figures can be
compared side by side and the growth with alpha is visible.
"""
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

MODEL, CROP, PATCH = 'aurora', 720, 'Z1000'
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MID = [0.2, 0.4, 0.6, 0.8]          # these live in the dose_response directory
FIELDS = ['Q1000', 'T1000', 'U1000', 'V1000']
LEADS = [6, 24]

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
BASE_DIR = os.path.expanduser('~/weather-interpretability/results/patching_19var')
DR_DIR = os.path.expanduser(f'~/weather-interpretability/results/dose_response_z1000_{MODEL}')
OUT_DIR = os.path.expanduser('~/weather-interpretability/results/composite_bias')
FIG_DIR = os.path.expanduser('~/weather-interpretability/figures/08_input_patching')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

VARS = {'Q1000': ('specific_humidity', 1000, 'кг/кг'), 'T1000': ('temperature', 1000, 'К'),
        'U1000': ('u_component_of_wind', 1000, 'м/с'), 'V1000': ('v_component_of_wind', 1000, 'м/с')}

_tri = [(2024, m, d) for m in range(1, 13) for d in (4, 11, 18, 25)]
_hrs = [0, 6, 12, 18] * (len(_tri) // 4)
ALL_DATES = {f'{y}{m:02d}{d:02d}_{h:02d}': np.datetime64(f'{y}-{m:02d}-{d:02d}T{h:02d}:00')
             for (y, m, d), h in zip(_tri, _hrs)}

ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None)
lat = ds_upper.latitude.values[:CROP]
lon = ds_upper.longitude.values

dates = sorted(os.path.basename(f).replace(f'_{MODEL}.npz', '')
               for f in glob.glob(os.path.join(DR_DIR, f'*_{MODEL}.npz')))
dates = [d for d in dates if os.path.exists(os.path.join(BASE_DIR, f'{d}_{MODEL}.npz'))]
N = len(dates)
print(f'{N} дат', flush=True)

# accumulate sum of squared errors per pixel: sse[lead][alpha][field]
sse = {L: {a: {f: np.zeros((CROP, lon.size), dtype=np.float64) for f in FIELDS}
           for a in ALPHAS} for L in LEADS}

for i, label in enumerate(dates, 1):
    init = ALL_DATES[label]
    b = np.load(os.path.join(BASE_DIR, f'{label}_{MODEL}.npz'))
    d = np.load(os.path.join(DR_DIR, f'{label}_{MODEL}.npz'))
    for L in LEADS:
        valid = init + np.timedelta64(L, 'h')
        for f in FIELDS:
            var, lv, _ = VARS[f]
            truth = ds_upper.sel(time=valid, level=lv)[var].values[:CROP]
            base = b[f'baseline_lead{L}_{f}']
            for a in ALPHAS:
                if a == 0.0:
                    pred = base
                elif a == 1.0:
                    pred = b[f'patch{PATCH}_lead{L}_{f}']
                else:
                    pred = d[f'{PATCH}_a{a}_lead{L}_{f}']
                sse[L][a][f] += (pred.astype(np.float64) - truth) ** 2
    b.close(); d.close()
    if i % 8 == 0 or i == N:
        print(f'  {i}/{N}', flush=True)

# per-pixel RMSE over dates, then the difference against alpha=0
rmse = {L: {a: {f: np.sqrt(sse[L][a][f] / N) for f in FIELDS} for a in ALPHAS} for L in LEADS}
delta = {L: {a: {f: rmse[L][a][f] - rmse[L][0.0][f] for f in FIELDS} for a in ALPHAS} for L in LEADS}

np.savez_compressed(os.path.join(OUT_DIR, f'composite_rmse_{PATCH}_{MODEL}_n{N}.npz'),
                    **{f'rmse_lead{L}_a{a}_{f}': rmse[L][a][f]
                       for L in LEADS for a in ALPHAS for f in FIELDS},
                    lat=lat, lon=lon, n_dates=np.array([N]))
print('сохранены попиксельные RMSE', flush=True)

# one shared colour scale per (lead, field), set by the strongest case alpha=1
SCALE = {(L, f): np.percentile(np.abs(delta[L][1.0][f]), 99) for L in LEADS for f in FIELDS}

for L in LEADS:
    for a in ALPHAS:
        fig, axes = plt.subplots(2, 2, figsize=(17, 10.5),
                                 subplot_kw={'projection': ccrs.Robinson(central_longitude=0)})
        fig.subplots_adjust(left=0.02, right=0.97, top=0.88, bottom=0.03,
                            wspace=0.08, hspace=0.22)
        for ax, f in zip(axes.ravel(), FIELDS):
            D = delta[L][a][f]
            v = SCALE[(L, f)]
            im = ax.pcolormesh(lon, lat, D, transform=ccrs.PlateCarree(), cmap='RdBu_r',
                               vmin=-v, vmax=v, shading='auto', rasterized=True)
            ax.coastlines(linewidth=0.4, color='0.25')
            ax.set_global()
            cb = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.72, pad=0.02)
            cb.set_label(f'Δ RMSE, {VARS[f][2]}', fontsize=8)
            cb.ax.tick_params(labelsize=7)
            worse = float((D > 0).mean()) * 100
            ratio = float(rmse[L][a][f].mean() / rmse[L][0.0][f].mean())
            ax.set_title(f'{f}:  RMSE(α={a:g}) − RMSE(α=0)\n'
                         f'хуже на {worse:.0f}% узлов,  средняя RMSE ×{ratio:.2f}',
                         fontsize=9.5, pad=8)
        extra = '   (контроль: тождественно нуль)' if a == 0.0 else ''
        fig.suptitle(f'Aurora, композит по {N} датам, лид +{L}ч:  подмена {PATCH} '
                     f'с долей климатологии α = {a:g}{extra}\n'
                     f'попиксельная RMSE по датам, минус то же при α=0   '
                     f'(красное — стало хуже; шкала общая для всех α)', fontsize=12, y=0.97)
        out = f'{FIG_DIR}/composite_bias_{PATCH}_lead{L}_alpha{a:g}_n{N}.png'
        plt.savefig(out, dpi=140)
        plt.close(fig)
        print('saved', out, flush=True)

print('\n=== средняя по глобусу попиксельная RMSE, отношение к α=0 ===')
for L in LEADS:
    for f in FIELDS:
        r = [rmse[L][a][f].mean() / rmse[L][0.0][f].mean() for a in ALPHAS]
        print(f'  +{L:2d}ч {f:6s}: ' + '  '.join(f'{a:g}:{x:.2f}x' for a, x in zip(ALPHAS, r)))
