"""Dose-response matrices for the Z1000 blend, Aurora.

Rows    = alpha in {0, 0.2, 0.4, 0.6, 0.8, 1.0}   (0 = real field, 1 = full climatology)
Columns = all 19 forecast fields

Four panels: RMSE and ACC, each at +6h and +24h.

Stored (pickle) per (date, lead, alpha, output):
    rmse_raw   -- cos(phi)-weighted RMSE against ERA5, in the field's own units
    rmse_norm  -- same divided by sigma_w(truth)   <- what the figure colours show
    acc        -- cos(phi)-weighted ACC against ERA5 w.r.t. climatology
    d_rmse     -- rmse_raw(alpha) - rmse_raw(0)
    d_acc      -- acc(alpha)      - acc(0)
Raw units are kept so the figure can be redrawn any other way without recomputing.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL = 'aurora'
CROP = 720
PATCH = 'Z1000'
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MID = [0.2, 0.4, 0.6, 0.8]
LEADS = [6, 24]

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
BASE_DIR = os.path.expanduser('~/weather-interpretability/results/patching_19var')
DR_DIR = os.path.expanduser(f'~/weather-interpretability/results/dose_response_z1000_{MODEL}')
FIG_DIR = os.path.expanduser('~/weather-interpretability/figures/08_input_patching')
os.makedirs(FIG_DIR, exist_ok=True)

NAMES = ['MSLP', 'U10', 'V10', 'T2M']
for s in ('Z', 'Q', 'T', 'U', 'V'):
    NAMES += [f'{s}1000', f'{s}850', f'{s}500']
GROUP_EDGES = [4, 7, 10, 13, 16]
UNITS = {'MSLP': 'Па', 'U10': 'м/с', 'V10': 'м/с', 'T2M': 'К'}
for _lv in (1000, 850, 500):
    UNITS[f'Z{_lv}'] = 'м²/с²'; UNITS[f'Q{_lv}'] = 'кг/кг'
    UNITS[f'T{_lv}'] = 'К'; UNITS[f'U{_lv}'] = 'м/с'; UNITS[f'V{_lv}'] = 'м/с'

VARS = {'MSLP': ('surf', 'mean_sea_level_pressure', None), 'U10': ('surf', '10m_u_component_of_wind', None),
        'V10': ('surf', '10m_v_component_of_wind', None), 'T2M': ('surf', '2m_temperature', None)}
for _s, _l in (('Z', 'geopotential'), ('Q', 'specific_humidity'), ('T', 'temperature'),
               ('U', 'u_component_of_wind'), ('V', 'v_component_of_wind')):
    for _lv in (1000, 850, 500):
        VARS[f'{_s}{_lv}'] = ('upper', _l, _lv)

_tri = [(2024, m, d) for m in range(1, 13) for d in (4, 11, 18, 25)]
_hrs = [0, 6, 12, 18] * (len(_tri) // 4)
ALL_DATES = {f'{y}{m:02d}{d:02d}_{h:02d}': np.datetime64(f'{y}-{m:02d}-{d:02d}T{h:02d}:00')
             for (y, m, d), h in zip(_tri, _hrs)}

ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None)
clim_surf = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_surface.zarr', chunks=None)
clim_upper = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_upper_500_850_1000.zarr', chunks=None)
_lat = ds_surf.latitude.values[:CROP]
W = np.cos(np.deg2rad(_lat))[:, None] * np.ones((1, ds_surf.longitude.size))
W = W / W.sum()


def truth_clim(name, valid):
    kind, var, lv = VARS[name]
    ts = pd.Timestamp(valid)
    doy, hour = min(ts.dayofyear, 366), (ts.hour // 6) * 6
    if kind == 'surf':
        return (ds_surf.sel(time=valid)[var].values[:CROP],
                clim_surf.sel(dayofyear=doy, hour=hour)[var].values[:CROP])
    return (ds_upper.sel(time=valid, level=lv)[var].values[:CROP],
            clim_upper.sel(dayofyear=doy, hour=hour, level=lv)[var].values[:CROP])


def rmse_w(a, b):
    return float(np.sqrt((W * (a - b) ** 2).sum()))


def std_w(a):
    m = (W * a).sum()
    return float(np.sqrt((W * (a - m) ** 2).sum()))


def acc_w(p, t, c):
    a, b = p - c, t - c
    den = np.sqrt((W * a ** 2).sum() * (W * b ** 2).sum())
    return float((W * a * b).sum() / den) if den > 1e-12 else np.nan


dates = sorted(os.path.basename(f).replace(f'_{MODEL}.npz', '')
               for f in glob.glob(os.path.join(DR_DIR, f'*_{MODEL}.npz')))
print(f'{len(dates)} дат в {DR_DIR}', flush=True)

rows = []
for label in dates:
    bfp = os.path.join(BASE_DIR, f'{label}_{MODEL}.npz')
    if not os.path.exists(bfp):
        print(f'  {label}: нет baseline, пропуск', flush=True); continue
    try:
        b = np.load(bfp)
        d = np.load(os.path.join(DR_DIR, f'{label}_{MODEL}.npz'))
        init = ALL_DATES[label]
        for lead in LEADS:
            valid = init + np.timedelta64(lead, 'h')
            for out in NAMES:
                tr, cl = truth_clim(out, valid)
                sdw = std_w(tr)
                base = b[f'baseline_lead{lead}_{out}']
                r0, a0 = rmse_w(base, tr), acc_w(base, tr, cl)
                for al in ALPHAS:
                    if al == 0.0:
                        pred = base
                    elif al == 1.0:
                        pred = b[f'patch{PATCH}_lead{lead}_{out}']
                    else:
                        pred = d[f'{PATCH}_a{al}_lead{lead}_{out}']
                    r, ac = rmse_w(pred, tr), acc_w(pred, tr, cl)
                    rows.append({'date': label, 'lead': lead, 'alpha': al, 'output': out,
                                 'rmse_raw': r, 'rmse_norm': r / sdw if sdw > 1e-9 else np.nan,
                                 'acc': ac, 'd_rmse': r - r0, 'd_acc': ac - a0})
        b.close(); d.close()
        print(f'  {label} ok', flush=True)
    except Exception as e:
        print(f'  {label}: ПРОПУСК ({str(e)[:50]})', flush=True)

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit('нет данных')
df.to_pickle(os.path.join(DR_DIR, f'dose_matrix_{MODEL}.pkl'))
n = df.date.nunique()
print(f'{n} дат, {len(df)} строк', flush=True)


def draw(ax, mat, title, cmap, cbar_label, fmt='{:.3f}', symmetric=False):
    v = np.nanmax(np.abs(mat))
    lo = -v if symmetric else np.nanmin(mat)
    im = ax.imshow(mat, cmap=cmap, vmin=lo, vmax=v, aspect='auto')
    ax.set_xticks(range(len(NAMES))); ax.set_xticklabels(NAMES, rotation=90, fontsize=8)
    ax.set_yticks(range(len(ALPHAS)))
    ax.set_yticklabels([f'{PATCH}({a:g})' for a in ALPHAS], fontsize=9)
    for e in GROUP_EDGES:
        ax.axvline(e - 0.5, color='k', lw=0.6, alpha=0.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                rel = (val - lo) / (v - lo) if v > lo else 0.5
                ax.text(j, i, fmt.format(val), ha='center', va='center', fontsize=6,
                        color='white' if (rel > 0.65 or rel < 0.15) else 'black')
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.015)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return im


fig, axes = plt.subplots(4, 1, figsize=(15, 16))
for k, lead in enumerate(LEADS):
    sub = df[df.lead == lead]
    m_r = sub.groupby(['alpha', 'output'])['rmse_norm'].mean().unstack().reindex(
        index=ALPHAS, columns=NAMES).values
    m_a = sub.groupby(['alpha', 'output'])['acc'].mean().unstack().reindex(
        index=ALPHAS, columns=NAMES).values
    draw(axes[2 * k], m_r,
         f'RMSE / $\\sigma_w$(правда), взвеш. по cos φ  —  Aurora +{lead}ч  (n={n} дат)',
         'viridis', 'RMSE в долях СКО поля')
    draw(axes[2 * k + 1], m_a,
         f'ACC против правды отн. климатологии, взвеш.  —  Aurora +{lead}ч  (n={n} дат)',
         'viridis_r', 'ACC')
axes[-1].set_xlabel('поле прогноза', fontsize=10)
fig.suptitle(f'Aurora: плавное замещение {PATCH} климатологией\n'
             f'строка $\\alpha$: 0 = реальные данные, 1 = полная климатология', fontsize=13, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.985])
out = f'{FIG_DIR}/dose_matrix_{PATCH}_{MODEL}_n{n}.png'
plt.savefig(out, dpi=150)
print('saved', out, flush=True)

# краткая сводка: во сколько раз растёт ошибка и как падает ACC
print('\n=== рост нормированной RMSE относительно alpha=0 ===')
for lead in LEADS:
    s = df[df.lead == lead].groupby('alpha')['rmse_norm'].mean()
    print(f'  +{lead:2d}ч: ' + '  '.join(f'a={a:g}:{s[a]/s[0.0]:.2f}x' for a in ALPHAS))
print('=== средний ACC по всем полям ===')
for lead in LEADS:
    s = df[df.lead == lead].groupby('alpha')['acc'].mean()
    print(f'  +{lead:2d}ч: ' + '  '.join(f'a={a:g}:{s[a]:.4f}' for a in ALPHAS))
