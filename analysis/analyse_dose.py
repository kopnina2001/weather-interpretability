"""Dose-response matrices for a blended input field.

    analyse_dose.py <model> [patch_var]        e.g.  analyse_dose.py pangu Z1000

Rows    = alpha in {0, 0.2, 0.4, 0.6, 0.8, 1.0}   (0 = real field, 1 = full climatology)
Columns = all 19 forecast fields

Four panels: RMSE and ACC, each at +6h and +24h.

Stored (pickle) per (date, lead, alpha, output):
    rmse_raw    -- cos(phi)-weighted RMSE against ERA5, in the field's own units
    rmse_anom   -- same divided by sigma_w(truth - climatology)   <- what the figure shows
    rmse_norm   -- same divided by sigma_w(truth)                 <- kept for continuity
    acc         -- cos(phi)-weighted ACC against ERA5 w.r.t. climatology
    d_rmse, d_acc, sig_truth, sig_anom

The RMSE panels are normalised by the ANOMALY std, not the std of the field itself.  The
ratio between the two runs from 1.0 (V500 -- the climatological mean meridional wind is
almost zero) to 5.8 (T2M -- dominated by the pole-equator gradient), so normalising by the
field std systematically deflates the thermodynamic variables relative to wind and reorders
which fields look most sensitive.  RMSE/sigma_anom = 1 is the no-skill line: the forecast is
no better than issuing climatology.
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

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'aurora'
PATCH = sys.argv[2] if len(sys.argv) > 2 else 'Z1000'
CROP = 720 if MODEL == 'aurora' else None
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
LEADS = [6, 24]

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
BASE_DIR = os.path.expanduser('~/weather-interpretability/results/patching/patching_19var'
                              + ('' if MODEL == 'aurora' else f'_{MODEL}'))
DR_DIR = os.path.expanduser(
    f'~/weather-interpretability/results/dose_response/{PATCH.lower()}/{MODEL}')
FIG_DIR = os.path.expanduser('~/weather-interpretability/figures/dose_response')
os.makedirs(FIG_DIR, exist_ok=True)

NAMES = ['MSLP', 'U10', 'V10', 'T2M']
for s in ('Z', 'Q', 'T', 'U', 'V'):
    NAMES += [f'{s}1000', f'{s}850', f'{s}500']
GROUP_EDGES = [4, 7, 10, 13, 16]

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
print(f'{MODEL} / {PATCH}: {len(dates)} дат в {DR_DIR}', flush=True)

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
                s_tr, s_an = std_w(tr), std_w(tr - cl)
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
                                 'rmse_raw': r,
                                 'rmse_anom': r / s_an if s_an > 1e-12 else np.nan,
                                 'rmse_norm': r / s_tr if s_tr > 1e-12 else np.nan,
                                 'acc': ac, 'd_rmse': r - r0, 'd_acc': ac - a0,
                                 'sig_truth': s_tr, 'sig_anom': s_an})
        b.close(); d.close()
        print(f'  {label} ok', flush=True)
    except Exception as e:
        print(f'  {label}: ПРОПУСК ({str(e)[:60]})', flush=True)

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit('нет данных')
df.to_pickle(os.path.join(DR_DIR, f'dose_matrix_{MODEL}.pkl'))
n = df.date.nunique()
print(f'{n} дат, {len(df)} строк', flush=True)

# Рисование вынесено в plots/plot_dose_matrix.py, чтобы гамма и нормировка задавались
# в одном месте и не расходились между запусками.

print('\n=== рост RMSE/σ_аном относительно alpha=0 ===')
for lead in LEADS:
    s = df[df.lead == lead].groupby('alpha')['rmse_anom'].mean()
    print(f'  +{lead:2d}ч: ' + '  '.join(f'a={a:g}:{s[a]/s[0.0]:.2f}x' for a in ALPHAS))
print('=== средний ACC по всем полям ===')
for lead in LEADS:
    s = df[df.lead == lead].groupby('alpha')['acc'].mean()
    print(f'  +{lead:2d}ч: ' + '  '.join(f'a={a:g}:{s[a]:.4f}' for a in ALPHAS))
