"""Dose-response analysis: how does forecast error of the OTHER fields grow as the patched
geopotential field is blended from real (alpha=0) towards climatology (alpha=1)?

alpha=0 and alpha=1 are read from the existing patching_19var* runs, the intermediate alphas
from dose_response_*. All metrics are cos(phi)-area-weighted.

For each (patched field, output field, lead) the curve  S(alpha)  is fitted with a power law
    S(alpha) = A * alpha**p
and the exponent p tells the shape:  p≈1 linear, p<1 concave/saturating, p>1 convex.
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
CROP = 720 if MODEL == 'aurora' else None
ALPHAS_MID = [0.1, 0.25, 0.5, 0.75]
ALPHAS = [0.0] + ALPHAS_MID + [1.0]
PATCH_VARS = ['Z1000', 'Z850', 'Z500']
LEADS = [6, 24]

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
BASE_DIR = os.path.expanduser('~/weather-interpretability/results/patching/patching_19var'
                              + ('' if MODEL == 'aurora' else f'_{MODEL}'))
DR_DIR = os.path.expanduser(f'~/weather-interpretability/results/dose_response/dose_response_{MODEL}')
FIG_DIR = os.path.expanduser('~/weather-interpretability/figures/dose_response')

NAMES = ['MSLP', 'U10', 'V10', 'T2M']
for s in ('Z', 'Q', 'T', 'U', 'V'):
    NAMES += [f'{s}1000', f'{s}850', f'{s}500']
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
print(f'{MODEL}: {len(dates)} дат в dose_response', flush=True)

rows, indiff = [], []
for label in dates:
    base_fp = os.path.join(BASE_DIR, f'{label}_{MODEL}.npz')
    if not os.path.exists(base_fp):
        print(f'  {label}: нет базового файла, пропуск', flush=True); continue
    b = np.load(base_fp)
    d = np.load(os.path.join(DR_DIR, f'{label}_{MODEL}.npz'))
    init = ALL_DATES[label]
    for pv in PATCH_VARS:
        m, s_, sd = d[f'inputdiff_{pv}']
        indiff.append({'date': label, 'patched': pv,
                       'input_pert_rel': float(m) / float(sd)})   # |clim-real| / sigma(field)
    for lead in LEADS:
        valid = init + np.timedelta64(lead, 'h')
        for out in NAMES:
            tr, cl = truth_clim(out, valid)
            base = b[f'baseline_lead{lead}_{out}']
            sdw, acc0 = std_w(tr), acc_w(base, tr, cl)
            for pv in PATCH_VARS:
                for a in ALPHAS:
                    if a == 0.0:
                        pred = base
                    elif a == 1.0:
                        pred = b[f'patch{pv}_lead{lead}_{out}']
                    else:
                        pred = d[f'{pv}_a{a}_lead{lead}_{out}']
                    rows.append({'date': label, 'lead': lead, 'patched': pv, 'output': out,
                                 'alpha': a,
                                 'rel_sens': rmse_w(pred, base) / sdw if sdw > 1e-9 else np.nan,
                                 'dacc': acc_w(pred, tr, cl) - acc0})
    b.close(); d.close()
    print(f'  {label} ok', flush=True)

df = pd.DataFrame(rows)
df.to_pickle(os.path.join(DR_DIR, f'dose_metrics_{MODEL}.pkl'))
pd.DataFrame(indiff).to_pickle(os.path.join(DR_DIR, f'input_pert_{MODEL}.pkl'))
n = df.date.nunique()
print(f'{n} дат, {len(df)} строк', flush=True)

# ---- power-law fit  S = A * alpha^p  (log-log least squares over alpha>0) ----------------
def fit_p(sub):
    g = sub.groupby('alpha')['rel_sens'].mean()
    x = np.array([a for a in ALPHAS if a > 0])
    y = np.array([g[a] for a in x])
    ok = y > 0
    if ok.sum() < 3:
        return np.nan, np.nan
    p, lnA = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
    return p, float(np.exp(lnA))


print('\n=== показатель степени p в S(alpha) = A*alpha^p ===')
print('p<1 вогнутая (рано насыщается) | p=1 линейная | p>1 выпуклая')
fits = []
for lead in LEADS:
    for pv in PATCH_VARS:
        sub_all = df[(df.lead == lead) & (df.patched == pv) & (df.output != pv)]
        p, A = fit_p(sub_all)
        fits.append({'lead': lead, 'patched': pv, 'p': p, 'A': A})
        print(f'  +{lead:2d}ч  {pv}: p = {p:.3f}   (среднее по всем прочим полям)')
pd.DataFrame(fits).to_pickle(os.path.join(DR_DIR, f'powerlaw_{MODEL}.pkl'))

# ---- figure: curves per patched field ---------------------------------------------------
TARGETS = ['Q1000', 'T1000', 'U1000', 'V1000', 'T2M', 'MSLP']
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for r, lead in enumerate(LEADS):
    for c, pv in enumerate(PATCH_VARS):
        ax = axes[r, c]
        for tgt in TARGETS:
            g = (df[(df.lead == lead) & (df.patched == pv) & (df.output == tgt)]
                 .groupby('alpha')['rel_sens'].mean())
            ax.plot(g.index, g.values, 'o-', ms=4, lw=1.2, label=tgt)
        p = [f for f in fits if f['lead'] == lead and f['patched'] == pv][0]['p']
        aa = np.linspace(0.02, 1, 50)
        gm = (df[(df.lead == lead) & (df.patched == pv) & (df.output != pv)]
              .groupby('alpha')['rel_sens'].mean())
        ax.plot(aa, gm[1.0] * aa, 'k--', lw=0.9, alpha=0.6, label='линейный ход')
        ax.set_title(f'подмена {pv}, +{lead}ч   (p = {p:.2f})', fontsize=10)
        ax.set_xlabel(r'$\alpha$  (0 = реальные данные, 1 = климатология)')
        ax.set_ylabel('rel. sensitivity')
        ax.grid(alpha=0.3)
        if r == 0 and c == 0:
            ax.legend(fontsize=7.5, ncol=2)
fig.suptitle(f'{MODEL.capitalize()}: рост ошибки при плавном замещении геопотенциала '
             f'климатологией (n={n} дат)\nчёрный пунктир — линейный ход через точку '
             r'$\alpha=1$;  кривая выше него = вогнутая (ранний выход на насыщение)',
             fontsize=12, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.955])
out = f'{FIG_DIR}/dose_response_{MODEL}_n{n}.png'
plt.savefig(out, dpi=150)
print('saved', out, flush=True)
