"""Does the choice of normalising sigma change the conclusions?

sigma_truth  = weighted spatial std of the ERA5 field itself      (what was used)
sigma_anom   = weighted spatial std of (ERA5 - climatology)       (skill-referenced)

RMSE / sigma_anom = 1 is the no-skill line (a climatology forecast), so that ratio is the
one with a meaningful zero point.  This script recomputes the slope k of RMSE vs alpha under
both normalisations and prints them side by side.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

MODEL, PATCH = 'aurora', 'Z1000'
CROP, LEADS = 720, [6, 24]
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
DR_DIR = os.path.expanduser(f'~/weather-interpretability/results/dose_response_z1000_{MODEL}')

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


def std_w(a):
    m = (W * a).sum()
    return float(np.sqrt((W * (a - m) ** 2).sum()))


def truth_clim(name, valid):
    kind, var, lv = VARS[name]
    ts = pd.Timestamp(valid)
    doy, hour = min(ts.dayofyear, 366), (ts.hour // 6) * 6
    if kind == 'surf':
        return (ds_surf.sel(time=valid)[var].values[:CROP],
                clim_surf.sel(dayofyear=doy, hour=hour)[var].values[:CROP])
    return (ds_upper.sel(time=valid, level=lv)[var].values[:CROP],
            clim_upper.sel(dayofyear=doy, hour=hour, level=lv)[var].values[:CROP])


df = pd.read_pickle(os.path.join(DR_DIR, f'dose_matrix_{MODEL}.pkl'))
dates = sorted(df.date.unique())
print(f'{len(dates)} дат, считаю sigma_anom...', flush=True)

rows = []
for i, label in enumerate(dates, 1):
    init = ALL_DATES[label]
    for lead in LEADS:
        valid = init + np.timedelta64(lead, 'h')
        for out in NAMES:
            tr, cl = truth_clim(out, valid)
            rows.append({'date': label, 'lead': lead, 'output': out,
                         'sig_truth': std_w(tr), 'sig_anom': std_w(tr - cl)})
    if i % 12 == 0:
        print(f'  {i}/{len(dates)}', flush=True)

sig = pd.DataFrame(rows)
m = df.merge(sig, on=['date', 'lead', 'output'])
m['rmse_anom'] = m.rmse_raw / m.sig_anom
m.to_pickle(os.path.join(DR_DIR, f'dose_matrix_sigma_{MODEL}.pkl'))

A = np.array(ALPHAS)
for lead in LEADS:
    sub = m[m.lead == lead]
    g_t = sub.groupby(['output', 'alpha'])['rmse_norm'].mean().unstack()
    g_a = sub.groupby(['output', 'alpha'])['rmse_anom'].mean().unstack()
    ratio = sub.groupby('output')[['sig_truth', 'sig_anom']].mean()
    print(f'\n=== +{lead}ч ===')
    print(f'{"поле":7s} {"σ_прав":>9s} {"σ_аном":>9s} {"отн":>5s} | '
          f'{"k(σ_прав)":>9s} {"k(σ_аном)":>9s} | {"RMSE/σa при α=1":>15s}')
    res = []
    for f in NAMES:
        kt = np.polyfit(A, [g_t.loc[f, a] for a in ALPHAS], 1)[0]
        ka = np.polyfit(A, [g_a.loc[f, a] for a in ALPHAS], 1)[0]
        st, sa = ratio.loc[f, 'sig_truth'], ratio.loc[f, 'sig_anom']
        res.append((f, kt, ka))
        print(f'{f:7s} {st:9.3g} {sa:9.3g} {st/sa:5.1f} | {kt:9.3f} {ka:9.3f} | '
              f'{g_a.loc[f,1.0]:15.3f}')
    print('  топ-5 по k(σ_правда): ',
          ', '.join(f for f, _, _ in sorted(res, key=lambda r: -r[1])[:5]))
    print('  топ-5 по k(σ_аном):   ',
          ', '.join(f for f, _, _ in sorted(res, key=lambda r: -r[2])[:5]))
    rk_t = {f: i for i, (f, _, _) in enumerate(sorted(res, key=lambda r: -r[1]))}
    rk_a = {f: i for i, (f, _, _) in enumerate(sorted(res, key=lambda r: -r[2]))}
    sp = np.corrcoef([rk_t[f] for f in NAMES], [rk_a[f] for f in NAMES])[0, 1]
    print(f'  корреляция рангов: {sp:.3f}')
