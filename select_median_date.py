"""Step 1 of the bias-map analysis: rank all 48 init dates by the BASELINE (unpatched) ACC of
the Z1000 field, plot the sorted curve, and identify the median date -- the "typical quality"
day that the bias maps will then be built on.
"""
import glob
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'aurora'
CROP = 720 if MODEL == 'aurora' else None
DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
RES_DIR = os.path.expanduser('~/weather-interpretability/results/patching_19var'
                             + ('' if MODEL == 'aurora' else f'_{MODEL}'))
FIG_DIR = os.path.expanduser('~/weather-interpretability/figures/08_input_patching')
os.makedirs(FIG_DIR, exist_ok=True)
FIELD = 'Z1000'
ERA_VAR, LEVEL = 'geopotential', 1000
LEADS = [6, 24]

_triples = [(2024, m, d) for m in range(1, 13) for d in (4, 11, 18, 25)]
_hours = [0, 6, 12, 18] * (len(_triples) // 4)
ALL_DATES = {f'{y}{m:02d}{d:02d}_{h:02d}': np.datetime64(f'{y}-{m:02d}-{d:02d}T{h:02d}:00')
             for (y, m, d), h in zip(_triples, _hours)}

ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None)
clim_upper = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_upper_500_850_1000.zarr', chunks=None)


def acc(pred, tr, cl):
    a, b = (pred - cl).ravel(), (tr - cl).ravel()
    return float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum()))


rows = []
for fp in sorted(glob.glob(os.path.join(RES_DIR, f'*_{MODEL}.npz'))):
    label = os.path.basename(fp).replace(f'_{MODEL}.npz', '')
    init = ALL_DATES[label]
    d = np.load(fp)
    for lead in LEADS:
        valid = init + np.timedelta64(lead, 'h')
        ts = pd.Timestamp(valid)
        tr = ds_upper.sel(time=valid, level=LEVEL)[ERA_VAR].values[:CROP]
        cl = clim_upper.sel(dayofyear=min(ts.dayofyear, 366), hour=(ts.hour // 6) * 6,
                            level=LEVEL)[ERA_VAR].values[:CROP]
        rows.append({'date': label, 'lead': lead,
                     'acc_baseline': acc(d[f'baseline_lead{lead}_{FIELD}'], tr, cl)})
    d.close()

df = pd.DataFrame(rows)
df.to_pickle(os.path.join(RES_DIR, f'baseline_acc_{FIELD}_{MODEL}.pkl'))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
chosen = {}
for ax, lead in zip(axes, LEADS):
    s = df[df.lead == lead].sort_values('acc_baseline').reset_index(drop=True)
    n = len(s)
    mid = n // 2                      # median position in the sorted list
    chosen[lead] = s.loc[mid]
    ax.plot(range(n), s.acc_baseline, 'o-', ms=4, lw=1, color='tab:blue')
    ax.axvline(mid, color='tab:red', ls='--', lw=1.2)
    ax.plot(mid, s.loc[mid, 'acc_baseline'], 'o', ms=11, mfc='none', mec='tab:red', mew=2)
    ax.annotate(f"медиана: {s.loc[mid,'date']}\nACC = {s.loc[mid,'acc_baseline']:.4f}",
                xy=(mid, s.loc[mid, 'acc_baseline']), xytext=(0.05, 0.86),
                textcoords='axes fraction', color='tab:red', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='tab:red', lw=1))
    ax.set_xlabel('дата (отсортировано по возрастанию ACC)')
    ax.set_ylabel(f'baseline ACC, {FIELD}')
    ax.set_title(f'{MODEL.capitalize()} +{lead}ч — базовый ACC поля {FIELD}, n={n} дат')
    ax.grid(alpha=0.3)
    lo, hi = s.acc_baseline.iloc[0], s.acc_baseline.iloc[-1]
    ax.text(0.05, 0.06, f'min {lo:.4f}   max {hi:.4f}   размах {hi-lo:.4f}',
            transform=ax.transAxes, fontsize=9, color='0.35')

plt.tight_layout()
out = f'{FIG_DIR}/baseline_acc_sorted_{FIELD}_{MODEL}.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print('saved', out)
for lead in LEADS:
    c = chosen[lead]
    print(f'+{lead}ч -> медианная дата: {c.date}  (ACC={c.acc_baseline:.4f})')
