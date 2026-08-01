"""Compute the real 7x7 diagonal matrices (DeltaRMSE vs truth, DeltaACC) from whichever
multiseason Aurora patching dates have finished so far -- works on ANY subset (partial results
while the job is still running), clearly labeling how many of the 48 dates were available.
"""
import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
RESULTS_DIR = os.path.expanduser('~/weather-interpretability/results/patching_experiment_multiseason')
FIG_DIR = os.path.expanduser('~/weather-interpretability/figures/08_input_patching')
os.makedirs(FIG_DIR, exist_ok=True)

COMPONENTS = ['MSLP', 'T2M', 'Z500', 'Z850', 'T850', 'U850', 'V850']
VAR_INFO = {
    'MSLP': ('mean_sea_level_pressure', 'surf', None, 100.0, 'hPa'),
    'T2M': ('2m_temperature', 'surf', None, 1.0, 'K'),
    'Z500': ('geopotential', 'upper', 500, 9.80665, 'gpm'),
    'Z850': ('geopotential', 'upper', 850, 9.80665, 'gpm'),
    'T850': ('temperature', 'upper', 850, 1.0, 'K'),
    'U850': ('u_component_of_wind', 'upper', 850, 1.0, 'm/s'),
    'V850': ('v_component_of_wind', 'upper', 850, 1.0, 'm/s'),
}
LEADS = [6, 24]

# --- rebuild the same 48-date label -> datetime mapping used by the run script ---
date_triples = [(y, m, d) for y in (2024, 2025) for m in range(1, 13) for d in (8, 23)]
hours_cycle = [0, 6, 12, 18] * (len(date_triples) // 4)
ALL_DATES = {f'{y}{m:02d}{d:02d}_{h:02d}': np.datetime64(f'{y}-{m:02d}-{d:02d}T{h:02d}:00')
             for (y, m, d), h in zip(date_triples, hours_cycle)}

print('Loading ERA5 + climatology...', flush=True)
ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None)
clim_surf_ds = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_surface.zarr', chunks=None)
clim_upper_ds = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_upper_500_850_1000.zarr', chunks=None)


def get_truth(var, kind, level, valid_time):
    if kind == 'surf':
        return ds_surf.sel(time=valid_time)[var].values[:720]
    return ds_upper.sel(time=valid_time, level=level)[var].values[:720]


def get_clim(var, kind, level, valid_time):
    ts = pd.Timestamp(valid_time)
    doy = min(ts.dayofyear, 366)
    hour = (ts.hour // 6) * 6
    if kind == 'surf':
        return clim_surf_ds.sel(dayofyear=doy, hour=hour)[var].values[:720]
    return clim_upper_ds.sel(dayofyear=doy, hour=hour, level=level)[var].values[:720]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def corr_raw(a, b):
    a, b = a.ravel(), b.ravel()
    num = np.sum(a * b)
    den = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    return float(num / den) if den > 1e-9 else float('nan')


def acc(pred, truth, clim):
    return corr_raw(pred - clim, truth - clim)


available_files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*_aurora.npz')))
available_labels = [os.path.basename(f).replace('_aurora.npz', '') for f in available_files]
print(f'Found {len(available_labels)}/48 dates finished: {available_labels}', flush=True)

if len(available_labels) == 0:
    print('No dates finished yet -- nothing to plot.', flush=True)
    raise SystemExit(0)

rows = []
skipped = []
for label in available_labels:
    init_time = ALL_DATES[label]
    try:
        data = np.load(os.path.join(RESULTS_DIR, f'{label}_aurora.npz'))
        date_rows = []
        for lead in LEADS:
            valid_time = init_time + np.timedelta64(lead, 'h')
            for out_var in COMPONENTS:
                var, kind, level, scale, unit = VAR_INFO[out_var]
                truth = get_truth(var, kind, level, valid_time)
                clim = get_clim(var, kind, level, valid_time)
                baseline = data[f'baseline_lead{lead}_{out_var}']
                acc_baseline = acc(baseline, truth, clim)
                var_std = float(np.std(truth))
                for patched_var in COMPONENTS:
                    patched = data[f'patch{patched_var}_lead{lead}_{out_var}']
                    rmse_patched_baseline = rmse(patched, baseline)
                    acc_patched = acc(patched, truth, clim)
                    date_rows.append({
                        'date': label, 'lead': lead, 'patched': patched_var, 'output': out_var,
                        'relative_sensitivity': rmse_patched_baseline / var_std if var_std > 1e-9 else np.nan,
                        'delta_acc': acc_patched - acc_baseline,
                    })
        rows.extend(date_rows)
        del data
    except Exception:
        # file still being written by the running job (race condition) -- skip it this pass
        skipped.append(label)
        continue

if skipped:
    print(f'Skipped {len(skipped)} file(s) still being written: {skipped}', flush=True)

df = pd.DataFrame(rows)
df.to_pickle(os.path.join(RESULTS_DIR, 'partial_metrics.pkl'))
n_dates = df['date'].nunique()
print(f'Computed metrics for {n_dates} dates, {len(df)} rows total.', flush=True)


def plot_matrix(ax, mat, title, cmap='RdBu_r', fmt='{:.2f}'):
    vmax = np.nanmax(np.abs(mat.values))
    if cmap == 'RdBu_r':
        im = ax.imshow(mat.values, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(COMPONENTS))); ax.set_xticklabels(COMPONENTS, rotation=45, ha='right')
    ax.set_yticks(range(len(COMPONENTS))); ax.set_yticklabels(COMPONENTS)
    ax.set_xlabel('output variable (affected)')
    ax.set_ylabel('patched (perturbed) variable')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=8,
                    color='white' if abs(v) > vmax * 0.6 else 'black')
    ax.set_title(title)
    return im


fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for col, lead in enumerate(LEADS):
    sub = df[df.lead == lead]
    sens_mat = sub.groupby(['patched', 'output'])['relative_sensitivity'].mean().unstack().reindex(
        index=COMPONENTS, columns=COMPONENTS)
    acc_mat = sub.groupby(['patched', 'output'])['delta_acc'].mean().unstack().reindex(
        index=COMPONENTS, columns=COMPONENTS)
    im1 = plot_matrix(axes[0, col], sens_mat,
                       f'Relative sensitivity RMSE(patched,baseline)/std(truth) -- Aurora +{lead}h\n'
                       f'(REAL DATA, n={n_dates}/48 dates so far)', cmap='viridis')
    plt.colorbar(im1, ax=axes[0, col], shrink=0.85)
    im2 = plot_matrix(axes[1, col], acc_mat,
                       f'ΔACC vs baseline -- Aurora +{lead}h\n(REAL DATA, n={n_dates}/48 dates so far)')
    plt.colorbar(im2, ax=axes[1, col], shrink=0.85)

plt.tight_layout()
out_path = f'{FIG_DIR}/diagonal_matrix_PARTIAL_n{n_dates}.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved {out_path}', flush=True)
