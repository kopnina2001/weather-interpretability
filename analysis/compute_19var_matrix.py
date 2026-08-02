"""Build the full 19x19 sensitivity / dACC matrices from the patching_19var runs.

Works on partial results (skips dates not finished yet, and any file whose array count is not
the expected 760 -- guards against the truncated-file problem we hit during the disk-full
incident).
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

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'aurora'
# Aurora crops the south pole (720 lat); Pangu keeps the full 721-lat grid.
CROP = 720 if MODEL == 'aurora' else None
RES_DIR = os.path.expanduser(
    '~/weather-interpretability/results/patching/patching_19var'
    + ('' if MODEL == 'aurora' else f'_{MODEL}'))
FIG_DIR = os.path.expanduser('~/weather-interpretability/figures/patching_matrices')
os.makedirs(FIG_DIR, exist_ok=True)
EXPECTED_ARRAYS = 760          # 20 runs x 2 leads x 19 fields
LEADS = [6, 24]

# display order: surface block, then one block per upper variable (1000/850/500)
NAMES = ['MSLP', 'U10', 'V10', 'T2M']
for s in ('Z', 'Q', 'T', 'U', 'V'):
    NAMES += [f'{s}1000', f'{s}850', f'{s}500']
GROUP_EDGES = [4, 7, 10, 13, 16]   # where to draw separators

VARS = {
    'MSLP': ('surf', 'mean_sea_level_pressure', None),
    'U10': ('surf', '10m_u_component_of_wind', None),
    'V10': ('surf', '10m_v_component_of_wind', None),
    'T2M': ('surf', '2m_temperature', None),
}
for _s, _l in (('Z', 'geopotential'), ('Q', 'specific_humidity'), ('T', 'temperature'),
               ('U', 'u_component_of_wind'), ('V', 'v_component_of_wind')):
    for _lv in (1000, 850, 500):
        VARS[f'{_s}{_lv}'] = ('upper', _l, _lv)

_triples = [(2024, m, d) for m in range(1, 13) for d in (4, 11, 18, 25)]
_hours = [0, 6, 12, 18] * (len(_triples) // 4)
ALL_DATES = {f'{y}{m:02d}{d:02d}_{h:02d}': np.datetime64(f'{y}-{m:02d}-{d:02d}T{h:02d}:00')
             for (y, m, d), h in zip(_triples, _hours)}

print('Opening ERA5 + climatology...', flush=True)
ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None)
clim_surf = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_surface.zarr', chunks=None)
clim_upper = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_upper_500_850_1000.zarr', chunks=None)

# --- area weights w = cos(phi), the WeatherBench-2 / ECMWF convention -----------------
_lat = ds_surf.latitude.values[:CROP]
W = np.cos(np.deg2rad(_lat))[:, None] * np.ones((1, ds_surf.longitude.size))
W = W / W.sum()          # normalised once; every weighted sum below divides by sum(w)=1

_truth_cache: dict = {}


def truth_and_clim(name, valid):
    key = (name, str(valid))
    if key in _truth_cache:
        return _truth_cache[key]
    kind, var, lvl = VARS[name]
    ts = pd.Timestamp(valid)
    doy, hour = min(ts.dayofyear, 366), (ts.hour // 6) * 6
    if kind == 'surf':
        tr = ds_surf.sel(time=valid)[var].values[:CROP]
        cl = clim_surf.sel(dayofyear=doy, hour=hour)[var].values[:CROP]
    else:
        tr = ds_upper.sel(time=valid, level=lvl)[var].values[:CROP]
        cl = clim_upper.sel(dayofyear=doy, hour=hour, level=lvl)[var].values[:CROP]
    if len(_truth_cache) > 400:
        _truth_cache.clear()
    _truth_cache[key] = (tr, cl)
    return tr, cl


def rmse(a, b):
    """Unweighted RMSE (kept for comparison with the earlier, non-standard version)."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rmse_w(a, b):
    return float(np.sqrt((W * (a - b) ** 2).sum()))


def std_w(a):
    m = (W * a).sum()
    return float(np.sqrt((W * (a - m) ** 2).sum()))


def acc(pred, tr, cl):
    a, b = (pred - cl).ravel(), (tr - cl).ravel()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 1e-9 else np.nan


def acc_w(pred, tr, cl):
    a, b = pred - cl, tr - cl
    den = np.sqrt((W * a ** 2).sum() * (W * b ** 2).sum())
    return float((W * a * b).sum() / den) if den > 1e-12 else np.nan


N_WORKERS = 6      # npz zlib decompression is single-threaded per file, so scale by processes


def process_file(fp):
    """Decompress one date's npz and return its metric rows. Runs in a worker process."""
    label = os.path.basename(fp).replace(f'_{MODEL}.npz', '')
    try:
        d = np.load(fp)
        if len(d.files) != EXPECTED_ARRAYS:
            return label, None, f'{len(d.files)} arrays'
        init = ALL_DATES[label]
        out_rows = []
        for lead in LEADS:
            valid = init + np.timedelta64(lead, 'h')
            for out in NAMES:
                tr, cl = truth_and_clim(out, valid)
                base = d[f'baseline_lead{lead}_{out}']
                acc_base, acc_base_w = acc(base, tr, cl), acc_w(base, tr, cl)
                sd, sd_w = float(np.std(tr)), std_w(tr)
                for pat in NAMES:
                    p_ = d[f'patch{pat}_lead{lead}_{out}']
                    out_rows.append({
                        'date': label, 'lead': lead, 'patched': pat, 'output': out,
                        'rel_sens': rmse(p_, base) / sd if sd > 1e-9 else np.nan,
                        'dacc': acc(p_, tr, cl) - acc_base,
                        'rel_sens_w': rmse_w(p_, base) / sd_w if sd_w > 1e-9 else np.nan,
                        'dacc_w': acc_w(p_, tr, cl) - acc_base_w,
                        'acc_base_w': acc_base_w,
                    })
        d.close()
        return label, out_rows, None
    except Exception as e:
        return label, None, str(e)[:40]


if __name__ == '__main__':
    import multiprocessing as mp
    files = sorted(glob.glob(os.path.join(RES_DIR, f'*_{MODEL}.npz')))
    rows, skipped = [], []
    with mp.Pool(N_WORKERS) as pool:
        for label, out_rows, err in pool.imap_unordered(process_file, files):
            if err is not None:
                skipped.append((label, err))
            else:
                rows.extend(out_rows)
                print(f'  {label} ok ({len(rows)} rows so far)', flush=True)

    if skipped:
        print(f'skipped {len(skipped)}: {skipped}', flush=True)
    if not rows:
        raise SystemExit('no complete dates yet')

    df = pd.DataFrame(rows)
    n = df.date.nunique()
    df.to_pickle(os.path.join(RES_DIR, f'metrics_19var_{MODEL}.pkl'))
    print(f'{n} dates, {len(df)} rows -> metrics_19var.pkl', flush=True)


    def draw(ax, mat, title, cmap, symmetric):
        v = np.nanmax(np.abs(mat))
        im = ax.imshow(mat, cmap=cmap, vmin=(-v if symmetric else 0), vmax=v)
        ax.set_xticks(range(19)); ax.set_xticklabels(NAMES, rotation=90, fontsize=7)
        ax.set_yticks(range(19)); ax.set_yticklabels(NAMES, fontsize=7)
        ax.set_xlabel('output variable (affected)', fontsize=8)
        ax.set_ylabel('patched (perturbed) variable', fontsize=8)
        for e in GROUP_EDGES:
            ax.axhline(e - 0.5, color='k', lw=0.6, alpha=0.5)
            ax.axvline(e - 0.5, color='k', lw=0.6, alpha=0.5)
        for i in range(19):
            for j in range(19):
                val = mat[i, j]
                if not np.isfinite(val):
                    continue
                ax.text(j, i, f'{val:.2f}'.lstrip('0') if abs(val) < 1 else f'{val:.1f}',
                        ha='center', va='center', fontsize=4.2,
                        color='white' if abs(val) > v * 0.55 else 'black')
        ax.set_title(title, fontsize=9)
        return im


    fig, axes = plt.subplots(2, 2, figsize=(19, 18))
    for col, lead in enumerate(LEADS):
        sub = df[df.lead == lead]
        m1 = sub.groupby(['patched', 'output'])['rel_sens_w'].mean().unstack().reindex(
            index=NAMES, columns=NAMES).values
        m2 = sub.groupby(['patched', 'output'])['dacc_w'].mean().unstack().reindex(
            index=NAMES, columns=NAMES).values
        m1u = sub.groupby(['patched', 'output'])['rel_sens'].mean().unstack().reindex(
            index=NAMES, columns=NAMES).values
        off = ~np.eye(len(NAMES), dtype=bool)
        print(f'  lead +{lead}h: weighted vs unweighted sensitivity, off-diagonal '
              f'r={np.corrcoef(m1[off], m1u[off])[0,1]:.4f}, '
              f'mass {m1[off].sum():.1f} vs {m1u[off].sum():.1f}', flush=True)
        im1 = draw(axes[0, col], m1,
                   f'Relative sensitivity  RMSE$_w$(patched,baseline)/$\\sigma_w$(truth), cos$\\varphi$-weighted  --  {MODEL.capitalize()} +{lead}h   (n={n} dates)',
                   'viridis', False)
        plt.colorbar(im1, ax=axes[0, col], shrink=0.8)
        im2 = draw(axes[1, col], m2,
                   f'$\\Delta$ACC$_w$ vs baseline, cos$\\varphi$-weighted  --  {MODEL.capitalize()} +{lead}h   (n={n} dates)', 'RdBu_r', True)
        plt.colorbar(im2, ax=axes[1, col], shrink=0.8)

    plt.tight_layout()
    out = f'{FIG_DIR}/matrix_19var_{MODEL}_n{n}_weighted.png'
    plt.savefig(out, dpi=160, bbox_inches='tight')
    print('saved', out, flush=True)
