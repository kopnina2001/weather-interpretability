"""Compute all metrics for the Stormer input-patching experiment
(results/legacy/patching_experiment_stormer/*.npz) -- companion to compute_patching_metrics.py (Aurora),
adapted to Stormer's coarser 1.40625deg (128x256) regridded grid and the flat surf/upper variable
lists used by run_patching_experiment_stormer.py. Saves a tidy summary table + the raw fields
needed for error maps, so the notebook only needs to load small cached arrays.
"""
import os
import pickle
import numpy as np
import pandas as pd
import xarray as xr

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
RESULTS_DIR = os.path.expanduser('~/weather-interpretability/results/legacy/patching_experiment_stormer')
OUT_SUMMARY = os.path.join(RESULTS_DIR, 'metrics_summary.pkl')
OUT_MAPS = os.path.join(RESULTS_DIR, 'error_maps.pkl')

CHINA_EXTENT = [95, 130, 15, 45]
DATES = {'storm': np.datetime64('2024-09-15T00:00'), 'quiet': np.datetime64('2024-11-15T00:00')}
COMPONENTS = ['surface_mslp', 'wind', 'mass_field', 'temperature', 't2m']
SCOPES = ['global', 'regional']

SURFACE_VARS = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
UPPER_VARS = ['geopotential', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity']
PRESSURE_LEVELS_HPA = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
CLIM_LEVELS = [500, 850, 1000]

# The core output variables we report in the sensitivity matrix / summary table.
CORE_VARS = [
    ('mean_sea_level_pressure', 'surf', None, 'MSLP', 100.0, 'hPa'),
    ('2m_temperature', 'surf', None, 'T2M', 1.0, 'K'),
    ('geopotential', 'upper', 500, 'Z500', 9.80665, 'gpm'),
    ('geopotential', 'upper', 850, 'Z850', 9.80665, 'gpm'),
    ('temperature', 'upper', 850, 'T850', 1.0, 'K'),
    ('u_component_of_wind', 'upper', 850, 'U850', 1.0, 'm/s'),
    ('v_component_of_wind', 'upper', 850, 'V850', 1.0, 'm/s'),
]

TARGET_LAT = np.linspace(-90 + 1.40625 / 2, 90 - 1.40625 / 2, 128)
TARGET_LON = np.arange(0, 360, 1.40625)


def regrid(da):
    return da.interp(latitude=TARGET_LAT, longitude=TARGET_LON)


ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None).sel(level=PRESSURE_LEVELS_HPA)
clim_surf_ds = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_surface.zarr', chunks=None)
clim_upper_ds = xr.open_zarr(f'{DATA_ROOT}/climatology_1990-2019_upper_500_850_1000.zarr', chunks=None)

lat_full = TARGET_LAT
lon_full = TARGET_LON
lat_mask = (lat_full >= CHINA_EXTENT[2]) & (lat_full <= CHINA_EXTENT[3])
lon_mask = (lon_full >= CHINA_EXTENT[0]) & (lon_full <= CHINA_EXTENT[1])
region_idx = np.ix_(lat_mask, lon_mask)
lat_weights_global = np.cos(np.deg2rad(lat_full))[:, None] * np.ones((1, len(lon_full)))


def get_field(var, kind, level, date_arrays, key, step_idx):
    """Extract a single (lat, lon) field for one variable/level from a loaded npz dict."""
    if kind == 'surf':
        vi = SURFACE_VARS.index(var)
        return date_arrays[f'{key}_surf'][step_idx, vi]
    else:
        vi = UPPER_VARS.index(var)
        li = PRESSURE_LEVELS_HPA.index(level)
        return date_arrays[f'{key}_upper'][step_idx, vi, li]


def get_truth(var, kind, level, valid_time):
    if kind == 'surf':
        return regrid(ds_surf.sel(time=valid_time))[var].values
    else:
        return regrid(ds_upper.sel(time=valid_time, level=level))[var].values


def get_climatology_field(var, kind, level, valid_time):
    ts = pd.Timestamp(valid_time)
    doy = min(ts.dayofyear, 366)
    hour = (ts.hour // 6) * 6
    if kind == 'surf':
        return regrid(clim_surf_ds.sel(dayofyear=doy, hour=hour))[var].values
    else:
        return regrid(clim_upper_ds.sel(dayofyear=doy, hour=hour, level=level))[var].values


def rmse(a, b, weights=None):
    if weights is None:
        return float(np.sqrt(np.mean((a - b) ** 2)))
    return float(np.sqrt(np.average((a - b) ** 2, weights=weights)))


def corr_raw(a, b):
    a, b = a.ravel(), b.ravel()
    num = np.sum(a * b)
    den = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    return float(num / den) if den > 1e-9 else float('nan')


def acc(pred, truth, clim):
    return corr_raw(pred - clim, truth - clim)


def storm_weight(valid_time, lat, lon):
    """Gaussian weight centered on the truth MSLP minimum inside CHINA_EXTENT (storm track
    proxy), sigma=5 degrees, combined with cos(lat) area weighting."""
    mslp = regrid(ds_surf.sel(time=valid_time))['mean_sea_level_pressure'].values
    sub = mslp[region_idx]
    iy, ix = np.unravel_index(np.argmin(sub), sub.shape)
    center_lat = lat[lat_mask][iy]
    center_lon = lon[lon_mask][ix]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    dist2 = (lat_grid - center_lat) ** 2 + (lon_grid - center_lon) ** 2
    gauss = np.exp(-dist2 / (2 * 5.0 ** 2))
    return gauss * np.cos(np.deg2rad(lat_grid))


rows = []
maps_store = {}

for date_label, init_time in DATES.items():
    print(f'=== {date_label} ===', flush=True)
    npz_path = os.path.join(RESULTS_DIR, f'{date_label}.npz')
    data = np.load(npz_path)

    for step_idx in range(4):
        lead_hours = (step_idx + 1) * 6
        valid_time = init_time + np.timedelta64(lead_hours, 'h')
        print(f'  lead +{lead_hours}h ({valid_time})', flush=True)

        w_area = lat_weights_global
        w_storm = storm_weight(valid_time, lat_full, lon_full) if date_label == 'storm' else None

        for var, kind, level, var_label, unit_scale, unit in CORE_VARS:
            truth = get_truth(var, kind, level, valid_time)
            clim = get_climatology_field(var, kind, level, valid_time)
            baseline = get_field(var, kind, level, data, 'baseline', step_idx)
            var_std = float(np.std(truth))

            rmse_baseline_truth = rmse(baseline, truth)
            acc_baseline = acc(baseline, truth, clim)

            for component in COMPONENTS:
                for scope in SCOPES:
                    key = f'{component}_{scope}'
                    patched = get_field(var, kind, level, data, key, step_idx)

                    rmse_patched_baseline = rmse(patched, baseline)
                    rmse_patched_truth = rmse(patched, truth)
                    acc_patched = acc(patched, truth, clim)
                    rmse_w_area = rmse(patched, baseline, weights=w_area)
                    rmse_w_storm = rmse(patched, baseline, weights=w_storm) if w_storm is not None else np.nan

                    rows.append({
                        'date': date_label, 'lead_hours': lead_hours, 'component': component,
                        'scope': scope, 'var': var_label, 'unit': unit,
                        'metric1_rmse_patch_baseline': rmse_patched_baseline / unit_scale,
                        'metric2_delta_rmse_vs_truth': (rmse_patched_truth - rmse_baseline_truth) / unit_scale,
                        'metric3_delta_acc': acc_patched - acc_baseline,
                        'metric4_relative_sensitivity': rmse_patched_baseline / var_std if var_std > 1e-9 else np.nan,
                        'weighted_rmse_area': rmse_w_area / unit_scale,
                        'weighted_rmse_storm': rmse_w_storm / unit_scale if not np.isnan(rmse_w_storm) else np.nan,
                        'rmse_baseline_truth': rmse_baseline_truth / unit_scale,
                        'rmse_patched_truth': rmse_patched_truth / unit_scale,
                    })

            # Cache full 2D fields for error maps: only for MSLP and Z850, at +24h, global scope
            if lead_hours == 24 and var_label in ('MSLP', 'Z850'):
                for component in COMPONENTS:
                    key = f'{component}_global'
                    patched = get_field(var, kind, level, data, key, step_idx)
                    maps_store[(date_label, var_label, component)] = {
                        'truth': truth / unit_scale, 'baseline': baseline / unit_scale,
                        'patched': patched / unit_scale, 'unit': unit,
                    }
    del data

df = pd.DataFrame(rows)
df.to_pickle(OUT_SUMMARY)
print(f'Saved metrics summary ({len(df)} rows) to {OUT_SUMMARY}', flush=True)

with open(OUT_MAPS, 'wb') as f:
    pickle.dump({'maps': maps_store, 'lat': lat_full, 'lon': lon_full}, f)
print(f'Saved error map cache to {OUT_MAPS}', flush=True)
