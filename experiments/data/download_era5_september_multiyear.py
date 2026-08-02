"""Download ERA5 (ARCO) for the specific September windows needed for the multi-storm
skeleton-decomposition robustness check (7 storms, one per year 2018-2025, same region/season
as Bebinca). Downloads only the days actually needed per storm (a few days before landfall for
Aurora's t-6h input timestep, plus 48h after for the rollout leads), not full months, to keep
this cheap. Reuses the same month-by-month/day-by-day `.load()` + `chunks=None` pattern that
avoided the dask memory blowups during the original 2024-2025 download (see README: sortby on
the full un-sliced dataset built a graph over the whole 85-year archive).

Writes to NEW zarr stores (separate from era5_2024_2025_6h_*.zarr) so the existing production
data is untouched:
  era5_multistorm_6h_surface.zarr
  era5_multistorm_6h_upper.zarr
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

OUT_ROOT = '/srv/exw/data/irina_weather_interpretability'
SURF_OUT = f'{OUT_ROOT}/era5_multistorm_6h_surface.zarr'
UPPER_OUT = f'{OUT_ROOT}/era5_multistorm_6h_upper.zarr'

PRESSURE_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SURFACE_VARS = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'mean_sea_level_pressure']
UPPER_VARS = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']

# (storm name, landfall/peak reference date) -- download window = ref_date - 2 days to ref_date + 3 days
STORMS = {
    'mangkhut_2018': '2018-09-16',
    'lingling_2019': '2019-09-05',
    'chanthu_2021': '2021-09-12',
    'muifa_2022': '2022-09-13',
    'haikui_2023': '2023-09-05',
}

print('Opening ARCO ERA5...', flush=True)
ds_full = xr.open_zarr(
    'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
    storage_options=dict(token='anon'), chunks=None)

first_surf, first_upper = True, True
for storm_name, ref_date in STORMS.items():
    ref = pd.Timestamp(ref_date)
    start = ref - pd.Timedelta(days=2)
    end = ref + pd.Timedelta(days=3)
    print(f'=== {storm_name}: {start} to {end} ===', flush=True)

    # positional time slicing (not .sel with explicit timestamp list) to stay lazy, per the
    # lesson learned during the original download
    time_index = pd.date_range(start, end, freq='6h')

    surf_snap = ds_full[SURFACE_VARS].sel(time=time_index, method='nearest').load()
    surf_snap = surf_snap.assign_coords(time=time_index)
    mode = 'w' if first_surf else 'a'
    surf_snap.to_zarr(SURF_OUT, mode=mode, append_dim=None if first_surf else 'time')
    first_surf = False
    print(f'  surface saved ({len(time_index)} steps)', flush=True)

    upper_snap = ds_full[UPPER_VARS].sel(time=time_index, level=PRESSURE_LEVELS_HPA, method='nearest').load()
    upper_snap = upper_snap.assign_coords(time=time_index)
    mode = 'w' if first_upper else 'a'
    upper_snap.to_zarr(UPPER_OUT, mode=mode, append_dim=None if first_upper else 'time')
    first_upper = False
    print(f'  upper saved ({len(time_index)} steps)', flush=True)

print('All done.', flush=True)
