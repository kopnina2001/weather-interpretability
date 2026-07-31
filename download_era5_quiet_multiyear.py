"""Download ERA5 (ARCO) for 5 quiet-control windows (one per year, same September as each
matched storm, verified no tropical cyclone in the wider China-coast region via IBTrACS) --
paired controls for the multi-storm robustness check. Appends to the same
era5_multistorm_6h_{surface,upper}.zarr created for the storm cases.
"""
import pandas as pd
import xarray as xr

OUT_ROOT = '/srv/exw/data/irina_weather_interpretability'
SURF_OUT = f'{OUT_ROOT}/era5_multistorm_6h_surface.zarr'
UPPER_OUT = f'{OUT_ROOT}/era5_multistorm_6h_upper.zarr'

PRESSURE_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SURFACE_VARS = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'mean_sea_level_pressure']
UPPER_VARS = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']

QUIET_DATES = {
    'quiet_2018': '2018-09-22',
    'quiet_2019': '2019-09-14',
    'quiet_2021': '2021-09-20',
    'quiet_2022': '2022-09-20',
    'quiet_2023': '2023-09-16',
}

print('Opening ARCO ERA5...', flush=True)
ds_full = xr.open_zarr(
    'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
    storage_options=dict(token='anon'), chunks=None)

for label, ref_date in QUIET_DATES.items():
    ref = pd.Timestamp(ref_date)
    start = ref - pd.Timedelta(days=2)
    end = ref + pd.Timedelta(days=3)
    print(f'=== {label}: {start} to {end} ===', flush=True)

    time_index = pd.date_range(start, end, freq='6h')

    surf_snap = ds_full[SURFACE_VARS].sel(time=time_index, method='nearest').load()
    surf_snap = surf_snap.assign_coords(time=time_index)
    surf_snap.to_zarr(SURF_OUT, mode='a', append_dim='time')
    print(f'  surface appended ({len(time_index)} steps)', flush=True)

    upper_snap = ds_full[UPPER_VARS].sel(time=time_index, level=PRESSURE_LEVELS_HPA, method='nearest').load()
    upper_snap = upper_snap.assign_coords(time=time_index)
    upper_snap.to_zarr(UPPER_OUT, mode='a', append_dim='time')
    print(f'  upper appended ({len(time_index)} steps)', flush=True)

print('All done.', flush=True)
