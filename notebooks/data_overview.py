# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # ERA5 Visualization — Weather Model Interpretability Project
#
# Визуализация трёх синоптических кейсов из ARCO ERA5:
#
# 1. **Storm Eunice** — 2022-02-18 12:00 UTC (взрывной циклогенез, Северная Атлантика/Британия)
# 2. **Европейская волна тепла / блокинг** — 2022-07-18 12:00 UTC
# 3. **Спокойный baseline** — 2021-10-05 12:00 UTC
#
# Источник: `gs://gcp-public-data-arco-era5` (Analysis-Ready Cloud-Optimized ERA5, Google Research).
#

# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.rcParams['figure.dpi'] = 110


# %% [markdown]
# ## 1. Открываем ARCO ERA5
#
# Основной массив: 31 приповерхностная переменная + переменные на 37 уровнях давления, 0.25°, часовое разрешение.
#

# %%
ARCO_URL = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'

ds = xr.open_zarr(ARCO_URL, storage_options=dict(token='anon'), chunks=None)

# ARCO stores longitude as 0..359.75; convert to -180..180 and sort so that
# regional slices crossing the prime meridian (e.g. North Atlantic/Europe) don't
# produce a seam artifact at lon=0.
ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby('longitude')

print(ds)


# %% [markdown]
# ## 2. Маппинг имён переменных: ARCO (длинные CDS-имена) → короткие GRIB-имена
#
# Этот словарь переиспользуется в `02_saliency_experiment.ipynb` — модели (Pangu-Weather, Aurora) ожидают короткие имена.
#

# %%
# Короткие имена, которые ожидают Pangu-Weather / Aurora
ARCO_TO_SHORT = {
    '2m_temperature': 't2m',
    '10m_u_component_of_wind': 'u10',
    '10m_v_component_of_wind': 'v10',
    'mean_sea_level_pressure': 'msl',
    'geopotential': 'z',
    'specific_humidity': 'q',
    'temperature': 't',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
}

# 13 уровней давления (гПа), которые требуются Pangu-Weather / Aurora
PRESSURE_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

SURFACE_VARS = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure']
UPPER_AIR_VARS = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']

print('Surface vars available:', all(v in ds.data_vars for v in SURFACE_VARS))
print('Upper-air vars available:', all(v in ds.data_vars for v in UPPER_AIR_VARS))
print('Levels present in dataset:', ds.level.values if 'level' in ds.coords else 'N/A')


# %% [markdown]
# ## 3. Три синоптических кейса
#

# %%
CASES = {
    'storm_eunice':   {'time': '2022-02-18T12:00', 'label': 'Storm Eunice (18 Feb 2022, 12:00 UTC)'},
    'heatwave_block': {'time': '2022-07-18T12:00', 'label': 'European heatwave / blocking (18 Jul 2022, 12:00 UTC)'},
    'quiet_baseline': {'time': '2021-10-05T12:00', 'label': 'Quiet baseline (5 Oct 2021, 12:00 UTC)'},
}


# %% [markdown]
# ## 4. Вырезаем нужные переменные и уровни для каждого кейса
#
# Тянем только 13 нужных уровней (не все 37) и только нужный временной срез — иначе скачивание будет неоправданно долгим.
#

# %%
def load_case(time_str):
    """Load surface + upper-air fields for one timestamp, sliced to the 13 required pressure levels."""
    snapshot = ds.sel(time=time_str)
    surface = snapshot[SURFACE_VARS].load()
    upper_air = snapshot[UPPER_AIR_VARS].sel(level=PRESSURE_LEVELS_HPA).load()
    return surface, upper_air

case_data = {}
for key, meta in CASES.items():
    print(f'Loading {key} ({meta["time"]})...')
    surface, upper_air = load_case(meta['time'])
    case_data[key] = {'surface': surface, 'upper_air': upper_air}
print('Done.')


# %% [markdown]
# ## 5. Визуализация: T2M, MSLP, ветер (10м), Z500
#
# Для каждого кейса — 4 карты в один ряд, чтобы легко сравнивать три режима между собой.
#

# %%
def plot_case(key):
    meta = CASES[key]
    surface = case_data[key]['surface']
    upper_air = case_data[key]['upper_air']

    t2m = surface['2m_temperature'] - 273.15  # K -> C
    msl = surface['mean_sea_level_pressure'] / 100  # Pa -> hPa
    u10 = surface['10m_u_component_of_wind']
    v10 = surface['10m_v_component_of_wind']
    wind_speed = np.sqrt(u10**2 + v10**2)
    z500 = upper_air['geopotential'].sel(level=500) / 9.80665  # m^2/s^2 -> geopotential meters

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(meta['label'], fontsize=14, y=1.05)

    panels = [
        (t2m, 'T2M (°C)', 'RdBu_r'),
        (msl, 'MSLP (hPa)', 'viridis'),
        (wind_speed, '10m wind speed (m/s)', 'plasma'),
        (z500, 'Z500 (gpm)', 'cividis'),
    ]

    for ax, (field, title, cmap) in zip(axes, panels):
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        im = field.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.85)

    plt.tight_layout()
    plt.savefig(f'figures/data_overview/{key}_overview.png', dpi=150, bbox_inches='tight')
    plt.show()

import os
os.makedirs('figures/data_overview', exist_ok=True)

for key in CASES:
    plot_case(key)


# %% [markdown]
# ## 6. Региональный zoom (Северная Атлантика/Европа) для Storm Eunice и волны тепла
#
# Глобальный вид скрывает детали — для двух экстремальных кейсов даём региональный крупный план.
#

# %%
def plot_regional_zoom(key, lon_range=(-40, 30), lat_range=(30, 70)):
    meta = CASES[key]
    surface = case_data[key]['surface']
    msl = surface['mean_sea_level_pressure'] / 100
    u10 = surface['10m_u_component_of_wind']
    v10 = surface['10m_v_component_of_wind']

    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([*lon_range, *lat_range], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.6)

    cf = msl.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), levels=20, cmap='viridis', add_colorbar=True,
                            cbar_kwargs={'label': 'MSLP (hPa)', 'shrink': 0.8})
    msl.plot.contour(ax=ax, transform=ccrs.PlateCarree(), levels=20, colors='k', linewidths=0.4)

    step = 4  # subsample wind barbs for readability
    ax.quiver(u10.longitude[::step], u10.latitude[::step],
              u10.values[::step, ::step], v10.values[::step, ::step],
              transform=ccrs.PlateCarree(), scale=400, width=0.002, alpha=0.7)

    ax.set_title(f"{meta['label']} — regional zoom (MSLP + 10m wind)")
    plt.savefig(f'figures/data_overview/{key}_regional_zoom.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_regional_zoom('storm_eunice')
plot_regional_zoom('heatwave_block')


# %% [markdown]
# ## 7. Сохранение подготовленных данных для следующего ноутбука
#
# Сохраняем сырые срезы (surface + upper_air на 13 уровнях) в **Zarr** (не NetCDF — cloud-native chunked формат, компактнее за счёт компрессии по умолчанию, тот же формат, что и сам ARCO ERA5), чтобы `02_saliency_experiment.ipynb` не тянул их из ARCO заново.
#

# %%
import os
os.makedirs('data', exist_ok=True)

for key in CASES:
    case_data[key]['surface'].to_zarr(f'data/{key}_surface.zarr', mode='w')
    case_data[key]['upper_air'].to_zarr(f'data/{key}_upper_air.zarr', mode='w')

print('Saved:', os.listdir('data'))

