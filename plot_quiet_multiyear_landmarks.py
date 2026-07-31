import pickle
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
RESULTS_DIR = '/home/irina/weather-interpretability/results/skeleton_decomposition'
FIG_DIR = '/home/irina/weather-interpretability/figures/07_skeleton_decomposition'
CHINA_EXTENT = [95, 130, 15, 45]

QUIET_CASES = {
    'quiet_2018': np.datetime64('2018-09-21'),  # init = ref - 1 day
    'quiet_2019': np.datetime64('2019-09-13'),
    'quiet_2021': np.datetime64('2021-09-19'),
    'quiet_2022': np.datetime64('2022-09-19'),
    'quiet_2023': np.datetime64('2023-09-15'),
}

ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_multistorm_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_multistorm_6h_upper.zarr', chunks=None)
lat_full = ds_surf.latitude.values
lon_full = ds_surf.longitude.values
lat_crop = lat_full[:720]


def stage0_latlon():
    H_s, W_s = 90, 180
    factor_h = len(lat_crop) // H_s
    factor_w = len(lon_full) // W_s
    return lat_crop[::factor_h][:H_s], lon_full[::factor_w][:W_s], H_s, W_s


def relative_vorticity(u, v, lat, lon):
    R = 6.371e6
    dlat = np.deg2rad(lat[1] - lat[0])
    dlon = np.deg2rad(lon[1] - lon[0])
    dv_dx = np.gradient(v, axis=1) / (dlon * R * np.cos(np.deg2rad(lat))[:, None])
    du_dy = np.gradient(u, axis=0) / (dlat * R)
    return dv_dx - du_dy


lat_s, lon_s, H_s, W_s = stage0_latlon()

fig, axes = plt.subplots(2, 3, figsize=(19, 12), subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()
for ax, (label, init_time) in zip(axes, QUIET_CASES.items()):
    valid_time = init_time + np.timedelta64(24, 'h')
    u850 = ds_upper.sel(time=valid_time, level=850)['u_component_of_wind'].values
    v850 = ds_upper.sel(time=valid_time, level=850)['v_component_of_wind'].values
    vort = relative_vorticity(u850, v850, lat_full, lon_full)

    with open(f'{RESULTS_DIR}/{label}_aurora.pkl', 'rb') as f:
        r = pickle.load(f)
    dec = [d for d in r['stages'][0] if d['lead_hours'] == 24][0]
    landmarks = dec['row_landmarks']
    lm_lat = lat_s[landmarks // W_s]
    lm_lon = lon_s[landmarks % W_s]

    ax.set_extent(CHINA_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
    im = ax.pcolormesh(lon_full, lat_full, vort * 1e5, transform=ccrs.PlateCarree(),
                        cmap='RdBu_r', vmin=-15, vmax=15)
    ax.scatter(lm_lon, lm_lat, transform=ccrs.PlateCarree(), c='black', s=40, marker='x', linewidths=1.5)
    ax.set_title(f'{label} (valid={str(valid_time)[:10]}) -- landmarks @ +24h')

axes[-1].axis('off')
fig.suptitle('Quiet-day landmarks across 5 years (2018,2019,2021,2022,2023) -- what do they target '
             'when there is no storm?', y=1.01, fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/quiet_multiyear_landmarks.png', dpi=150, bbox_inches='tight')
print('saved')
