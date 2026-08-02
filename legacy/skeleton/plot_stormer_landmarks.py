import pickle
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
RESULTS_DIR = '/home/irina/weather-interpretability/results/legacy/skeleton_decomposition'
FIG_DIR = '/home/irina/weather-interpretability/legacy/figures/07_skeleton_decomposition'
INIT_TIME_STORM = np.datetime64('2024-09-15T00:00')
INIT_TIME_QUIET = np.datetime64('2024-11-15T00:00')
CHINA_EXTENT = [95, 130, 15, 45]

TOKEN_LAT = np.linspace(-90 + 1.40625 / 2, 90 - 1.40625 / 2, 128).reshape(64, 2).mean(axis=1)
TOKEN_LON = np.arange(0, 360, 1.40625).reshape(128, 2).mean(axis=1)
H_TOK, W_TOK = 64, 128

ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None)


def relative_vorticity(u, v, lat, lon):
    R = 6.371e6
    dlat = np.deg2rad(lat[0] - lat[1])  # TOKEN_LAT is ascending; keep sign convention simple
    dlon = np.deg2rad(lon[1] - lon[0])
    dv_dx = np.gradient(v, axis=1) / (dlon * R * np.cos(np.deg2rad(lat))[:, None])
    du_dy = np.gradient(u, axis=0) / (np.deg2rad(lat[1] - lat[0]) * R)
    return dv_dx - du_dy


def load_vorticity(init_time):
    valid_time = init_time + np.timedelta64(24, 'h')
    u850 = ds_upper.sel(time=valid_time, level=850)['u_component_of_wind'].interp(
        latitude=TOKEN_LAT, longitude=TOKEN_LON).values
    v850 = ds_upper.sel(time=valid_time, level=850)['v_component_of_wind'].interp(
        latitude=TOKEN_LAT, longitude=TOKEN_LON).values
    return relative_vorticity(u850, v850, TOKEN_LAT, TOKEN_LON)


BLOCK_LABELS = {'shallow': 'Block 0 (shallow)', 'middle': 'Block 11 (middle)', 'deep': 'Block 23 (deep)'}

for date_label, init_time in [('storm', INIT_TIME_STORM), ('quiet', INIT_TIME_QUIET)]:
    with open(f'{RESULTS_DIR}/{date_label}_stormer.pkl', 'rb') as f:
        results = pickle.load(f)
    vort = load_vorticity(init_time)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    for ax, block_name in zip(axes, ['shallow', 'middle', 'deep']):
        dec = [d for d in results['stages'][block_name] if d['lead_hours'] == 24][0]
        landmarks = dec['row_landmarks']
        lm_lat = TOKEN_LAT[landmarks // W_TOK]
        lm_lon = TOKEN_LON[landmarks % W_TOK]

        ax.set_extent(CHINA_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
        im = ax.pcolormesh(TOKEN_LON, TOKEN_LAT, vort * 1e5, transform=ccrs.PlateCarree(),
                            cmap='RdBu_r', vmin=-15, vmax=15)
        ax.scatter(lm_lon, lm_lat, transform=ccrs.PlateCarree(), c='black', s=60, marker='x',
                   linewidths=2, label='landmark grid point')
        ax.set_title(f'Stormer {BLOCK_LABELS[block_name]} ({date_label}) -- landmarks @ +24h\n'
                     f'(background: 850hPa rel. vorticity x1e5, k={results["k"]} of 8192)')
        ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/stormer_landmarks_on_vorticity_24h_{date_label}.png', dpi=150, bbox_inches='tight')
    print(f'saved stormer_landmarks_on_vorticity_24h_{date_label}.png')
