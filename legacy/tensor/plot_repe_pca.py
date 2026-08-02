import pickle
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
RESULTS_PATH = '/home/irina/weather-interpretability/results/legacy/tensor_decomposition/repe_pca.pkl'
FIG_DIR = '/home/irina/weather-interpretability/legacy/figures/09_tensor_decomposition'
INIT_TIME = np.datetime64('2024-09-15T00:00')
CHINA_EXTENT = [95, 130, 15, 45]

with open(RESULTS_PATH, 'rb') as f:
    results = pickle.load(f)

STAGE_RES_LOOKUP = {0: (90, 180), 1: (45, 90), 2: (45, 90)}
STAGE_LABELS = {0: 'Stage 0 (finest)', 1: 'Stage 1', 2: 'Stage 2 (bottleneck)'}

ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None)
lat_full = ds_surf.latitude.values
lon_full = ds_surf.longitude.values
lat_crop = lat_full[:720]


def stage_latlon(stage_idx):
    H_s, W_s = STAGE_RES_LOOKUP[stage_idx]
    factor_h = len(lat_crop) // H_s
    factor_w = len(lon_full) // W_s
    lat_s = lat_crop[::factor_h][:H_s]
    lon_s = lon_full[::factor_w][:W_s]
    return lat_s, lon_s


def relative_vorticity(u, v, lat, lon):
    R = 6.371e6
    dlat = np.deg2rad(lat[1] - lat[0])
    dlon = np.deg2rad(lon[1] - lon[0])
    dv_dx = np.gradient(v, axis=1) / (dlon * R * np.cos(np.deg2rad(lat))[:, None])
    du_dy = np.gradient(u, axis=0) / (dlat * R)
    return dv_dx - du_dy


valid_time_24h = INIT_TIME + np.timedelta64(24, 'h')
u850 = ds_upper.sel(time=valid_time_24h, level=850)['u_component_of_wind'].values
v850 = ds_upper.sel(time=valid_time_24h, level=850)['v_component_of_wind'].values
vort_24 = relative_vorticity(u850, v850, lat_full, lon_full)

fig, axes = plt.subplots(2, 3, figsize=(16, 10), subplot_kw={'projection': ccrs.PlateCarree()})
for col, stage_idx in enumerate([0, 1, 2]):
    lat_s, lon_s = stage_latlon(stage_idx)
    H_s, W_s = STAGE_RES_LOOKUP[stage_idx]
    step24 = [d for d in results[stage_idx] if d['lead_hours'] == 24][0]
    for row, pc in enumerate([0, 1]):
        ax = axes[row, col]
        score_map = step24['scores'][:, pc].reshape(H_s, W_s)
        ax.set_extent(CHINA_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
        vmax = np.abs(score_map).max()
        im = ax.pcolormesh(lon_s, lat_s, score_map, transform=ccrs.PlateCarree(),
                            cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.85)
        ev = step24['explained_var'][pc]
        ax.set_title(f'{STAGE_LABELS[stage_idx]} -- PC{pc+1} score (expl.var={ev:.2f})')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/repe_pca_scores.png', dpi=150, bbox_inches='tight')
print('saved')
