import pickle
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
RESULTS_PATH = '/home/irina/weather-interpretability/results/skeleton_decomposition/quiet_aurora.pkl'
FIG_DIR = '/home/irina/weather-interpretability/figures/07_skeleton_decomposition'
INIT_TIME = np.datetime64('2024-11-15T00:00')
CHINA_EXTENT = [95, 130, 15, 45]

with open(RESULTS_PATH, 'rb') as f:
    results = pickle.load(f)

STAGE_RES = results['stage_res']
stages = sorted(results['stages'].keys())
STAGE_LABELS = {stages[0]: 'Stage 0 (finest)', stages[-1]: 'Stage %d (bottleneck)' % stages[-1]}
for s in stages:
    STAGE_LABELS.setdefault(s, f'Stage {s}')

ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None)
lat_full = ds_surf.latitude.values
lon_full = ds_surf.longitude.values
lat_crop = lat_full[:720]
lon_crop = lon_full


def stage_latlon(stage_idx):
    last = stages[-1]
    res_idx = stage_idx if stage_idx == last else stage_idx + 1
    _, H_s, W_s = STAGE_RES[res_idx]
    factor_h = len(lat_crop) // H_s
    factor_w = len(lon_crop) // W_s
    lat_s = lat_crop[::factor_h][:H_s]
    lon_s = lon_crop[::factor_w][:W_s]
    return lat_s, lon_s, H_s, W_s


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

fig, axes = plt.subplots(1, len(stages), figsize=(6.5 * len(stages), 6),
                         subplot_kw={'projection': ccrs.PlateCarree()})
if len(stages) == 1:
    axes = [axes]
for ax, s in zip(axes, stages):
    lat_s, lon_s, H_s, W_s = stage_latlon(s)
    dec = [d for d in results['stages'][s] if d['lead_hours'] == 24][0]
    landmarks = dec['row_landmarks']
    lm_lat = lat_s[landmarks // W_s]
    lm_lon = lon_s[landmarks % W_s]

    ax.set_extent(CHINA_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
    im = ax.pcolormesh(lon_full, lat_full, vort_24 * 1e5, transform=ccrs.PlateCarree(),
                        cmap='RdBu_r', vmin=-15, vmax=15)
    ax.scatter(lm_lon, lm_lat, transform=ccrs.PlateCarree(), c='black', s=20,
               marker='x', linewidths=1.2, label='landmark grid point')
    ax.set_title(f'{STAGE_LABELS[s]} (QUIET control, 2024-11-15) — landmarks @ +24h\n'
                 f'(background: 850hPa rel. vorticity x1e5, k={results["k"]})')
    ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/landmarks_on_vorticity_24h_QUIET.png', dpi=150, bbox_inches='tight')
print('Saved landmarks_on_vorticity_24h_QUIET.png')
