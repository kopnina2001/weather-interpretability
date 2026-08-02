import pickle
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
INIT_TIME = np.datetime64('2024-09-15T00:00')
RESULTS_PATH = '/home/irina/weather-interpretability/results/legacy/skeleton_decomposition/bebinca_attention.pkl'
FIG_DIR = '/home/irina/weather-interpretability/legacy/figures/07_skeleton_decomposition'
WS = (2, 6, 12)

with open(RESULTS_PATH, 'rb') as f:
    results = pickle.load(f)

ds_upper = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_upper.zarr', chunks=None)

for stage_idx, stage_data in results['stages'].items():
    row0, col0 = stage_data['row0'], stage_data['col0']
    lat_s, lon_s = stage_data['lat_s'], stage_data['lon_s']
    win_lat = lat_s[row0:row0 + WS[1]]
    win_lon = lon_s[col0:col0 + WS[2]]

    step24 = [s for s in stage_data['steps'] if s['lead_hours'] == 24][0]
    n_heads = len(step24['heads'])

    valid_time = INIT_TIME + np.timedelta64(24, 'h')
    u850 = ds_upper.sel(time=valid_time, level=850, latitude=win_lat, longitude=win_lon)['u_component_of_wind'].values
    v850 = ds_upper.sel(time=valid_time, level=850, latitude=win_lat, longitude=win_lon)['v_component_of_wind'].values

    ncols = 4
    nrows = int(np.ceil(n_heads / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)
    lon_grid, lat_grid = np.meshgrid(win_lon, win_lat)
    for h in range(n_heads):
        ax = axes[h]
        dec = step24['heads'][h]
        ax.quiver(lon_grid, lat_grid, u850, v850, color='lightgray', scale=200)
        for lm in dec['row_landmarks']:
            c_off, h_off, w_off = np.unravel_index(lm, WS)
            ax.scatter(win_lon[w_off], win_lat[h_off], color='red', marker='x', s=60,
                       label='query landmark' if lm == dec['row_landmarks'][0] else None)
        for lm in dec['col_landmarks']:
            c_off, h_off, w_off = np.unravel_index(lm, WS)
            ax.scatter(win_lon[w_off], win_lat[h_off], facecolors='none', edgecolors='blue',
                       marker='o', s=90, label='key landmark' if lm == dec['col_landmarks'][0] else None)
        ax.set_title(f'head {h}\nrow_err={dec["row_rel_error"]:.2f} col_err={dec["col_rel_error"]:.2f}',
                     fontsize=9)
        if h == 0:
            ax.legend(fontsize=7, loc='upper right')
    for h in range(n_heads, len(axes)):
        axes[h].axis('off')
    fig.suptitle(f'Stage {stage_idx} block0 window attention landmarks @ +24h\n'
                 f'(background: 850hPa wind in-window; k={results["k"]} of 144)', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/attention_landmarks_stage{stage_idx}_24h.png', dpi=140, bbox_inches='tight')
    print(f'Saved attention_landmarks_stage{stage_idx}_24h.png')

    # overlap metric: how much do query-landmark and key-landmark SETS overlap per head?
    for h in range(n_heads):
        dec = step24['heads'][h]
        overlap = len(set(dec['row_landmarks'].tolist()) & set(dec['col_landmarks'].tolist()))
        print(f'  stage {stage_idx} head {h}: query/key landmark overlap = {overlap}/{results["k"]}')
