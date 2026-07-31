"""Compute the storm-relative landmark concentration metric across all 6 storm cases
(Bebinca 2024 + 5 new: Mangkhut 2018, Lingling 2019, Chanthu 2021, Muifa 2022, Haikui 2023)
plus the single quiet control -- a multi-case robustness check for the notebook 07 finding
that skeleton-decomposition landmarks cluster at the storm centre.

Metric: minimum haversine distance (km) from any of the k=256 stage-0 row-landmarks to the
real storm centre (from IBTrACS) at the +24h valid time. Small distance = a landmark sits
right on/near the storm centre.
"""
import pickle
import numpy as np
import xarray as xr

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
RESULTS_DIR = '/home/irina/weather-interpretability/results/skeleton_decomposition'

STORM_CENTERS = {
    'mangkhut_2018': (20.7, 115.3),
    'lingling_2019': (24.2, 125.3),
    'chanthu_2021': (23.8, 122.3),
    'muifa_2022': (25.7, 124.2),
    'haikui_2023': (23.5, 116.9),
    'bebinca_2024': (30.9, 121.8),  # note: bebinca_aurora.pkl uses key 'bebinca' internally but file differs
}
FILE_MAP = {
    'mangkhut_2018': 'mangkhut_2018_aurora.pkl',
    'lingling_2019': 'lingling_2019_aurora.pkl',
    'chanthu_2021': 'chanthu_2021_aurora.pkl',
    'muifa_2022': 'muifa_2022_aurora.pkl',
    'haikui_2023': 'haikui_2023_aurora.pkl',
    'bebinca_2024': 'bebinca_aurora.pkl',
}
# quiet control reference point: nearest comparably-strong vorticity extremum used in notebook 07
# (near Taiwan, ~22.5N 121E) -- not a "storm center" but the point we tested landmarks against
QUIET_REF = (22.5, 121.0)

ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
lat_full = ds_surf.latitude.values
lon_full = ds_surf.longitude.values
lat_crop = lat_full[:720]


def stage0_latlon():
    H_s, W_s = 90, 180  # stage0 output resolution
    factor_h = len(lat_crop) // H_s
    factor_w = len(lon_full) // W_s
    return lat_crop[::factor_h][:H_s], lon_full[::factor_w][:W_s], H_s, W_s


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


lat_s, lon_s, H_s, W_s = stage0_latlon()
lat_spacing_deg = abs(lat_s[1] - lat_s[0])
lon_spacing_deg = abs(lon_s[1] - lon_s[0])
print(f'stage0 grid spacing: {lat_spacing_deg:.3f} deg lat, {lon_spacing_deg:.3f} deg lon')

RADIUS_KM = 300.0  # ~1.5 grid cells at this resolution

# Null expectation: with k=256 landmarks spread over the FULL global stage0 grid (H_s*W_s
# points), how many would we expect to fall by pure chance within RADIUS_KM of a fixed point,
# given the grid's local cell density near that latitude?
def expected_count_by_chance(clat, k, total_points, lat_spacing_deg, lon_spacing_deg):
    cell_width_km = haversine_km(clat, 0, clat, lon_spacing_deg)
    cell_height_km = haversine_km(clat - lat_spacing_deg / 2, 0, clat + lat_spacing_deg / 2, 0)
    cells_in_radius = (np.pi * RADIUS_KM ** 2) / (cell_width_km * cell_height_km)
    return k * cells_in_radius / total_points


total_points = H_s * W_s
print(f'\n{"case":18s} {"center_lat":>10s} {"center_lon":>10s} {"n_within_300km":>15s} {"expected_by_chance":>18s}')
results = {}
for name, (clat, clon) in STORM_CENTERS.items():
    with open(f'{RESULTS_DIR}/{FILE_MAP[name]}', 'rb') as f:
        r = pickle.load(f)
    dec = [d for d in r['stages'][0] if d['lead_hours'] == 24][0]
    landmarks = dec['row_landmarks']
    lm_lat = lat_s[landmarks // W_s]
    lm_lon = lon_s[landmarks % W_s]
    dists = haversine_km(lm_lat, lm_lon, clat, clon)
    n_within = int((dists <= RADIUS_KM).sum())
    expected = expected_count_by_chance(clat, len(landmarks), total_points, lat_spacing_deg, lon_spacing_deg)
    results[name] = (n_within, expected)
    print(f'{name:18s} {clat:10.1f} {clon:10.1f} {n_within:15d} {expected:18.3f}')

with open(f'{RESULTS_DIR}/quiet_aurora.pkl', 'rb') as f:
    r = pickle.load(f)
dec = [d for d in r['stages'][0] if d['lead_hours'] == 24][0]
landmarks = dec['row_landmarks']
lm_lat = lat_s[landmarks // W_s]
lm_lon = lon_s[landmarks % W_s]
dists = haversine_km(lm_lat, lm_lon, QUIET_REF[0], QUIET_REF[1])
n_within_quiet = int((dists <= RADIUS_KM).sum())
expected_quiet = expected_count_by_chance(QUIET_REF[0], len(landmarks), total_points, lat_spacing_deg, lon_spacing_deg)
print(f'{"quiet_control":18s} {QUIET_REF[0]:10.1f} {QUIET_REF[1]:10.1f} {n_within_quiet:15d} {expected_quiet:18.3f}')

storm_counts = [v[0] for v in results.values()]
print(f'\nStorm cases: n_within_300km values = {storm_counts}, mean = {np.mean(storm_counts):.2f}')
print(f'Quiet control: {n_within_quiet}')
print(f'Expected by pure chance (uniform random landmarks): ~{expected:.3f} (near 0)')
n_storms_with_hit = sum(1 for c in storm_counts if c >= 1)
print(f'\n{n_storms_with_hit}/{len(storm_counts)} storms have >=1 landmark within {RADIUS_KM:.0f}km of the real centre '
      f'(vs expected ~{expected:.2f} by chance, vs quiet control = {n_within_quiet})')
