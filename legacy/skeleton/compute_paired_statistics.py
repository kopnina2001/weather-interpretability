"""Paired statistical test: for each of 6 storms (Mangkhut 2018, Lingling 2019, Chanthu 2021,
Muifa 2022, Haikui 2023, Bebinca 2024), compare the landmark concentration metric
(n landmarks within 300km of the REAL storm centre) against its own paired quiet control from
the SAME year -- evaluated at the *same* real storm-centre location (i.e. we ask: does the
quiet-day run also place landmarks near where the storm would later be, or is that
concentration specific to the storm run itself?).
"""
import pickle
import numpy as np
import xarray as xr

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
RESULTS_DIR = '/home/irina/weather-interpretability/results/legacy/skeleton_decomposition'

STORM_CENTERS = {
    'mangkhut_2018': (20.7, 115.3),
    'lingling_2019': (24.2, 125.3),
    'chanthu_2021': (23.8, 122.3),
    'muifa_2022': (25.7, 124.2),
    'haikui_2023': (23.5, 116.9),
    'bebinca_2024': (30.9, 121.8),
}
STORM_FILE = {
    'mangkhut_2018': 'mangkhut_2018_aurora.pkl',
    'lingling_2019': 'lingling_2019_aurora.pkl',
    'chanthu_2021': 'chanthu_2021_aurora.pkl',
    'muifa_2022': 'muifa_2022_aurora.pkl',
    'haikui_2023': 'haikui_2023_aurora.pkl',
    'bebinca_2024': 'bebinca_aurora.pkl',
}
QUIET_FILE = {
    'mangkhut_2018': 'quiet_2018_aurora.pkl',
    'lingling_2019': 'quiet_2019_aurora.pkl',
    'chanthu_2021': 'quiet_2021_aurora.pkl',
    'muifa_2022': 'quiet_2022_aurora.pkl',
    'haikui_2023': 'quiet_2023_aurora.pkl',
    'bebinca_2024': 'quiet_aurora.pkl',
}

ds_surf = xr.open_zarr(f'{DATA_ROOT}/era5_2024_2025_6h_surface.zarr', chunks=None)
lat_full = ds_surf.latitude.values
lon_full = ds_surf.longitude.values
lat_crop = lat_full[:720]


def stage0_latlon():
    H_s, W_s = 90, 180
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
RADIUS_KM = 300.0


def n_within(pkl_path, clat, clon):
    with open(f'{RESULTS_DIR}/{pkl_path}', 'rb') as f:
        r = pickle.load(f)
    dec = [d for d in r['stages'][0] if d['lead_hours'] == 24][0]
    landmarks = dec['row_landmarks']
    lm_lat = lat_s[landmarks // W_s]
    lm_lon = lon_s[landmarks % W_s]
    dists = haversine_km(lm_lat, lm_lon, clat, clon)
    return int((dists <= RADIUS_KM).sum())


print(f'{"case":18s} {"storm_n":>8s} {"quiet_n":>8s} {"diff":>6s}')
diffs = []
storm_ns, quiet_ns = [], []
for name, (clat, clon) in STORM_CENTERS.items():
    sn = n_within(STORM_FILE[name], clat, clon)
    qn = n_within(QUIET_FILE[name], clat, clon)
    diffs.append(sn - qn)
    storm_ns.append(sn)
    quiet_ns.append(qn)
    print(f'{name:18s} {sn:8d} {qn:8d} {sn - qn:6d}')

diffs = np.array(diffs)
n_positive = int((diffs > 0).sum())
n_negative = int((diffs < 0).sum())
n_zero = int((diffs == 0).sum())
print(f'\nPaired differences (storm - quiet): {diffs.tolist()}')
print(f'Sign test: {n_positive} positive, {n_negative} negative, {n_zero} zero (n={len(diffs)})')
print(f'Mean storm n_within = {np.mean(storm_ns):.2f}, mean quiet n_within = {np.mean(quiet_ns):.2f}')

# exact binomial sign-test p-value (two-sided), ignoring zeros
from scipy.stats import binomtest
n_nonzero = n_positive + n_negative
if n_nonzero > 0:
    res = binomtest(n_positive, n_nonzero, 0.5, alternative='two-sided')
    print(f'Sign test p-value (two-sided, n_nonzero={n_nonzero}): {res.pvalue:.4f}')
else:
    print('All differences zero -- sign test undefined')

# paired bootstrap on the mean difference
rng = np.random.default_rng(0)
boot_means = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(10000)]
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
print(f'Bootstrap 95% CI on mean(storm-quiet) diff: [{ci_low:.2f}, {ci_high:.2f}]')
