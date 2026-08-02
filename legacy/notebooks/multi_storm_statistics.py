# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 10 — Multi-storm statistics: does the skeleton-decomposition finding generalize?
#
# Notebook 07 found that skeleton-decomposition landmarks (Aurora backbone hidden state)
# cluster tightly on the Typhoon Bebinca centre (2024), confirmed against a single quiet
# control. That is an n=1 case study. Here we test whether the effect **generalizes** across
# independent storms, years, and a properly paired (not single-reference) quiet control.
#
# ## Design
#
# - **Region fixed**: China coast / Western Pacific (same crop as Bebinca).
# - **Season fixed**: September (removes the seasonal confound notebook 09 ran into with RepE).
# - **One storm per year, 2018-2023 + 2024 (Bebinca)** = 6 real storms (IBTrACS-verified centre
#   coordinates/times), spanning very different intensities (40-115 kt) and 6 different years
#   (reduces the risk that results reflect one particular year's climate/ENSO state).
# - **Paired quiet control per storm**: for each storm year, a quiet date in the *same* September
#   (IBTrACS-verified: no cyclone anywhere in a wide China-coast box), same region.
# - **Data**: small multi-year ERA5 windows (`era5_multistorm_6h_{surface,upper}.zarr`, only the
#   ~5-day window around each storm/quiet reference date -- not full months, kept cheap) plus the
#   existing `era5_2024_2025` cache and WB2 climatology already in place for Bebinca.
# - **Metric**: number of stage-0 (k=256) row-landmarks within 300 km of the *real* storm centre
#   (from IBTrACS) at the +24h valid time -- compared, for each pair, storm-day vs quiet-day
#   landmark count **at the same geographic coordinates**.

# %%
import os
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import binomtest

DATA_ROOT = '/srv/exw/data/irina_weather_interpretability'
RESULTS_DIR = os.path.expanduser('~/weather-interpretability/results/legacy/skeleton_decomposition')

STORM_CENTERS = {
    'mangkhut_2018': (20.7, 115.3),
    'lingling_2019': (24.2, 125.3),
    'chanthu_2021': (23.8, 122.3),
    'muifa_2022': (25.7, 124.2),
    'haikui_2023': (23.5, 116.9),
    'bebinca_2024': (30.9, 121.8),
}
STORM_FILE = {k: f'{k}_aurora.pkl' for k in STORM_CENTERS}
STORM_FILE['bebinca_2024'] = 'bebinca_aurora.pkl'
QUIET_FILE = {
    'mangkhut_2018': 'quiet_2018_aurora.pkl', 'lingling_2019': 'quiet_2019_aurora.pkl',
    'chanthu_2021': 'quiet_2021_aurora.pkl', 'muifa_2022': 'quiet_2022_aurora.pkl',
    'haikui_2023': 'quiet_2023_aurora.pkl', 'bebinca_2024': 'quiet_aurora.pkl',
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


rows = []
for name, (clat, clon) in STORM_CENTERS.items():
    sn = n_within(STORM_FILE[name], clat, clon)
    qn = n_within(QUIET_FILE[name], clat, clon)
    rows.append({'storm': name, 'storm_day_n': sn, 'quiet_day_n': qn, 'diff': sn - qn})

df = pd.DataFrame(rows)
df

# %% [markdown]
# ## Sign test + bootstrap on the paired differences

# %%
diffs = df['diff'].values
n_positive = int((diffs > 0).sum())
n_negative = int((diffs < 0).sum())
n_zero = int((diffs == 0).sum())
n_nonzero = n_positive + n_negative

print(f'Storm-day mean n_within_300km = {df.storm_day_n.mean():.2f}')
print(f'Quiet-day mean n_within_300km = {df.quiet_day_n.mean():.2f}')
print(f'Paired diffs: {diffs.tolist()}')
print(f'Sign test: {n_positive} positive / {n_negative} negative / {n_zero} zero (n={len(diffs)})')

res = binomtest(n_positive, n_nonzero, 0.5, alternative='two-sided')
print(f'Sign test p-value (two-sided): {res.pvalue:.4f}')

rng = np.random.default_rng(0)
boot_means = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(10000)]
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
print(f'Bootstrap 95% CI on mean(storm-quiet): [{ci_low:.2f}, {ci_high:.2f}]')

# %% [markdown]
# **Result**: 6/6 storms show more landmarks near the real storm centre on the storm day than
# on a matched quiet day in the same year/season/region (sign test p=0.031; bootstrap 95% CI on
# the mean difference excludes 0). This is the first result in this project with actual N>1
# statistical support, not a single-case demonstration.

# %% [markdown]
# ## What do quiet-day landmarks target instead?

# %% [markdown]
# ![](legacy/figures/07_skeleton_decomposition/quiet_multiyear_landmarks.png)
#

# %% [markdown]
# Quiet-day landmarks are not scattered randomly either -- they cluster on whatever the
# strongest vorticity feature happens to be that day (extratropical fronts, cut-off lows,
# anticyclones), just not at the storm's coordinates (because there is nothing unusual there on
# a quiet day). This refines rather than weakens the finding: the model appears to generically
# route its representation toward the locally strongest dynamical feature, and tropical cyclones
# are simply the most intense, persistent example of such a feature in this region/season --
# which is why they win the paired comparison so consistently.

# %% [markdown]
# ## Caveats
#
# - N=6 is enough for a sign test but still small; a stronger claim would use more storms/years.
# - All 6 storms + quiet controls share the same region and season (by design, to control
#   confounds) -- generalization to other basins/seasons is untested.
# - Only stage 0 (finest resolution) of Aurora's backbone was checked here; the earlier
#   per-stage comparison (notebook 07) should ideally be repeated across all 6 storms too.
# - Training-data leakage was not checked: whether any of these storms (especially 2024)
#   fell inside Aurora's pretraining/fine-tuning window is unverified -- see README limitations.
