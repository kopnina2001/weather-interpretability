import pandas as pd
import numpy as np

df = pd.read_csv('/tmp/ibtracs_wp.csv', skiprows=[1], low_memory=False)
df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
df['LON'] = pd.to_numeric(df['LON'], errors='coerce')

# wider region to be safe (any storm even a bit outside our crop could still matter)
WIDE = dict(lat_min=15, lat_max=38, lon_min=105, lon_max=128)

YEARS = [2018, 2019, 2021, 2022, 2023]

for year in YEARS:
    sub = df[(df.SEASON == year) & (df.ISO_TIME.dt.month == 9) &
             (df.LAT >= WIDE['lat_min']) & (df.LAT <= WIDE['lat_max']) &
             (df.LON >= WIDE['lon_min']) & (df.LON <= WIDE['lon_max'])]
    busy_days = sorted(sub.ISO_TIME.dt.day.dropna().unique().astype(int).tolist())
    all_days = set(range(1, 31))
    quiet_days = sorted(all_days - set(busy_days))
    # find a quiet day with quiet day before/after too (need init-1day and +2days after for rollout)
    candidates = [d for d in quiet_days if (d - 1) in quiet_days and (d + 1) in quiet_days
                  and (d + 2) in quiet_days and (d - 2) in quiet_days]
    print(f'{year}: busy days = {busy_days}')
    print(f'{year}: quiet 5-day-window candidates (day +-2 all quiet) = {candidates}')
