import pandas as pd
import numpy as np

df = pd.read_csv('/tmp/ibtracs_wp.csv', skiprows=[1], low_memory=False)
df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
df['LON'] = pd.to_numeric(df['LON'], errors='coerce')

STORMS = {
    'mangkhut_2018': ('2018250N12170', np.datetime64('2018-09-16')),
    'lingling_2019': ('2019243N06136', np.datetime64('2019-09-05')),
    'chanthu_2021': ('2021248N12141', np.datetime64('2021-09-12')),
    'muifa_2022': ('2022247N26147', np.datetime64('2022-09-13')),
    'haikui_2023': ('2023239N18144', np.datetime64('2023-09-05')),
    'bebinca_2024': ('2024253N11148', np.datetime64('2024-09-16')),
}

for name, (sid, valid_time) in STORMS.items():
    sub = df[df.SID == sid].copy()
    if len(sub) == 0:
        # SID guess might be wrong -- try matching by name/season instead
        print(f'{name}: SID {sid} not found, skipping (need manual lookup)')
        continue
    sub['dt'] = (sub.ISO_TIME - pd.Timestamp(valid_time)).abs()
    row = sub.sort_values('dt').iloc[0]
    print(f'{name}: closest track point to {valid_time} -> time={row.ISO_TIME}, lat={row.LAT}, lon={row.LON}, dt={row["dt"]}')
