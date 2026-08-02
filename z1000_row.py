import os
import numpy as np
import pandas as pd

RES = os.path.expanduser('~/weather-interpretability/results/patching_19var/metrics_19var_aurora.pkl')
df = pd.read_pickle(RES)
TG = ['Q1000', 'T1000', 'U1000', 'V1000', 'Z850', 'Z500', 'MSLP', 'U10', 'V10', 'T2M']

for lead in (6, 24):
    d = df[(df.lead == lead) & (df.patched == 'Z1000') & (df.output.isin(TG))]
    g = d.groupby('output')[['rel_sens_w', 'dacc_w']].mean().reindex(TG)
    print(f'\n=== подмена Z1000, лид +{lead}ч (взвешенные, среднее по 48 датам) ===')
    print(f'{"поле":8s} {"rel.sens":>9s} {"dACC":>9s}')
    for k, r in g.iterrows():
        print(f'{k:8s} {r.rel_sens_w:9.3f} {r.dacc_w:+9.4f}')

# how much of the Z1000 row's total effect lands on each field (share of the row)
d24 = df[(df.lead == 24) & (df.patched == 'Z1000')].groupby('output')['rel_sens_w'].mean()
print(f'\nсумма строки Z1000 (+24ч, без диагонали): '
      f'{d24.drop("Z1000").sum():.2f}; диагональ Z1000->Z1000 = {d24["Z1000"]:.3f}')
