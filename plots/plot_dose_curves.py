"""Dose-response curves: RMSE vs blend fraction alpha, one line per weather field.

Reads the metrics table produced by analyse_dose_z1000.py -- no model runs, no npz re-reading.

The fields shown are the TOP-N most affected, ranked by the growth of the normalised RMSE
from alpha=0 to alpha=1.  The patched field itself (Z1000) is excluded from that ranking --
it is trivially first -- and drawn as a dashed reference line instead.

Y axis is RMSE / sigma_w(truth): the raw RMSEs cannot share an axis because the fields carry
different units (kg/kg vs m2/s2 vs K vs m/s).  Raw values are printed in the legend.

One figure per lead time.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'aurora'
PATCH = sys.argv[2] if len(sys.argv) > 2 else 'Z1000'
TOP_N = 5
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
LEADS = [6, 24]

DR_DIR = os.path.expanduser(
    f'~/weather-interpretability/results/dose_response/{PATCH.lower()}/{MODEL}')
FIG_DIR = os.path.expanduser('~/weather-interpretability/figures/dose_response')
os.makedirs(FIG_DIR, exist_ok=True)

UNITS = {'MSLP': 'Па', 'U10': 'м/с', 'V10': 'м/с', 'T2M': 'К'}
for _lv in (1000, 850, 500):
    UNITS[f'Z{_lv}'] = 'м²/с²'; UNITS[f'Q{_lv}'] = 'кг/кг'
    UNITS[f'T{_lv}'] = 'К'; UNITS[f'U{_lv}'] = 'м/с'; UNITS[f'V{_lv}'] = 'м/с'

TITLE = {'aurora': 'Aurora', 'pangu': 'Pangu-Weather', 'stormer': 'Stormer'}[MODEL]
df = pd.read_pickle(os.path.join(DR_DIR, f'dose_matrix_{MODEL}.pkl'))
n = df.date.nunique()
print(f'{n} дат', flush=True)

for lead in LEADS:
    sub = df[df.lead == lead]
    g = sub.groupby(['output', 'alpha'])['rmse_anom'].mean().unstack()
    growth = (g[1.0] - g[0.0]).drop(PATCH).sort_values(ascending=False)
    top = list(growth.index[:TOP_N])
    print(f'\n+{lead}ч, топ-{TOP_N} по приросту RMSE/σ:')
    for f in top:
        print(f'   {f:6s}  {g.loc[f,0.0]:.3f} -> {g.loc[f,1.0]:.3f}   '
              f'(+{growth[f]:.3f}, ×{g.loc[f,1.0]/g.loc[f,0.0]:.2f})')

    raw = sub.groupby(['output', 'alpha'])['rmse_raw'].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 6.2))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, f in enumerate(top):
        y = [g.loc[f, a] for a in ALPHAS]
        ax.plot(ALPHAS, y, 'o-', ms=6, lw=2, color=colors[i],
                label=f'{f}  ({raw.loc[f,0.0]:.3g} → {raw.loc[f,1.0]:.3g} {UNITS[f]})')
    yz = [g.loc[PATCH, a] for a in ALPHAS]
    ax.plot(ALPHAS, yz, 's--', ms=5, lw=1.6, color='0.35',
            label=f'{PATCH} (само подменяемое поле)')

    ax.set_xlabel(r'$\alpha$ — доля климатологии во входном поле ' + PATCH, fontsize=11)
    ax.set_ylabel(r'RMSE / $\sigma_w$(аномалии)', fontsize=11)
    ax.set_title(f'{TITLE} +{lead}ч: рост ошибки при вливании зашумлённых данных\n'
                 f'топ-{TOP_N} наиболее затронутых полей, среднее по {n} датам', fontsize=12)
    ax.set_xticks(ALPHAS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc='upper left', title='поле (сырая RMSE при α=0 → α=1)',
              title_fontsize=8.5)
    plt.tight_layout()
    out = f'{FIG_DIR}/dose_curves_{PATCH}_lead{lead}_{MODEL}_n{n}.png'
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print('saved', out, flush=True)
