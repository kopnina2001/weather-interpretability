"""Pick the median date using the cos(phi)-WEIGHTED baseline ACC of Z1000, and plot the
sorted curve. Reads the metrics pickle (acc_base_w is stored there now), so no npz re-reading.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'aurora'
RES_DIR = os.path.expanduser('~/weather-interpretability/results/patching_19var'
                             + ('' if MODEL == 'aurora' else f'_{MODEL}'))
FIG_DIR = os.path.expanduser('~/weather-interpretability/figures/08_input_patching')
FIELD, TARGETS, LEADS = 'Z1000', ['Q1000', 'T1000', 'U1000', 'V1000'], [6, 24]

df = pd.read_pickle(os.path.join(RES_DIR, f'metrics_19var_{MODEL}.pkl'))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
chosen = {}
for ax, lead in zip(axes, LEADS):
    # baseline ACC is identical across 'patched', so take it per (date) for output=Z1000
    a = (df[(df.lead == lead) & (df.output == FIELD)]
         .groupby('date')['acc_base_w'].first().rename('acc'))
    # damage from the Z1000 patch, averaged over the four fields the bias maps will show
    dmg = (df[(df.lead == lead) & (df.patched == FIELD) & (df.output.isin(TARGETS))]
           .groupby('date')['dacc_w'].mean().rename('dacc'))
    t = pd.concat([a, dmg], axis=1).sort_values('acc').reset_index()
    n, mid = len(t), len(t) // 2
    chosen[lead] = t.loc[mid]

    ax.plot(range(n), t.acc, 'o-', ms=4, lw=1, color='tab:blue')
    ax.axvline(mid, color='tab:red', ls='--', lw=1.2)
    ax.plot(mid, t.loc[mid, 'acc'], 'o', ms=11, mfc='none', mec='tab:red', mew=2)
    ax.annotate(f"медиана: {t.loc[mid,'date']}\nACC$_w$ = {t.loc[mid,'acc']:.4f}\n"
                f"$\\Delta$ACC$_w$ = {t.loc[mid,'dacc']:+.4f}",
                xy=(mid, t.loc[mid, 'acc']), xytext=(0.04, 0.80),
                textcoords='axes fraction', color='tab:red', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='tab:red', lw=1))
    ax.set_xlabel('дата (отсортировано по возрастанию ACC$_w$)')
    ax.set_ylabel(f'базовый ACC$_w$ поля {FIELD} (взвеш. по cos φ)')
    ax.set_title(f'{MODEL.capitalize()} +{lead}ч — качество базового прогноза {FIELD}, n={n}')
    ax.grid(alpha=0.3)
    ax.text(0.04, 0.06, f'min {t.acc.iloc[0]:.4f}   max {t.acc.iloc[-1]:.4f}',
            transform=ax.transAxes, fontsize=9, color='0.35')

plt.tight_layout()
out = f'{FIG_DIR}/baseline_accw_sorted_{FIELD}_{MODEL}.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print('saved', out)
for lead in LEADS:
    c = chosen[lead]
    print(f'+{lead}ч -> {c.date}   ACC_w={c.acc:.4f}   dACC_w={c.dacc:+.4f}')
