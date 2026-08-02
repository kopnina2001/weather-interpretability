"""Compare two ways of picking the "typical" date for the bias maps:
  (a) by baseline ACC of Z1000   -- typical FORECAST QUALITY
  (b) by dACC caused by the Z1000 patch -- typical DAMAGE from the patch
and show how different the resulting dates are.
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
TARGETS = ['Q1000', 'T1000', 'U1000', 'V1000']   # fields the bias maps will show
LEADS = [6, 24]

dacc = pd.read_pickle(os.path.join(RES_DIR, f'metrics_19var_{MODEL}.pkl'))
accb = pd.read_pickle(os.path.join(RES_DIR, f'baseline_acc_Z1000_{MODEL}.pkl'))

fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))
summary = {}
for ax, lead in zip(axes, LEADS):
    d = dacc[(dacc.lead == lead) & (dacc.patched == 'Z1000')]
    # damage the Z1000 patch does to the four fields we will map
    mean_dmg = (d[d.output.isin(TARGETS)].groupby('date')['dacc'].mean()
                .rename('dacc_mean_targets'))
    self_dmg = (d[d.output == 'Z1000'].set_index('date')['dacc']
                .rename('dacc_Z1000_self'))
    a = accb[accb.lead == lead].set_index('date')['acc_baseline']
    t = pd.concat([mean_dmg, self_dmg, a], axis=1).sort_values('dacc_mean_targets')
    t = t.reset_index().rename(columns={'index': 'date'})
    n = len(t)
    mid = n // 2

    ax.plot(range(n), t.dacc_mean_targets, 'o-', ms=4, lw=1,
            color='tab:purple', label=f'$\\Delta$ACC, среднее по {", ".join(TARGETS)}')
    ax.plot(range(n), t.dacc_Z1000_self, 's-', ms=3, lw=0.8, alpha=0.55,
            color='tab:orange', label='$\\Delta$ACC на самом Z1000')
    ax.axvline(mid, color='tab:red', ls='--', lw=1.2)
    ax.plot(mid, t.loc[mid, 'dacc_mean_targets'], 'o', ms=11, mfc='none',
            mec='tab:red', mew=2)

    # where does the ACC-chosen date sit in THIS ordering?
    acc_sorted = accb[accb.lead == lead].sort_values('acc_baseline').reset_index(drop=True)
    acc_date = acc_sorted.loc[len(acc_sorted) // 2, 'date']
    pos_of_acc_date = int(t.index[t.date == acc_date][0])
    ax.plot(pos_of_acc_date, t.loc[pos_of_acc_date, 'dacc_mean_targets'], '*',
            ms=17, color='tab:green', zorder=5)

    ax.annotate(f'медиана по $\\Delta$ACC: {t.loc[mid,"date"]}',
                xy=(mid, t.loc[mid, 'dacc_mean_targets']), xytext=(0.04, 0.30),
                textcoords='axes fraction', color='tab:red', fontsize=9.5,
                arrowprops=dict(arrowstyle='->', color='tab:red', lw=1))
    ax.annotate(f'дата, выбранная по ACC: {acc_date}\n(позиция {pos_of_acc_date} из {n})',
                xy=(pos_of_acc_date, t.loc[pos_of_acc_date, 'dacc_mean_targets']),
                xytext=(0.04, 0.10), textcoords='axes fraction', color='green', fontsize=9.5,
                arrowprops=dict(arrowstyle='->', color='green', lw=1))
    ax.set_xlabel('дата (отсортировано по возрастанию $\\Delta$ACC = от худшего ущерба к слабейшему)')
    ax.set_ylabel('$\\Delta$ACC от подмены Z1000')
    ax.set_title(f'{MODEL.capitalize()} +{lead}ч — ущерб от подмены Z1000, n={n} дат')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')

    summary[lead] = dict(dacc_median_date=t.loc[mid, 'date'],
                         dacc_median_val=t.loc[mid, 'dacc_mean_targets'],
                         acc_date=acc_date, acc_date_pos=pos_of_acc_date,
                         acc_date_dacc=t.loc[pos_of_acc_date, 'dacc_mean_targets'],
                         spread=(t.dacc_mean_targets.min(), t.dacc_mean_targets.max()))

plt.tight_layout()
out = f'{FIG_DIR}/date_selection_dacc_vs_acc_{MODEL}.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print('saved', out)
for lead, s in summary.items():
    print(f'\n+{lead}ч:')
    print(f'  по dACC медиана : {s["dacc_median_date"]}  (dACC={s["dacc_median_val"]:+.4f})')
    print(f'  по ACC  выбрана : {s["acc_date"]}  -> в порядке dACC стоит на позиции '
          f'{s["acc_date_pos"]}/48, dACC={s["acc_date_dacc"]:+.4f}')
    print(f'  размах dACC     : {s["spread"][0]:+.4f} .. {s["spread"][1]:+.4f}')
