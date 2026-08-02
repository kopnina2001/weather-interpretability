"""Redraw the dose-response matrix from the pickle written by analyse_dose.py -- no npz reads.

    plot_dose_matrix.py <model> [patch_var]

Rows    = alpha in {0, 0.2, 0.4, 0.6, 0.8, 1.0}   (0 = real field, 1 = full climatology)
Columns = the 19 forecast fields

Top panel per lead:     RMSE / sigma_w(anomaly), sequential viridis from 0
Bottom panel per lead:  ACC(alpha) - ACC(alpha=0), diverging RdBu_r on a symmetric scale

Same colour convention as the 19x19 patching matrices (analysis/compute_19var_matrix.py), so
the two figures read the same way.  The bottom panel shows the DELTA rather than the raw ACC:
a diverging palette needs a meaningful zero, and raw ACC sits in 0.75-1.0 with no such point.
Row alpha=0 is therefore identically white, which doubles as a wiring check.  Absolute ACC
stays in the pickle (column `acc`).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'aurora'
PATCH = sys.argv[2] if len(sys.argv) > 2 else 'Z1000'
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
LEADS = [6, 24]

DR_DIR = os.path.expanduser(
    f'~/weather-interpretability/results/dose_response/{PATCH.lower()}/{MODEL}')
FIG_DIR = os.path.expanduser('~/weather-interpretability/figures/dose_response')
os.makedirs(FIG_DIR, exist_ok=True)

NAMES = ['MSLP', 'U10', 'V10', 'T2M']
for s in ('Z', 'Q', 'T', 'U', 'V'):
    NAMES += [f'{s}1000', f'{s}850', f'{s}500']
GROUP_EDGES = [4, 7, 10, 13, 16]
TITLE = {'aurora': 'Aurora', 'pangu': 'Pangu-Weather', 'stormer': 'Stormer'}[MODEL]

df = pd.read_pickle(os.path.join(DR_DIR, f'dose_matrix_{MODEL}.pkl'))
n = df.date.nunique()
print(f'{MODEL} / {PATCH}: {n} дат', flush=True)


def draw(ax, mat, title, cmap, cbar_label, symmetric, fmt='{:.3f}'):
    if symmetric:
        v = np.nanmax(np.abs(mat)); lo = -v
    else:
        v, lo = np.nanmax(mat), 0.0
    im = ax.imshow(mat, cmap=cmap, vmin=lo, vmax=v, aspect='auto')
    ax.set_xticks(range(len(NAMES))); ax.set_xticklabels(NAMES, rotation=90, fontsize=8)
    ax.set_yticks(range(len(ALPHAS)))
    ax.set_yticklabels([f'{PATCH}({a:g})' for a in ALPHAS], fontsize=9)
    for e in GROUP_EDGES:
        ax.axvline(e - 0.5, color='k', lw=0.6, alpha=0.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isfinite(val):
                continue
            rel = (val - lo) / (v - lo) if v > lo else 0.5
            # white text only where the underlying colour is dark
            dark = (rel > 0.78 or rel < 0.22) if symmetric else (rel > 0.65 or rel < 0.15)
            ax.text(j, i, fmt.format(val), ha='center', va='center', fontsize=6,
                    color='white' if dark else 'black')
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.015)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


fig, axes = plt.subplots(4, 1, figsize=(15, 16))
for k, lead in enumerate(LEADS):
    sub = df[df.lead == lead]
    m_r = sub.groupby(['alpha', 'output'])['rmse_anom'].mean().unstack().reindex(
        index=ALPHAS, columns=NAMES).values
    m_d = sub.groupby(['alpha', 'output'])['d_acc'].mean().unstack().reindex(
        index=ALPHAS, columns=NAMES).values
    draw(axes[2 * k], m_r,
         f'RMSE / $\\sigma_w$(аномалии), взвеш. по cos φ  —  {TITLE} +{lead}ч  (n={n} дат)',
         'viridis', 'RMSE в долях СКО аномалии', symmetric=False)
    draw(axes[2 * k + 1], m_d,
         f'$\\Delta$ACC$_w$ относительно α=0, взвеш.  —  {TITLE} +{lead}ч  (n={n} дат)',
         'RdBu_r', 'ΔACC  (красное — хуже)', symmetric=True)
axes[-1].set_xlabel('поле прогноза', fontsize=10)
fig.suptitle(f'{TITLE}: плавное замещение {PATCH} климатологией\n'
             f'строка $\\alpha$: 0 = реальные данные, 1 = полная климатология.  '
             f'RMSE нормирована на СКО аномалии (1.0 = уровень климатологического прогноза)',
             fontsize=13, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.985])
out = f'{FIG_DIR}/dose_matrix_{PATCH}_{MODEL}_n{n}.png'
plt.savefig(out, dpi=150)
print('saved', out, flush=True)

print('\n=== ΔACC при α=1, худшие 6 полей ===')
for lead in LEADS:
    s = (df[(df.lead == lead) & (df.alpha == 1.0)]
         .groupby('output')['d_acc'].mean().sort_values())
    print(f'  +{lead:2d}ч: ' + '  '.join(f'{f}:{v:+.3f}' for f, v in s.head(6).items()))
