"""Dose-response, all 19 fields at once: one panel per variable letter, markers only plus a
least-squares line through the six alpha points.

Panels:  Z | Q | T | U | V  (levels 1000/850/500 together)  and one for the surface fields.
Within a panel the levels share a colour code (1000 blue, 850 orange, 500 green) so the
panels can be read against each other.

The points are NOT joined -- the dashed line is the fit  y = k*alpha + b  and k is printed in
the legend, so the legend doubles as a table of sensitivities.

Y axis is RMSE / sigma_w(truth): inside the surface panel the fields carry different units
(Pa, m/s, K) and inside Z/Q/T the magnitudes differ several-fold between levels.

One figure per lead time.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL, PATCH = 'aurora', 'Z1000'
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
LEADS = [6, 24]

DR_DIR = os.path.expanduser(f'~/weather-interpretability/results/dose_response/dose_response_z1000_{MODEL}')
FIG_DIR = os.path.expanduser('~/weather-interpretability/figures/dose_response')
os.makedirs(FIG_DIR, exist_ok=True)

# panel title -> list of (field, colour, marker)
LV_COLOR = {1000: 'tab:blue', 850: 'tab:orange', 500: 'tab:green'}
LV_MARK = {1000: 'o', 850: 's', 500: '^'}
PANELS = []
for s, lbl in (('Z', 'Геопотенциал Z'), ('Q', 'Уд. влажность Q'), ('T', 'Температура T'),
               ('U', 'Зональный ветер U'), ('V', 'Меридиональный ветер V')):
    PANELS.append((lbl, [(f'{s}{lv}', LV_COLOR[lv], LV_MARK[lv]) for lv in (1000, 850, 500)]))
PANELS.append(('Приземные поля', [('MSLP', 'tab:red', 'o'), ('U10', 'tab:blue', 's'),
                                  ('V10', 'tab:orange', '^'), ('T2M', 'tab:green', 'D')]))

df = pd.read_pickle(os.path.join(DR_DIR, f'dose_matrix_{MODEL}.pkl'))
n = df.date.nunique()
print(f'{n} дат', flush=True)

A = np.array(ALPHAS)
for lead in LEADS:
    g = df[df.lead == lead].groupby(['output', 'alpha'])['rmse_norm'].mean().unstack()
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9))
    print(f'\n+{lead}ч, наклон k прямой RMSE/σ по α:')
    for ax, (title, items) in zip(axes.ravel(), PANELS):
        for f, col, mk in items:
            y = np.array([g.loc[f, a] for a in ALPHAS])
            k, b = np.polyfit(A, y, 1)
            ax.plot(A, y, mk, ms=7, color=col, mfc='none', mew=1.6, linestyle='none')
            ax.plot(A, k * A + b, '--', lw=1.3, color=col, alpha=0.85)
            tag = ' (подменяемое)' if f == PATCH else ''
            ax.plot([], [], mk, ms=6, color=col, mfc='none', mew=1.6, linestyle='none',
                    label=f'{f}{tag}:  k = {k:.3f}')
            print(f'   {f:6s} k = {k:+.4f}')
        ax.set_title(title, fontsize=11)
        ax.set_xticks(ALPHAS)
        ax.set_xlim(-0.06, 1.06)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
    for ax in axes[1]:
        ax.set_xlabel(r'$\alpha$ — доля климатологии во входном ' + PATCH, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'RMSE / $\sigma_w$(правда)', fontsize=10)
    fig.suptitle(f'Aurora +{lead}ч: все 19 полей прогноза при плавной подмене {PATCH} '
                 f'климатологией  (среднее по {n} датам)\n'
                 f'маркеры — измеренные точки;  пунктир — МНК-прямая, k — её наклон '
                 f'(чувствительность поля к порче {PATCH})', fontsize=12.5)
    plt.tight_layout(rect=[0, 0, 1, 0.935])
    out = f'{FIG_DIR}/dose_panels_{PATCH}_lead{lead}_{MODEL}_n{n}.png'
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print('saved', out, flush=True)
