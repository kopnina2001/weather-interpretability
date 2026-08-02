"""Pull the geopotential rows out of the 19x19 delta-ACC matrix and redraw them alone.

    plot_acc_strip.py [aurora|pangu|both]

The full matrix (analysis/compute_19var_matrix.py) sets its colour scale from the strongest
row, which is MSLP; the Z1000/Z850/Z500 rows end up nearly white and unreadable.  This script
extracts those three rows and rescales the colour to them, so the figure can go into the text
on its own.

Same convention as the parent figure: rows = the patched input field, columns = the forecast
field, values = ACC_w(patched) - ACC_w(baseline), diverging RdBu_r on a symmetric scale.

The scale is set from the OFF-DIAGONAL cells only.  The self-effect cells (patch Z1000, look
at Z1000) are an order of magnitude larger and would flatten everything else -- the whole
point of the extraction.  They are allowed to saturate and get a black outline to mark that
their colour is clipped; the printed number is still exact.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

WHICH = sys.argv[1] if len(sys.argv) > 1 else 'both'
ROWS = ['Z1000', 'Z850', 'Z500']
LEADS = [6, 24]

RES = os.path.expanduser('~/weather-interpretability/results/patching')
FIG_DIR = os.path.expanduser('~/weather-interpretability/figures/patching_matrices')
PKL = {'aurora': f'{RES}/patching_19var/metrics_19var_aurora.pkl',
       'pangu': f'{RES}/patching_19var_pangu/metrics_19var_pangu.pkl'}
TITLE = {'aurora': 'Aurora', 'pangu': 'Pangu-Weather'}

NAMES = ['MSLP', 'U10', 'V10', 'T2M']
for s in ('Z', 'Q', 'T', 'U', 'V'):
    NAMES += [f'{s}1000', f'{s}850', f'{s}500']
GROUP_EDGES = [4, 7, 10, 13, 16]


def matrix(df, lead):
    return (df[df.lead == lead].groupby(['patched', 'output'])['dacc_w'].mean()
            .unstack().reindex(index=ROWS, columns=NAMES).values)


def draw(ax, mat, title, v):
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
    ax.set_xticks(range(len(NAMES))); ax.set_xticklabels(NAMES, rotation=90, fontsize=9)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([f'подменён {r}' for r in ROWS], fontsize=10)
    for e in GROUP_EDGES:
        ax.axvline(e - 0.5, color='k', lw=0.7, alpha=0.5)
    for i, r in enumerate(ROWS):
        for j, c in enumerate(NAMES):
            val = mat[i, j]
            if not np.isfinite(val):
                continue
            clipped = abs(val) > v
            rel = np.clip((val + v) / (2 * v), 0, 1)
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7.5,
                    color='white' if (rel > 0.80 or rel < 0.20) else 'black')
            if clipped:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       edgecolor='k', lw=1.6))
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.012)
    cb.set_label('ΔACC$_w$  (синее — хуже)', fontsize=9)
    cb.ax.tick_params(labelsize=8)


def scale(mats):
    """symmetric limit from the off-diagonal cells of all panels that share the colour bar"""
    vals = []
    for m in mats:
        for i, r in enumerate(ROWS):
            for j, c in enumerate(NAMES):
                if c != r and np.isfinite(m[i, j]):
                    vals.append(abs(m[i, j]))
    return float(np.max(vals))


data = {m: pd.read_pickle(p) for m, p in PKL.items()
        if WHICH in ('both', m) and os.path.exists(p)}
n = {m: df.date.nunique() if 'date' in df else df['init'].nunique() for m, df in data.items()}
print({m: f'{k} дат' for m, k in n.items()}, flush=True)

# one figure per model, both leads stacked
for m, df in data.items():
    mats = [matrix(df, L) for L in LEADS]
    v = scale(mats)
    fig, axes = plt.subplots(2, 1, figsize=(14, 5.4))
    for ax, L, mat in zip(axes, LEADS, mats):
        draw(ax, mat, f'{TITLE[m]}  +{L}ч   (n={n[m]} дат)', v)
    axes[-1].set_xlabel('поле прогноза', fontsize=10)
    fig.suptitle(f'{TITLE[m]}: падение ACC при подмене геопотенциала на климатологию\n'
                 f'шкала задана по недиагональным ячейкам; ячейки в рамке выходят за неё',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = f'{FIG_DIR}/acc_strip_Z_{m}_n{n[m]}.png'
    plt.savefig(out, dpi=160)
    plt.close(fig)
    print('saved', out, flush=True)

# combined: both models, both leads, one shared scale
if len(data) == 2:
    panels = [(m, L, matrix(data[m], L)) for m in ('aurora', 'pangu') for L in LEADS]
    v = scale([p[2] for p in panels])
    fig, axes = plt.subplots(4, 1, figsize=(14, 10.2))
    for ax, (m, L, mat) in zip(axes, panels):
        draw(ax, mat, f'{TITLE[m]}  +{L}ч   (n={n[m]} дат)', v)
    axes[-1].set_xlabel('поле прогноза', fontsize=10)
    fig.suptitle('Падение ACC при подмене геопотенциала на климатологию: Aurora против '
                 'Pangu-Weather\nобщая шкала у всех четырёх панелей, задана по '
                 'недиагональным ячейкам', fontsize=12.5)
    plt.tight_layout(rect=[0, 0, 1, 0.945])
    out = f'{FIG_DIR}/acc_strip_Z_both_n48.png'
    plt.savefig(out, dpi=160)
    print('saved', out, flush=True)

for m, df in data.items():
    print(f'\n=== {TITLE[m]}: сильнейшие недиагональные эффекты ===')
    for L in LEADS:
        mat = matrix(df, L)
        cells = [(mat[i, j], ROWS[i], NAMES[j]) for i in range(3) for j in range(len(NAMES))
                 if NAMES[j] != ROWS[i] and np.isfinite(mat[i, j])]
        cells.sort()
        print(f'  +{L:2d}ч: ' + '  '.join(f'{r}→{c}:{v:+.3f}' for v, r, c in cells[:5]))
