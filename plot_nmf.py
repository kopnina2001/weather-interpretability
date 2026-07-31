import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = '/home/irina/weather-interpretability/results/tensor_decomposition'
FIG_DIR = '/home/irina/weather-interpretability/figures/09_tensor_decomposition'
import os
os.makedirs(FIG_DIR, exist_ok=True)

with open(f'{OUT_DIR}/nmf_results.pkl', 'rb') as f:
    results = pickle.load(f)

WS = (2, 6, 12)  # window shape, 144 = 2*6*12 tokens

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for row, date_label in enumerate(['storm', 'quiet']):
    for col, stage_idx in enumerate([0, 1]):
        ax = axes[row, col]
        r = results[(date_label, stage_idx)]
        H = r['H']  # (K, 144)
        K = H.shape[0]
        # reshape each basis pattern (144,) -> average over the tiny c-dim -> (6,12) spatial-in-window
        H_grid = H.reshape(K, WS[0], WS[1], WS[2]).mean(axis=1)  # (K, 6, 12)
        vmax = H_grid.max()
        for k in range(K):
            sub_ax_x = k % 3
            sub_ax_y = k // 3
        # simple: tile K patterns side by side within this subplot using imshow grid
        tiled = np.concatenate([H_grid[k] for k in range(K)], axis=1)
        im = ax.imshow(tiled, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
        for k in range(K):
            ax.axvline(k * WS[2] - 0.5, color='white', linewidth=0.5)
        ax.set_title(f'{date_label}, stage{stage_idx} -- {K} NMF basis attention patterns '
                     f'(recon_err={r["recon_err"]:.2f})')
        ax.set_yticks([])
        ax.set_xticks([])
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/nmf_basis_patterns.png', dpi=150, bbox_inches='tight')
print('saved')

# Also: which heads load most on which component (W matrix), per stage/date
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))
for row, date_label in enumerate(['storm', 'quiet']):
    for col, stage_idx in enumerate([0, 1]):
        ax = axes2[row, col]
        r = results[(date_label, stage_idx)]
        W = r['W']  # (heads*144, K)
        n_heads = r['n_heads']
        W_per_head = W.reshape(n_heads, 144, -1).mean(axis=1)  # (heads, K) avg loading per head
        im = ax.imshow(W_per_head, cmap='magma', aspect='auto')
        ax.set_xlabel('NMF component')
        ax.set_ylabel('head')
        ax.set_title(f'{date_label}, stage{stage_idx}: mean component loading per head')
        plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/nmf_head_loadings.png', dpi=150, bbox_inches='tight')
print('saved2')
