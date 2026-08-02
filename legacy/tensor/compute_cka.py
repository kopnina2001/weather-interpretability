"""Task 2: linear CKA (Centered Kernel Alignment, Kornblith et al. 2019) between Aurora
attention heads, per stage/date, at +24h. Tests whether different heads compute redundant
representations of the same 144 window tokens."""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.expanduser('~/weather-interpretability/results/legacy/tensor_decomposition')
FIG_DIR = os.path.expanduser('~/weather-interpretability/legacy/figures/09_tensor_decomposition')
os.makedirs(FIG_DIR, exist_ok=True)
STAGES = [0, 1]
STEP_IDX = 3  # +24h


def linear_cka(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(Y.T @ X, 'fro') ** 2
    var1 = np.linalg.norm(X.T @ X, 'fro')
    var2 = np.linalg.norm(Y.T @ Y, 'fro')
    denom = var1 * var2
    return float(hsic / denom) if denom > 1e-12 else float('nan')


results = {}
for date_label in ['storm', 'quiet']:
    data = np.load(os.path.join(OUT_DIR, f'{date_label}_head_outputs.npz'))
    for stage_idx in STAGES:
        arr = data[f'stage{stage_idx}_step{STEP_IDX}']  # (heads, 144, head_dim)
        n_heads = arr.shape[0]
        cka_mat = np.zeros((n_heads, n_heads))
        for i in range(n_heads):
            for j in range(n_heads):
                cka_mat[i, j] = linear_cka(arr[i], arr[j])
        results[(date_label, stage_idx)] = cka_mat
        print(f'{date_label} stage{stage_idx}: {n_heads} heads, '
              f'mean off-diag CKA={ (cka_mat.sum()-np.trace(cka_mat))/(n_heads**2-n_heads):.3f}', flush=True)

with open(os.path.join(OUT_DIR, 'cka_matrices.pkl'), 'wb') as f:
    pickle.dump(results, f)

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for row, date_label in enumerate(['storm', 'quiet']):
    for col, stage_idx in enumerate(STAGES):
        ax = axes[row, col]
        mat = results[(date_label, stage_idx)]
        im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'{date_label}, stage {stage_idx} ({mat.shape[0]} heads)')
        ax.set_xlabel('head')
        ax.set_ylabel('head')
        plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/cka_heads_matrix.png', dpi=150, bbox_inches='tight')
print('saved figure')
