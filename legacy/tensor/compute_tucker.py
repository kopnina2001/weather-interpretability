"""Task 3: Tucker decomposition of the (heads, query, key) attention tensor -- how many
head-mode components are needed to reconstruct all heads' attention patterns to a given
fidelity? Low required rank = heads are redundant combinations of a few shared attention
"basis" patterns; high required rank = heads are largely independent/specialized."""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import tucker

OUT_DIR = os.path.expanduser('~/weather-interpretability/results/legacy/tensor_decomposition')
FIG_DIR = os.path.expanduser('~/weather-interpretability/legacy/figures/09_tensor_decomposition')
os.makedirs(FIG_DIR, exist_ok=True)
STAGES = [0, 1]
STEP_IDX = 3  # +24h

results = {}
for date_label in ['storm', 'quiet']:
    data = np.load(os.path.join(OUT_DIR, f'{date_label}_attn_tensor.npz'))
    for stage_idx in STAGES:
        T = data[f'stage{stage_idx}_step{STEP_IDX}'].astype(np.float64)  # (heads, 144, 144)
        n_heads = T.shape[0]
        rel_errors = []
        head_ranks = list(range(1, n_heads + 1))
        for r in head_ranks:
            core, factors = tucker(tl.tensor(T), rank=[r, T.shape[1], T.shape[2]])
            recon = tl.tucker_to_tensor((core, factors))
            err = float(np.linalg.norm(recon - T) / np.linalg.norm(T))
            rel_errors.append(err)
        results[(date_label, stage_idx)] = {'head_ranks': head_ranks, 'rel_errors': rel_errors}
        eff_r = next((r for r, e in zip(head_ranks, rel_errors) if e < 0.10), None)
        print(f'{date_label} stage{stage_idx}: n_heads={n_heads}, '
              f'head-rank for <10% error = {eff_r}', flush=True)

with open(os.path.join(OUT_DIR, 'tucker_results.pkl'), 'wb') as f:
    pickle.dump(results, f)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, stage_idx in zip(axes, STAGES):
    for date_label, color in [('storm', 'tab:red'), ('quiet', 'tab:blue')]:
        r = results[(date_label, stage_idx)]
        ax.plot(r['head_ranks'], r['rel_errors'], marker='o', label=date_label, color=color)
    ax.axhline(0.10, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Tucker head-mode rank')
    ax.set_ylabel('relative reconstruction error')
    ax.set_title(f'Stage {stage_idx}: heads needed to reconstruct attention tensor')
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/tucker_head_rank.png', dpi=150, bbox_inches='tight')
print('saved figure')
