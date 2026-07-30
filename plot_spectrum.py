import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = '/home/irina/weather-interpretability/results/skeleton_decomposition'
FIG_DIR = '/home/irina/weather-interpretability/figures/09_tensor_decomposition'
import os
os.makedirs(FIG_DIR, exist_ok=True)

with open(f'{RESULTS_DIR}/spectrum_comparison.pkl', 'rb') as f:
    results = pickle.load(f)

STAGE_LABELS = {0: 'Stage 0 (finest)', 1: 'Stage 1', 2: 'Stage 2 (bottleneck)'}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, stage_idx in zip(axes, [0, 1, 2]):
    for case_label, color in [('storm', 'tab:red'), ('quiet', 'tab:blue')]:
        step = [d for d in results[case_label][stage_idx] if d['lead_hours'] == 24][0]
        ax.plot(np.arange(1, len(step['cum_energy']) + 1), step['cum_energy'],
                label=case_label, color=color)
    ax.axhline(0.9, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('rank k')
    ax.set_ylabel('cumulative energy fraction')
    ax.set_title(f'{STAGE_LABELS[stage_idx]} @ +24h')
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/spectrum_storm_vs_quiet.png', dpi=150, bbox_inches='tight')
print('saved')

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(3)
width = 0.35
for i, case_label in enumerate(['storm', 'quiet']):
    vals = [results[case_label][s][3]['eff_ranks'][0.90] for s in [0, 1, 2]]
    dims = [results[case_label][s][3]['total_dim'] for s in [0, 1, 2]]
    frac = [v / d for v, d in zip(vals, dims)]
    ax.bar(x + (i - 0.5) * width, frac, width, label=case_label)
ax.set_xticks(x)
ax.set_xticklabels([STAGE_LABELS[s] for s in [0, 1, 2]])
ax.set_ylabel('effective rank (90% energy) / total dim')
ax.set_title('Effective rank fraction @ +24h -- storm vs quiet')
ax.legend()
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/effective_rank_fraction.png', dpi=150, bbox_inches='tight')
print('saved2')
