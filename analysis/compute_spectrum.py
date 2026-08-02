"""Task 1: singular value spectrum / effective rank of Aurora backbone hidden states, comparing
the Bebinca storm case against the quiet control -- reuses the raw activation matrices already
cached by run_skeleton_case.py / run_skeleton_case_quiet.py (no model rerun needed).

Effective rank = smallest k such that the top-k singular values capture >= 90% of the total
squared Frobenius norm ("energy"). A flatter/slower-decaying spectrum -> higher effective rank
-> more genuinely high-dimensional / less compressible information content.
"""
import os
import pickle
import numpy as np

RESULTS_DIR = os.path.expanduser('~/weather-interpretability/results/legacy/skeleton_decomposition')
OUT_PATH = os.path.join(RESULTS_DIR, 'spectrum_comparison.pkl')

CASES = {'storm': 'bebinca_aurora_raw_X.npz', 'quiet': 'quiet_aurora_raw_X.npz'}
ENERGY_THRESHOLDS = [0.80, 0.90, 0.95]

results = {}
for case_label, fname in CASES.items():
    data = np.load(os.path.join(RESULTS_DIR, fname))
    stage_keys = sorted({k.split('_step')[0] for k in data.files}, key=lambda s: int(s.replace('stage', '')))
    case_result = {}
    for stage_key in stage_keys:
        stage_idx = int(stage_key.replace('stage', ''))
        step_results = []
        for step_idx in range(4):
            X = data[f'{stage_key}_step{step_idx}'].astype(np.float64)
            # economy SVD, singular values only (fast, no need for U/V)
            s = np.linalg.svd(X, compute_uv=False)
            energy = s ** 2
            cum_energy = np.cumsum(energy) / np.sum(energy)
            eff_ranks = {}
            for thresh in ENERGY_THRESHOLDS:
                k = int(np.searchsorted(cum_energy, thresh) + 1)
                eff_ranks[thresh] = k
            step_results.append({
                'lead_hours': (step_idx + 1) * 6,
                'singular_values': s,
                'cum_energy': cum_energy,
                'eff_ranks': eff_ranks,
                'total_dim': min(X.shape),
            })
            print(f'{case_label} {stage_key} step+{(step_idx+1)*6}h: '
                  f'shape={X.shape} eff_rank(90%)={eff_ranks[0.90]}/{min(X.shape)}', flush=True)
        case_result[stage_idx] = step_results
    results[case_label] = case_result

with open(OUT_PATH, 'wb') as f:
    pickle.dump(results, f)
print(f'Saved to {OUT_PATH}', flush=True)
