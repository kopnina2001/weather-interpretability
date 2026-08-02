"""Task 3 (RepE-style): PCA on the storm-minus-quiet hidden-state DIFFERENCE, following
Representation Engineering (Zou et al. 2023) -- instead of decomposing one run's activations,
decompose the *difference* between a real-storm run and a climatologically "blank" quiet run
to find "storm concept directions" in channel space, and see where in space those directions
concentrate. Reuses the raw hidden-state matrices already cached by run_skeleton_case.py
(bebinca_aurora_raw_X.npz / quiet_aurora_raw_X.npz) -- no model rerun needed.
"""
import os
import pickle
import numpy as np

RESULTS_DIR = os.path.expanduser('~/weather-interpretability/results/legacy/skeleton_decomposition')
OUT_PATH = os.path.expanduser('~/weather-interpretability/results/legacy/tensor_decomposition/repe_pca.pkl')
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

K = 5  # top concept directions to keep
STAGES = [0, 1, 2]

storm_data = np.load(os.path.join(RESULTS_DIR, 'bebinca_aurora_raw_X.npz'))
quiet_data = np.load(os.path.join(RESULTS_DIR, 'quiet_aurora_raw_X.npz'))

results = {}
for stage_idx in STAGES:
    step_results = []
    for step_idx in range(4):
        key = f'stage{stage_idx}_step{step_idx}'
        X_storm = storm_data[key].astype(np.float64)
        X_quiet = quiet_data[key].astype(np.float64)
        diff = X_storm - X_quiet  # (N_spatial, D)
        diff_c = diff - diff.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(diff_c, full_matrices=False)
        scores = U[:, :K] * S[:K]  # (N_spatial, K) -- spatial map of alignment with each direction
        explained_var = (S[:K] ** 2) / np.sum(S ** 2)
        step_results.append({
            'lead_hours': (step_idx + 1) * 6,
            'scores': scores.astype(np.float32),
            'explained_var': explained_var,
            'components': Vt[:K].astype(np.float32),  # (K, D) concept directions in channel space
        })
        print(f'stage{stage_idx} step+{(step_idx+1)*6}h: top-{K} explained var = '
              f'{explained_var.round(3)}', flush=True)
    results[stage_idx] = step_results

with open(OUT_PATH, 'wb') as f:
    pickle.dump(results, f)
print(f'Saved to {OUT_PATH}', flush=True)
