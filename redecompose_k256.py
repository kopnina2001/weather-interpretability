"""Rebuild the skeleton decomposition results with a chosen k=256 (found via rank-vs-error
sweep: stage0 crosses rel_error<0.15 around k~192-256; deeper/bottleneck stages don't reach
that threshold even at k=512, but 256 is a reasonable shared budget) -- reusing the raw hidden
state matrices cached by run_skeleton_case.py, no GPU/model rerun needed.
"""
import os
import pickle
import numpy as np
import scipy.linalg.interpolative as sli

RAW_X_PATH = os.path.expanduser('~/weather-interpretability/results/skeleton_decomposition/bebinca_aurora_raw_X.npz')
OLD_PKL_PATH = os.path.expanduser('~/weather-interpretability/results/skeleton_decomposition/bebinca_aurora.pkl')
OUT_PATH = OLD_PKL_PATH
K_LANDMARKS = 256

with open(OLD_PKL_PATH, 'rb') as f:
    old_results = pickle.load(f)

stage_res = old_results['stage_res']
raw_X = np.load(RAW_X_PATH)


def skeleton_decompose(X, k):
    X64 = X.astype(np.float64)
    idx_r, proj_r = sli.interp_decomp(X64.T, k)
    row_landmarks = idx_r[:k]
    P_row = sli.reconstruct_interp_matrix(idx_r, proj_r)
    Xhat_row = (P_row.T @ X64[row_landmarks, :])
    row_rel_error = float(np.linalg.norm(Xhat_row - X64) / np.linalg.norm(X64))

    idx_c, proj_c = sli.interp_decomp(X64, k)
    col_landmarks = idx_c[:k]
    P_col = sli.reconstruct_interp_matrix(idx_c, proj_c)
    Xhat_col = X64[:, col_landmarks] @ P_col
    col_rel_error = float(np.linalg.norm(Xhat_col - X64) / np.linalg.norm(X64))

    residual_row = (X64 - Xhat_row).astype(np.float32)
    return {
        'row_landmarks': row_landmarks,
        'col_landmarks': col_landmarks,
        'row_reconstruction': Xhat_row.astype(np.float32),
        'row_residual': residual_row,
        'row_rel_error': row_rel_error,
        'col_rel_error': col_rel_error,
    }


results = {'k': K_LANDMARKS, 'stage_res': stage_res, 'stages': {}}
for stage_key in sorted({k.split('_step')[0] for k in raw_X.files}, key=lambda s: int(s.replace('stage', ''))):
    stage_idx = int(stage_key.replace('stage', ''))
    old_stage_out = old_results['stages'][stage_idx]
    stage_out = []
    for step_idx, old_dec in enumerate(old_stage_out):
        X = raw_X[f'{stage_key}_step{step_idx}']
        dec = skeleton_decompose(X, K_LANDMARKS)
        dec['lead_hours'] = old_dec['lead_hours']
        dec['H'], dec['W'] = old_dec['H'], old_dec['W']
        stage_out.append(dec)
        print(f'  stage {stage_idx} step +{dec["lead_hours"]}h: '
              f'row_rel_error={dec["row_rel_error"]:.4f} col_rel_error={dec["col_rel_error"]:.4f}', flush=True)
    results['stages'][stage_idx] = stage_out

with open(OUT_PATH, 'wb') as f:
    pickle.dump(results, f)
print(f'Saved (k={K_LANDMARKS}) to {OUT_PATH}', flush=True)
