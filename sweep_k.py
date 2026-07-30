"""Rank-vs-error sweep for the skeleton (interpolative) decomposition: loads the cached raw
hidden-state matrices (bebinca_aurora_raw_X.npz, produced by run_skeleton_case.py) and, for a
range of k values, computes the row-skeleton and column-skeleton reconstruction relative error
at each backbone stage. Finds the smallest k where error drops below a target threshold
(knee-finding), per stage. No GPU / model rerun needed.
"""
import os
import numpy as np
import scipy.linalg.interpolative as sli

RAW_X_PATH = os.path.expanduser('~/weather-interpretability/results/skeleton_decomposition/bebinca_aurora_raw_X.npz')
K_VALUES = [8, 16, 24, 40, 64, 96, 128, 192, 256, 384, 512]
TARGET_ERROR = 0.15
STEP_IDX_FOR_SWEEP = 3  # +24h (index 3 of 4 rollout steps)


def rel_error_for_k(X64, k, transpose):
    A = X64.T if transpose else X64
    idx, proj = sli.interp_decomp(A, k)
    P = sli.reconstruct_interp_matrix(idx, proj)
    if transpose:
        Xhat = (P.T @ X64[idx[:k], :])
    else:
        Xhat = X64[:, idx[:k]] @ P
    return float(np.linalg.norm(Xhat - X64) / np.linalg.norm(X64))


data = np.load(RAW_X_PATH)
stage_keys = sorted({k.split('_step')[0] for k in data.files}, key=lambda s: int(s.replace('stage', '')))

print(f'{"stage":8s} {"k":>5s} {"row_err":>10s} {"col_err":>10s}')
knees = {}
for stage in stage_keys:
    X = data[f'{stage}_step{STEP_IDX_FOR_SWEEP}'].astype(np.float64)
    row_knee = None
    col_knee = None
    for k in K_VALUES:
        row_err = rel_error_for_k(X, k, transpose=True)
        col_err = rel_error_for_k(X, k, transpose=False)
        print(f'{stage:8s} {k:5d} {row_err:10.4f} {col_err:10.4f}', flush=True)
        if row_knee is None and row_err < TARGET_ERROR:
            row_knee = k
        if col_knee is None and col_err < TARGET_ERROR:
            col_knee = k
    knees[stage] = {'row_k': row_knee, 'col_k': col_knee}
    print(f'  -> knee (first k with error < {TARGET_ERROR}): row_k={row_knee}, col_k={col_knee}', flush=True)

print('\nSummary:', knees)
