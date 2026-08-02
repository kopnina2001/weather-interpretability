"""Task 4: Non-negative Matrix Factorization (NMF, Lee & Seung 1999) as an alternative,
"parts-based" decomposition of Aurora's window attention matrices -- contrasted with the
skeleton/interpolative decomposition applied to the same object in notebook 07 Section 5,
which gave a negative result (landmarks dominated by a QR-pivoting edge-selection artifact).
Attention weights are already non-negative (softmax outputs), a natural fit for NMF with no
data transformation needed. We ask: does NMF recover a cleaner, less artifact-prone additive
basis of attention patterns than skeleton decomposition did?
"""
import os
import pickle
import numpy as np
from sklearn.decomposition import NMF

OUT_DIR = os.path.expanduser('~/weather-interpretability/results/legacy/tensor_decomposition')
STAGES = [0, 1]
STEP_IDX = 3  # +24h
K_COMPONENTS = 6

results = {}
for date_label in ['storm', 'quiet']:
    data = np.load(os.path.join(OUT_DIR, f'{date_label}_attn_tensor.npz'))
    for stage_idx in STAGES:
        T = data[f'stage{stage_idx}_step{STEP_IDX}'].astype(np.float64)  # (heads, 144, 144)
        n_heads = T.shape[0]
        # Stack all heads' attention matrices as rows: (heads*144, 144) -- NMF finds a shared
        # basis of k "attention pattern" row-vectors (each a distribution over 144 key
        # positions) plus per-(head,query) non-negative loadings on that basis.
        X = T.reshape(-1, T.shape[-1])  # (heads*144, 144), non-negative (softmax rows)
        nmf = NMF(n_components=K_COMPONENTS, init='nndsvda', max_iter=500, random_state=0)
        W = nmf.fit_transform(X)  # (heads*144, K)
        H = nmf.components_       # (K, 144) -- the shared basis attention patterns
        recon_err = float(np.linalg.norm(X - W @ H) / np.linalg.norm(X))
        results[(date_label, stage_idx)] = {'W': W.astype(np.float32), 'H': H.astype(np.float32),
                                             'recon_err': recon_err, 'n_heads': n_heads}
        print(f'{date_label} stage{stage_idx}: recon_err={recon_err:.4f}', flush=True)

with open(os.path.join(OUT_DIR, 'nmf_results.pkl'), 'wb') as f:
    pickle.dump(results, f)
print('saved')
