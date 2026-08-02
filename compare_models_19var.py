"""Numerical Aurora-vs-Pangu comparison of the 19x19 patching matrices."""
import os
import numpy as np
import pandas as pd

NAMES = ['MSLP', 'U10', 'V10', 'T2M']
for s in ('Z', 'Q', 'T', 'U', 'V'):
    NAMES += [f'{s}1000', f'{s}850', f'{s}500']
N = len(NAMES)

paths = {
    'aurora': '~/weather-interpretability/results/patching_19var/metrics_19var_aurora.pkl',
    'pangu': '~/weather-interpretability/results/patching_19var_pangu/metrics_19var_pangu.pkl',
}
M = {}
for k, p in paths.items():
    df = pd.read_pickle(os.path.expanduser(p))
    M[k] = {}
    for lead in (6, 24):
        sub = df[df.lead == lead]
        M[k][lead] = {
            'sens': sub.groupby(['patched', 'output'])['rel_sens'].mean().unstack()
                       .reindex(index=NAMES, columns=NAMES).values,
            'dacc': sub.groupby(['patched', 'output'])['dacc'].mean().unstack()
                       .reindex(index=NAMES, columns=NAMES).values,
        }
    print(f'{k}: {df.date.nunique()} dates')

for lead in (6, 24):
    print(f'\n{"="*70}\nLEAD +{lead}h\n{"="*70}')
    A, P = M['aurora'][lead]['sens'], M['pangu'][lead]['sens']

    print('\n-- INFLUENCE OUT (row sum excl. diagonal): what disturbs everything else')
    print(f'{"var":8s} {"Aurora":>8s} {"Pangu":>8s} {"P/A":>6s}')
    ro = [(NAMES[i], A[i].sum() - A[i, i], P[i].sum() - P[i, i]) for i in range(N)]
    for nm, a, p in sorted(ro, key=lambda x: -x[2])[:8]:
        print(f'{nm:8s} {a:8.3f} {p:8.3f} {p/max(a,1e-9):6.2f}')

    print('\n-- VULNERABILITY IN (col sum excl. diagonal): what gets disturbed')
    print(f'{"var":8s} {"Aurora":>8s} {"Pangu":>8s} {"P/A":>6s}')
    ci = [(NAMES[j], A[:, j].sum() - A[j, j], P[:, j].sum() - P[j, j]) for j in range(N)]
    for nm, a, p in sorted(ci, key=lambda x: -x[2])[:8]:
        print(f'{nm:8s} {a:8.3f} {p:8.3f} {p/max(a,1e-9):6.2f}')

    print('\n-- SELF-EFFECT (diagonal): information not recoverable from other fields')
    print(f'{"var":8s} {"Aurora":>8s} {"Pangu":>8s}')
    for i in range(N):
        print(f'{NAMES[i]:8s} {A[i,i]:8.3f} {P[i,i]:8.3f}')

    print('\n-- TOP ASYMMETRIES  A->B vs B->A  (Aurora)')
    pr = [(NAMES[i], NAMES[j], A[i, j], A[j, i]) for i in range(N) for j in range(i+1, N)]
    for a_, b_, ab, ba in sorted(pr, key=lambda x: -(x[2]-x[3]))[:6]:
        print(f'   {a_}->{b_} = {ab:.3f}  vs  {b_}->{a_} = {ba:.3f}   ({ab/max(ba,1e-6):.1f}x)')
    print('-- TOP ASYMMETRIES (Pangu)')
    pr = [(NAMES[i], NAMES[j], P[i, j], P[j, i]) for i in range(N) for j in range(i+1, N)]
    for a_, b_, ab, ba in sorted(pr, key=lambda x: -(x[2]-x[3]))[:6]:
        print(f'   {a_}->{b_} = {ab:.3f}  vs  {b_}->{a_} = {ba:.3f}   ({ab/max(ba,1e-6):.1f}x)')

    # structural agreement
    off = ~np.eye(N, dtype=bool)
    r = np.corrcoef(A[off], P[off])[0, 1]
    print(f'\n-- structural agreement of off-diagonal sensitivity: r = {r:.3f}')
    print(f'-- total off-diagonal mass: Aurora {A[off].sum():.1f}, Pangu {P[off].sum():.1f} '
          f'(Pangu/Aurora = {P[off].sum()/A[off].sum():.2f})')

    print('\n-- ACC: which patch hurts each output most (non-self marked)')
    AA, PA = M['aurora'][lead]['dacc'], M['pangu'][lead]['dacc']
    for j in range(N):
        ia, ip = int(np.argmin(AA[:, j])), int(np.argmin(PA[:, j]))
        ta = '' if ia == j else ' *'
        tp = '' if ip == j else ' *'
        print(f'   {NAMES[j]:8s} Aurora<-{NAMES[ia]:7s}({AA[ia,j]:+.3f}){ta:2s}  '
              f'Pangu<-{NAMES[ip]:7s}({PA[ip,j]:+.3f}){tp}')
