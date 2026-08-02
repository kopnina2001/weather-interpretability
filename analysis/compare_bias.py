"""Compare the composite bias maps of Aurora and Pangu side by side.

Ratios alone are misleading: a model with a small baseline error can show a huge multiplier
and still end up more accurate than the other.  This prints the absolute RMSE at alpha=0 and
alpha=1 in the field's own units, and two measures of how concentrated the damage is:

    top10  -- share of the total damage that sits in the worst 10% of grid points
    gini   -- Gini coefficient of the damage across grid points (0 = uniform, 1 = one point)

Both are area-weighted by cos(phi), otherwise the polar grid rows dominate the counts.
"""
import os
import numpy as np

BASE = os.path.expanduser('~/weather-interpretability/results/dose_response/z1000')
FIELDS = ['Q1000', 'T1000', 'U1000', 'V1000']
UNITS = {'Q1000': 'кг/кг', 'T1000': 'К', 'U1000': 'м/с', 'V1000': 'м/с'}
LEADS = [6, 24]


def load(model):
    p = f'{BASE}/composite_bias_{model}/composite_rmse_{model}_Z1000_n48.npz'
    if not os.path.exists(p):          # aurora was written before the name was parameterised
        p = f'{BASE}/composite_bias_{model}/composite_rmse_Z1000_{model}_n48.npz'
    return np.load(p)


def stats(d, L, f):
    lat = d['lat']
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, d['lon'].size))
    w = w / w.sum()
    r0, r1 = d[f'rmse_lead{L}_a0.0_{f}'], d[f'rmse_lead{L}_a1.0_{f}']
    dmg = r1 - r0
    m0, m1 = float((w * r0).sum()), float((w * r1).sum())

    # concentration of the damage, area-weighted
    flat = dmg.ravel(); wf = w.ravel()
    order = np.argsort(flat)[::-1]
    fs, ws = flat[order], wf[order]
    cw, cd = np.cumsum(ws), np.cumsum(fs * ws)
    total = cd[-1]
    top10 = float(cd[np.searchsorted(cw, 0.10)] / total)
    # Gini over the area-weighted damage distribution
    asc = np.argsort(flat)
    fa, wa = flat[asc], wf[asc]
    cwa = np.cumsum(wa); cda = np.cumsum(fa * wa) / total
    gini = float(1 - 2 * (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(cda, cwa))
    worse = float((w * (dmg > 0)).sum())
    return m0, m1, top10, gini, worse


data = {m: load(m) for m in ('aurora', 'pangu')}
for L in LEADS:
    print(f'\n=== лид +{L}ч ===')
    print(f'{"поле":7s} {"модель":7s} {"RMSE α=0":>11s} {"RMSE α=1":>11s} {"рост":>6s} '
          f'{"топ-10%":>8s} {"Джини":>7s} {"хуже":>6s}')
    for f in FIELDS:
        for m in ('aurora', 'pangu'):
            m0, m1, t10, g, w_ = stats(data[m], L, f)
            print(f'{f if m=="aurora" else "":7s} {m:7s} {m0:11.4g} {m1:11.4g} '
                  f'{m1/m0:5.2f}x {t10*100:7.1f}% {g:7.3f} {w_*100:5.1f}%')
