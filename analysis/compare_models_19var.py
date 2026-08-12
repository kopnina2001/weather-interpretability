"""Compare Aurora and Pangu 19x19 field-patching matrices.

The paper reports area-weighted diagnostics, so this script uses ``rel_sens_w``
and ``dacc_w`` by default. Pass ``--unweighted`` only for the legacy comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NAMES = ["MSLP", "U10", "V10", "T2M"]
for short_name in ("Z", "Q", "T", "U", "V"):
    NAMES += [f"{short_name}{level}" for level in (1000, 850, 500)]
N = len(NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aurora",
        type=Path,
        default=ROOT / "results/patching/patching_19var/metrics_19var_aurora.pkl",
    )
    parser.add_argument(
        "--pangu",
        type=Path,
        default=ROOT / "results/patching/patching_19var_pangu/metrics_19var_pangu.pkl",
    )
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="Use legacy rel_sens/dacc columns instead of area-weighted metrics.",
    )
    return parser.parse_args()


def load_matrices(path: Path, sens_col: str, dacc_col: str) -> tuple[dict, int]:
    if not path.exists():
        raise FileNotFoundError(f"metrics file not found: {path}")
    df = pd.read_pickle(path)
    required = {"date", "lead", "patched", "output", sens_col, dacc_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    matrices: dict[int, dict[str, np.ndarray]] = {}
    for lead in (6, 24):
        sub = df[df.lead == lead]
        matrices[lead] = {
            "sens": sub.groupby(["patched", "output"])[sens_col]
            .mean()
            .unstack()
            .reindex(index=NAMES, columns=NAMES)
            .to_numpy(),
            "dacc": sub.groupby(["patched", "output"])[dacc_col]
            .mean()
            .unstack()
            .reindex(index=NAMES, columns=NAMES)
            .to_numpy(),
        }
    return matrices, int(df.date.nunique())


def main() -> None:
    args = parse_args()
    sens_col = "rel_sens" if args.unweighted else "rel_sens_w"
    dacc_col = "dacc" if args.unweighted else "dacc_w"
    metric_label = "unweighted" if args.unweighted else "cos(latitude)-weighted"

    matrices = {}
    for model, path in (("aurora", args.aurora), ("pangu", args.pangu)):
        matrices[model], n_dates = load_matrices(path, sens_col, dacc_col)
        print(f"{model}: {n_dates} dates; {metric_label} metrics")

    for lead in (6, 24):
        print(f'\n{"=" * 70}\nLEAD +{lead}h\n{"=" * 70}')
        aurora = matrices["aurora"][lead]["sens"]
        pangu = matrices["pangu"][lead]["sens"]

        print("\n-- INFLUENCE OUT (row sum excluding diagonal)")
        print(f'{"var":8s} {"Aurora":>8s} {"Pangu":>8s} {"P/A":>6s}')
        rows = [
            (NAMES[i], aurora[i].sum() - aurora[i, i], pangu[i].sum() - pangu[i, i])
            for i in range(N)
        ]
        for name, a_value, p_value in sorted(rows, key=lambda row: -row[2])[:8]:
            print(f"{name:8s} {a_value:8.3f} {p_value:8.3f} {p_value / max(a_value, 1e-9):6.2f}")

        print("\n-- VULNERABILITY IN (column sum excluding diagonal)")
        print(f'{"var":8s} {"Aurora":>8s} {"Pangu":>8s} {"P/A":>6s}')
        columns = [
            (NAMES[j], aurora[:, j].sum() - aurora[j, j], pangu[:, j].sum() - pangu[j, j])
            for j in range(N)
        ]
        for name, a_value, p_value in sorted(columns, key=lambda row: -row[2])[:8]:
            print(f"{name:8s} {a_value:8.3f} {p_value:8.3f} {p_value / max(a_value, 1e-9):6.2f}")

        print("\n-- SELF-EFFECT (diagonal)")
        print(f'{"var":8s} {"Aurora":>8s} {"Pangu":>8s}')
        for i, name in enumerate(NAMES):
            print(f"{name:8s} {aurora[i, i]:8.3f} {pangu[i, i]:8.3f}")

        for model, matrix in (("Aurora", aurora), ("Pangu", pangu)):
            print(f"\n-- TOP ASYMMETRIES ({model})")
            pairs = [
                (NAMES[i], NAMES[j], matrix[i, j], matrix[j, i])
                for i in range(N)
                for j in range(i + 1, N)
            ]
            for source, target, forward, reverse in sorted(
                pairs, key=lambda row: -(row[2] - row[3])
            )[:6]:
                print(
                    f"   response of {target} to replaced {source} = {forward:.3f}; "
                    f"reverse response = {reverse:.3f}   "
                    f"({forward / max(reverse, 1e-6):.1f}x)"
                )

        off_diagonal = ~np.eye(N, dtype=bool)
        correlation = np.corrcoef(aurora[off_diagonal], pangu[off_diagonal])[0, 1]
        print(f"\n-- cross-model off-diagonal sensitivity: r = {correlation:.3f}")
        print(
            f"-- total off-diagonal mass: Aurora {aurora[off_diagonal].sum():.1f}, "
            f"Pangu {pangu[off_diagonal].sum():.1f} "
            f"(Pangu/Aurora = {pangu[off_diagonal].sum() / aurora[off_diagonal].sum():.2f})"
        )

        print("\n-- dACC: patch that hurts each output most (non-self marked)")
        aurora_dacc = matrices["aurora"][lead]["dacc"]
        pangu_dacc = matrices["pangu"][lead]["dacc"]
        for j, name in enumerate(NAMES):
            aurora_idx = int(np.argmin(aurora_dacc[:, j]))
            pangu_idx = int(np.argmin(pangu_dacc[:, j]))
            aurora_mark = "" if aurora_idx == j else " *"
            pangu_mark = "" if pangu_idx == j else " *"
            print(
                f"   {name:8s} Aurora<-{NAMES[aurora_idx]:7s}"
                f"({aurora_dacc[aurora_idx, j]:+.3f}){aurora_mark:2s}  "
                f"Pangu<-{NAMES[pangu_idx]:7s}"
                f"({pangu_dacc[pangu_idx, j]:+.3f}){pangu_mark}"
            )


if __name__ == "__main__":
    main()
