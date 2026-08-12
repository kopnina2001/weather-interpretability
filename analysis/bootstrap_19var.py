"""Month-block bootstrap for the Aurora/Pangu 19-field patching audit.

The script consumes the per-date metric pickles produced by
``analysis/compute_19var_matrix.py``. It does not need the original forecast
arrays. Months are resampled as paired blocks, so the four initializations in a
month stay together and the same bootstrap sample is used for both models.

Outputs
-------
weighted_matrix_ci.csv
    Point estimate and 95% interval for every matrix cell, model, lead, and
    area-weighted metric.
key_results_bootstrap.csv
    Cross-model correlation, selected directional effects, asymmetry, and
    Z500 attenuation reported in the paper. If the input tables include
    ``delta_rmse_w_anom``, the file also records truth-relative Z500 recovery.
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
NAME_TO_INDEX = {name: index for index, name in enumerate(NAMES)}
REQUIRED_METRICS = ("rel_sens_w", "dacc_w")
OPTIONAL_METRICS = ("delta_rmse_w_anom",)
LEADS = (6, 24)


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
        "--output-dir",
        type=Path,
        default=ROOT / "results/patching/patching_19var/bootstrap",
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def parse_month(label: str) -> str:
    digits = "".join(character for character in str(label) if character.isdigit())
    if len(digits) < 6:
        raise ValueError(f"cannot parse YYYYMM from date label {label!r}")
    return digits[:6]


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"metrics file not found: {path}")
    frame = pd.read_pickle(path)
    required = {"date", "lead", "patched", "output", *REQUIRED_METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    columns = list(required) + [metric for metric in OPTIONAL_METRICS if metric in frame.columns]
    frame = frame.loc[frame.lead.isin(LEADS), columns].copy()
    frame["date"] = frame.date.astype(str)
    return frame


def common_complete_dates(frames: dict[str, pd.DataFrame]) -> list[str]:
    complete_sets = []
    expected_rows = len(LEADS) * len(NAMES) ** 2
    for model, frame in frames.items():
        counts = frame.groupby("date").size()
        complete = set(counts[counts == expected_rows].index)
        if not complete:
            raise ValueError(
                f"{model} has no dates with {expected_rows} rows "
                f"({len(LEADS)} leads x {len(NAMES)} x {len(NAMES)} fields)"
            )
        complete_sets.append(complete)
    dates = sorted(set.intersection(*complete_sets))
    if not dates:
        raise ValueError("Aurora and Pangu have no common complete dates")
    return dates


def date_cube(frame: pd.DataFrame, dates: list[str], lead: int, metric: str) -> np.ndarray:
    cubes = []
    for date in dates:
        subset = frame[(frame.date == date) & (frame.lead == lead)]
        matrix = (
            subset.pivot(index="patched", columns="output", values=metric)
            .reindex(index=NAMES, columns=NAMES)
            .to_numpy(dtype=float)
        )
        if matrix.shape != (len(NAMES), len(NAMES)) or not np.isfinite(matrix).all():
            raise ValueError(f"incomplete or non-finite {metric} matrix for {date}, lead {lead}")
        cubes.append(matrix)
    return np.stack(cubes)


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.nanquantile(values, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100")

    frames = {
        "aurora": load_metrics(args.aurora),
        "pangu": load_metrics(args.pangu),
    }
    dates = common_complete_dates(frames)
    for model in frames:
        frames[model] = frames[model][frames[model].date.isin(dates)].copy()
    metrics = list(REQUIRED_METRICS)
    metrics.extend(
        metric
        for metric in OPTIONAL_METRICS
        if all(metric in frame.columns for frame in frames.values())
    )

    month_to_indices: dict[str, list[int]] = {}
    for index, date in enumerate(dates):
        month_to_indices.setdefault(parse_month(date), []).append(index)
    months = sorted(month_to_indices)
    if len(months) < 2:
        raise ValueError("month-block bootstrap needs at least two distinct months")

    rng = np.random.default_rng(args.seed)
    bootstrap_indices = []
    for _ in range(args.n_bootstrap):
        sampled_months = rng.choice(months, size=len(months), replace=True)
        bootstrap_indices.append(
            np.concatenate([month_to_indices[month] for month in sampled_months])
        )

    cubes = {
        model: {
            lead: {metric: date_cube(frame, dates, lead, metric) for metric in metrics}
            for lead in LEADS
        }
        for model, frame in frames.items()
    }

    matrix_rows = []
    bootstrap_means: dict[str, dict[int, dict[str, np.ndarray]]] = {
        model: {lead: {} for lead in LEADS} for model in frames
    }
    for model in frames:
        for lead in LEADS:
            for metric in metrics:
                cube = cubes[model][lead][metric]
                point = cube.mean(axis=0)
                boot = np.stack([cube[indices].mean(axis=0) for indices in bootstrap_indices])
                bootstrap_means[model][lead][metric] = boot
                lows = np.quantile(boot, 0.025, axis=0)
                highs = np.quantile(boot, 0.975, axis=0)
                for i, patched in enumerate(NAMES):
                    for j, output in enumerate(NAMES):
                        matrix_rows.append(
                            {
                                "metric": metric,
                                "model": model,
                                "lead_h": lead,
                                "patched": patched,
                                "output": output,
                                "estimate": point[i, j],
                                "ci_low": lows[i, j],
                                "ci_high": highs[i, j],
                                "n_dates": len(dates),
                                "bootstrap_block": "calendar_month",
                                "n_bootstrap": args.n_bootstrap,
                                "seed": args.seed,
                            }
                        )

    key_rows = []

    def add_key(
        diagnostic: str,
        model: str,
        lead: int | str,
        estimate: float,
        samples: np.ndarray,
    ) -> None:
        low, high = interval(samples)
        key_rows.append(
            {
                "diagnostic": diagnostic,
                "model": model,
                "lead_h": lead,
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "n_dates": len(dates),
                "bootstrap_block": "calendar_month",
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
            }
        )

    off_diagonal = ~np.eye(len(NAMES), dtype=bool)
    for lead in LEADS:
        aurora_cube = cubes["aurora"][lead]["rel_sens_w"]
        pangu_cube = cubes["pangu"][lead]["rel_sens_w"]
        point_a = aurora_cube.mean(axis=0)
        point_p = pangu_cube.mean(axis=0)
        point_r = float(np.corrcoef(point_a[off_diagonal], point_p[off_diagonal])[0, 1])
        boot_r = []
        for indices in bootstrap_indices:
            matrix_a = aurora_cube[indices].mean(axis=0)
            matrix_p = pangu_cube[indices].mean(axis=0)
            boot_r.append(np.corrcoef(matrix_a[off_diagonal], matrix_p[off_diagonal])[0, 1])
        add_key("cross_model_off_diagonal_r", "aurora_vs_pangu", lead, point_r, np.asarray(boot_r))

    z850 = NAME_TO_INDEX["Z850"]
    v850 = NAME_TO_INDEX["V850"]
    z500 = NAME_TO_INDEX["Z500"]
    for model in frames:
        for lead in LEADS:
            cube = cubes[model][lead]["rel_sens_w"]
            point = cube.mean(axis=0)
            boot = bootstrap_means[model][lead]["rel_sens_w"]
            add_key("Z850_to_V850", model, lead, point[z850, v850], boot[:, z850, v850])
            add_key("V850_to_Z850", model, lead, point[v850, z850], boot[:, v850, z850])
            difference = point[z850, v850] - point[v850, z850]
            difference_samples = boot[:, z850, v850] - boot[:, v850, z850]
            add_key("Z850_V850_asymmetry_difference", model, lead, difference, difference_samples)

        point_6 = cubes[model][6]["rel_sens_w"].mean(axis=0)[z500, z500]
        point_24 = cubes[model][24]["rel_sens_w"].mean(axis=0)[z500, z500]
        attenuation = point_24 / point_6
        boot_6 = bootstrap_means[model][6]["rel_sens_w"][:, z500, z500]
        boot_24 = bootstrap_means[model][24]["rel_sens_w"][:, z500, z500]
        add_key("Z500_diagonal_attenuation_24_over_6", model, "24/6", attenuation, boot_24 / boot_6)

        if "delta_rmse_w_anom" in metrics:
            truth_6 = cubes[model][6]["delta_rmse_w_anom"].mean(axis=0)[z500, z500]
            truth_24 = cubes[model][24]["delta_rmse_w_anom"].mean(axis=0)[z500, z500]
            truth_boot_6 = bootstrap_means[model][6]["delta_rmse_w_anom"][:, z500, z500]
            truth_boot_24 = bootstrap_means[model][24]["delta_rmse_w_anom"][:, z500, z500]
            add_key("Z500_truth_relative_delta_rmse", model, 6, truth_6, truth_boot_6)
            add_key("Z500_truth_relative_delta_rmse", model, 24, truth_24, truth_boot_24)
            add_key(
                "Z500_truth_relative_change_24_minus_6",
                model,
                "24-6",
                truth_24 - truth_6,
                truth_boot_24 - truth_boot_6,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / "weighted_matrix_ci.csv"
    key_path = args.output_dir / "key_results_bootstrap.csv"
    pd.DataFrame(matrix_rows).to_csv(matrix_path, index=False)
    pd.DataFrame(key_rows).to_csv(key_path, index=False)
    print(
        f"paired month-block bootstrap: {len(dates)} dates, {len(months)} months, "
        f"{args.n_bootstrap} replicates"
    )
    print(f"wrote {matrix_path}")
    print(f"wrote {key_path}")


if __name__ == "__main__":
    main()
