"""Build the anonymous supplementary repository used for workshop review."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "weather-forecast-response-audit-anonymous"

FILES = (
    "paper/neurips2026_ccai/main_en.tex",
    "paper/neurips2026_ccai/main_ru.tex",
    "paper/neurips2026_ccai/references.bib",
    "paper/neurips2026_ccai/tackling_climate_workshop_style.sty",
    "analysis/bootstrap_19var.py",
    "analysis/compare_models_19var.py",
    "plots/translate_paper_figures.py",
    "figures/patching_matrices/matrix_19var_aurora_n48_weighted.png",
    "figures/patching_matrices/matrix_19var_pangu_n48_weighted.png",
    "figures/patching_matrices/diagonal_matrix_seasonal_48dates.png",
    "figures/patching_matrices/acc_strip_Z_both_n48_en.png",
    "figures/dose_response/dose_panels_anom_Z1000_lead6_aurora_n48_en.png",
    "figures/dose_response/dose_panels_anom_Z1000_lead24_aurora_n48_en.png",
    "figures/dose_response/dose_panels_anom_Z1000_lead6_pangu_n48_en.png",
    "figures/dose_response/dose_panels_anom_Z1000_lead24_pangu_n48_en.png",
    "figures/bias_maps/composite_bias_Z1000_lead6_alpha1_n48_en.png",
    "figures/bias_maps/composite_bias_Z1000_lead24_alpha1_n48_en.png",
    "figures/bias_maps/composite_bias_pangu_Z1000_lead6_alpha1_n48_en.png",
    "figures/bias_maps/composite_bias_pangu_Z1000_lead24_alpha1_n48_en.png",
    "figures/bias_maps/composite_bias_Z1000_lead6_alpha1_n48_t1000_en.png",
    "figures/bias_maps/composite_bias_Z1000_lead24_alpha1_n48_t1000_en.png",
    "figures/bias_maps/composite_bias_pangu_Z1000_lead6_alpha1_n48_t1000_en.png",
    "figures/bias_maps/composite_bias_pangu_Z1000_lead24_alpha1_n48_t1000_en.png",
    "figures/bias_maps/composite_bias_Z1000_lead6_alpha1_n48_t1000_ru.png",
    "figures/bias_maps/composite_bias_Z1000_lead24_alpha1_n48_t1000_ru.png",
    "figures/bias_maps/composite_bias_pangu_Z1000_lead6_alpha1_n48_t1000_ru.png",
    "figures/bias_maps/composite_bias_pangu_Z1000_lead24_alpha1_n48_t1000_ru.png",
)

MANUSCRIPTS = (
    "paper/neurips2026_ccai/main_en.tex",
    "paper/neurips2026_ccai/main_ru.tex",
)
INCLUDE_GRAPHICS = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")

TEXT_SUFFIXES = {".md", ".py", ".tex", ".bib", ".txt"}
FORBIDDEN_PATTERNS = {
    "home path": re.compile(r"/(?:home|Users)/[^/\s]+/", re.IGNORECASE),
    "run-store path": re.compile(r"/(?:srv|data|mnt)/[^\s'\"]+", re.IGNORECASE),
    "email": re.compile(r"(?<![\w.-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "repository URL": re.compile(r"https?://(?:www\.)?(?:github|gitlab)\.com/", re.IGNORECASE),
    "local identity": re.compile(r"(?:dsuhoi|kopnina|irina)", re.IGNORECASE),
}

README = """# Anonymous supplementary repository

This package accompanies the anonymous workshop submission
"Cross-Field Forecast Responses to Climatological Input Replacement in Aurora
and Pangu-Weather."

## Contents

- `paper/neurips2026_ccai/`: anonymous English submission source, Russian content
  mirror, bibliography, and the official workshop style file;
- `analysis/`: scripts that compare area-weighted response matrices and compute
  paired calendar-month bootstrap intervals from per-date metric tables;
- `plots/`: deterministic translation/cropping code for the raster figures;
- `figures/`: every raster figure referenced by either manuscript.

The package contains the reported aggregate figures but not the large ERA5,
WeatherBench 2, model-weight, forecast, or per-date metric stores. Those inputs
cannot be redistributed here. Consequently, the 95% intervals described in the
appendix cannot be recomputed from this archive alone. The paper labels all
reported values as point estimates.

## Expected metric tables

`analysis/bootstrap_19var.py` accepts two pickled pandas tables, one per model.
Each row must contain `date`, `lead`, `patched`, `output`, `rel_sens_w`, and
`dacc_w`; `delta_rmse_w_anom` is optional. Run:

```bash
python analysis/bootstrap_19var.py \
  --aurora path/to/metrics_19var_aurora.pkl \
  --pangu path/to/metrics_19var_pangu.pkl \
  --output-dir bootstrap
```

## Build the paper

From `paper/neurips2026_ccai/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main_en.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error main_ru.tex
```

The review PDF is the English version. Both manuscripts use anonymous submission
mode. No Git history, author metadata, host path, or public repository URL is
included in this package. Install the small analysis environment with
`python -m pip install -r requirements.txt`. No license is assigned in this
review package; add an author-approved license before public release.
"""

REQUIREMENTS = """numpy>=2.0
pandas>=2.0
Pillow>=10.0
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "anonymous_repository",
    )
    return parser.parse_args()


def package_files() -> tuple[str, ...]:
    """Return declared files plus every raster referenced by either manuscript."""
    discovered: list[str] = list(FILES)
    for relative in MANUSCRIPTS:
        manuscript = ROOT / relative
        for match in INCLUDE_GRAPHICS.finditer(manuscript.read_text(encoding="utf-8")):
            figure = (manuscript.parent / match.group(1)).resolve()
            discovered.append(figure.relative_to(ROOT).as_posix())
    return tuple(dict.fromkeys(discovered))


def copy_files(destination: Path) -> None:
    files = package_files()
    missing = [relative for relative in files if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"required files are missing: {missing}")
    for relative in files:
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (destination / "README.md").write_text(README, encoding="utf-8")
    (destination / "requirements.txt").write_text(REQUIREMENTS, encoding="ascii")


def scan(destination: Path) -> None:
    findings: list[str] = []
    for path in sorted(destination.rglob("*")):
        relative = path.relative_to(destination).as_posix()
        if path.is_symlink():
            findings.append(f"symlink: {relative}")
            continue
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(relative):
                findings.append(f"{label} in member name: {relative}")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in FORBIDDEN_PATTERNS.items():
                if pattern.search(text):
                    findings.append(f"{label} in text: {relative}")
    if findings:
        raise RuntimeError("anonymous-package scan failed:\n" + "\n".join(findings))


def write_archive(destination: Path) -> tuple[Path, Path]:
    archive = destination.parent / f"{PACKAGE_NAME}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                relative = path.relative_to(destination)
                handle.write(path, Path(PACKAGE_NAME) / relative)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    sums = destination.parent / "SHA256SUMS"
    sums.write_text(
        f"{digest}  {archive.relative_to(ROOT).as_posix()}\n", encoding="ascii"
    )
    return archive, sums


def main() -> None:
    args = parse_args()
    destination = args.output_dir.resolve() / PACKAGE_NAME
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    copy_files(destination)
    scan(destination)
    archive, sums = write_archive(destination)
    print(f"anonymous repository: {destination}")
    print(f"archive: {archive}")
    print(f"checksums: {sums}")


if __name__ == "__main__":
    main()
