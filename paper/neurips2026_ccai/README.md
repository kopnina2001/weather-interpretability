# NeurIPS 2026 CCAI workshop paper

Two anonymous workshop-style manuscripts are provided:

- `main_en.tex` -- English submission draft;
- `main_ru.tex` -- Russian content draft.

The Russian draft is the content source. The English manuscript mirrors its
section order, experiments, figures, claims, limitations, and citations.

Both use the unmodified `tackling_climate_workshop_style.sty` from
`TCCML_NeurIPS_2026_Style_File.zip`. The IMRAD main text occupies four pages;
references start on page 5 and the supplementary material follows them.

## Build

From this directory:

```bash
typst compile --root . --input lang=en figures/method_pipeline.typ figures/method_pipeline_en.pdf
typst compile --root . --input lang=ru figures/method_pipeline.typ figures/method_pipeline_ru.pdf
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/en main_en.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build/ru main_ru.tex
```

The English version uses pdfLaTeX as requested by the template. The Russian
mirror uses XeLaTeX so Cyrillic glyphs remain embedded and searchable.
Figure 1 is maintained as one bilingual Typst/CeTZ source. Linux Biolinum is
used for labels and New Computer Modern Math for equations; both fonts are
embedded in the vector PDFs. Arrow ports and routes are fixed rather than
automatically re-laid out during the LaTeX build.

Final checked PDFs are copied to:

- `../../output/pdf/weather_interpretability_ccai2026_en.pdf`;
- `../../output/pdf/weather_interpretability_ccai2026_ru.pdf`.

## Evidence status

The committed code, README, and rendered figures were checked together. The
`data` and `results` symlinks point to `/srv/exw` and were unavailable on the
authoring host, so raw arrays and confidence intervals were not recomputed.
The manuscript says this explicitly and treats reported values as descriptive
point estimates.

`analysis/compare_models_19var.py` now uses `rel_sens_w` and `dacc_w` by
default. The previously reported cross-model `r=0.913` came from the unweighted
comparison and is not promoted in the four-page paper. After reconnecting the
run store, execute:

```bash
python analysis/bootstrap_19var.py --n-bootstrap 5000
```

The script resamples calendar-month blocks and writes one table for all
area-weighted matrix cells and one table for the key directional and attenuation
results. A fresh `compute_19var_matrix.py` run also adds normalized
truth-relative delta RMSE; the bootstrap then reports whether the Z500 error
against ERA5 decreases from +6 to +24 h. Replace the point estimates with these
intervals before submission.

The repository still lacks three confirmatory results: truth-relative recovery
across at least three leads, an orography/below-ground mask, and a second
season-matched corruption. These are described in Appendix A and must not be
presented as completed experiments.

The English PDF is anonymous and formatted for the Papers track. Replace the
author block only after acceptance or when building a `[preprint]` version.
All figures referenced by `main_en.tex` have English labels. The deterministic
`plots/translate_paper_figures.py` script recreates the English raster copies
and the dedicated bilingual `T1000` panels used in Figure 3 without changing
any numerical panel or map; the Russian source figures remain available to
`main_ru.tex`.
