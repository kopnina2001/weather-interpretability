# NeurIPS 2026 CCAI workshop paper

Two anonymous workshop-style manuscripts are provided:

- `main_en.tex` -- English submission draft;
- `main_ru.tex` -- Russian review/communication mirror.

Both use the unmodified `tackling_climate_workshop_style.sty` from
`TCCML_NeurIPS_2026_Style_File.zip`. The core manuscript follows IMRAD and is
separated from references and supplementary figures with `\clearpage`.

## Build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/en main_en.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build/ru main_ru.tex
```

The English version uses pdfLaTeX as requested by the template. The Russian
mirror uses XeLaTeX so Cyrillic glyphs remain embedded and searchable.

Final checked PDFs are copied to:

- `../../output/pdf/weather_interpretability_ccai2026_en.pdf`;
- `../../output/pdf/weather_interpretability_ccai2026_ru.pdf`.

## Evidence status

The committed code, README, and rendered figures were checked together. The
`data` and `results` symlinks point to `/srv/exw` and were unavailable on the
authoring host, so raw arrays and confidence intervals were not recomputed.
The manuscript says this explicitly and treats reported values as descriptive
point estimates. Before submission, reconnect the run store and add date-block
bootstrap confidence intervals for the promoted numerical claims.

The English PDF is anonymous and formatted for the Papers track. Replace the
author block only after acceptance or when building a `[preprint]` version.
The English supplement reuses two committed dose-response panels whose embedded
plot annotations are in Russian; their paper captions and all main-text figures
are English.
