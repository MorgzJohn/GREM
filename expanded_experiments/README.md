# Sensitivity study of GREM under garment-only degradations

Pre-specified analysis plan: `config/analysis_plan.json` (version 2, SHA-256 `8295cdc3...`, stored in `results/run_meta.json`).
Scoring code: `../code/grem_core.py` (SHA-256 `723b3b09...`, also stored in `results/run_meta.json`). Nothing in GREM was changed for these experiments.

## What is in each folder
* `config/` : `outfits.csv` (one row per generated output: files, quality and mask ratings, analysis set, wrong-garment partner), `paths.json`, `analysis_plan.json`.
* `scripts/` : `00_validate_dataset.py` (files, masks, flags), `update_outfits.py`, `exp_lib.py`, `01_make_degradations.py`, `02_run_batch.py`,
  `03_verify.py`, `04_analyze.py`, `05_figures.py`, `06_paper_figure.py`, `check_rows.py`.
* `validation/dataset_report.csv`, `degraded/manifest.csv` + `meta.json` : mask sizes, flags, and the record of every degraded image (SHA-256 of each file).
* `results/` : raw scores (`raw_results.csv`), mismatched-reference scores, expected failures, run metadata, verification report.
* `analysis/`, `figures/` : statistics tables and figures used in the paper.

## Re-running without images
```bash
pip install -r ../requirements-experiments.txt
python3 scripts/04_analyze.py      # reads results/, validation/, config/ ; writes analysis/
python3 scripts/05_figures.py      # writes figures/
python3 scripts/06_paper_figure.py
```

## Re-running everything (needs your own images)
Edit `config/paths.json` (folder of person / garment / output photos) and `config/outfits.csv`, then:
```bash
python3 scripts/00_validate_dataset.py      # label mask_quality in config/outfits.csv by eye from the overlay sheets
python3 scripts/update_outfits.py --recompute
python3 scripts/01_make_degradations.py
python3 scripts/02_run_batch.py
python3 scripts/03_verify.py                # must end with ALL CHECKS PASSED
python3 scripts/04_analyze.py && python3 scripts/05_figures.py && python3 scripts/06_paper_figure.py
```
Degradations change only pixels inside the estimated garment region; `03_verify.py` checks that pixels outside are bit-identical to the original.
