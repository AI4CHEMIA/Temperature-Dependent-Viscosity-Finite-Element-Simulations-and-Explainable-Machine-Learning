# Interpretable ML surrogates for avgNu and |ψ|max

Reproducible code, data, trained models, predictions and figures for the machine-learning
part of the paper. The surrogates predict the mean Nusselt number (avgNu) and the maximum
absolute stream function (|ψ|max) from the inputs Ra, Da, R, A and C. They are valid only
at Pr = 7 and N = 3, the fixed values in the dataset.

## Quick start

```bash
pip install -r requirements.txt          # Python 3.13; versions pinned to those used for the paper
./run_all.sh                             # full pipeline from the raw spreadsheet (~15 min, 2 CPU cores)
python scripts/07_verify.py              # verification only (~1 min), using the shipped models
```

Predict new cases (columns `Ra,Da,R,A,C`):

```bash
python scripts/predict.py my_cases.csv --model GP  --out my_predictions.csv
python scripts/predict.py my_cases.csv --model MLP --out my_predictions.csv
python scripts/predict.py examples/example_inputs.csv --model MLP   # worked example
```

`predict.py` also adds a `validated_regime` flag. It is false when a case lies outside the
sampled ranges, or varies more than one parameter from the base case: such combinations
were never simulated, so those predictions are unvalidated.

## Pipeline

| Step | Script | What it does | 
|---|---|---|
| 1 | `scripts/01_preprocess.py` | Cleans the sheet, drops constant inputs (Pr, N), removes 4 base-case replicates, labels every run by its sweep, builds the deterministic sweep-aware split |
| 2 | `scripts/02_train.py` | Grid search per model family on **validation only** (score = ½ MAPE val-interp + ½ MAPE val-extrap), with and without the log10(Ra·Da) feature; refits the winner on train+val; saves models and predictions |
| 3 | `scripts/03_evaluate.py` | Single evaluation on the isolated test set (interpolation / extrapolation / all, per sweep, bootstrap 95 % CI); figures 2–4 | 
| 4 | `scripts/04_extrapolation_study.py` | Extrapolation stress test: removes the outer 25 % of each sweep from train+val and measures error vs distance; figure 5 | 
| 5 | `scripts/05_shap_analysis.py` | Exact SHAP for RF, MLP and GP: base-case reference (on-manifold) and Sobol design-space background (off-manifold); additivity and cross-model agreement; figures 6–8 |
| 6 | `scripts/06_physics_insights.py` | Local scaling exponents, elasticities at the base case, Darcy–Rayleigh similarity test, ψ extrema, heat-transfer/circulation coupling, physics-feature ablation; figures 9–11 |
| 7 | `scripts/07_verify.py` | 9 independent checks (checksum, leakage, extrapolation placement, reproduction of predictions, metrics and SHAP) | 

## Folder contents

```
data/raw/data_ML_paper1.xlsx                 original spreadsheet (unchanged)
data/processed/dataset_clean.csv             312 unique runs + 4 replicates, with sweep / subset / extrapolation distance
data/processed/split_summary.csv             counts per sweep and subset
data/processed/preprocessing_log.json        base case, dropped inputs, replicates, ranges, SHA-256 of the raw file
surrogate/                                   config, feature transforms, model factories, MLP ensemble, metrics, plot style
scripts/                                     steps 1-7 + predict.py
models/<target>__<model>.joblib              models fitted on train+val (the ones evaluated on the test set)
models/deploy/<target>__<model>.joblib       same configuration refitted on all 312 runs (for use; not independently tested)
results/tuning/                              every validation configuration tried + the selected ones
results/predictions/predictions_all_models.csv     every run x target x model, with 95 % intervals and subset labels
results/predictions/predictions_test_set_wide.csv  test runs, all models side by side
results/predictions/extrapolation_stress_test_predictions.csv
results/metrics/                             test metrics (+CI), per-sweep metrics, stress-test summaries, PI coverage
results/shap/                                SHAP values (both backgrounds), importance, rank agreement, faithfulness, additivity
results/physics/                             exponents, elasticities, coupling, similarity, ablation, physics_summary.json
results/figures/                             mlfig02-fig11
examples/example_inputs.csv                  sample input file for predict.py (last row = an unvalidated multi-parameter case)
```

## Subset definitions

Each sweep is ordered by its coordinate (log10 for Ra and Da):

- `test_extrap`: the outermost 7 % of points at each end. These lie outside the range used to fit the final models.
- `val_extrap`: the next 5 % inward at each end. These are outside the training range during model selection.
- `test_interp` / `val_interp`: every 9th interior point (offsets 4 and 8), spread over the whole sweep.
- `train`: everything else, including the base case.

Split sizes are 183 train, 59 validation and 70 test runs.

## Reproducibility notes

- The split is deterministic, and all models use fixed seeds (`SEED = 2026` in `surrogate/config.py`).
- Re-running `run_all.sh` on the pinned versions reproduces the shipped results. `07_verify.py` checks the shipped models against the shipped CSVs to 1e-9.
- Metrics are computed in physical units after inverting the log10 and standardisation transforms.
