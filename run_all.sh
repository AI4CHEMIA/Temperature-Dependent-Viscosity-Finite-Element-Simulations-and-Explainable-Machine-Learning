#!/usr/bin/env bash
# Reproduce every result, table and figure from the raw spreadsheet (~15 min on 2 CPU cores).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0
python scripts/01_preprocess.py          # clean data, sweep-aware split
python scripts/02_train.py               # validation-only model selection, refit, save models + predictions
python scripts/03_evaluate.py            # single evaluation on the isolated test set + figures 2-4
python scripts/04_extrapolation_study.py # extrapolation stress test on train+val only + figure 5
python scripts/05_shap_analysis.py       # exact SHAP (base-case & design-space) + figures 6-8
python scripts/06_physics_insights.py    # scaling exponents, elasticities, similarity + figures 9-11
python scripts/07_verify.py              # independent verification (9 checks)
