"""Predict avgNu and |psi|_max for new input combinations with the trained surrogates.

    python scripts/predict.py new_inputs.csv --model GP --out predictions.csv
    python scripts/predict.py new_inputs.csv --model MLP --deploy

new_inputs.csv needs the columns Ra, Da, R, A, C (Pr = 7 and N = 3 are implied).
Output columns: <target>_pred and, where available, a 95 % interval
<target>_PI95_low / _PI95_high (GP posterior, MLP-ensemble or RF-tree spread),
plus a flag telling whether the point lies inside the sampled design and
whether it varies more than one parameter from the base case (combinations
of several non-base parameters were never simulated: treat those predictions
as unvalidated - see the paper's interpretability section).
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surrogate.config import MODELS, PROCESSED, TARGETS  # noqa: E402
from surrogate.features import predict_log_mean_std  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs")
    ap.add_argument("--model", default="GP", choices=["GP", "MLP", "RF", "PolyRidge", "GBM",
                                                      "ExtraTrees"])
    ap.add_argument("--deploy", action="store_true",
                    help="use the model refitted on all 312 runs (not independently tested)")
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_args()

    log = json.loads((PROCESSED / "preprocessing_log.json").read_text())
    F, base, rng = log["features"], log["base_case"], log["sweep_ranges"]
    X = pd.read_csv(a.inputs)
    out = X.copy()
    for t in TARGETS:
        path = (MODELS / "deploy" if a.deploy else MODELS) / f"{t}__{a.model}.joblib"
        mdl = joblib.load(path)["model"]
        out[f"{t}_pred"] = mdl.predict(X[F])
        mu, sd = predict_log_mean_std(mdl, X[F])
        if np.isfinite(sd).all():
            out[f"{t}_PI95_low"] = 10 ** (mu - 1.96 * sd)
            out[f"{t}_PI95_high"] = 10 ** (mu + 1.96 * sd)
    inside = np.all([(X[p] >= rng[p]["min"]) & (X[p] <= rng[p]["max"]) for p in F], axis=0)
    n_off = np.sum([~np.isclose(X[p], base[p], rtol=1e-6) for p in F], axis=0)
    out["inside_sampled_ranges"] = inside
    out["n_params_off_base"] = n_off
    out["validated_regime"] = inside & (n_off <= 1)
    out.to_csv(a.out, index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
