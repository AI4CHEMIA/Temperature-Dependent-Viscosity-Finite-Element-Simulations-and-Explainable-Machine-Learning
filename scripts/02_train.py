"""Step 2 - hyperparameter selection on the VALIDATION set only, then refit.

Protocol (per target, per model family)
  1. every grid configuration (x {raw features, + log10(Ra*Da)}) is fitted on
     `train` and scored on `val_interp` and `val_extrap`;
  2. selection score = 0.5 * MAPE(val_interp) + 0.5 * MAPE(val_extrap), so the
     chosen model must both interpolate and extrapolate well;
  3. the selected configuration is refitted on train + val (the test set is
     never touched here) and saved to models/;
  4. a deployment copy refitted on ALL unique runs is saved to models/deploy/
     (not evaluated - for downstream use only).

The test set is read only by 03_evaluate.py.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surrogate.config import MODELS, PROCESSED, RESULTS, TARGETS  # noqa: E402
from surrogate.features import (FAMILIES, FINAL_MLP_MEMBERS, GRIDS,  # noqa: E402
                                build_model, make_pipeline, predict_log_mean_std)
from surrogate.metrics import metrics  # noqa: E402

warnings.filterwarnings("ignore")


def load():
    d = pd.read_csv(PROCESSED / "dataset_clean.csv")
    d = d[~d.is_replicate].reset_index(drop=True)
    feats = json.loads((PROCESSED / "preprocessing_log.json").read_text())["features"]
    return d, feats


def main():
    d, F = load()
    tr = d[d.subset == "train"]
    vi = d[d.subset == "val_interp"]
    ve = d[d.subset == "val_extrap"]
    trval = d[d.subset.isin(["train", "val_interp", "val_extrap"])]
    (MODELS / "deploy").mkdir(parents=True, exist_ok=True)
    (RESULTS / "tuning").mkdir(parents=True, exist_ok=True)

    tuning_rows, selected = [], {}
    for target in TARGETS:
        for fam in FAMILIES:
            t0 = time.time()
            best = None
            for params in GRIDS[fam]:
                for phys in (False, True):
                    pipe = make_pipeline(build_model(fam, params, len(F) + phys), F, phys)
                    pipe.fit(tr[F], tr[target])
                    m_i = metrics(vi[target], pipe.predict(vi[F]))
                    m_e = metrics(ve[target], pipe.predict(ve[F]))
                    score = 0.5 * m_i["MAPE_%"] + 0.5 * m_e["MAPE_%"]
                    row = {"target": target, "family": fam, "params": json.dumps(params),
                           "darcy_rayleigh_feature": phys, "score": score,
                           "val_interp_MAPE_%": m_i["MAPE_%"], "val_extrap_MAPE_%": m_e["MAPE_%"],
                           "val_interp_R2": m_i["R2"], "val_extrap_R2": m_e["R2"]}
                    tuning_rows.append(row)
                    if best is None or score < best["score"]:
                        best = row
            selected[(target, fam)] = best
            print(f"{target:8s} {fam:10s} best score={best['score']:.4f}% "
                  f"(interp {best['val_interp_MAPE_%']:.4f}%, extrap {best['val_extrap_MAPE_%']:.4f}%) "
                  f"phys={best['darcy_rayleigh_feature']} {best['params']}  [{time.time()-t0:.0f}s]")

    tun = pd.DataFrame(tuning_rows)
    tun.to_csv(RESULTS / "tuning" / "validation_search_all_configs.csv", index=False)
    sel = pd.DataFrame(selected.values())
    sel.to_csv(RESULTS / "tuning" / "selected_hyperparameters.csv", index=False)

    # ---- refit selected configs on train+val (evaluated) and on all data (deploy)
    pred_rows = []
    for (target, fam), b in selected.items():
        params = json.loads(b["params"])
        if fam == "MLP":
            params["n_members"] = FINAL_MLP_MEMBERS
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        phys = bool(b["darcy_rayleigh_feature"])
        pipe = make_pipeline(build_model(fam, params, len(F) + phys), F, phys)
        pipe.fit(trval[F], trval[target])
        joblib.dump({"model": pipe, "features": F, "target": target, "family": fam,
                     "params": params, "darcy_rayleigh_feature": phys,
                     "fitted_on": "train+val"}, MODELS / f"{target}__{fam}.joblib", compress=3)
        dep = make_pipeline(build_model(fam, params, len(F) + phys), F, phys)
        dep.fit(d[F], d[target])
        joblib.dump({"model": dep, "features": F, "target": target, "family": fam,
                     "params": params, "darcy_rayleigh_feature": phys,
                     "fitted_on": "all unique runs (deployment; not evaluated)"},
                    MODELS / "deploy" / f"{target}__{fam}.joblib", compress=3)

        mu, sd = predict_log_mean_std(pipe, d[F])
        yhat = pipe.predict(d[F])
        out = d[["Run_ID", "sweep", "subset", "sweep_coord", "extrap_dist"] + F].copy()
        out["target"], out["model"] = target, fam
        out["y_true"], out["y_pred"] = d[target].values, yhat
        out["abs_rel_err_%"] = np.abs(yhat / d[target].values - 1) * 100
        out["log10_std"] = sd
        out["PI95_low"] = np.power(10.0, mu - 1.96 * sd)
        out["PI95_high"] = np.power(10.0, mu + 1.96 * sd)
        out["seen_in_training"] = d.subset.isin(["train", "val_interp", "val_extrap"]).values
        pred_rows.append(out)
        print(f"refit {target:8s} {fam:10s} done")

    preds = pd.concat(pred_rows, ignore_index=True)
    (RESULTS / "predictions").mkdir(parents=True, exist_ok=True)
    preds.to_csv(RESULTS / "predictions" / "predictions_all_models.csv", index=False)
    print("saved predictions and models")


if __name__ == "__main__":
    main()
