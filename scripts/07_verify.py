"""Step 7 - independent verification of the delivered results.

Checks (each prints PASS/FAIL; exit code 1 if any check fails):
  V1  raw data checksum matches the one recorded at preprocessing
  V2  split integrity: every unique run in exactly one subset, no duplicate
      inputs shared between training (train+val) and test, base replicates excluded
  V3  every test-extrapolation point lies OUTSIDE the train+val range of its sweep
  V4  hyperparameters were selected without test data (selection table has no test columns)
  V5  reloaded models reproduce the saved predictions (max rel. diff < 1e-9)
  V6  test metrics recomputed from the reloaded models match results/metrics/test_metrics.csv
  V7  SHAP additivity: sum(phi) + base value == log10(prediction) for every run
      (tolerance 1e-6 in log10, i.e. 2e-4 % - GP kernel solves carry ~1e-8 round-off)
  V8  SHAP faithfulness: base-reference SHAP of the varied input equals log10 f(x)/f(base)
  V9  deployment models (refitted on all runs) load and give finite predictions; the
      accurate surrogates (GP, MLP) reproduce every CFD run within 1 %

Usage:  python scripts/07_verify.py
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surrogate.config import MODELS, PROCESSED, RAW_XLSX, RESULTS, TARGETS  # noqa: E402
from surrogate.features import FAMILIES  # noqa: E402
from surrogate.metrics import metrics  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def main():
    log = json.loads((PROCESSED / "preprocessing_log.json").read_text())
    F = log["features"]
    sha = hashlib.sha256(RAW_XLSX.read_bytes()).hexdigest()
    check("V1 raw data checksum", sha == log["raw_sha256"], sha[:16] + "...")

    D = pd.read_csv(PROCESSED / "dataset_clean.csv")
    U = D[~D.is_replicate]
    ok = U.Run_ID.is_unique and U.subset.isin(
        ["train", "val_interp", "val_extrap", "test_interp", "test_extrap"]).all()
    seen = U[~U.subset.str.startswith("test")]
    test = U[U.subset.str.startswith("test")]
    key = lambda df: set(map(tuple, np.round(df[F].values, 12)))  # noqa: E731
    shared = key(seen) & key(test)
    check("V2 split integrity / no input shared between train+val and test",
          ok and not shared and (D[D.is_replicate].subset == "excluded_replicate").all(),
          f"train={sum(U.subset=='train')}, val={seen.shape[0]-sum(U.subset=='train')}, "
          f"test={len(test)}, shared={len(shared)}")

    bad = 0
    for p in F:
        s = U[U.sweep == p]
        rng = s[~s.subset.str.startswith("test")][p]
        te = s[s.subset == "test_extrap"][p]
        bad += int(((te >= rng.min()) & (te <= rng.max())).sum())
    check("V3 extrapolation test points outside train+val range", bad == 0, f"violations={bad}")

    sel = pd.read_csv(RESULTS / "tuning" / "selected_hyperparameters.csv")
    check("V4 selection used validation only", not any("test" in c for c in sel.columns),
          "selection columns: " + ", ".join(c for c in sel.columns if "val" in c))

    P = pd.read_csv(RESULTS / "predictions" / "predictions_all_models.csv")
    M = pd.read_csv(RESULTS / "metrics" / "test_metrics.csv")
    worst, worst_m = 0.0, 0.0
    for t in TARGETS:
        for fam in FAMILIES:
            obj = joblib.load(MODELS / f"{t}__{fam}.joblib")
            mdl = obj["model"]
            g = P[(P.target == t) & (P.model == fam)].set_index("Run_ID").loc[U.Run_ID]
            yhat = mdl.predict(U[F])
            worst = max(worst, np.max(np.abs(yhat / g.y_pred.values - 1)))
            for sub, subs in [("test_interp", ["test_interp"]), ("test_extrap", ["test_extrap"]),
                              ("test_all", ["test_interp", "test_extrap"])]:
                k = U.subset.isin(subs).values
                m = metrics(U[t].values[k], yhat[k])
                ref = M[(M.target == t) & (M.model == fam) & (M.subset == sub)].iloc[0]
                worst_m = max(worst_m, abs(m["MAPE_%"] - ref["MAPE_%"]), abs(m["R2"] - ref["R2"]))
    check("V5 reloaded models reproduce saved predictions", worst < 1e-9, f"max rel diff={worst:.2e}")
    check("V6 recomputed test metrics match reported metrics", worst_m < 1e-9,
          f"max abs diff={worst_m:.2e}")

    worst_add, worst_faith = 0.0, 0.0
    for t in TARGETS:
        for fam in ["RF", "MLP", "GP"]:
            S = pd.read_csv(RESULTS / "shap" / f"shap_basecase_reference__{t}__{fam}.csv")
            mdl = joblib.load(MODELS / f"{t}__{fam}.joblib")["model"]
            flog = np.log10(mdl.predict(S[F]))
            phi = S[[f"phi_{p}" for p in F]].values
            worst_add = max(worst_add, np.abs(phi.sum(1) + S.base_value_log10 - flog).max())
            fb = np.log10(mdl.predict(pd.DataFrame([log["base_case"]])[F]))[0]
            for i, sw in enumerate(S.sweep):
                if sw in F:
                    worst_faith = max(worst_faith, abs(phi[i, F.index(sw)] - (flog[i] - fb)))
    check("V7 SHAP additivity (log10 space)", worst_add < 1e-6, f"max |error|={worst_add:.2e}")
    check("V8 SHAP base-reference = exact OAT main effect", worst_faith < 1e-6,
          f"max |error|={worst_faith:.2e}")

    finite, worst_dep = True, 0.0
    for t in TARGETS:
        for fam in FAMILIES:
            mdl = joblib.load(MODELS / "deploy" / f"{t}__{fam}.joblib")["model"]
            yhat = mdl.predict(U[F])
            finite &= bool(np.isfinite(yhat).all())
            if fam in ("GP", "MLP"):
                worst_dep = max(worst_dep, np.max(np.abs(yhat / U[t].values - 1)) * 100)
    check("V9 deployment models load; GP/MLP reproduce all CFD runs", finite and worst_dep < 1.0,
          f"max in-sample APE (GP, MLP) = {worst_dep:.3f} %")

    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
