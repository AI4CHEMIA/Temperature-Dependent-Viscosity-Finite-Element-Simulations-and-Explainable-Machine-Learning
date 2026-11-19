"""Step 4 - extrapolation stress test (uses train+val ONLY; test stays isolated).

For each sweep p, the outer 25 % (by normalised sweep coordinate) of the
train+val points of p is removed at BOTH ends; the selected configuration of
each model family is refitted on what remains (all other sweeps intact) and
asked to predict the removed points. The absolute relative error is then
analysed as a function of the normalised extrapolation distance
    delta = distance beyond the retained range / full span of the sweep
(log10 coordinate for Ra and Da).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surrogate import plotstyle as ps  # noqa: E402
from surrogate.config import FIG, PROCESSED, RESULTS, TARGET_LABEL, TARGETS  # noqa: E402
from surrogate.features import build_model, make_pipeline  # noqa: E402

warnings.filterwarnings("ignore")
plt = ps.plt
FAMS = ["RF", "MLP", "GP", "PolyRidge"]
CUT = 0.25
BINS = [0, 0.05, 0.10, 0.15, 0.20, 0.25]


def main():
    ps.apply()
    d = pd.read_csv(PROCESSED / "dataset_clean.csv")
    d = d[~d.is_replicate]
    d = d[d.subset.isin(["train", "val_interp", "val_extrap"])].reset_index(drop=True)
    F = json.loads((PROCESSED / "preprocessing_log.json").read_text())["features"]
    sel = pd.read_csv(RESULTS / "tuning" / "selected_hyperparameters.csv")

    rows = []
    for t in TARGETS:
        for fam in FAMS:
            b = sel[(sel.target == t) & (sel.family == fam)].iloc[0]
            params = json.loads(b.params)
            if fam == "MLP":
                params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
                params["n_members"] = 5
            phys = bool(b.darcy_rayleigh_feature)
            for p in F:
                s = d[d.sweep == p]
                c = s.sweep_coord
                lo, hi = c.min() + CUT * (c.max() - c.min()), c.max() - CUT * (c.max() - c.min())
                held = s[(c < lo) | (c > hi)]
                train = d.drop(held.index)
                pipe = make_pipeline(build_model(fam, params, len(F) + phys), F, phys)
                pipe.fit(train[F], train[t])
                yhat = pipe.predict(held[F])
                kept = train[train.sweep == p].sweep_coord
                dist = np.where(held.sweep_coord < kept.min(), kept.min() - held.sweep_coord,
                                held.sweep_coord - kept.max())
                for rid, ytrue, yp, dd, side in zip(held.Run_ID, held[t], yhat, dist,
                                                    np.where(held.sweep_coord < kept.min(),
                                                             "low", "high")):
                    rows.append({"target": t, "model": fam, "sweep": p, "Run_ID": rid,
                                 "side": side, "extrap_dist": dd, "y_true": ytrue, "y_pred": yp,
                                 "APE_%": abs(yp / ytrue - 1) * 100})
            print(f"{t} {fam} done")

    R = pd.DataFrame(rows)
    (RESULTS / "metrics").mkdir(exist_ok=True, parents=True)
    R.to_csv(RESULTS / "predictions" / "extrapolation_stress_test_predictions.csv", index=False)
    R["dist_bin"] = pd.cut(R.extrap_dist, BINS, include_lowest=True)
    summ = (R.groupby(["target", "model", "dist_bin"], observed=True)["APE_%"]
            .agg(["mean", "median", "max", "count"]).reset_index())
    summ.to_csv(RESULTS / "metrics" / "extrapolation_stress_test_by_distance.csv", index=False)
    by_sweep = (R.groupby(["target", "model", "sweep"])["APE_%"].agg(["mean", "max"])
                .reset_index())
    by_sweep.to_csv(RESULTS / "metrics" / "extrapolation_stress_test_by_sweep.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.pivot_table(index=["target", "model"], columns="dist_bin", values="mean",
                           observed=True).round(3))
    print(by_sweep.pivot_table(index=["target", "model"], columns="sweep", values="mean").round(3))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    mids = [(a + b) / 2 * 100 for a, b in zip(BINS[:-1], BINS[1:])]
    for ax, t in zip(axes, TARGETS):
        for fam in FAMS:
            g = summ[(summ.target == t) & (summ.model == fam)]
            ax.plot(mids[:len(g)], g["mean"], marker="o", ms=4.5, color=ps.MODEL_COLOR[fam],
                    label=ps.MODEL_LABEL[fam])
        ax.set_yscale("log")
        ax.set_xlabel("Extrapolation distance beyond training range (% of sweep span)")
        ax.set_ylabel("Mean absolute relative error (%)")
        ax.set_title(f"{TARGET_LABEL[t]} — stress test (outer 25 % of each sweep removed)")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig05_extrapolation_stress_test.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
