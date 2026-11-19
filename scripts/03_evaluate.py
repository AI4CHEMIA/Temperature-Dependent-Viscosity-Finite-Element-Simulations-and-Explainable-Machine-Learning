"""Step 3 - single evaluation of the frozen models on the isolated test set.

Reports, per target and model:
  * test_interp  - interior points of every sweep never seen in training
  * test_extrap  - outermost 7 % of every sweep, OUTSIDE the train+val range
  * test_all     - both
with 95 % percentile-bootstrap confidence intervals, a per-sweep breakdown,
and the in-sample (train+val) fit for an over-fitting check.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surrogate import plotstyle as ps  # noqa: E402
from surrogate.config import (FIG, LOG_INPUTS, MODELS, PROCESSED,  # noqa: E402
                              RESULTS, TARGET_LABEL, TARGETS)
from surrogate.features import FAMILIES, predict_log_mean_std  # noqa: E402
from surrogate.metrics import bootstrap_ci, metrics  # noqa: E402

plt = ps.plt
SUBSETS = {"test_interp": ["test_interp"], "test_extrap": ["test_extrap"],
           "test_all": ["test_interp", "test_extrap"],
           "fit_trainval": ["train", "val_interp", "val_extrap"]}


def metric_tables(P):
    rows, by_sweep = [], []
    for (t, m), g in P.groupby(["target", "model"]):
        for name, subs in SUBSETS.items():
            s = g[g.subset.isin(subs)]
            r = {"target": t, "model": m, "subset": name, **metrics(s.y_true, s.y_pred)}
            if name.startswith("test"):
                ci = bootstrap_ci(s.y_true.values, s.y_pred.values, seed=7)
                for k, (lo, hi) in ci.items():
                    r[k.replace("_CI", "_CI95_low")], r[k.replace("_CI", "_CI95_high")] = lo, hi
            rows.append(r)
        for (sw, sub), s in g[g.subset.str.startswith("test")].groupby(["sweep", "subset"]):
            by_sweep.append({"target": t, "model": m, "sweep": sw, "subset": sub,
                             **metrics(s.y_true, s.y_pred)})
    return pd.DataFrame(rows), pd.DataFrame(by_sweep)


def fig_parity(P):
    models = ["RF", "MLP", "GP"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
    for i, t in enumerate(TARGETS):
        for j, m in enumerate(models):
            ax = axes[i, j]
            g = P[(P.target == t) & (P.model == m)]
            tr = g[g.seen_in_training]
            ax.scatter(tr.y_true, tr.y_pred, s=8, color=ps.DATA_GREY, alpha=0.35, lw=0,
                       label="train + val (fitted)")
            for sub, mk, lab in [("test_interp", "o", "test - interpolation"),
                                 ("test_extrap", "D", "test - extrapolation")]:
                s = g[g.subset == sub]
                ax.scatter(s.y_true, s.y_pred, s=26, marker=mk,
                           facecolor=ps.MODEL_COLOR[m] if sub == "test_extrap" else "white",
                           edgecolor=ps.MODEL_COLOR[m], lw=1.1, label=lab, zorder=3)
            lo, hi = g[["y_true", "y_pred"]].min().min(), g[["y_true", "y_pred"]].max().max()
            ax.plot([lo, hi], [lo, hi], color=ps.INK2, lw=0.8, ls="--", zorder=1)
            ax.set_xscale("log"); ax.set_yscale("log")
            te = g[g.subset.str.startswith("test")]
            mi = metrics(g[g.subset == "test_interp"].y_true, g[g.subset == "test_interp"].y_pred)
            me = metrics(g[g.subset == "test_extrap"].y_true, g[g.subset == "test_extrap"].y_pred)
            ax.set_title(f"{ps.MODEL_LABEL[m]} — {TARGET_LABEL[t]}\n"
                         f"test MAPE: interp {mi['MAPE_%']:.3f}%, extrap {me['MAPE_%']:.3f}%",
                         fontsize=8.5)
            ax.set_xlabel(f"CFD {TARGET_LABEL[t]}")
            ax.set_ylabel(f"Predicted {TARGET_LABEL[t]}")
            if i == 0 and j == 0:
                ax.legend(loc="upper left", fontsize=7)
            del te
    fig.tight_layout()
    fig.savefig(FIG / "fig02_parity_test.png")
    plt.close(fig)


def fig_sweeps(D, F, base):
    """Model response along each sweep vs CFD, including the extrapolation zones."""
    loaded = {(t, m): joblib.load(MODELS / f"{t}__{m}.joblib")["model"]
              for t in TARGETS for m in ["RF", "MLP", "GP"]}
    for t in TARGETS:
        fig, axes = plt.subplots(1, len(F), figsize=(3.0 * len(F), 3.1), sharey=False)
        for ax, p in zip(axes, F):
            s = D[(D.sweep == p) | (D.sweep == "base")]
            lo, hi = s[p].min(), s[p].max()
            if p in LOG_INPUTS:
                grid = np.logspace(np.log10(lo) - 0.15, np.log10(hi) + 0.15, 300)
            else:
                pad = 0.08 * (hi - lo)
                grid = np.linspace(max(lo - pad, 0) if lo >= 0 else lo - pad, hi + pad, 300)
            X = pd.DataFrame([base] * len(grid))
            X[p] = grid
            # shade the test-extrapolation zones (outside the train+val range)
            seen = s[s.subset.isin(["train", "val_interp", "val_extrap"])][p]
            for a, b in [(grid.min(), seen.min()), (seen.max(), grid.max())]:
                ax.axvspan(a, b, color="#f1efe8", zorder=0, lw=0)
            for m in ["RF", "GP", "MLP"]:
                mdl = loaded[(t, m)]
                y = mdl.predict(X[F])
                if m == "MLP":
                    mu, sd = predict_log_mean_std(mdl, X[F])
                    ax.fill_between(grid, 10 ** (mu - 1.96 * sd), 10 ** (mu + 1.96 * sd),
                                    color=ps.MODEL_COLOR[m], alpha=0.15, lw=0)
                ax.plot(grid, y, color=ps.MODEL_COLOR[m], lw=1.3, label=ps.MODEL_LABEL[m],
                        zorder=2)
            tr = s[s.subset.isin(["train", "val_interp", "val_extrap"])]
            ax.scatter(tr[p], tr[t], s=9, color=ps.DATA_GREY, lw=0, zorder=3,
                       label="CFD (train+val)")
            for sub, mk, fc in [("test_interp", "o", "white"), ("test_extrap", "D", ps.INK)]:
                q = s[s.subset == sub]
                ax.scatter(q[p], q[t], s=20, marker=mk, facecolor=fc, edgecolor=ps.INK,
                           lw=0.9, zorder=4, label=f"CFD test ({sub.split('_')[1]})")
            if p in LOG_INPUTS:
                ax.set_xscale("log")
            ax.set_xlabel(ps.FEAT_LABEL[p])
            ax.set_title(f"{p} sweep", fontsize=9)
        axes[0].set_ylabel(TARGET_LABEL[t])
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06), fontsize=7.5)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(FIG / f"fig03_sweep_response_{t}.png")
        plt.close(fig)


def fig_error_bars(M):
    order = ["GP", "MLP", "PolyRidge", "ExtraTrees", "GBM", "RF"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for ax, t in zip(axes, TARGETS):
        g = M[(M.target == t)]
        x = np.arange(len(order))
        for k, (sub, off, hatch) in enumerate([("test_interp", -0.19, None),
                                               ("test_extrap", 0.19, None)]):
            v = [g[(g.model == m) & (g.subset == sub)]["MAPE_%"].iloc[0] for m in order]
            lo = [g[(g.model == m) & (g.subset == sub)]["MAPE_%_CI95_low"].iloc[0] for m in order]
            hi = [g[(g.model == m) & (g.subset == sub)]["MAPE_%_CI95_high"].iloc[0] for m in order]
            cols = [ps.MODEL_COLOR[m] for m in order]
            ax.bar(x + off, v, width=0.36, color=cols, alpha=0.45 if k == 0 else 1.0,
                   edgecolor="white", lw=1)
            ax.errorbar(x + off, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt="none",
                        ecolor=ps.INK2, lw=0.8, capsize=2)
        ax.set_yscale("log")
        ax.set_xticks(x, [ps.MODEL_LABEL[m].replace(" (deep ensemble)", "") for m in order],
                      rotation=25, ha="right")
        ax.set_ylabel("Test MAPE (%) — log scale")
        ax.set_title(f"{TARGET_LABEL[t]}: light = interpolation, solid = extrapolation")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig04_test_mape_by_model.png")
    plt.close(fig)


def main():
    ps.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics").mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(RESULTS / "predictions" / "predictions_all_models.csv")
    D = pd.read_csv(PROCESSED / "dataset_clean.csv")
    D = D[~D.is_replicate]
    log = json.loads((PROCESSED / "preprocessing_log.json").read_text())
    F, base = log["features"], log["base_case"]

    M, S = metric_tables(P)
    M.to_csv(RESULTS / "metrics" / "test_metrics.csv", index=False)
    S.to_csv(RESULTS / "metrics" / "test_metrics_by_sweep.csv", index=False)
    show = M[M.subset != "fit_trainval"].pivot_table(index=["target", "model"], columns="subset",
                                                     values=["MAPE_%", "R2", "MaxAPE_%"])
    pd.set_option("display.width", 220)
    print(show.round(4).to_string())
    print(M[M.subset == "fit_trainval"][["target", "model", "MAPE_%", "R2"]].round(4).to_string())

    # calibration of the 95 % predictive intervals on the test set
    cov = []
    for (t, m), g in P[P.subset.str.startswith("test")].groupby(["target", "model"]):
        if g.log10_std.isna().all():
            continue
        for sub, s in list(g.groupby("subset")) + [("test_all", g)]:
            inside = (s.y_true >= s.PI95_low) & (s.y_true <= s.PI95_high)
            cov.append({"target": t, "model": m, "subset": sub, "PI95_coverage_%": inside.mean() * 100,
                        "median_PI95_halfwidth_%": float(np.median((s.PI95_high - s.PI95_low) / 2 / s.y_pred * 100))})
    pd.DataFrame(cov).to_csv(RESULTS / "metrics" / "test_PI95_coverage.csv", index=False)
    print(pd.DataFrame(cov).round(2).to_string())

    # paper-friendly wide table of test-set predictions (all models side by side)
    W = P[P.subset.str.startswith("test")].pivot_table(
        index=["target", "Run_ID", "sweep", "subset", "extrap_dist"] + F,
        columns="model", values="y_pred").reset_index()
    yt = P[P.subset.str.startswith("test")].drop_duplicates(["target", "Run_ID"])[
        ["target", "Run_ID", "y_true"]]
    W = W.merge(yt, on=["target", "Run_ID"]).sort_values(["target", "sweep", "Run_ID"])
    W.to_csv(RESULTS / "predictions" / "predictions_test_set_wide.csv", index=False)

    fig_parity(P)
    fig_sweeps(D, F, base)
    fig_error_bars(M)
    print("figures written")


if __name__ == "__main__":
    main()
