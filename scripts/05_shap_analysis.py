"""Step 5 - SHAP interpretability (exact Shapley values; 5 inputs -> 32 coalitions).

Explained quantity: log10(y_hat). SHAP is additive in this space, so in physical
units each feature acts as a MULTIPLICATIVE factor 10**phi_j on the reference
value - a natural reading for power-law-like heat-transfer correlations.

Two complementary analyses
--------------------------
(A) Base-case reference, explained on the CFD runs (on-manifold).
    With the base case as the single reference point, every coalition that the
    exact Shapley computation evaluates for a one-at-a-time run is either the
    run itself or the base case, i.e. a point that exists in the CFD data.
    Hence phi_j(x) = log10 f(x) - log10 f(x_base) for the varied parameter and
    0 for the others - an exactly identified main effect that can be checked
    against the CFD data (faithfulness test).

(B) Design-space (Sobol) background, explained on a Sobol sample of the full
    5-D box (off-manifold). This is the conventional global SHAP view of the
    learned response surface, but combinations of several non-base parameters
    were never simulated, so interaction structure there reflects each model's
    inductive bias, not physics. We quantify this with (i) cross-model agreement
    and (ii) how well an additive (OAT) decomposition reproduces each model.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau, qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surrogate import plotstyle as ps  # noqa: E402
from surrogate.config import (FIG, LOG_INPUTS, MODELS, PROCESSED, RESULTS,  # noqa: E402
                              SEED, TARGET_LABEL, TARGETS)

warnings.filterwarnings("ignore")
plt = ps.plt
EXPLAINED = ["RF", "MLP", "GP"]
SHAPDIR = RESULTS / "shap"


def f_log(model, F):
    def f(X):
        return np.log10(model.predict(pd.DataFrame(np.asarray(X, float), columns=F)))
    return f


def sobol_box(F, ranges, n, seed):
    s = qmc.Sobol(len(F), scramble=True, seed=seed).random(n)
    X = np.empty_like(s)
    for j, p in enumerate(F):
        lo, hi = ranges[p]["min"], ranges[p]["max"]
        X[:, j] = (10 ** (np.log10(lo) + s[:, j] * (np.log10(hi) - np.log10(lo)))
                   if p in LOG_INPUTS else lo + s[:, j] * (hi - lo))
    return X


def main():
    ps.apply()
    SHAPDIR.mkdir(parents=True, exist_ok=True)
    log = json.loads((PROCESSED / "preprocessing_log.json").read_text())
    F, base, ranges = log["features"], log["base_case"], log["sweep_ranges"]
    D = pd.read_csv(PROCESSED / "dataset_clean.csv")
    D = D[~D.is_replicate].reset_index(drop=True)
    xb = np.array([[base[p] for p in F]])
    Xd = D[F].values
    y_base = D.loc[D.sweep == "base", TARGETS].iloc[0]

    Xbg = sobol_box(F, ranges, 64, SEED)          # background for (B)
    Xev = sobol_box(F, ranges, 512, SEED + 1)     # explained points for (B)
    Xadd = sobol_box(F, ranges, 2048, SEED + 2)   # additivity / agreement test

    imp_rows, add_rows, faith_rows = [], [], []
    store_A, store_B, preds_off = {}, {}, {}
    for t in TARGETS:
        cfd_ref = np.log10(D[t] / y_base[t]).values      # CFD log-ratio to base case
        for m in EXPLAINED:
            mdl = joblib.load(MODELS / f"{t}__{m}.joblib")["model"]
            f = f_log(mdl, F)

            # ---------------- (A) base-case reference, on CFD runs
            exA = shap.explainers.Exact(f, shap.maskers.Independent(xb, max_samples=1))
            sA = exA(Xd, silent=True)
            phiA = sA.values
            assert np.allclose(phiA.sum(1) + sA.base_values, f(Xd), atol=1e-8)  # additivity
            store_A[(t, m)] = phiA
            dfA = D[["Run_ID", "sweep", "subset"] + F].copy()
            for j, p in enumerate(F):
                dfA[f"phi_{p}"] = phiA[:, j]
            dfA["base_value_log10"] = sA.base_values
            dfA["f_log10"] = f(Xd)
            dfA["cfd_log10_ratio_to_base"] = cfd_ref
            dfA.to_csv(SHAPDIR / f"shap_basecase_reference__{t}__{m}.csv", index=False)
            # faithfulness: SHAP main effect vs CFD log-ratio (held-out test runs only)
            te = D.subset.str.startswith("test").values
            tot = phiA.sum(1)
            for sub in ["test_interp", "test_extrap"]:
                k = (D.subset == sub).values
                res = tot[k] - cfd_ref[k]
                faith_rows.append({"target": t, "model": m, "subset": sub,
                                   "R2": 1 - (res ** 2).sum() / ((cfd_ref[k] - cfd_ref[k].mean()) ** 2).sum(),
                                   "max_abs_err_log10": np.abs(res).max(),
                                   "max_factor_err_%": (10 ** np.abs(res).max() - 1) * 100})
            off_axis = np.abs(phiA[np.arange(len(D)), :]).copy()
            for i, sw in enumerate(D.sweep):
                if sw in F:
                    off_axis[i, F.index(sw)] = 0
            assert off_axis.max() < 1e-9, "non-varied features received attribution"
            del te

            # ---------------- (B) Sobol background, design-space sample
            exB = shap.explainers.Exact(f, shap.maskers.Independent(Xbg, max_samples=len(Xbg)))
            sB = exB(Xev, silent=True)
            store_B[(t, m)] = sB
            pd.DataFrame(sB.values, columns=[f"phi_{p}" for p in F]).assign(
                **{p: Xev[:, j] for j, p in enumerate(F)},
                base_value_log10=sB.base_values).to_csv(
                SHAPDIR / f"shap_sobol_designspace__{t}__{m}.csv", index=False)

            # additive (OAT) decomposition of the model on the 5-D box
            fl = f(Xadd)
            fb = f(xb)[0]
            g = np.full(len(Xadd), fb)
            for j in range(len(F)):
                Xj = np.repeat(xb, len(Xadd), 0)
                Xj[:, j] = Xadd[:, j]
                g += f(Xj) - fb
            preds_off[(t, m)] = fl
            add_rows.append({"target": t, "model": m,
                             "R2_additive_vs_model": 1 - ((fl - g) ** 2).sum() / ((fl - fl.mean()) ** 2).sum(),
                             "median_abs_interaction_log10": np.median(np.abs(fl - g)),
                             "p95_abs_interaction_factor_%": (10 ** np.percentile(np.abs(fl - g), 95) - 1) * 100})

            # ---------------- importance summaries
            for j, p in enumerate(F):
                k = (D.sweep == p).values
                imp_rows.append({"target": t, "source": m, "feature": p,
                                 "onmanifold_mean_abs_phi": np.abs(phiA[k, j]).mean(),
                                 "onmanifold_span_log10": phiA[k, j].max() - phiA[k, j].min(),
                                 "designspace_mean_abs_phi": np.abs(sB.values[:, j]).mean()})
            print(f"SHAP {t} {m} done")
        # CFD-direct "ground truth" main effects (no model)
        for p in F:
            k = (D.sweep == p).values
            imp_rows.append({"target": t, "source": "CFD", "feature": p,
                             "onmanifold_mean_abs_phi": np.abs(cfd_ref[k]).mean(),
                             "onmanifold_span_log10": cfd_ref[k].max() - cfd_ref[k].min(),
                             "designspace_mean_abs_phi": np.nan})

    IMP = pd.DataFrame(imp_rows)
    IMP["span_factor"] = 10 ** IMP.onmanifold_span_log10
    IMP.to_csv(SHAPDIR / "importance_summary.csv", index=False)
    ADD = pd.DataFrame(add_rows)
    # cross-model disagreement off-manifold (where no CFD exists)
    for t in TARGETS:
        for a, b in [("MLP", "GP"), ("RF", "GP"), ("RF", "MLP")]:
            dlt = np.abs(preds_off[(t, a)] - preds_off[(t, b)])
            ADD.loc[len(ADD)] = {"target": t, "model": f"{a} vs {b}",
                                 "R2_additive_vs_model": np.nan,
                                 "median_abs_interaction_log10": np.median(dlt),
                                 "p95_abs_interaction_factor_%": (10 ** np.percentile(dlt, 95) - 1) * 100}
    ADD.to_csv(SHAPDIR / "additivity_and_offmanifold_agreement.csv", index=False)
    FA = pd.DataFrame(faith_rows)
    FA.to_csv(SHAPDIR / "shap_faithfulness_vs_cfd.csv", index=False)

    # rank agreement of the on-manifold importance
    rk = []
    for t in TARGETS:
        piv = IMP[IMP.target == t].pivot(index="feature", columns="source",
                                         values="onmanifold_mean_abs_phi")
        for m in EXPLAINED:
            rk.append({"target": t, "model": m, "kendall_tau_vs_CFD": kendalltau(piv[m], piv["CFD"])[0],
                       "ranking": " > ".join(piv[m].sort_values(ascending=False).index)})
        rk.append({"target": t, "model": "CFD", "kendall_tau_vs_CFD": 1.0,
                   "ranking": " > ".join(piv["CFD"].sort_values(ascending=False).index)})
        pivB = IMP[(IMP.target == t) & (IMP.source != "CFD")].pivot(
            index="feature", columns="source", values="designspace_mean_abs_phi")
        for m in EXPLAINED:
            rk.append({"target": t, "model": f"{m} (design-space SHAP)",
                       "kendall_tau_vs_CFD": kendalltau(pivB[m], piv["CFD"])[0],
                       "ranking": " > ".join(pivB[m].sort_values(ascending=False).index)})
    RK = pd.DataFrame(rk)
    RK.to_csv(SHAPDIR / "importance_rank_agreement.csv", index=False)
    pd.set_option("display.width", 220)
    print(IMP.pivot_table(index=["target", "feature"], columns="source",
                          values="onmanifold_mean_abs_phi").round(4))
    print(RK.to_string())
    print(ADD.round(4).to_string())
    print(FA.round(5).to_string())

    # ======================= figures
    # Fig 6: SHAP main effects (base reference) vs CFD
    fig, axes = plt.subplots(2, len(F), figsize=(3.0 * len(F), 5.6))
    for i, t in enumerate(TARGETS):
        cfd_ref = np.log10(D[t] / y_base[t]).values
        for j, p in enumerate(F):
            ax = axes[i, j]
            k = ((D.sweep == p) | (D.sweep == "base")).values
            o = np.argsort(D.loc[k, p].values)
            xv = D.loc[k, p].values[o]
            ax.axhline(0, color=ps.INK2, lw=0.6)
            ax.scatter(xv, cfd_ref[k][o], s=10, color=ps.DATA_GREY, lw=0, label="CFD (log ratio to base)",
                       zorder=2)
            for m in ["RF", "MLP"]:
                ax.plot(xv, store_A[(t, m)][k, j][o], color=ps.MODEL_COLOR[m], lw=1.4,
                        label=f"SHAP — {ps.MODEL_LABEL[m]}", zorder=3)
            if p in LOG_INPUTS:
                ax.set_xscale("log")
            ax.set_xlabel(p)
            if j == 0:
                ax.set_ylabel(f"φ for {TARGET_LABEL[t]}  [log$_{{10}}$ factor]")
            sec = ax.secondary_yaxis("right", functions=(lambda v: 10 ** v, lambda v: np.log10(np.clip(v, 1e-9, None))))
            sec.tick_params(labelsize=7, colors=ps.INK2)
            if j == len(F) - 1:
                sec.set_ylabel("multiplicative factor on base", fontsize=7.5, color=ps.INK2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig06_shap_main_effects_basecase.png")
    plt.close(fig)

    # Fig 7: importance (on-manifold mean |phi|) per feature, CFD vs models
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    srcs = ["CFD", "RF", "MLP", "GP"]
    col = {"CFD": ps.INK2, **ps.MODEL_COLOR}
    for ax, t in zip(axes, TARGETS):
        piv = IMP[IMP.target == t].pivot(index="feature", columns="source",
                                         values="onmanifold_mean_abs_phi")
        order = piv["CFD"].sort_values(ascending=False).index
        x = np.arange(len(order))
        for k, s in enumerate(srcs):
            ax.bar(x + (k - 1.5) * 0.2, piv.loc[order, s], width=0.19, color=col[s],
                   label="CFD data (direct)" if s == "CFD" else ps.MODEL_LABEL[s],
                   edgecolor="white", lw=0.8)
        ax.set_xticks(x, order)
        ax.set_ylabel("mean |φ|  [log$_{10}$ units]")
        ax.set_title(f"{TARGET_LABEL[t]}: global importance (base-case reference)")
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig07_shap_global_importance.png")
    plt.close(fig)

    # Fig 8: conventional beeswarm (design-space SHAP), MLP and RF
    for m in ["MLP", "RF"]:
        for t in TARGETS:
            sB = store_B[(t, m)]
            ex = shap.Explanation(values=sB.values, base_values=sB.base_values,
                                  data=np.column_stack([np.log10(Xev[:, j]) if p in LOG_INPUTS
                                                        else Xev[:, j] for j, p in enumerate(F)]),
                                  feature_names=[f"log10 {p}" if p in LOG_INPUTS else p for p in F])
            plt.figure(figsize=(5.2, 3.2))
            shap.plots.beeswarm(ex, show=False, color_bar=True, s=8)
            ax = plt.gca()
            ax.set_xlabel(f"SHAP value on log$_{{10}}$ {TARGET_LABEL[t]}")
            ax.set_title(f"{ps.MODEL_LABEL[m]} — design-space SHAP for {TARGET_LABEL[t]}", fontsize=9)
            plt.savefig(FIG / f"fig08_shap_beeswarm_designspace__{t}__{m}.png")
            plt.close("all")
    print("SHAP figures written")


if __name__ == "__main__":
    main()
