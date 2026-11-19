"""Step 6 - physically interpretable quantities derived from the CFD data and
the validated surrogates (GP = most accurate; MLP = smooth differentiable check).

  * local scaling exponents  n = d ln y / d ln Ra  (and Da)
  * elasticities at the base case  E_j = d ln y / d ln x_j
  * onset of convective enhancement and Darcy-Rayleigh (Ra*Da) similarity test
  * location of the |psi|_max peak in Ra and its minimum in A
  * coupling between heat transfer and circulation strength along each sweep
  * validation ablation of the physics feature log10(Ra*Da)
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surrogate import plotstyle as ps  # noqa: E402
from surrogate.config import (FIG, LOG_INPUTS, MODELS, PROCESSED, RESULTS,  # noqa: E402
                              TARGET_LABEL, TARGETS)

warnings.filterwarnings("ignore")
plt = ps.plt
OUT = RESULTS / "physics"
SWEEP_COLOR = {"Ra": "#2a78d6", "Da": "#eb6834", "R": "#1baf7a", "A": "#eda100", "C": "#e87ba4"}


def main():
    ps.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    log = json.loads((PROCESSED / "preprocessing_log.json").read_text())
    F, base = log["features"], log["base_case"]
    D = pd.read_csv(PROCESSED / "dataset_clean.csv")
    D = D[~D.is_replicate]
    B = D[D.sweep == "base"].iloc[0]
    gp = {t: joblib.load(MODELS / f"{t}__GP.joblib")["model"] for t in TARGETS}
    mlp = {t: joblib.load(MODELS / f"{t}__MLP.joblib")["model"] for t in TARGETS}
    res = {}

    def sweep(p):
        s = pd.concat([D[D.sweep == p], D[D.sweep == "base"]]).sort_values(p)
        return s

    def grid_frame(p, values):
        X = pd.DataFrame([base] * len(values))
        X[p] = values
        return X[F]

    # ---------------------------------------------------------------- 1. exponents
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0))
    expo_rows = []
    for j, p in enumerate(["Ra", "Da"]):
        s = sweep(p)
        lx = np.log10(s[p].values)
        g = np.logspace(lx.min(), lx.max(), 400)
        for i, t in enumerate(TARGETS):
            ax = axes[i, j]
            n_cfd = np.gradient(np.log10(s[t].values), lx)
            yg = gp[t].predict(grid_frame(p, g))
            n_gp = np.gradient(np.log10(yg), np.log10(g))
            ym = mlp[t].predict(grid_frame(p, g))
            n_mlp = np.gradient(np.log10(ym), np.log10(g))
            ax.axhline(0, color=ps.INK2, lw=0.6)
            ax.plot(g, n_mlp, color=ps.MODEL_COLOR["MLP"], lw=1.2, label="MLP surrogate")
            ax.plot(g, n_gp, color=ps.MODEL_COLOR["GP"], lw=1.6, label="GP surrogate")
            ax.scatter(s[p], n_cfd, s=9, color=ps.INK2, lw=0, label="CFD (finite difference)",
                       zorder=3)
            ax.set_xscale("log")
            ax.set_xlabel(p)
            ax.set_ylabel(f"local exponent d ln {TARGET_LABEL[t]} / d ln {p}")
            k = np.argmax(n_gp)
            expo_rows.append({"target": t, "parameter": p, "max_local_exponent_GP": n_gp[k],
                              f"at_{p}": g[k], "exponent_at_min_GP": n_gp[0],
                              "exponent_at_max_GP": n_gp[-1]})
            for a, b in [(1e3, 1e4), (1e4, 1e5), (1e5, 1e6)] if p == "Ra" else \
                        [(1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2)]:
                m = (s[p] >= a * 0.999) & (s[p] <= b * 1.001)
                slope = np.polyfit(np.log10(s.loc[m, p]), np.log10(s.loc[m, t]), 1)[0]
                expo_rows.append({"target": t, "parameter": p, "decade": f"{a:g}-{b:g}",
                                  "power_law_exponent_CFD_fit": slope})
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig09_local_scaling_exponents.png")
    plt.close(fig)
    pd.DataFrame(expo_rows).to_csv(OUT / "scaling_exponents.csv", index=False)

    # -------------------------------------------------------------- 2. elasticities
    el = []
    for t in TARGETS:
        for p in F:
            s = sweep(p).reset_index(drop=True)
            i = s.index[s.Run_ID == B.Run_ID][0]
            x0, y0 = s.loc[i, p], s.loc[i, t]
            xm, xp = s.loc[i - 1, p], s.loc[i + 1, p]
            ym, yp = s.loc[i - 1, t], s.loc[i + 1, t]
            e_cfd = (np.log(yp) - np.log(ym)) / (np.log(xp) - np.log(xm))
            h = 0.005
            Xp, Xm = grid_frame(p, [x0 * (1 + h)]), grid_frame(p, [x0 * (1 - h)])
            e_gp = (np.log(gp[t].predict(Xp)[0]) - np.log(gp[t].predict(Xm)[0])) / \
                   (np.log(1 + h) - np.log(1 - h))
            e_mlp = (np.log(mlp[t].predict(Xp)[0]) - np.log(mlp[t].predict(Xm)[0])) / \
                    (np.log(1 + h) - np.log(1 - h))
            el.append({"target": t, "parameter": p, "base_value": x0,
                       "elasticity_CFD": e_cfd, "elasticity_GP": e_gp, "elasticity_MLP": e_mlp,
                       "pct_change_in_y_for_+10pct_x_CFD": (1.1 ** e_cfd - 1) * 100})
    EL = pd.DataFrame(el)
    EL.to_csv(OUT / "elasticities_at_base_case.csv", index=False)

    # ------------------------------------- 3. onset + Darcy-Rayleigh similarity test
    nu_c = D[D.sweep == "Ra"].sort_values("Ra").avgNu.iloc[0]
    onset = {}
    for p in ["Ra", "Da"]:
        s = sweep(p)
        thr = 1.05 * s.avgNu.min()
        x = np.interp(np.log10(thr), np.log10(s.avgNu.values), np.log10(s[p].values))
        onset[p] = {"Nu_conduction_limit": s.avgNu.min(), "onset_value_5pct": 10 ** x,
                    "onset_Ra_m": (10 ** x) * (base["Da"] if p == "Ra" else base["Ra"])}
    res["onset_of_convective_enhancement"] = onset
    sRa, sDa = sweep("Ra"), sweep("Da")
    ram_ra = np.log10(sRa.Ra * base["Da"]).values
    ram_da = np.log10(sDa.Da * base["Ra"]).values
    common = np.linspace(max(ram_ra.min(), ram_da.min()), min(ram_ra.max(), ram_da.max()), 300)
    ratio = {}
    for t in TARGETS:
        a = 10 ** np.interp(common, ram_ra, np.log10(sRa[t].values))
        b = 10 ** np.interp(common, ram_da, np.log10(sDa[t].values))
        ratio[t] = a / b
    cross = common[np.where(np.diff(np.sign(ratio["avgNu"] - 1)))[0]]
    res["darcy_rayleigh_similarity"] = {
        "Ra_m_range_compared": [10 ** common[0], 10 ** common[-1]],
        "Nu_ratio_Rasweep_over_Dasweep_min": float(ratio["avgNu"].min()),
        "Nu_ratio_Rasweep_over_Dasweep_max": float(ratio["avgNu"].max()),
        "Nu_ratio_at_Ra_m_10": float(np.interp(1, common, ratio["avgNu"])),
        "Nu_ratio_at_Ra_m_1000": float(ratio["avgNu"][-1]),
        "Nu_crossover_Ra_m": [float(10 ** c) for c in cross],
        "psi_ratio_min": float(ratio["psi_max"].min()), "psi_ratio_max": float(ratio["psi_max"].max())}

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))
    for ax, t in zip(axes[:2], TARGETS):
        ax.plot(10 ** ram_ra, sRa[t], color=SWEEP_COLOR["Ra"], marker="o", ms=2.5, lw=1.2,
                label=f"Ra sweep (Da = {base['Da']:g})")
        ax.plot(10 ** ram_da, sDa[t], color=SWEEP_COLOR["Da"], marker="o", ms=2.5, lw=1.2,
                label=f"Da sweep (Ra = {base['Ra']:g})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Darcy–Rayleigh number Ra·Da")
        ax.set_ylabel(TARGET_LABEL[t])
    axes[0].legend(fontsize=7)
    for t, ls in zip(TARGETS, ["-", "--"]):
        axes[2].plot(10 ** common, ratio[t], color=ps.INK, ls=ls, lw=1.3, label=TARGET_LABEL[t])
    axes[2].axhline(1, color=ps.INK2, lw=0.6)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Ra·Da")
    axes[2].set_ylabel("ratio: Ra sweep / Da sweep at equal Ra·Da")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_darcy_rayleigh_similarity.png")
    plt.close(fig)

    # -------------------------------------------------- 4. extrema of |psi|_max
    g = np.logspace(3, 6, 4000)
    yg = gp["psi_max"].predict(grid_frame("Ra", g))
    k = int(np.argmax(yg))
    s = sweep("Ra")
    kc = int(np.argmax(s.psi_max.values))
    nu_at_peak = gp["avgNu"].predict(grid_frame("Ra", [g[k]]))[0]
    gA = np.linspace(0.01, 0.10, 4000)
    yA = gp["psi_max"].predict(grid_frame("A", gA))
    kA = int(np.argmin(yA))
    res["psi_max_extrema"] = {
        "Ra_at_psi_peak_GP": float(g[k]), "psi_peak_GP": float(yg[k]),
        "Ra_at_psi_peak_CFD_grid": float(s.Ra.values[kc]), "psi_peak_CFD": float(s.psi_max.values[kc]),
        "Nu_at_psi_peak_GP": float(nu_at_peak),
        "psi_drop_from_peak_to_Ra1e6_%": float((1 - s.psi_max.values[-1] / s.psi_max.values[kc]) * 100),
        "Nu_gain_from_peak_to_Ra1e6_%": float((s.avgNu.values[-1] / nu_at_peak - 1) * 100),
        "A_at_psi_min_GP": float(gA[kA]), "psi_min_A_GP": float(yA[kA])}

    # ------------------------------------- 5. heat transfer vs circulation coupling
    cp = []
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for p in F:
        s = sweep(p)
        dnu = np.log(s.avgNu.values[-1] / s.avgNu.values[0])
        dps = np.log(s.psi_max.values[-1] / s.psi_max.values[0])
        cp.append({"sweep": p, "Nu_at_min": s.avgNu.values[0], "Nu_at_max": s.avgNu.values[-1],
                   "psi_at_min": s.psi_max.values[0], "psi_at_max": s.psi_max.values[-1],
                   "Nu_change_%": (np.exp(dnu) - 1) * 100, "psi_change_%": (np.exp(dps) - 1) * 100,
                   "spearman_Nu_psi": spearmanr(s.avgNu, s.psi_max)[0],
                   "dlnNu_over_dlnpsi_end_to_end": dnu / dps if abs(dps) > 1e-12 else np.nan})
        ax.plot(s.psi_max, s.avgNu, color=SWEEP_COLOR[p], lw=1.6, label=f"{p} sweep")
        ax.annotate("", xy=(s.psi_max.values[-1], s.avgNu.values[-1]),
                    xytext=(s.psi_max.values[-4], s.avgNu.values[-4]),
                    arrowprops=dict(arrowstyle="-|>", color=SWEEP_COLOR[p], lw=1.2))
    ax.scatter([B.psi_max], [B.avgNu], s=40, color=ps.INK, zorder=5, label="base case")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"{TARGET_LABEL['psi_max']} (circulation strength)")
    ax.set_ylabel(f"{TARGET_LABEL['avgNu']} (heat transfer)")
    ax.set_title("Heat transfer vs circulation along each sweep (arrow = increasing parameter)",
                 fontsize=8.5)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig11_heat_transfer_vs_circulation.png")
    plt.close(fig)
    CP = pd.DataFrame(cp)
    CP.to_csv(OUT / "heat_transfer_circulation_coupling.csv", index=False)

    # ---------------------------------- 6. physics-feature ablation (validation only)
    tun = pd.read_csv(RESULTS / "tuning" / "validation_search_all_configs.csv")
    abl = (tun.groupby(["target", "family", "darcy_rayleigh_feature"])
           [["score", "val_interp_MAPE_%", "val_extrap_MAPE_%"]].min().reset_index())
    abl.to_csv(OUT / "physics_feature_ablation_validation.csv", index=False)

    (OUT / "physics_summary.json").write_text(json.dumps(res, indent=2, default=float))
    pd.set_option("display.width", 220)
    print(json.dumps(res, indent=2, default=float))
    print(EL.round(4).to_string())
    print(pd.DataFrame(expo_rows).round(4).to_string())
    print(CP.round(4).to_string())
    print(abl.pivot_table(index=["target", "family"], columns="darcy_rayleigh_feature",
                          values=["val_interp_MAPE_%", "val_extrap_MAPE_%"]).round(4))
    del nu_c


if __name__ == "__main__":
    main()
