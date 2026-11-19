"""Step 1 - clean the raw sheet, identify the one-at-a-time sweeps and build
the sweep-aware train / validation / test partition.

Outputs
-------
data/processed/dataset_clean.csv      all runs, with sweep label and subset
data/processed/split_summary.csv      counts per sweep and subset
data/processed/preprocessing_log.json what was dropped and why
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surrogate.config import (ALL_INPUTS, EXT_TEST_FRAC, EXT_VAL_FRAC,  # noqa: E402
                              INTERP_PERIOD, INTERP_TEST_PHASE, INTERP_VAL_PHASE,
                              LOG_INPUTS, OUTPUTS, PROCESSED, RAW_XLSX, REL_TOL)


def nround(x):  # round half up (Python's round() is banker's rounding)
    return int(np.floor(x + 0.5))


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(RAW_XLSX, header=1)
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.dropna(how="all")
    df = raw[["Run_ID"] + ALL_INPUTS + list(OUTPUTS)].rename(columns=OUTPUTS).copy()
    df["Run_ID"] = df["Run_ID"].astype(int)
    log = {"raw_file": RAW_XLSX.name,
           "raw_sha256": hashlib.sha256(RAW_XLSX.read_bytes()).hexdigest(),
           "n_raw_runs": int(len(df))}

    # ---- 1. constant inputs carry no information for a data-driven model
    const = [c for c in ALL_INPUTS if df[c].nunique() == 1]
    features = [c for c in ALL_INPUTS if c not in const]
    log["constant_inputs_dropped"] = {c: float(df[c].iloc[0]) for c in const}
    log["features"] = features

    # ---- 2. base case = Run_ID 1; label every run by the parameter it varies
    base = df.loc[df.Run_ID == 1, features].iloc[0]
    log["base_case"] = {c: float(base[c]) for c in features}
    differs = ~np.isclose(df[features].values, base.values, rtol=REL_TOL, atol=0)
    n_diff = differs.sum(axis=1)
    if (n_diff > 1).any():
        raise ValueError("Runs varying more than one parameter found: not an OAT design")
    df["sweep"] = np.where(n_diff == 0, "base",
                           np.array(features)[differs.argmax(axis=1)])

    # ---- 3. replicates of the base case inside sweeps -> keep Run 1 only
    rep = df[(df.sweep == "base") & (df.Run_ID != 1)]
    y_base = df.loc[df.Run_ID == 1, ["avgNu", "psi_max"]].iloc[0]
    log["base_replicates_removed"] = rep.Run_ID.tolist()
    log["base_replicate_max_rel_dev"] = {
        t: float((rep[t] / y_base[t] - 1).abs().max()) for t in ["avgNu", "psi_max"]}
    df["is_replicate"] = (df.sweep == "base") & (df.Run_ID != 1)

    # ---- 4. sweep-aware split (deterministic, no randomness)
    df["subset"] = "train"
    df["sweep_coord"] = np.nan        # position along the sweep, normalised to [0, 1]
    df.loc[df.is_replicate, "subset"] = "excluded_replicate"
    for p in features:
        idx = df.index[(df.sweep == p) & ~df.is_replicate]
        v = df.loc[idx, p].astype(float)
        v = np.log10(v) if p in LOG_INPUTS else v
        order = v.sort_values().index
        n = len(order)
        coord = (v - v.min()) / (v.max() - v.min())
        df.loc[idx, "sweep_coord"] = coord
        k_t, k_v = nround(EXT_TEST_FRAC * n), nround(EXT_VAL_FRAC * n)
        ext_test = list(order[:k_t]) + list(order[n - k_t:])
        ext_val = list(order[k_t:k_t + k_v]) + list(order[n - k_t - k_v:n - k_t])
        interior = list(order[k_t + k_v:n - k_t - k_v])
        df.loc[ext_test, "subset"] = "test_extrap"
        df.loc[ext_val, "subset"] = "val_extrap"
        for i, r in enumerate(interior):
            if i % INTERP_PERIOD == INTERP_TEST_PHASE:
                df.loc[r, "subset"] = "test_interp"
            elif i % INTERP_PERIOD == INTERP_VAL_PHASE:
                df.loc[r, "subset"] = "val_interp"

    # ---- 5. extrapolation distance of each held-out edge point, measured in
    # normalised sweep coordinate beyond the range seen by the model that
    # predicts it (train for validation points, train+val for test points)
    df["extrap_dist"] = 0.0
    for p in features:
        s = df[(df.sweep == p) & ~df.is_replicate]
        for subset, seen in [("val_extrap", ["train"]),
                             ("test_extrap", ["train", "val_interp", "val_extrap"])]:
            lo, hi = s[s.subset.isin(seen)].sweep_coord.agg(["min", "max"])
            m = s.subset == subset
            c = s.loc[m, "sweep_coord"]
            df.loc[c.index, "extrap_dist"] = np.where(c < lo, lo - c, c - hi)

    df.to_csv(PROCESSED / "dataset_clean.csv", index=False)
    summ = (df[~df.is_replicate].groupby(["sweep", "subset"]).size()
            .unstack(fill_value=0))
    summ.loc["TOTAL"] = summ.sum()
    summ["total"] = summ.sum(axis=1)
    summ.to_csv(PROCESSED / "split_summary.csv")
    log["n_unique_runs"] = int((~df.is_replicate).sum())
    log["subset_counts"] = df[~df.is_replicate].subset.value_counts().to_dict()
    ranges = {}
    for p in features:
        s = df[df.sweep == p]
        ranges[p] = {"n": int(len(s[~s.is_replicate])), "min": float(s[p].min()),
                     "max": float(s[p].max())}
    log["sweep_ranges"] = ranges
    (PROCESSED / "preprocessing_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))
    print(summ)
    plot_split(df[~df.is_replicate], features)


def plot_split(df, features):
    """Figure 1: where each subset sits along every sweep."""
    from surrogate import plotstyle as ps
    from surrogate.config import FIG
    ps.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    style = {"train": ("o", ps.DATA_GREY, "train"),
             "val_interp": ("o", "#2a78d6", "validation - interpolation"),
             "val_extrap": ("s", "#2a78d6", "validation - extrapolation"),
             "test_interp": ("o", "#eb6834", "test - interpolation"),
             "test_extrap": ("D", "#eb6834", "test - extrapolation")}
    fig, ax = ps.plt.subplots(figsize=(8.6, 2.9))
    for i, p in enumerate(features):
        s = df[df.sweep == p]
        for sub, (mk, col, lab) in style.items():
            q = s[s.subset == sub]
            ax.scatter(q.sweep_coord, np.full(len(q), i), marker=mk, s=22 if sub != "train" else 10,
                       color=col, facecolor=col if "extrap" in sub or sub == "train" else "white",
                       lw=1.0, label=lab if i == 0 else None, zorder=3)
        b = df[df.sweep == "base"].iloc[0][p]
        lo, hi = df[df.sweep == p][p].min(), df[df.sweep == p][p].max()
        bc = ((np.log10(b) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)) if p in LOG_INPUTS
              else (b - lo) / (hi - lo))
        ax.scatter([bc], [i], marker="*", s=90, color=ps.INK, zorder=4,
                   label="base case" if i == 0 else None)
    ax.set_yticks(range(len(features)),
                  [f"{p} sweep" + (" (log)" if p in LOG_INPUTS else "") for p in features])
    ax.set_xlabel("normalised position along the sweep (0 = lowest value, 1 = highest)")
    ax.set_ylim(-0.6, len(features) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig01_sweep_aware_split.png")
    ps.plt.close(fig)


if __name__ == "__main__":
    main()
