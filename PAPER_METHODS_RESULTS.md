# Interpretable machine-learning surrogates for avgNu and |ψ|max

*Paper-ready description of methods, results, interpretability and physical insights.
Every number below can be regenerated with `./run_all.sh` and checked with `scripts/07_verify.py`.
Figures are in `results/figures/`, and tables in `results/metrics/`, `results/shap/` and `results/physics/`.*

---

## Summary

We trained data-driven surrogates for the mean Nusselt number (avgNu) and the maximum stream-function magnitude (|ψ|max) on 312 unique CFD runs. The inputs are Ra, Da, R, A and C, and the dataset holds Pr = 7 and N = 3 fixed. The data are a one-at-a-time (OAT) design: a base case plus five single-parameter sweeps. We therefore isolated a sweep-aware test set with two parts: 26 interior points that test **interpolation**, and the outermost 7% of every sweep (44 points), which lie outside the training range and test **extrapolation**. Hyperparameters were chosen on a separate validation set with the same structure, and the test set was used exactly once.

An MLP deep ensemble predicts the held-out data with these mean absolute percentage errors (MAPE):

| Output | Interpolation | Extrapolation |
|---|---|---|
| avgNu | 0.025% | 0.32% |
| \|ψ\|max | 0.054% | 0.49% |

Its R² is ≥ 0.9997, and its 95% predictive intervals cover 90–94% of test points. A Gaussian process (GP) benchmark is about 10× more accurate (extrapolation MAPE 0.03%), but its intervals are over-confident. A random forest (RF) interpolates acceptably (MAPE ≈ 1%). It cannot extrapolate: extrapolation MAPE is 3.2% for avgNu and 7.9% for |ψ|max, with worst-case errors of 17% and 68%.

Exact SHAP values, computed against the base case, reproduce the CFD main effects (test R² ≥ 0.9999 for MLP and GP). All models agree with the CFD data on the feature ranking (Kendall τ = 1):

- **avgNu:** Ra > Da > C > A > R.
- **|ψ|max:** Da > C > Ra > R > A.

Local elasticities at the base case rank differently. C is the strongest local lever: +10% in C raises avgNu by 6.0% and |ψ|max by 8.4%. Off the sampled OAT manifold, the models disagree by a median of 20–42%, which shows that the design cannot identify parameter interactions.

The surrogates also yield physical results:

- The onset of convective enhancement is set by the Darcy–Rayleigh number (Ra·Da ≈ 2.1–2.5).
- The Nu–Ra exponent passes through a Darcy-like maximum of ≈ 0.50 at Ra ≈ 2×10⁴.
- |ψ|max peaks at Ra ≈ 5.3×10⁴ and then decouples from heat transfer.
- |ψ|max ∝ Da (Darcy's law) at low Da.
- A changes heat transfer by 36% while leaving circulation strength unchanged.

---

## 1. Dataset and design of experiments

The 316 runs comprise one base case (Ra = 10⁵, Da = 10⁻³, R = 0.2, A = 0.05, C = 3, Pr = 7, N = 3) and five sweeps. In each sweep, one parameter varies and the others stay at their base values (Table 1).

**Table 1 – Design of experiments.**

| Parameter | Base | Range | Spacing | Unique runs |
|---|---|---|---|---|
| Ra | 1×10⁵ | 1×10³ – 1×10⁶ | 0.05 decade | 60 |
| Da | 1×10⁻³ | 1×10⁻⁵ – 1×10⁻² | 0.05 decade | 60 |
| R | 0.2 | 0.10 – 0.50 | 0.008 | 51 |
| A | 0.05 | 0.010 – 0.100 | 0.001 | 90 |
| C | 3 | 0 – 5 | 0.1 | 50 |
| Pr, N | 7, 3 | fixed | – | – |

The outputs span 0.975–10.67 (avgNu, ×11) and 4.80×10⁻⁴–1.94×10⁻² (|ψ|max, ×41). Four runs (IDs 42, 103, 215 and 296) repeat the base case and agree with Run 1 to within 1.5×10⁻⁶ relative. We removed them so that no input appears in both the training and test sets, which leaves 312 unique runs.

Two consequences follow, and the rest of the study is built around them:

1. Pr and N carry no information, so the surrogates apply only at Pr = 7 and N = 3.
2. An OAT design identifies each main effect exactly along its sweep. It contains no run in which two parameters leave their base values together, so it **cannot identify interactions**.

## 2. Sweep-aware partition: interpolation vs extrapolation

A random split would put nearly every test point between two densely spaced training neighbours, which would overstate how well the models generalise. We partitioned each sweep deterministically instead (Fig. 1).

1. Order the sweep by its coordinate ξ (log₁₀ for Ra and Da) and normalise it to [0, 1].
2. **Test – extrapolation:** the outermost k_t = round(0.07 n) points at *each* end.
3. **Validation – extrapolation:** the next k_v = round(0.05 n) points inward at each end.
4. **Interior points:** every 9th interior point (offset 4) goes to **test – interpolation**, and every 9th (offset 8) to **validation – interpolation**. Interpolation points therefore cover the whole sweep.
5. All other points, including the base case, form the training set.

The result is 183 training, 59 validation (25 interpolation + 34 extrapolation) and 70 test runs (26 + 44); Table 2 gives the per-sweep counts. Test-extrapolation points lie 1.1–8.0% of the sweep span beyond the range of the final (train+val) models. Validation-extrapolation points sit at the same distance beyond the training range used during model selection, so the validation set mimics the test task.

**Table 2 – Partition per sweep.**

| Sweep | Train | Val-interp | Val-extrap | Test-interp | Test-extrap |
|---|---|---|---|---|---|
| Ra | 36 | 5 | 6 | 5 | 8 |
| Da | 36 | 5 | 6 | 5 | 8 |
| R | 29 | 4 | 6 | 4 | 8 |
| A | 53 | 7 | 10 | 8 | 12 |
| C | 28 | 4 | 6 | 4 | 8 |
| base | 1 | – | – | – | – |
| **Total** | **183** | **25** | **34** | **26** | **44** |

## 3. Models, preprocessing and selection

**Transforms.** Ra and Da enter as log₁₀ values, and all inputs are then standardised. As a candidate physics-informed feature, we optionally add log₁₀(Ra·Da), the Darcy–Rayleigh number. Both outputs are modelled as standardised log₁₀(y): the responses look like power laws, and this keeps relative errors homoscedastic. Every transform is fitted inside the training pipeline, and predictions are converted back to physical units before scoring.

**Model families.** Each target gets its own model.

- **Random forest (RF):** 500 trees, an interpretable baseline that admits exact TreeSHAP.
- **MLP deep ensemble:** L-BFGS training, 3 members during search and 10 in the final model; the mean is the prediction and the member spread is the epistemic uncertainty.
- **Benchmarks:** Gaussian process with an anisotropic Matérn-3/2, Matérn-5/2 or RBF kernel plus a white-noise term; polynomial ridge (degree 2–4); gradient boosting; extra trees.

**Selection.** Each family's grid, crossed with {with, without} log₁₀(Ra·Da), was fitted on the training set and scored on the validation set:

S = ½ MAPE(val-interp) + ½ MAPE(val-extrap),  where MAPE = (100/n) Σ |ŷᵢ − yᵢ| / |yᵢ|.

The criterion rewards interpolation and extrapolation equally. The winning configuration (Table 3) was refitted on train+val and evaluated once on the test set. A copy refitted on all 312 runs is provided for downstream use and is not used for evaluation.

**Table 3 – Selected configurations (validation MAPE, %).**

| Target | Model | Configuration | log(Ra·Da) | Val interp | Val extrap |
|---|---|---|---|---|---|
| avgNu | RF | max_features 0.6, min_leaf 1 | yes | 1.228 | 3.382 |
| avgNu | MLP | 3×64 tanh, α = 1e-5 | yes | 0.032 | 0.180 |
| avgNu | GP | RBF-ARD + noise | yes | 0.0006 | 0.023 |
| \|ψ\|max | RF | max_features 1.0, min_leaf 1 | yes | 0.932 | 6.663 |
| \|ψ\|max | MLP | 3×64 tanh, α = 1e-3 | no | 0.034 | 0.184 |
| \|ψ\|max | GP | Matérn-5/2-ARD + noise | no | 0.002 | 0.041 |

**Metrics.** We report MAPE, maximum APE, RMSE, MAE and R² in physical units. Each MAPE carries a 95% CI from a 2,000-sample percentile bootstrap over test points. Predictive 95% intervals come from the tree spread (RF), the member spread (MLP) or the posterior standard deviation (GP), all in log space.

## 4. Predictive performance

**Table 4 – Test-set MAPE in % [95% bootstrap CI]; worst-case APE and R² over the whole test set.**

| Target | Model | Interpolation (n = 26) | Extrapolation (n = 44) | Max APE % | R² |
|---|---|---|---|---|---|
| avgNu | GP | 0.0003 [0.0002, 0.0003] | 0.027 [0.013, 0.044] | 0.30 | >0.9999 |
| avgNu | **MLP** | **0.025 [0.016, 0.035]** | **0.32 [0.21, 0.46]** | 2.12 | 0.9997 |
| avgNu | Poly ridge | 0.24 [0.11, 0.38] | 2.15 [0.90, 3.61] | 19.8 | 0.9996 |
| avgNu | Extra trees | 0.062 [0.039, 0.088] | 2.58 [1.74, 3.60] | 14.2 | 0.9858 |
| avgNu | Grad. boosting | 1.13 [0.61, 1.79] | 2.61 [1.77, 3.62] | 14.3 | 0.9853 |
| avgNu | **RF** | **1.05 [0.64, 1.53]** | **3.18 [2.17, 4.39]** | 16.5 | 0.9787 |
| \|ψ\|max | GP | 0.0033 [0.0008, 0.0079] | 0.029 [0.014, 0.050] | 0.32 | >0.9999 |
| \|ψ\|max | **MLP** | **0.054 [0.033, 0.079]** | **0.49 [0.26, 0.80]** | 5.40 | 0.9999 |
| \|ψ\|max | Poly ridge | 0.14 [0.06, 0.23] | 0.87 [0.48, 1.32] | 6.91 | 0.9997 |
| \|ψ\|max | Extra trees | 0.12 [0.05, 0.20] | 6.30 [3.30, 9.88] | 57.0 | 0.9939 |
| \|ψ\|max | Grad. boosting | 1.25 [0.61, 2.04] | 6.32 [3.32, 9.90] | 57.1 | 0.9936 |
| \|ψ\|max | **RF** | **1.01 [0.52, 1.58]** | **7.93 [4.32, 12.25]** | 67.5 | 0.9908 |

**Interpolation.** Every smooth model interpolates within 0.25%. The CFD responses are smooth, so the remaining surrogate error is far below typical CFD discretisation uncertainty.

**Extrapolation separates the model classes.** Tree ensembles are piecewise constant, so outside the training range they repeat the last training value (Fig. 3, flat RF segments). The error then grows with the local gradient, which is why RF errors are largest where the response is steepest:

- |ψ|max at low Da: 22.8% MAPE on that sweep's extrapolation points.
- |ψ|max at low Ra: 12.4%.
- avgNu at high Ra: 6.5%.

The MLP ensemble and the GP follow the CFD curvature beyond the data. The MLP's largest per-sweep extrapolation errors are 0.94% (avgNu, Ra sweep) and 1.62% (|ψ|max, Da sweep).

**Uncertainty calibration.** On the test set, the MLP ensemble's 95% intervals cover 90.0% (avgNu) and 94.3% (|ψ|max) of points; on extrapolation points alone, 86.4% and 93.2%. Their median half-widths are 0.27% and 0.36%, so they are informative and close to nominal. The GP is the most accurate model, but its intervals are over-confident: half-widths ≤ 0.02% and extrapolation coverage of 79.6% and 88.6%. RF intervals collapse outside the data (extrapolation coverage 23–36%).

**Extrapolation stress test (train+val only; Fig. 5, Table 5).** To probe distances beyond the test set, we removed the outer 25% of one sweep at a time from train+val, refitted with the selected hyperparameters, and predicted the removed points.

**Table 5 – Stress test: mean APE (%) vs extrapolation distance (% of the sweep span).**

| Target | Model | 0–5 | 5–10 | 10–15 | 15–20 | 20–25 |
|---|---|---|---|---|---|---|
| avgNu | GP | 0.004 | 0.054 | 0.34 | 0.98 | 2.42 |
| avgNu | MLP | 0.24 | 0.98 | 2.56 | 4.47 | 6.83 |
| avgNu | Poly ridge | 0.04 | 0.17 | 1.04 | 2.46 | 5.91 |
| avgNu | RF | 3.80 | 8.63 | 12.7 | 15.6 | 18.9 |
| \|ψ\|max | GP | 0.005 | 0.032 | 0.14 | 0.37 | 0.80 |
| \|ψ\|max | MLP | 0.31 | 1.19 | 3.71 | 5.33 | 9.94 |
| \|ψ\|max | Poly ridge | 0.06 | 0.20 | 0.69 | 1.22 | 2.40 |
| \|ψ\|max | RF | 4.58 | 12.3 | 25.5 | 31.2 | 50.2 |

A practical rule follows. Within about 5% of the sweep span beyond the data, MLP and GP stay below 0.35% error. Within 10%, the MLP stays below about 1.2%. Further out, error grows roughly linearly with distance.

**Physics-feature ablation (validation only).** Adding log₁₀(Ra·Da) reduced the MLP's validation extrapolation error for avgNu from 0.259% to 0.180% (−31%). For |ψ|max it increased it from 0.184% to 0.325%, so the feature was rejected there. The data thus indicate that the Darcy–Rayleigh number is a natural coordinate for heat transfer but not for circulation strength (see §6).

**Recommended surrogate.** RF admits exact TreeSHAP, but its inability to extrapolate (test errors up to 68%) disqualifies it as a predictive surrogate on this design. We recommend the MLP deep ensemble as the primary interpretable surrogate: sub-percent extrapolation error and calibrated intervals. The GP serves as the high-accuracy benchmark.

## 5. Interpretability (SHAP)

We explain f(x) = log₁₀ ŷ(x), so SHAP's additivity becomes **multiplicative** in physical units:

ŷ(x) = ŷ_ref · Π_j 10^{φ_j(x)}.

Each φ_j is therefore the factor by which parameter j scales the output, matching power-law correlation practice. With five inputs, we computed **exact** Shapley values (2⁵ = 32 coalitions) for RF, MLP and GP; no sampling approximation was needed. Two backgrounds answer different questions.

**(A) Base-case reference (on-manifold; Figs. 6–7).** With x_b (the base case) as the single reference, every coalition evaluated for an OAT run is either that run or the base case. Both are points that exist in the CFD data. The attribution is therefore exactly identified:

φ_j(x) = log₁₀ ŷ(x) − log₁₀ ŷ(x_b) for the varied parameter, and 0 for all others (verified to < 3×10⁻⁸).

We can compare it directly with the CFD log-ratio log₁₀[y(x)/y(x_b)]. On held-out test runs this SHAP decomposition reaches R² ≥ 0.9999 for MLP and GP. The largest multiplicative error is ≤ 0.32% for GP, 2.2% for MLP on avgNu and 5.4% for MLP on |ψ|max (both in extrapolation), and 20% and 68% for RF. We define global importance as I_j = mean |φ_j| over the sweep of j (Table 6).

**Table 6 – On-manifold SHAP importance I_j (log₁₀ units) and the CFD end-to-end change factor.**

| Target | Feature | CFD | RF | MLP | GP | CFD factor across range |
|---|---|---|---|---|---|---|
| avgNu | Ra | 0.356 | 0.349 | 0.355 | 0.356 | ×10.9 |
| avgNu | Da | 0.294 | 0.296 | 0.294 | 0.294 | ×6.09 |
| avgNu | C | 0.124 | 0.120 | 0.124 | 0.124 | ×2.80 |
| avgNu | A | 0.054 | 0.053 | 0.054 | 0.054 | ×1.57 (decreasing) |
| avgNu | R | 0.017 | 0.017 | 0.017 | 0.017 | ×1.13 |
| \|ψ\|max | Da | 0.431 | 0.420 | 0.430 | 0.431 | ×36.9 |
| \|ψ\|max | C | 0.166 | 0.161 | 0.165 | 0.166 | ×3.93 |
| \|ψ\|max | Ra | 0.112 | 0.105 | 0.111 | 0.112 | ×3.10 (non-monotonic) |
| \|ψ\|max | R | 0.007 | 0.007 | 0.007 | 0.007 | ×1.05 |
| \|ψ\|max | A | 0.001 | 0.001 | 0.001 | 0.001 | ×1.006 |

All three models reproduce the CFD ranking exactly (Kendall τ = 1.0 for both outputs).

**(B) Design-space background (off-manifold; Fig. 8).** The conventional global view uses a 64-point scrambled Sobol background and explains 512 Sobol points that fill the whole 5-D box. The rankings agree (τ = 1, except RF on |ψ|max, τ = 0.8). The magnitudes differ between models, however. For mean |φ| on avgNu, Ra scores 0.107 for GP and 0.223 for MLP, and C scores 0.070 for GP and 0.015 for RF.

To quantify this, we measured how much of each model's 5-D response an additive (OAT) decomposition explains. For |ψ|max, R² = 0.99 (GP), 0.90 (MLP) and 0.74 (RF); for avgNu, 0.51 (MLP), −0.23 (RF) and −4.1 (GP). The pairwise disagreement between models at those multi-parameter points has a median of 25–32% (avgNu) and 20–42% (|ψ|max), with 95th percentiles of 71–290%. **Interactions between parameters are not identified by the OAT data. Each model fills them in according to its own inductive bias.** On-manifold SHAP (A) is physically meaningful and verifiable. Off-manifold SHAP interaction structure (B) is a property of the model, not of the flow, and must not be reported as physics.

**Local sensitivities.** Elasticities at the base case, E_j = ∂ln y/∂ln x_j, agree across CFD central differences, GP and MLP to within 0.007 (except E_A for |ψ|max, which is ≈ 0), as Table 7 shows.

**Table 7 – Elasticities at the base case (CFD | GP | MLP) and the response to a +10% change.**

| Parameter | avgNu E | +10% → avgNu | \|ψ\|max E | +10% → \|ψ\|max |
|---|---|---|---|---|
| C | 0.608 \| 0.609 \| 0.609 | +6.0% | 0.846 \| 0.847 \| 0.840 | +8.4% |
| Ra | 0.428 \| 0.428 \| 0.426 | +4.2% | −0.056 \| −0.056 \| −0.056 | −0.53% |
| A | −0.276 \| −0.276 \| −0.277 | −2.6% | 0.0007 \| 0.0014 \| 0.0008 | ≈ 0 |
| Da | 0.256 \| 0.256 \| 0.256 | +2.5% | 0.297 \| 0.297 \| 0.297 | +2.9% |
| R | 0.065 \| 0.065 \| 0.064 | +0.62% | 0.029 \| 0.029 \| 0.028 | +0.27% |

Local and global rankings differ, so importance depends on scale. Globally, Ra and Da dominate because they were swept over three decades. At the operating point, C is the strongest lever for both outputs. A paper should report both views and state the explored ranges.

## 6. Physical insights for discussion

1. **The Darcy–Rayleigh number sets the onset of convection.** avgNu exceeds its conduction value (0.975–0.980) by 5% at Ra·Da = 2.5 in the Ra sweep (Da = 10⁻³) and at Ra·Da = 2.1 in the Da sweep (Ra = 10⁵) (Fig. 10). Two independent paths reach the same threshold, which supports Ra_m = Ra·Da as the governing group at low Ra_m.
2. **Ra·Da similarity breaks down as non-Darcy effects grow.** At equal Ra·Da, avgNu along the Ra sweep is 10–12% *lower* than along the Da sweep for Ra·Da ≲ 10². This is consistent with stronger Brinkman (viscous) resistance at the larger Da of the Ra sweep. The curves cross at Ra·Da ≈ 99, and at Ra·Da = 10³ the ratio reaches 1.79, because the Da sweep saturates toward the clear-fluid limit at fixed Ra = 10⁵. |ψ|max does not collapse at all (ratio 8.3 → 0.53). The ML ablation in §4 independently confirms this asymmetry: log(Ra·Da) helps predict avgNu and hurts predictions of |ψ|max.
3. **Heat-transfer scaling has a regime transition.** The local exponent n = d ln Nu/d ln Ra (Fig. 9) rises from ≈ 0.04 (conduction) to a maximum of 0.505 at Ra ≈ 2.2×10⁴, then falls to 0.32 at Ra = 10⁶. Decade-wise CFD fits give 0.18, 0.49 and 0.37. The maximum of ≈ ½ matches the classical Darcy boundary-layer scaling Nu ∼ Ra_m^{1/2}. The decline toward ⅓ indicates a shift to a Brinkman/clear-fluid-type laminar regime. The exponent with respect to Da shows the same pattern (maximum 0.52 at Da ≈ 1.5×10⁻⁴, falling to 0.04 at Da = 10⁻²). The surrogates provide these exponents as smooth, differentiable functions, so no curve fitting is needed.
4. **Darcy's law appears in the circulation.** At low Da, d ln|ψ|max/d ln Da → 0.97, so |ψ|max ∝ Da: velocity is proportional to permeability, as Darcy's law requires. The exponent decays to 0.06 at Da = 10⁻², where viscous diffusion takes over.
5. **Circulation strength peaks, then decouples from heat transfer.** |ψ|max is maximal at Ra ≈ 5.3×10⁴ (GP; |ψ|max = 0.01239). From there to Ra = 10⁶ it falls by 25% while avgNu rises by 211%. The likely explanation is that the flow concentrates in thinning boundary layers while the core stagnates. Fig. 11 shows that |ψ|max is a valid proxy for heat transfer only below about Ra ≈ 5×10⁴.
6. **A acts on the thermal path, not on the flow.** Across A = 0.01–0.10, avgNu falls by 36% (5.36 → 3.42) while |ψ|max changes by only +0.13%: a shallow minimum near A ≈ 0.045, with a total variation of about 0.5%. The elasticities are −0.276 for avgNu and ≈ 0 for |ψ|max. Through its specific mechanism (to be named from the definition of A), A changes heat transfer without changing circulation strength. This gives a clean lever for tuning heat transfer at constant flow.
7. **C drives both transports, and the coupling exponent differs by mechanism.** Raising C from 0 to 5 increases avgNu by 180% and |ψ|max by 293%. The end-to-end coupling exponent d ln Nu/d ln|ψ|max is 0.75 along C and 0.50 along Da. The value ½ is consistent with boundary-layer scaling (Nu ∼ Pe^{1/2}). The larger exponent along C suggests that C also strengthens the thermal driving directly, beyond its effect on the flow.
8. **R is a weak, nearly linear modifier** (+13% in avgNu and +5% in |ψ|max over 0.1–0.5; elasticity 0.065).

*Note: the physical names of R, A and C are not part of the dataset. Insights 6–8 are stated quantitatively and should be phrased in the paper using those definitions.*

## 7. Limitations and recommendations

- **Interactions are unidentified.** Predictions for cases where two or more parameters leave the base case are unvalidated, and the models disagree there by a median of 20–42%. A space-filling supplement of 50–100 CFD runs (Sobol or Latin hypercube) would identify the interactions and enable a full Sobol-index variance decomposition. `predict.py` flags such inputs.
- **Pr and N are fixed.** No statement can be made about them.
- **Extrapolation has been verified up to 8% of the sweep span on the test set and probed up to 25% in the stress test.** Errors grow beyond about 10%.
- **GP intervals are over-confident.** For uncertainty statements, use the MLP ensemble's intervals.
- The CFD data are deterministic. The surrogate errors in interpolation (< 0.06%) lie below typical mesh-discretisation uncertainty, so a reported CFD grid-convergence error should be quoted as the effective accuracy floor.

## 8. Reproducibility

The accompanying package (`README.md`) contains:

- the raw spreadsheet (SHA-256 recorded);
- the cleaned data with subset labels;
- all code: preprocessing, training, evaluation, stress test, SHAP, physics analysis, verification and prediction;
- trained models, evaluated and deployment versions;
- all predictions with 95% intervals;
- every table above as CSV, and all figures at 300 dpi.

`scripts/07_verify.py` runs nine independent checks, all of which pass:

1. Raw-data checksum.
2. Split integrity (no input shared between train+val and test).
3. Every extrapolation test point lies outside the training range.
4. Model selection used validation data only.
5. Reloaded models reproduce the saved predictions (< 1.3×10⁻¹³).
6. Recomputed test metrics match the reported ones (< 1.6×10⁻¹²).
7. SHAP additivity holds (< 2×10⁻⁸).
8. Base-reference SHAP equals the exact OAT main effect (< 3×10⁻⁸).
9. The deployment models reproduce the CFD data (GP and MLP within 0.54%).

The full pipeline runs in about 15 minutes on 2 CPU cores with the pinned package versions (Python 3.13, scikit-learn 1.9.1, SHAP 0.54.1).

## Figure captions

- **Fig. 1.** Sweep-aware partition. Each row is one OAT sweep on a normalised coordinate (log scale for Ra and Da). Diamonds: extrapolation test points (outer 7% at each end). Squares: extrapolation validation points (next 5%). Open circles: interpolation test and validation points. Star: base case.
- **Fig. 2.** Parity plots of the RF, MLP ensemble and GP for avgNu (top) and |ψ|max (bottom). Open circles: interpolation test points. Filled diamonds: extrapolation test points. Grey: training fit.
- **Fig. 3.** Responses along each sweep (other parameters at base) for avgNu and |ψ|max. CFD data (grey: train+val; markers: test) against RF, GP and MLP (with 95% band). Shaded regions lie outside the training range. RF goes flat in them.
- **Fig. 4.** Test MAPE per model (log scale; light = interpolation, solid = extrapolation) with bootstrap 95% CIs.
- **Fig. 5.** Extrapolation stress test: mean APE against distance beyond the training range, as a percentage of the sweep span.
- **Fig. 6.** Exact SHAP main effects relative to the base case (log₁₀ factor on the left axis; multiplicative factor on the right) for RF and MLP, against CFD log-ratios.
- **Fig. 7.** On-manifold global importance (mean |φ|) from CFD directly, RF, MLP and GP.
- **Fig. 8.** Design-space SHAP beeswarm plots (Sobol background) for MLP and RF. Interaction structure here is model-dependent; see §5(B).
- **Fig. 9.** Local scaling exponents d ln y/d ln Ra and d ln y/d ln Da from CFD finite differences and the GP and MLP surrogates.
- **Fig. 10.** Darcy–Rayleigh similarity test: avgNu and |ψ|max against Ra·Da for the Ra and Da sweeps, with their ratio at equal Ra·Da.
- **Fig. 11.** Heat transfer (avgNu) against circulation strength (|ψ|max) along each sweep. Arrows point toward increasing parameter value.
