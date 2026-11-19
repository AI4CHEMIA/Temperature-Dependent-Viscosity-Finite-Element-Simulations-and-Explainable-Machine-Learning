"""Feature/target transforms, model factories and the MLP deep ensemble.

All transforms are fitted inside scikit-learn pipelines, i.e. on the training
data of each fit only, so no information from validation/test leaks in.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel, Matern,
                                              WhiteKernel)
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, PolynomialFeatures,
                                   StandardScaler)

from .config import LOG_INPUTS, SEED


# --------------------------------------------------------------------------
# Input transform
# --------------------------------------------------------------------------
class PhysicsFeatures(BaseEstimator, TransformerMixin):
    """log10 on decade-spanning inputs; optionally append log10(Ra*Da).

    Ra*Da is the Darcy-Rayleigh (modified Rayleigh) number Ra_m that governs
    convection in the Darcy limit. Used only in the physics-feature ablation.
    """

    def __init__(self, feature_names=None, add_darcy_rayleigh: bool = False):
        self.feature_names = feature_names
        self.add_darcy_rayleigh = add_darcy_rayleigh

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        names = list(self.feature_names)
        for c in LOG_INPUTS:
            if c in names:
                X[:, names.index(c)] = np.log10(X[:, names.index(c)])
        if self.add_darcy_rayleigh:
            ram = X[:, names.index("Ra")] + X[:, names.index("Da")]  # log10(Ra*Da)
            X = np.column_stack([X, ram])
        return X


def pow10(z):
    """Inverse of log10 (module-level so fitted models can be pickled)."""
    return np.power(10.0, z)


def target_wrap(regressor, log_target: bool = True) -> TransformedTargetRegressor:
    """y -> log10(y) -> standardise -> regressor, inverted on predict."""
    inner = TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())
    if not log_target:
        return inner
    return TransformedTargetRegressor(regressor=inner, func=np.log10,
                                      inverse_func=pow10,
                                      check_inverse=False)


def make_pipeline(model, feature_names, add_darcy_rayleigh=False, log_target=True):
    x_pipe = Pipeline([
        ("phys", PhysicsFeatures(feature_names, add_darcy_rayleigh)),
        ("scale", StandardScaler()),
        ("model", model),
    ])
    return target_wrap(x_pipe, log_target)


# --------------------------------------------------------------------------
# Deep ensemble of MLPs (mean prediction + epistemic spread)
# --------------------------------------------------------------------------
class MLPEnsemble(BaseEstimator, RegressorMixin):
    """Average of `n_members` MLPs that differ only in their random init.

    The ensemble mean is smoother and more stable than a single network on
    small data; the member spread is a cheap epistemic-uncertainty proxy.
    """

    def __init__(self, hidden_layer_sizes=(64, 64), activation="tanh", alpha=1e-4,
                 n_members=5, max_iter=4000, random_state=SEED):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.n_members = n_members
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        self.members_ = []
        for k in range(self.n_members):
            m = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,
                             activation=self.activation, alpha=self.alpha,
                             solver="lbfgs", max_iter=self.max_iter, tol=1e-7,
                             random_state=self.random_state + k)
            self.members_.append(m.fit(X, y))
        return self

    def member_predictions(self, X):
        return np.stack([m.predict(X) for m in self.members_])

    def predict(self, X):
        return self.member_predictions(X).mean(axis=0)


# --------------------------------------------------------------------------
# Model families and their hyperparameter grids (searched on validation only)
# --------------------------------------------------------------------------
def gp_kernel(kind: str, d: int):
    ls = np.ones(d)
    base = {"matern15": Matern(ls, (1e-2, 1e3), nu=1.5),
            "matern25": Matern(ls, (1e-2, 1e3), nu=2.5),
            "rbf": RBF(ls, (1e-2, 1e3))}[kind]
    return ConstantKernel(1.0, (1e-3, 1e3)) * base + WhiteKernel(1e-5, (1e-10, 1e-1))


def build_model(family: str, params: dict, n_features: int):
    p = dict(params)
    if family == "RF":
        return RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=2, **p)
    if family == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=500, random_state=SEED, n_jobs=2, **p)
    if family == "GBM":
        return GradientBoostingRegressor(random_state=SEED, subsample=0.8, **p)
    if family == "MLP":
        return MLPEnsemble(random_state=SEED, **p)
    if family == "GP":
        return GaussianProcessRegressor(kernel=gp_kernel(p["kernel"], n_features),
                                        n_restarts_optimizer=4, random_state=SEED)
    if family == "PolyRidge":
        return Pipeline([("poly", PolynomialFeatures(p["degree"], include_bias=False)),
                         ("ridge", Ridge(alpha=p["alpha"]))])
    raise ValueError(family)


GRIDS = {
    # Primary interpretable models requested
    "RF": [dict(max_features=mf, min_samples_leaf=ml)
           for mf in (0.6, 0.8, 1.0) for ml in (1, 2, 3)],
    "MLP": [dict(hidden_layer_sizes=h, activation=a, alpha=al, n_members=3)
            for h in ((32, 32), (64, 64), (64, 64, 64))
            for a in ("tanh", "relu") for al in (1e-5, 1e-4, 1e-3)],
    # Benchmarks
    "ExtraTrees": [dict(max_features=mf, min_samples_leaf=ml)
                   for mf in (0.8, 1.0) for ml in (1, 2)],
    "GBM": [dict(n_estimators=n, learning_rate=0.05, max_depth=d)
            for n in (400, 800) for d in (2, 3, 4)],
    "GP": [dict(kernel=k) for k in ("matern15", "matern25", "rbf")],
    "PolyRidge": [dict(degree=d, alpha=a) for d in (2, 3, 4) for a in (1e-6, 1e-4, 1e-2)],
}
PRIMARY = ["RF", "MLP"]
FAMILIES = ["RF", "MLP", "GP", "PolyRidge", "GBM", "ExtraTrees"]
FINAL_MLP_MEMBERS = 10   # the selected MLP is refitted as a 10-member ensemble


def clone_pipeline(pipe):
    return clone(pipe)


# --------------------------------------------------------------------------
# Predictive uncertainty in log10(y) space
# --------------------------------------------------------------------------
def predict_log_mean_std(fitted, X):
    """Return (mean, std) of log10(y) for a fitted `make_pipeline` object.

    RF / ExtraTrees: spread over trees. MLP: spread over ensemble members.
    GP: posterior standard deviation. Other families: std = NaN.
    """
    inner = fitted.regressor_                     # TTR with StandardScaler on log10 y
    sc = inner.transformer_
    pipe = inner.regressor_
    Z = pipe[:-1].transform(np.asarray(X, float))
    model = pipe[-1]
    a, b = sc.scale_[0], sc.mean_[0]
    if hasattr(model, "estimators_") and isinstance(model, (RandomForestRegressor,
                                                              ExtraTreesRegressor)):
        P = np.stack([t.predict(Z) for t in model.estimators_]) * a + b
        return np.log10(fitted.predict(X)), P.std(axis=0)
    if isinstance(model, MLPEnsemble):
        P = model.member_predictions(Z) * a + b
        return P.mean(axis=0), P.std(axis=0)
    if isinstance(model, GaussianProcessRegressor):
        m, s = model.predict(Z, return_std=True)
        return m * a + b, s * a
    return np.log10(fitted.predict(X)), np.full(len(Z), np.nan)
