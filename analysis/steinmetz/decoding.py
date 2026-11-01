"""Population decoding of task variables from residual activity (Fig 4c, Ext 5g)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .config import DT

C_GRID = (0.01, 0.1, 1.0)  # not stated in the paper; chosen by inner cross-validation
OFFSETS = {"stim": np.round(np.arange(-0.05, 0.4 + 1e-9, DT), 3), "move": np.round(np.arange(-0.25, 1e-9, DT), 3)}
TRAIN_WIN = {"stim": (0.05, 0.15), "move": (-0.05, 0.0)}


def event_tensor(resid, rows, trnum, events, offsets, dt=DT):
    """trials x offsets x neurons of residuals, NaN where a sample lies outside the fitted trial window."""
    row_of = -np.ones(int(rows.max()) + 2, int)
    row_of[rows] = np.arange(len(rows))
    idx = np.floor((events[:, None] + offsets) / dt).astype(int)
    inside = (idx >= 0) & (idx < len(row_of))
    r = np.full(idx.shape, -1)
    r[inside] = row_of[idx[inside]]
    trials = np.arange(len(events))[:, None]
    r[(r >= 0) & (trnum[np.maximum(r, 0)] != trials)] = -1  # keep only samples of the trial's own window
    out = np.full((*idx.shape, resid.shape[1]), np.nan, np.float32)
    ok = r >= 0
    out[ok] = resid[r[ok]]
    return out


def _fit(x, y, rng_seed):
    """L1 logistic regression, with strength picked by inner 3-fold cross-validation."""
    best, best_score = C_GRID[0], -np.inf
    n_inner = min(3, np.bincount(y).min())
    if n_inner >= 2:
        for c in C_GRID:
            score = 0.0
            for tr, te in StratifiedKFold(n_inner, shuffle=True, random_state=rng_seed).split(x, y):
                m = LogisticRegression(C=c, l1_ratio=1.0, solver="liblinear", random_state=0).fit(x[tr], y[tr])
                score += m.score(x[te], y[te])
            if score > best_score:
                best, best_score = c, score
    return LogisticRegression(C=best, l1_ratio=1.0, solver="liblinear", random_state=0).fit(x, y).coef_.ravel()


def decode_curve(tensor, labels, offsets, train_win, folds=5, seed=0):
    """Mean test projection of class 1 minus class 0 at each offset."""
    y = labels.astype(int)
    k = min(folds, np.bincount(y, minlength=2).min())
    if k < 2:
        return None
    m = (offsets >= train_win[0]) & (offsets <= train_win[1])
    x_train = np.nan_to_num(np.nanmean(tensor[:, m], axis=1)) if m.any() else None
    proj = np.full(tensor.shape[:2], np.nan)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(x_train, y):
        w = _fit(x_train[tr], y[tr], seed)
        proj[te] = np.einsum("tbn,n->tb", np.nan_to_num(tensor[te]), w)
        proj[te][np.isnan(tensor[te]).all(2)] = np.nan
    return np.nanmean(proj[y == 1], 0) - np.nanmean(proj[y == 0], 0)


def normalize_groups(curves, groups, t_index):
    """Divide each recording by its group's mean at choice time, so the group mean is 1 there and the s.e.m. is not forced to 0."""
    curves = np.asarray(curves, float)
    groups = np.asarray(groups)
    out = curves.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] = curves[m] / np.nanmean(curves[m, t_index])
    return out


def timing_anova(curves, groups, times):
    """Two-way ANOVA on time and group for Fig 4c. Returns F and p for each term."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    n, b = curves.shape
    df = pd.DataFrame({"y": curves.ravel(), "t": np.tile(np.arange(b), n), "g": np.repeat(np.asarray(groups), b)}).dropna()
    table = sm.stats.anova_lm(smf.ols("y ~ C(t) * C(g)", df).fit(), typ=2)
    return {k: (float(table.loc[k, "F"]), float(table.loc[k, "PR(>F)"])) for k in table.index[:-1]}
