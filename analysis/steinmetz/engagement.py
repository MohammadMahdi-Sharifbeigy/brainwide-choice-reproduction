"""Pre-stimulus state analyses (Fig 5c-f, Ext 9)."""
import numpy as np
from scipy import stats

from .config import CONTRASTS


def nested_anova(task, passive):
    """Condition effect with condition, session and neuron-nested-in-session as factors.

    Inputs are log rates per neuron. Neuron identity absorbs session, so the condition F has N - 1 residual df and
    reduces to a paired comparison. Returns F and p.
    """
    d = task - passive
    n = len(d)
    if n < 2:
        return np.nan, np.nan
    ss_cond = n * d.mean() ** 2 / 2
    ss_res = ((d - d.mean()) ** 2).sum() / 2
    f = ss_cond / (ss_res / (n - 1)) if ss_res > 0 else np.inf
    return f, stats.f.sf(f, 1, n - 1)


def engagement_vector(active, passive):
    """x_n = mean active rate minus mean passive rate, at unit L2 length."""
    x = active.mean(0) - passive.mean(0)
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x


def match_go_miss(tr, rng, contrasts=CONTRASTS):
    """Equal Go and Miss trials at each contra contrast. Returns (go indices, miss indices)."""
    go, miss = [], []
    for c in contrasts:
        g = np.flatnonzero(tr["go"] & (tr["cr"] == c))
        m = np.flatnonzero(tr["miss"] & (tr["cr"] == c))
        n = min(len(g), len(m))
        go.append(rng.permutation(g)[:n])
        miss.append(rng.permutation(m)[:n])
    return np.sort(np.concatenate(go)).astype(int), np.sort(np.concatenate(miss)).astype(int)


def zscore(x):
    x = np.asarray(x, float)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 0 else x * 0


def previous_trial(tr):
    """Inter-trial interval and previous reward per trial; NaN for the first trial."""
    iti = np.r_[np.nan, tr["stim"][1:] - tr["fb_time"][:-1]]
    prev = np.r_[np.nan, (tr["fb"][:-1] == 1).astype(float)]
    return iti, prev


def split_difference(rates, value):
    """Neuron vector: mean rate on trials above the session's median of value, minus below it."""
    ok = np.isfinite(value)
    hi = value[ok] > np.median(value[ok])
    return rates[ok][hi].mean(0) - rates[ok][~hi].mean(0)


def fit_glm(y, table):
    """Binomial GLM on the columns of table (name -> array). Returns the statsmodels result."""
    import statsmodels.api as sm

    X = sm.add_constant(np.column_stack(list(table.values())))
    return sm.GLM(y, X, family=sm.families.Binomial()).fit()


def deviance_test(small, big, extra_df):
    """p-value for the drop in deviance from small to big."""
    return stats.chi2.sf(small.deviance - big.deviance, extra_df)


def panel_curve(result, names, which, grid, power2=None):
    """P(Go) along one predictor with the others at their mean of zero, with a 95% band. power2 names its squared term."""
    n = len(grid)
    X = np.zeros((n, len(names) + 1))
    X[:, 0] = 1
    X[:, names.index(which) + 1] = grid
    if power2 is not None:
        X[:, names.index(power2) + 1] = grid**2
    eta = X @ result.params
    se = np.sqrt(np.einsum("ij,jk,ik->i", X, result.cov_params(), X))
    f = lambda v: 1 / (1 + np.exp(-v))
    return f(eta), f(eta - 1.96 * se), f(eta + 1.96 * se)
