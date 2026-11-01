"""Behavior: choice maps, reaction times, psychometric fits, wheel analysis."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import WHEEL_MM_PER_UNIT

LEVELS = (0.0, 0.25, 0.5, 1.0)


def choice_map(cl, cr, choice):
    """Percent of Left, NoGo and Right choices per contrast pair, each 4 x 4 with rows of left contrast."""
    out = {k: np.full((4, 4), np.nan) for k in ("left", "nogo", "right")}
    for i, a in enumerate(LEVELS):
        for j, b in enumerate(LEVELS):
            m = (cl == a) & (cr == b)
            if m.any():
                out["left"][i, j] = 100 * (choice[m] == 1).mean()
                out["nogo"][i, j] = 100 * (choice[m] == 0).mean()
                out["right"][i, j] = 100 * (choice[m] == -1).mean()
    return out


def rt_table(tr, session, subject):
    """One row per Left or Right trial: chosen contrast, whether a competing stimulus was present, reaction time in ms."""
    go = tr["go"]
    left = tr["has_left"][go]
    cl, cr = tr["cl"][go], tr["cr"][go]
    chosen = np.where(left, cl, cr)
    other = np.where(left, cr, cl)
    use = chosen > 0
    return pd.DataFrame({"session": session, "subject": subject, "chosen": chosen[use], "two": other[use] > 0,
                         "rt": 1000 * tr["first_time"][go][use]})


def rt_summary(df):
    """Mean and s.e.m. across sessions per chosen contrast and stimulus count."""
    per = df.groupby(["session", "chosen", "two"])["rt"].median().reset_index()
    return per.groupby(["chosen", "two"])["rt"].agg(["mean", "sem"]).reset_index()


def rt_anova(df):
    """Multi-way ANOVA with chosen contrast and stimulus count as factors. The paper does not list its factors."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    table = sm.stats.anova_lm(smf.ols("rt ~ C(chosen) + C(two) + C(subject)", df).fit(), typ=2)
    return {k: float(table.loc[k, "PR(>F)"]) for k in table.index[:-1]}


def _probs(p, cl, cr):
    bl, sl, n, br, sr = p
    a = np.exp(bl + sl * np.power(cl, n))
    b = np.exp(br + sr * np.power(cr, n))
    d = 1 + a + b
    return a / d, b / d, 1 / d


def fit_psychometric(cl, cr, choice):
    """Multinomial logistic model of the Ext 1 legend. Returns bL, sL, n, bR, sR."""
    def nll(p):
        pl, pr, pn = _probs(p, cl, cr)
        ll = np.log(pl[choice == 1] + 1e-12).sum() + np.log(pr[choice == -1] + 1e-12).sum() + np.log(pn[choice == 0] + 1e-12).sum()
        return -ll

    best = minimize(nll, (0.0, 3.0, 0.7, 0.0, 3.0), method="L-BFGS-B", bounds=[(-8, 8), (0, 30), (0.1, 2), (-8, 8), (0, 30)])
    return best.x


def model_probs(params, cl, cr):
    return _probs(params, np.asarray(cl, float), np.asarray(cr, float))


def binomial_ci(k, n, z=1.96):
    """Wilson 95% interval."""
    n = np.maximum(n, 1)
    p = k / n
    den = 1 + z**2 / n
    mid = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return mid - half, mid + half


def wheel_mm(session):
    t, pos = session.wheel()
    return t, pos * WHEEL_MM_PER_UNIT


def direction_decoding(t, mm, onsets, types, offsets=np.arange(-0.1, 0.0301, 0.005)):
    """Fraction of left and right turns whose sign of position change over 10 ms matches their direction, per offset."""
    use = (types == 1) | (types == 2)
    on, right = onsets[use], types[use] == 2
    if len(on) == 0:
        return np.full(len(offsets), np.nan)
    now = np.interp(on[:, None] + offsets, t, mm)
    before = np.interp(on[:, None] + offsets - 0.01, t, mm)
    v = now - before
    sign = (v[right].mean(0) >= v[~right].mean(0)) * 2 - 1 if (~right).any() and right.any() else np.ones(len(offsets))
    pred_right = (v * sign) > 0
    tied = v == 0
    acc = (pred_right == right[:, None]).astype(float)
    acc[tied] = 0.5
    return acc.mean(0)
