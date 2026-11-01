"""Known-answer tests for decoding, jPECC, ccCP, engagement and behavior."""
import numpy as np
import pandas as pd
import pytest

from steinmetz import behavior, cccp, decoding, engagement, jpecc


def direct_cp(counts, group_a, cond, m_lists):
    """Authors' form: tiedrank then t(shufLabels), one neuron-bin at a time."""
    from scipy.stats import rankdata
    total, den = 0, 0
    for idx, lab, na, nb in m_lists:
        r = rankdata(counts[idx])
        total = total + np.array([r[row.astype(bool)].sum() for row in lab]) - na * (na + 1) / 2
        den += na * nb
    return total / den


def test_batched_cccp_equals_direct_form():
    rng = np.random.default_rng(0)
    cond = rng.integers(0, 6, 120)
    a = rng.random(120) < 0.5
    counts = rng.poisson(2, (120, 8)).astype(np.float32)
    plan = cccp.plan_shuffles(a, cond, rng, 200)
    cp, p = cccp.probability(counts, plan)
    for j in range(8):
        d = direct_cp(counts[:, j], a, cond, plan)
        assert cp[j] == pytest.approx(d[0], abs=1e-5)


def test_cp_is_half_without_effect_and_high_with_effect():
    rng = np.random.default_rng(1)
    cond = rng.integers(0, 4, 400)
    a = rng.random(400) < 0.5
    counts = rng.poisson(3, (400, 2)).astype(np.float32)
    counts[a, 1] += 4
    cp, p = cccp.probability(counts, cccp.plan_shuffles(a, cond, rng, 500))
    assert abs(cp[0] - 0.5) < 0.1 and 0.01 < p[0] < 0.99
    assert cp[1] > 0.8 and p[1] > 0.99


def test_one_sided_condition_contributes_nothing():
    rng = np.random.default_rng(2)
    cond = np.r_[np.zeros(40), np.ones(10)]
    a = np.r_[rng.random(40) < 0.5, np.ones(10, bool)]
    plan = cccp.plan_shuffles(a, cond, rng, 10)
    assert len(plan) == 1 and len(plan[0][0]) == 40


def test_shuffles_really_shuffle():
    rng = np.random.default_rng(3)
    a = np.r_[np.ones(10, bool), np.zeros(10, bool)]
    plan = cccp.plan_shuffles(a, np.zeros(20), rng, 50)
    assert not np.array_equal(plan[0][1][1], plan[0][1][0])
    assert plan[0][1][1, 10:].sum() > 0  # draws reach the second group, unlike a permutation of 1:nx


def test_sliding_counts_box():
    x = np.arange(12.0).reshape(1, 12, 1)
    got = cccp.sliding_counts(x, 10)
    assert got.shape[1] == 3
    assert got[0, 0, 0] == x[0, :10, 0].sum()


def planted_decoding(seed=0, trials=120, neurons=20, nb=30):
    rng = np.random.default_rng(seed)
    y = rng.random(trials) < 0.5
    w = rng.normal(size=neurons)
    t = np.linspace(-0.145, 0, nb)
    ramp = np.clip((t + 0.1) / 0.1, 0, 1)
    x = rng.normal(size=(trials, nb, neurons), scale=1.0)
    x += (y[:, None] * 2.0 - 1)[..., None] * (ramp[None, :, None] * w[None, None, :])
    return x.astype(np.float32), y, t


def test_decoding_recovers_a_planted_axis_and_its_timing():
    x, y, t = planted_decoding()
    curve = decoding.decode_curve(x, y, t, (-0.05, 0.0))
    assert curve[-1] > 5 * abs(curve[:5].mean())


def test_decoding_handles_too_few_trials():
    x, y, t = planted_decoding(trials=30)
    y[:] = False
    y[0] = True
    assert decoding.decode_curve(x, y, t, (-0.05, 0.0)) is None


def test_group_normalization_keeps_sem_at_choice_time():
    rng = np.random.default_rng(4)
    curves = rng.uniform(0.5, 2, (20, 10))
    groups = np.array(["f"] * 10 + ["m"] * 10)
    out = decoding.normalize_groups(curves, groups, -1)
    for g in "fm":
        assert out[groups == g, -1].mean() == pytest.approx(1)
        assert out[groups == g, -1].std() > 0.05


def test_event_tensor_keeps_only_own_trial():
    rows = np.array([10, 11, 12, 20, 21])
    trnum = np.array([0, 0, 0, 1, 1])
    resid = np.arange(5.0)[:, None]
    ev = np.array([0.0525, 0.1025])  # bins 10 and 20
    out = decoding.event_tensor(resid, rows, trnum, ev, np.array([0.0, 0.005, 0.05]))
    assert out[0, 0, 0] == 0 and out[0, 1, 0] == 1
    assert np.isnan(out[0, 2, 0])  # bin 20 belongs to trial 1
    assert out[1, 0, 0] == 3


def lagged(lag, trials=150, nb=30, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(trials, nb + 10))
    z = np.apply_along_axis(lambda v: np.convolve(v, np.ones(3) / 3, "same"), 1, z)  # rates are smoothed in time
    a = np.stack([z[:, 5:5 + nb] * rng.normal() + rng.normal(size=(trials, nb)) * 0.3 for _ in range(15)], axis=2)
    b = np.stack([z[:, 5 - lag:5 - lag + nb] * rng.normal() + rng.normal(size=(trials, nb)) * 0.3 for _ in range(15)], axis=2)
    return a, b


def test_jpecc_lag_sign():
    a, b = lagged(4)  # b at time t repeats a at time t-4, so a leads
    r, p = jpecc.jpecc(a, b)
    assert r[10, 14] > 0.5
    assert r[14, 10] < 0.3
    assert np.nanmean(jpecc.asymmetry(r)) > 0.05
    assert np.nanmean(jpecc.asymmetry(r.T)) < -0.05  # swapping the areas flips the sign


def test_nested_anova_equals_regression_reference():
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    rng = np.random.default_rng(5)
    base = rng.normal(size=25)
    task, passive = base + 0.2 + rng.normal(size=25) * 0.3, base + rng.normal(size=25) * 0.3
    df = pd.DataFrame({"y": np.r_[task, passive], "c": np.r_[np.ones(25), np.zeros(25)], "n": np.r_[np.arange(25), np.arange(25)]})
    t = sm.stats.anova_lm(smf.ols("y ~ C(c) + C(n)", df).fit(), typ=2)
    f, p = engagement.nested_anova(task, passive)
    assert f == pytest.approx(t.loc["C(c)", "F"])
    assert p == pytest.approx(t.loc["C(c)", "PR(>F)"])


def test_engagement_vector_recovers_direction_and_index_separates():
    rng = np.random.default_rng(6)
    d = rng.normal(size=40)
    active = rng.normal(size=(300, 40)) + d
    passive = rng.normal(size=(300, 40))
    x = engagement.engagement_vector(active, passive)
    assert np.linalg.norm(x) == pytest.approx(1)
    assert np.corrcoef(x, d)[0, 1] > 0.95


def test_previous_trial_definitions():
    tr = {"stim": np.array([10.0, 20.0]), "fb_time": np.array([11.0, 21.0]), "fb": np.array([1.0, -1.0])}
    iti, prev = engagement.previous_trial(tr)
    assert np.isnan(iti[0]) and iti[1] == 9
    assert prev[1] == 1


def test_psychometric_fit_recovers_parameters():
    rng = np.random.default_rng(7)
    levels = np.array(behavior.LEVELS)
    cl, cr = rng.choice(levels, 6000), rng.choice(levels, 6000)
    true = (-0.8, 4.0, 0.6, -0.5, 3.0)
    pl, pr, pn = behavior.model_probs(true, cl, cr)
    u = rng.random(6000)
    ch = np.where(u < pl, 1, np.where(u < pl + pr, -1, 0))
    got = behavior.fit_psychometric(cl, cr, ch)
    assert got == pytest.approx(true, abs=0.35)


def test_direction_decoding_is_chance_before_and_perfect_after():
    t = np.arange(0, 100, 0.001)
    on = np.arange(1, 90, 1.5)
    ty = np.where(np.arange(len(on)) % 2 == 0, 1, 2)
    mm = np.zeros_like(t)
    for o, k in zip(on, ty):
        s = (t >= o) & (t < o + 0.3)
        mm[s] += (1 if k == 1 else -1) * (t[s] - o) * 50
        mm[t >= o + 0.3] += (1 if k == 1 else -1) * 15
    acc = behavior.direction_decoding(t, mm, on, ty)
    assert acc[0] == pytest.approx(0.5, abs=0.05)
    assert acc[-1] == 1
