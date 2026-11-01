"""Solver, basis, predictors and known-kernel recovery."""
import numpy as np
import pytest
from sklearn.linear_model import ElasticNet

from steinmetz import regression as rr
from steinmetz.config import DT, LC, LD, NPRED


def glmnet_reference(X, y, lam=0.5, alpha=0.5):
    """Same objective through coordinate descent, with GLMnet.f's scaling written out."""
    xs, ys = X.std(0), y.std()
    m = ElasticNet(alpha=lam / ys, l1_ratio=alpha, tol=1e-10, max_iter=200000).fit(X / xs, y / ys)
    return ys * m.coef_ / xs


def test_fit_enet_matches_reference_across_scales():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 30)) * rng.uniform(0.5, 3, 30)
    w = np.zeros(30)
    w[:4] = [2, -1, 1.5, 3]
    Y = np.column_stack([X @ w * s + rng.normal(size=4000) * n for s, n in ((1, 12), (0.2, 3), (3, 40))])
    got = rr.fit_enet(X, Y, tol=1e-9)
    for j in range(3):
        assert got[:, j] == pytest.approx(glmnet_reference(X, Y[:, j]), abs=2e-4)


def test_keeping_lambda_at_half_with_scaled_y_is_wrong():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5000, 50)) * rng.uniform(0.5, 3, 50)
    y = X[:, :5].sum(1) * 2 + rng.normal(size=5000) * 12
    xs, ys = X.std(0), y.std()
    wrong = ElasticNet(alpha=0.5, l1_ratio=0.5, tol=1e-9, max_iter=100000).fit(X / xs, y / ys).coef_
    right = rr.fit_enet(X, y[:, None], tol=1e-9)[:, 0]
    assert (wrong != 0).sum() < (np.abs(right) > 1e-9).sum()


def test_low_variance_neuron_gets_zero_weights_not_nan():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(1000, 10))
    Y = np.column_stack([np.zeros(1000), rng.normal(size=1000) * 1e-3, X[:, 0] * 10])
    b = rr.fit_enet(X, Y)
    assert np.isfinite(b).all()
    assert (b[:, 0] == 0).all() and (b[:, 1] == 0).all()
    assert abs(b[0, 2]) > 1


def batch_canoncor(xs, ys):
    allx = np.vstack(xs)
    cxx = np.cov(allx.T) + 1e-7 * np.eye(allx.shape[1])
    cyx = np.vstack([(y - y.mean(0)).T @ (x - x.mean(0)) / (len(x) - 1) for x, y in zip(xs, ys)])
    e, v = np.linalg.eigh(cxx)
    mh = (v / np.sqrt(e)) @ v.T
    _, s, c = np.linalg.svd(cyx @ mh, full_matrices=False)
    return mh @ c.T, s


def test_streamed_basis_equals_batch():
    from scipy import sparse
    rng = np.random.default_rng(3)
    xs, ys = [], []
    for k, (t, n) in enumerate(((800, 30), (1200, 50), (600, 20))):
        x = (rng.random((t, 40)) < 0.03).astype(float)
        xs.append(x)
        ys.append(x @ rng.normal(size=(40, n)) + rng.normal(size=(t, n)) * 2 + k)
    b, _ = batch_canoncor(xs, ys)
    got = rr.solve_basis([rr.basis_terms(sparse.csr_matrix(x), y) for x, y in zip(xs, ys)])
    for j in range(5):  # same component up to sign
        a = got[:, j] / np.linalg.norm(got[:, j])
        c = b[:, j] / np.linalg.norm(b[:, j])
        assert abs(a @ c) == pytest.approx(1, abs=1e-6)


def fake_trials(n=120, seed=0, trial_gap=3.0):
    rng = np.random.default_rng(seed)
    stim = 5 + np.arange(n) * trial_gap + rng.uniform(0, 0.5, n)
    go = rng.random(n) < 0.7
    left = go & (rng.random(n) < 0.5)
    right = go & ~left
    ft = rng.uniform(0.15, 0.35, n)
    cl = rng.choice([0, 0.25, 0.5, 1.0], n)
    cr = rng.choice([0, 0.25, 0.5, 1.0], n)
    return dict(stim=stim, kernel_trial=np.ones(n, bool), has_left=left, has_right=right, move_abs=stim + ft, cl=cl, cr=cr)


def test_predictor_lag_indexing_matches_kernelmodel():
    tr = dict(stim=np.array([10.0025]), kernel_trial=np.ones(1, bool), has_left=np.zeros(1, bool), has_right=np.zeros(1, bool),
              move_abs=np.array([np.nan]), cl=np.array([0.25]), cr=np.array([0.0]))
    P, rows, *_ = rr.build_design(tr)
    assert P.shape[1] == NPRED == 650
    # round((t - dt/2 - 0.05)/dt) - 1 = 1989 falls just before the window, so its first bin (1990) carries lag 1
    assert rows[0] == 1990
    A = P.toarray()
    assert A[0, 1] == 1 and A[0, 0] == 0
    assert A[1, 2] == 1
    assert A[:, LC:].sum() == 0


def test_eval_masks_stop_at_movement():
    tr = fake_trials()
    P, rows, trnum, ef, en = rr.build_design(tr)
    tc = (rows + 0.5) * DT
    t = np.flatnonzero(tr["has_left"] | tr["has_right"])[0]
    m = trnum == t
    assert tc[m & ef].max() <= tr["move_abs"][t]
    assert tc[m & en].max() <= tr["move_abs"][t]


def simulate(tr, seed=1, n=60):
    """Rates built from planted kernels: neuron 0 prefers contra, 1 prefers ipsi, 2 is action-only, 3 vision-only, the rest noise."""
    rng = np.random.default_rng(seed)
    P, rows, trnum, ef, en = rr.build_design(tr)
    bump = np.exp(-(((np.arange(LD) - 35) / 8.0) ** 2))
    vb = np.exp(-(((np.arange(LC) - 30) / 8.0) ** 2))
    K = np.zeros((NPRED, n))
    K[rr.CHOICE[0]:rr.CHOICE[1], 0] = -20 * bump  # Right bigger: contra preference
    K[rr.ACTION[0]:rr.ACTION[1], 0] = 25 * bump
    K[rr.CHOICE[0]:rr.CHOICE[1], 1] = 20 * bump  # Left bigger: ipsi preference
    K[rr.ACTION[0]:rr.ACTION[1], 1] = 25 * bump
    K[rr.ACTION[0]:rr.ACTION[1], 2] = 30 * bump
    for c in range(3):
        K[270 + c * LC: 270 + (c + 1) * LC, 3] = (c + 1) * 15 * vb
    F = P @ K + rng.normal(size=(P.shape[0], n)) * 4
    return P, F, trnum, ef, en, K


@pytest.fixture(scope="module")
def planted():
    tr = fake_trials(n=240)
    P, F, trnum, ef, en, K = simulate(tr)
    B = rr.solve_basis([rr.basis_terms(P, F)])[:, :60]
    X = P @ B
    folds = rr.trial_folds(trnum, 5, np.random.default_rng(0))
    ve, betas = rr.score_ranks(X, F, folds, ef, 20)
    nd = ve.argmax(0) + 1
    Khat = rr.kernels(B, rr.fit_enet(X, F), nd)
    return tr, P, F, trnum, ef, en, K, ve.max(0), Khat


def test_full_model_explains_planted_neurons_only(planted):
    *_, ve, _ = planted
    assert (ve[:4] > 0.1).all()
    assert (ve[4:] < 0.02).all()


def test_contra_ipsi_sign_is_locked(planted):
    *_, Khat = planted
    pref = rr.prefers_contra(Khat)
    assert pref[0] and not pref[1]
    contra, ipsi = rr.rates_at_move(Khat, -0.02)
    assert contra[0] > ipsi[0] and ipsi[1] > contra[1]


def nested_ve(P, F, trnum, en, key, seed=0):
    a, b = rr.TESTS[key]
    other = np.r_[0:a, b:NPRED]
    Bno = rr.solve_basis([rr.basis_terms(P[:, other], F)])[:, :40]
    X0 = P[:, other] @ Bno
    res = np.asarray(F, float) - X0 @ rr.fit_enet(X0, F)
    Bk = rr.solve_basis([rr.basis_terms(P[:, a:b], res)])
    X1 = P[:, a:b] @ Bk
    folds = rr.trial_folds(trnum, 10, np.random.default_rng(seed))
    ve, _ = rr.score_ranks(X1, res, folds, en, 12)
    return ve.max(0)


def test_nested_choice_flags_choice_neurons_only(planted):
    tr, P, F, trnum, ef, en, *_ = planted
    ve = nested_ve(P[:], F[:, :4], trnum, en, "choice")
    assert ve[0] > 0.02 and ve[1] > 0.02
    assert ve[2] < 0.02 and ve[3] < 0.02


def test_shuffled_choice_lowers_choice_variance(planted):
    tr, P, F, trnum, ef, en, *_ = planted
    d = rr.shuffled_direction(tr, np.random.default_rng(5))
    Ps = rr.build_design(tr, d)[0]
    ve = nested_ve(Ps, F[:, :2], trnum, en, "choice")
    real = nested_ve(P, F[:, :2], trnum, en, "choice")
    assert (ve < real / 3).all()


def test_cosine_design_shape():
    tr = fake_trials()
    P, rows, trnum, *_ = rr.build_design(tr)
    assert rr.cosine_design(P, trnum).shape[1] == 130
