"""Reduced-rank kernel regression, following the authors' kernelFitBasic, regFullPipeline and nested-test code."""
import numpy as np
from scipy import sparse

from . import store
from .config import (BASIS_DIMS, DT, ENET_ALPHA, ENET_LAMBDA, FULL_FOLDS, FULL_RANKS, LC, LD, MOVE_WIN, NESTED_EVAL,
                     NESTED_FOLDS, NESTED_RANKS, NPRED, RESID_COMPS, STIM_WIN, VE_THRESHOLD)
from .rates import bin_rates, smooth

TESTS = {"vision_ipsi": (0, 270), "vision_contra": (270, 540), "action": (540, 595), "choice": (595, 650)}
ACTION = TESTS["action"]
CHOICE = TESTS["choice"]


def move_direction(tr):
    """D_m per trial: +1 Left (ipsi), -1 Right (contra), 0 without a movement."""
    return tr["has_left"].astype(float) - tr["has_right"].astype(float)


def shuffled_direction(tr, rng):
    d = move_direction(tr)
    go = np.flatnonzero((d != 0) & tr["kernel_trial"])
    d[go] = d[rng.permutation(go)]
    return d


def trial_geometry(tr):
    keep = tr["kernel_trial"]
    stim = tr["stim"][keep]
    has = (tr["has_left"] | tr["has_right"])[keep]
    mv = np.where(has, tr["move_abs"][keep], np.inf)
    end = np.minimum(stim + STIM_WIN[1], mv + MOVE_WIN[1])
    return stim, has, mv, stim + STIM_WIN[0], end


def build_design(tr, move_dir=None, dt=DT):
    """Predictors for samples inside trial windows. Returns P (rows x 650 csr), rows, trnum and the two evaluation masks.

    Lag indexing reproduces KernelModel: a kernel's first lag lands on bin round((t - dt/2 + w0)/dt) - 1, one bin
    earlier than its nominal time.
    """
    stim, has, mv, start, end = trial_geometry(tr)
    keep = tr["kernel_trial"]
    i0 = np.ceil(start / dt - 0.5).astype(int)
    i1 = np.floor(end / dt - 0.5).astype(int)
    rows = np.concatenate([np.arange(a, b + 1) for a, b in zip(i0, i1)])
    trnum = np.concatenate([np.full(b - a + 1, k) for k, (a, b) in enumerate(zip(i0, i1))])
    rows, first = np.unique(rows, return_index=True)
    trnum = trnum[first]
    nb = int(rows.max()) + 200
    row_of = -np.ones(nb, int)
    row_of[rows] = np.arange(len(rows))

    d = move_direction(tr) if move_dir is None else move_dir
    moved = (tr["has_left"] | tr["has_right"]) & keep
    events = []
    for side, con in (("cl", tr["cl"]), ("cr", tr["cr"])):
        for c in (0.25, 0.5, 1.0):
            m = keep & (con == c)
            events.append((tr["stim"][m], np.ones(m.sum()), STIM_WIN[0], LC))
    events.append((tr["move_abs"][moved], np.ones(moved.sum()), MOVE_WIN[0], LD))
    events.append((tr["move_abs"][moved], d[moved], MOVE_WIN[0], LD))

    r, c, v = [], [], []
    col = 0
    for times, values, w0, lags in events:
        base = np.round((times - dt / 2 + w0) / dt).astype(int) - 1
        idx = base[:, None] + np.arange(lags)
        val = np.repeat(values[:, None], lags, axis=1)
        cols = np.repeat(np.arange(col, col + lags)[None], len(times), axis=0)
        inside = (idx >= 0) & (idx < nb)
        row = np.full(idx.shape, -1)
        row[inside] = row_of[idx[inside]]
        ok = row >= 0
        r.append(row[ok]); c.append(cols[ok]); v.append(val[ok])
        col += lags
    P = sparse.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(len(rows), NPRED))

    tc = (rows + 0.5) * dt
    s, m_t = stim[trnum], mv[trnum]
    h = has[trnum]
    eval_full = (tc >= s) & (tc <= np.minimum(m_t, s + STIM_WIN[1]))
    (sa, sb), (ma, mb) = NESTED_EVAL
    eval_nested = ((tc >= s + sa) & (tc <= np.minimum(m_t, s + sb))) | (h & (tc >= np.maximum(s, m_t + ma)) & (tc <= m_t + mb))
    return P, rows, trnum, eval_full, eval_nested


def target_rates(st, clu, cluster_ids, tr, rows, dt=DT, batch=100):
    """Smoothed rates at `rows` minus each neuron's pre-stimulus mean, in neuron batches to bound memory. Returns F and the means."""
    stim = tr["stim"][tr["kernel_trial"]]
    off = np.arange(-0.2, 1e-9, dt)
    t_end = tr["fb_time"][-1] + 5
    F = np.zeros((len(rows), len(cluster_ids)), np.float32)
    means = np.zeros(len(cluster_ids), np.float32)
    for s in range(0, len(cluster_ids), batch):
        r = smooth(bin_rates(st, clu, cluster_ids[s:s + batch], t_end, dt), 0.025, axis=0, dt=dt)
        idx = np.clip(((stim[:, None] + off) / dt).astype(int), 0, len(r) - 1)
        bsl = r[idx.ravel()].mean(0)
        F[:, s:s + batch] = r[np.minimum(rows, len(r) - 1)] - bsl
        means[s:s + batch] = bsl
    return F, means


def basis_terms(P, F):
    """Sums one session contributes to the pooled basis; `CanonCor2all` needs the means, not only the products."""
    T = P.shape[0]
    ps = np.asarray(P.sum(0)).ravel()
    fs = np.asarray(F, float).sum(0)
    cyx = (P.T @ np.asarray(F, float) - np.outer(ps, fs) / T) / (T - 1)
    return (P.T @ P).toarray().astype(float), ps, T, cyx


def solve_basis(terms):
    """The authors' CanonCor2all from streamed terms. Returns b, predictors x components, best components first."""
    ptp = sum(t[0] for t in terms)
    ps = sum(t[1] for t in terms)
    n = sum(t[2] for t in terms)
    cyx = np.hstack([t[3] for t in terms])  # predictors x all neurons
    cxx = (ptp - np.outer(ps, ps) / n) / (n - 1) + 1e-7 * np.eye(len(ps))
    e, v = np.linalg.eigh(cxx)
    mh = (v / np.sqrt(np.maximum(e, 1e-12))) @ v.T
    m = mh @ cyx
    e2, c = np.linalg.eigh(m @ m.T)
    return mh @ c[:, ::-1]


def fit_enet(X, Y, lam=ENET_LAMBDA, alpha=ENET_ALPHA, tol=1e-6, maxit=20000):
    """Elastic net with glmnet's scaling, one weight column per neuron; intercepts are not returned.

    Columns of X and of Y are scaled by their 1/T standard deviations, and the penalty becomes lam/ys for each neuron,
    as in GLMnet.f, so the fit depends on the neuron's scale. Solved by accelerated proximal gradient on all neurons at
    once, because they share one Gram matrix.
    """
    T = len(X)
    Y = np.asarray(Y, float).reshape(T, -1)
    xs = X.std(0)
    use = xs > 0
    Xc = (X[:, use] - X[:, use].mean(0)) / xs[use]
    ys = Y.std(0)
    ok = np.flatnonzero(ys > 1e-12)
    beta = np.zeros((X.shape[1], Y.shape[1]))
    if len(ok) == 0 or not use.any():
        return beta
    G = Xc.T @ Xc / T
    c = Xc.T @ (Y[:, ok] - Y[:, ok].mean(0)) / T / ys[ok]
    step = 1 / np.linalg.eigvalsh(G)[-1]
    lam_s = lam / ys[ok]
    thresh = step * lam_s * alpha
    shrink = 1 / (1 + step * lam_s * (1 - alpha))

    W = np.zeros_like(c)
    Z = W.copy()
    t = np.ones(c.shape[1])
    active = np.arange(c.shape[1])
    for _ in range(maxit):
        z = Z[:, active]
        grad = G @ z - c[:, active]
        new = np.sign(z - step * grad) * np.maximum(np.abs(z - step * grad) - thresh[active], 0) * shrink[active]
        delta = new - W[:, active]
        W[:, active] = new
        t_new = (1 + np.sqrt(1 + 4 * t[active] ** 2)) / 2
        restart = np.sum((z - new) * delta, axis=0) > 0
        mom = (t[active] - 1) / t_new
        mom[restart] = 0
        t_new[restart] = 1
        Z[:, active] = new + delta * mom
        t[active] = t_new
        moving = np.abs(delta).max(0) > tol
        if not moving.any():
            break
        active = active[moving]
    beta[np.ix_(np.flatnonzero(use), ok)] = (W * ys[ok]) / xs[use][:, None]
    return beta


def trial_folds(trnum, k, rng):
    n = int(trnum.max()) + 1
    per_trial = rng.permutation(n) % k
    return per_trial[trnum]


def score_ranks(X, Y, folds, mask, ranks):
    """Cross-validated variance explained per rank, with one fit per fold and truncation (no refit), as in their code.

    Returns ve (ranks x neurons) and the per-fold weights.
    """
    N = Y.shape[1]
    Y = np.asarray(Y, float)
    sum_r, sum_r2 = np.zeros((ranks, N)), np.zeros((ranks, N))
    betas = []
    for k in np.unique(folds):
        te = folds == k
        beta = fit_enet(X[~te], Y[~te])
        betas.append(beta)
        use = te & mask
        Xte, res = X[use], Y[use].copy()
        for r in range(ranks):
            res -= np.outer(Xte[:, r], beta[r])
            sum_r[r] += res.sum(0)
            sum_r2[r] += (res**2).sum(0)
    n = mask.sum()
    var_r = (sum_r2 - sum_r**2 / n) / (n - 1)
    var_y = Y[mask].var(0, ddof=1)
    ve = np.zeros((ranks, N))
    good = var_y > 1e-3
    ve[:, good] = 1 - var_r[:, good] / var_y[good]
    return ve, betas


def predict_at_rank(X, betas, folds, nd):
    """Cross-validated prediction with each neuron at its own rank."""
    out = np.zeros((len(X), betas[0].shape[1]), np.float32)
    for k, beta in zip(np.unique(folds), betas):
        te = np.flatnonzero(folds == k)
        for r in np.unique(nd):
            n = nd == r
            out[np.ix_(te, n)] = X[te][:, :r] @ beta[:r][:, n]
    return out


def kernels(B, beta, nd):
    """Kernels, predictors x neurons, each at the neuron's rank."""
    out = np.zeros((B.shape[0], beta.shape[1]), np.float32)
    for r in np.unique(nd):
        n = nd == r
        out[:, n] = B[:, :r] @ beta[:r][:, n]
    return out


def split_kernels(K):
    """Named pieces of a 650 x N kernel matrix."""
    return {"vision_ipsi": K[:270].reshape(3, LC, -1), "vision_contra": K[270:540].reshape(3, LC, -1),
            "action": K[ACTION[0]:ACTION[1]], "choice": K[CHOICE[0]:CHOICE[1]]}


def rates_at_move(K, t=0.0):
    """Predicted contra and ipsi rate at time t from movement. D_m is +1 Left (ipsi), -1 Right (contra)."""
    lag = int(round((t - MOVE_WIN[0]) / DT))
    p = split_kernels(K)
    return p["action"][lag] - p["choice"][lag], p["action"][lag] + p["choice"][lag]


def prefers_contra(K):
    """True where the mean Choice kernel from -0.1 to 0 s is negative, meaning Action - Choice exceeds Action + Choice."""
    lags = np.arange(int(round((-0.1 - MOVE_WIN[0]) / DT)), int(round(-MOVE_WIN[0] / DT)))
    return split_kernels(K)["choice"][lags].mean(0) < 0


# ---- pooled stages

def _paths(cfg, variant):
    return cfg.results / "regression", cfg.results / "regression" / variant


def _design(cfg, s, variant):
    base = store.load(cfg.results / "regression" / "data" / f"{s}.npz")
    tr = store.load(cfg.results / "base" / f"{s}.npz")
    d = None
    if variant == "shuffle":
        d = shuffled_direction(tr, store.session_rng(cfg.seed, s, 7))
    P, rows, trnum, eval_full, eval_nested = build_design(tr, d)
    return P, base["F"], trnum, eval_full, eval_nested


def _full_terms(job):
    cfg, s, variant = job
    P, F, *_ = _design(cfg, s, variant)
    return basis_terms(P, F)


def _full_fit(job):
    cfg, s, variant = job
    _, root = _paths(cfg, variant)
    out = root / "full" / f"{s}.npz"
    if out.exists():
        return
    B = np.load(root / "basis_full.npy")
    P, F, trnum, eval_full, _ = _design(cfg, s, variant)
    X = P @ B
    folds = trial_folds(trnum, FULL_FOLDS, store.session_rng(cfg.seed, s, 1))
    ve, betas = score_ranks(X, F, folds, eval_full, FULL_RANKS)
    best, nd = ve.max(0), ve.argmax(0) + 1
    K = kernels(B, fit_enet(X, F), nd)
    store.save(out, ve=best.astype(np.float32), nd=nd, K=K, cvpred=predict_at_rank(X, betas, folds, nd))


def _included(cfg, s, variant):
    return store.load(cfg.results / "regression" / variant / "full" / f"{s}.npz")["ve"] >= VE_THRESHOLD


def _no_key_terms(job):
    cfg, s, variant, key = job
    a, b = TESTS[key]
    P, F, *_ = _design(cfg, s, variant)
    other = np.r_[0:a, b:NPRED]
    return basis_terms(P[:, other], F[:, _included(cfg, s, variant)])


def _residuals(job):
    cfg, s, variant, key = job
    _, root = _paths(cfg, variant)
    out = root / f"resid_{key}" / f"{s}.npz"
    if out.exists():
        return
    a, b = TESTS[key]
    B = np.load(root / f"basis_no_{key}.npy")
    P, F, *_ = _design(cfg, s, variant)
    inc = _included(cfg, s, variant)
    X = P[:, np.r_[0:a, b:NPRED]] @ B[:, :RESID_COMPS]
    Y = F[:, inc]
    store.save(out, resid=(Y - X @ fit_enet(X, Y)).astype(np.float32))


def _only_key_terms(job):
    cfg, s, variant, key = job
    a, b = TESTS[key]
    P, *_ = _design(cfg, s, variant)
    return basis_terms(P[:, a:b], store.load(cfg.results / "regression" / variant / f"resid_{key}" / f"{s}.npz")["resid"])


def _nested(job):
    cfg, s, variant, key = job
    _, root = _paths(cfg, variant)
    out = root / f"nested_{key}" / f"{s}.npz"
    if out.exists():
        return
    a, b = TESTS[key]
    B = np.load(root / f"basis_only_{key}.npy")
    P, _, trnum, _, eval_nested = _design(cfg, s, variant)
    Y = store.load(root / f"resid_{key}" / f"{s}.npz")["resid"]
    X = P[:, a:b] @ B
    folds = trial_folds(trnum, NESTED_FOLDS, store.session_rng(cfg.seed, s, 2))
    ve, _ = score_ranks(X, Y, folds, eval_nested, min(NESTED_RANKS, X.shape[1]))
    store.save(out, ve=ve.max(0).astype(np.float32), nd=ve.argmax(0) + 1)


def run_pooled(cfg, sessions, pmap, variant="real", tests=tuple(TESTS)):
    """Full model then nested tests for all sessions, one step at a time, skipping finished work."""
    _, root = _paths(cfg, variant)
    root.mkdir(parents=True, exist_ok=True)
    basis = root / "basis_full.npy"
    if not basis.exists():
        terms = pmap(_full_terms, [(cfg, s, variant) for s in sessions])
        np.save(basis, solve_basis(terms)[:, :BASIS_DIMS])
    pmap(_full_fit, [(cfg, s, variant) for s in sessions])
    for key in tests:
        jobs = [(cfg, s, variant, key) for s in sessions if _included(cfg, s, variant).any()]
        no = root / f"basis_no_{key}.npy"
        if not no.exists():
            np.save(no, solve_basis(pmap(_no_key_terms, jobs))[:, :RESID_COMPS])
        pmap(_residuals, jobs)
        only = root / f"basis_only_{key}.npy"
        if not only.exists():
            np.save(only, solve_basis(pmap(_only_key_terms, jobs)))
        pmap(_nested, jobs)


# ---- model comparison (Ext 5e, f)

def cosine_design(P, trnum):
    """The authors' makeCosReg: raised cosines 100 ms wide, spaced 25 ms, 130 predictors."""
    cos = (1 - np.cos(np.linspace(0, 2 * np.pi, 20))) / 2
    A = P.toarray()
    seeds = np.concatenate([np.arange(a + 2, b, 5) for a, b in
                            [(i * LC, (i + 1) * LC) for i in range(6)] + [(6 * LC, 6 * LC + LD), (6 * LC + LD, 6 * LC + 2 * LD)]])
    out = np.zeros((len(A), len(seeds)))
    for t in np.unique(trnum):
        idx = np.flatnonzero(trnum == t)
        for j, c in enumerate(seeds):
            if A[idx, c].any():
                out[idx, j] = np.convolve(A[idx, c], cos, "same")
    return out


def train_test_ve(X, Y, folds, mask):
    """Train and test variance explained, all columns, 5-fold."""
    Y = np.asarray(Y, float)
    pred_te = np.zeros_like(Y)
    tr_err = 0
    for k in np.unique(folds):
        te = folds == k
        beta = fit_enet(X[~te], Y[~te], tol=1e-5)
        pred_te[te] = X[te] @ beta
        m = ~te & mask
        tr_err = tr_err + (Y[m] - X[m] @ beta).var(0, ddof=1) / len(np.unique(folds))
    v = Y[mask].var(0, ddof=1)
    good = v > 1e-3
    test, train = np.zeros(Y.shape[1]), np.zeros(Y.shape[1])
    test[good] = 1 - (Y[mask] - pred_te[mask]).var(0, ddof=1)[good] / v[good]
    train[good] = 1 - tr_err[good] / v[good]
    return train, test


def compare_session(job):
    cfg, s = job
    out = cfg.results / "regression" / "compare" / f"{s}.npz"
    if out.exists():
        return
    B = np.load(cfg.results / "regression" / "real" / "basis_full.npy")[:, :NESTED_RANKS]
    P, F, trnum, eval_full, _ = _design(cfg, s, "real")
    folds = trial_folds(trnum, FULL_FOLDS, store.session_rng(cfg.seed, s, 3))
    res = {}
    for name, X in (("reduced", P @ B), ("toeplitz", P.toarray()), ("cosine", cosine_design(P, trnum))):
        res[f"{name}_train"], res[f"{name}_test"] = train_test_ve(X, F, folds, eval_full)
    store.save(out, **{k: v.astype(np.float32) for k, v in res.items()})
