"""Joint peri-event canonical correlation (Fig 4d, Ext 8), following the authors' jPECC.m."""
import numpy as np
from scipy import stats


def pc_scores(x, n):
    """x is trials x neurons; top n principal component scores."""
    x = x - x.mean(0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :n] * s[:n]


def _qr_stack(a):
    a = a - a.mean(1, keepdims=True)
    return np.linalg.qr(a)


def jpecc(sp1, sp2, kfold=10, lam=0.5, max_pc=10, seed=0):
    """Cross-validated first canonical correlation per pair of time bins.

    sp1 and sp2 are trials x bins x neurons. Ridge is by the augmented-matrix form of their code. Returns r and p, bins x bins.
    """
    n_tr, nb, _ = sp1.shape
    nd = min(max_pc, sp1.shape[2], sp2.shape[2], n_tr // 2)
    xs = np.stack([pc_scores(sp1[:, t].astype(float), nd) for t in range(nb)])
    ys = np.stack([pc_scores(sp2[:, t].astype(float), nd) for t in range(nb)])
    rng = np.random.default_rng(seed)
    fold = rng.permutation(n_tr) % kfold
    U, V = np.zeros((nb, nb, n_tr)), np.zeros((nb, nb, n_tr))
    eye = lam * np.eye(nd)
    zero = np.zeros((nd, nd))
    for k in range(kfold):
        te = fold == k
        ax = np.concatenate([xs[:, ~te], np.broadcast_to(eye, (nb, nd, nd)), np.broadcast_to(zero, (nb, nd, nd))], axis=1)
        ay = np.concatenate([ys[:, ~te], np.broadcast_to(zero, (nb, nd, nd)), np.broadcast_to(eye, (nb, nd, nd))], axis=1)
        qx, rx = _qr_stack(ax)
        qy, ry = _qr_stack(ay)
        d = qx.transpose(0, 2, 1)[:, None] @ qy[None]
        l, _, mh = np.linalg.svd(d, full_matrices=False)
        a = np.linalg.solve(rx[:, None], l[..., :, 0:1])
        b = np.linalg.solve(ry[None], mh.transpose(0, 1, 3, 2)[..., :, 0:1])
        U[:, :, te] = (xs[:, te][:, None] @ a)[..., 0]
        V[:, :, te] = (ys[:, te][None] @ b)[..., 0]
    U -= U.mean(2, keepdims=True)
    V -= V.mean(2, keepdims=True)
    su, sv = np.sqrt((U**2).sum(2)), np.sqrt((V**2).sum(2))
    r = np.divide((U * V).sum(2), su * sv, out=np.zeros((nb, nb)), where=su * sv > 0)
    t = r * np.sqrt((n_tr - 2) / np.maximum(1 - r**2, 1e-12))
    return r, 2 * stats.t.sf(np.abs(t), n_tr - 2)


def asymmetry(r, bin_size=0.01, span=0.05):
    """Lead-lag index per time. r[i, j] pairs area A at time i with area B at time j; positive means A leads."""
    k = int(round(span / bin_size))
    nb = len(r)
    out = np.full(nb, np.nan)
    for t in range(k, nb - k):
        steps = np.arange(1, k + 1)
        out[t] = r[t - steps, t + steps].mean() - r[t + steps, t - steps].mean()
    return out
