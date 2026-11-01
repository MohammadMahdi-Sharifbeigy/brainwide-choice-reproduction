"""Combined-conditions choice probability and detect probability (Ext 7), vectorized from the authors' choiceProbShuf."""
import numpy as np
from scipy.stats import rankdata

from .config import CCCP_SHUFFLES

STEP = 0.01
WIDTH = 0.1  # count window; a 100 ms box centered on 0 is the paper's -50 to +50 ms and on 0.2 its +150 to +250 ms
WINDOWS = {"stim": (-0.2, 0.3), "move": (-0.3, 0.25)}
BATCH = 100


def plan_shuffles(group_a, cond, rng, n_shuf=CCCP_SHUFFLES):
    """Per usable condition: trial indices with group A first, and a shuffles x trials 0/1 matrix.

    Row 0 is the true labelling. Labels are nA positions drawn from nA + nB (the commented randperm line in their code),
    shared across neurons and bins. A condition with only one group adds nothing and is skipped.
    """
    plan = []
    for c in np.unique(cond):
        a = np.flatnonzero((cond == c) & group_a)
        b = np.flatnonzero((cond == c) & ~group_a)
        if len(a) == 0 or len(b) == 0:
            continue
        total = len(a) + len(b)
        m = np.zeros((n_shuf + 1, total), np.float32)
        order = np.argsort(rng.random((n_shuf + 1, total)), axis=1)[:, : len(a)]
        order[0] = np.arange(len(a))
        m[np.arange(n_shuf + 1)[:, None], order] = 1
        plan.append((np.r_[a, b], m, len(a), len(b)))
    return plan


def probability(counts, plan):
    """counts is trials x K. Returns cp and p, length K, with p the rank of the true value among shuffles."""
    k = counts.shape[1]
    n_rep = plan[0][1].shape[0]
    numer = np.zeros((n_rep, k), np.float32)
    den = 0
    for idx, m, na, nb in plan:
        numer += m @ rankdata(counts[idx], axis=0).astype(np.float32) - na * (na + 1) / 2
        den += na * nb
    cp = numer / den
    below = (cp[1:] < cp[0]).sum(0)
    tied = (cp[1:] == cp[0]).sum(0)
    return cp[0], (below + (tied + 1) / 2) / n_rep


def scan(counts_3d, plan):
    """counts_3d is trials x bins x neurons. Returns cp and p as bins x neurons, batching neurons to bound memory."""
    n_tr, nb, n = counts_3d.shape
    cp, p = np.zeros((nb, n), np.float32), np.zeros((nb, n), np.float32)
    for s in range(0, n, BATCH):
        part = counts_3d[:, :, s:s + BATCH]
        c, q = probability(part.reshape(n_tr, -1), plan)
        cp[:, s:s + BATCH] = c.reshape(nb, -1)
        p[:, s:s + BATCH] = q.reshape(nb, -1)
    return cp, p


def sliding_counts(counts, width_bins):
    """Sum over a box of width_bins along axis 1; output bin j covers input bins j to j + width_bins - 1."""
    c = np.cumsum(np.concatenate([np.zeros_like(counts[:, :1]), counts], axis=1), axis=1)
    return c[:, width_bins:] - c[:, :-width_bins]
