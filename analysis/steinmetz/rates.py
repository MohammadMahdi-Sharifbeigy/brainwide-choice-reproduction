"""Spike binning, causal smoothing and event-aligned rates."""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import mannwhitneyu, wilcoxon

from .config import DT


def half_gaussian(sd, dt=DT):
    """Causal half-Gaussian: weight only at lag >= 0, summing to one."""
    t = np.arange(0, 5 * sd + dt / 2, dt)
    g = np.exp(-(t**2) / (2 * sd**2))
    return g / g.sum()


def smooth(x, sd, axis=0, dt=DT):
    return lfilter(half_gaussian(sd, dt).astype(x.dtype), 1.0, x, axis=axis)


def bin_rates(times, clusters, cluster_ids, t_end, dt=DT):
    """T x N rate matrix in sp/s for the given clusters, as float32."""
    nb = int(np.ceil(t_end / dt))
    lookup = -np.ones(int(max(clusters.max(), np.max(cluster_ids))) + 1, int)
    lookup[cluster_ids] = np.arange(len(cluster_ids))
    col = lookup[clusters]
    keep = (col >= 0) & (times > 0) & (times < t_end)
    row = np.minimum((times[keep] / dt).astype(int), nb - 1)
    out = np.zeros((nb, len(cluster_ids)), np.float32)
    np.add.at(out, (row, col[keep]), 1 / dt)
    return out


def aligned_counts(times, clusters, cluster_ids, events, win, dt=DT):
    """Spike counts as events x bins x neurons. `times` must be sorted."""
    edges = np.arange(int(round((win[1] - win[0]) / dt)) + 1) * dt + win[0]
    nb = len(edges) - 1
    lookup = -np.ones(int(max(clusters.max(), np.max(cluster_ids))) + 1, int)
    lookup[cluster_ids] = np.arange(len(cluster_ids))
    out = np.zeros((len(events), nb, len(cluster_ids)), np.float32)
    lo = np.searchsorted(times, events + edges[0])
    hi = np.searchsorted(times, events + edges[-1])
    for e, (a, b) in enumerate(zip(lo, hi)):
        col = lookup[clusters[a:b]]
        keep = col >= 0
        rows = np.minimum(((times[a:b][keep] - events[e] - edges[0]) / dt).astype(int), nb - 1)
        np.add.at(out[e], (rows, col[keep]), 1)
    return out, edges[:-1] + dt / 2


def aligned_rates(times, clusters, cluster_ids, events, win, sd, dt=DT):
    """Causally smoothed sp/s, events x bins x neurons, with padding so the first bin has full history."""
    pad = 5 * sd
    counts, t = aligned_counts(times, clusters, cluster_ids, events, (win[0] - pad, win[1]), dt)
    r = smooth(counts / dt, sd, axis=1, dt=dt).astype(np.float32)
    start = int(round(pad / dt))
    return r[:, start:], t[start:]


def display_psth(trial_rates, sd2=0.03, dt=DT):
    """Second smoothing step of the Methods: average trials, then smooth again."""
    return smooth(trial_rates.mean(0), sd2, axis=0, dt=dt)


def window_mean(rates, bin_centers, a, b):
    """Mean over bins whose center lies in [a, b], along axis 1 of events x bins x neurons."""
    m = (bin_centers >= a) & (bin_centers <= b)
    return rates[:, m].mean(1)


def significance_z(post, baseline):
    """Signed z of the post-stimulus mean against trial-by-trial baseline, using the s.e. across trials.

    post is trials x neurons (0 to 0.25 s), baseline trials x neurons (-0.1 to 0 s). Equals the paper's t-test form.
    """
    n = len(baseline)
    s = baseline.std(0, ddof=1) / np.sqrt(n)
    return np.divide(post.mean(0) - baseline.mean(0), s, out=np.zeros(post.shape[1]), where=s > 0)


def _sign_rank(x):
    """Two-sided p per column; columns of all zeros give NaN."""
    out = np.full(x.shape[1], np.nan)
    use = np.any(x != 0, axis=0)
    if use.any():
        out[use] = wilcoxon(x[:, use], axis=0, method="asymptotic", zero_method="wilcox").pvalue
    return out


def _rank_sum(a, b):
    out = np.full(a.shape[1], np.nan)
    use = np.any(a != 0, axis=0) | np.any(b != 0, axis=0)
    if use.any() and len(a) and len(b):
        out[use] = mannwhitneyu(a[:, use], b[:, use], axis=0, method="asymptotic").pvalue
    return out


def six_tests(tr_rate, bsl, stim_rate, mv_rate, mv_bsl, hL, hR, peri_rate, rew_rate):
    """The Methods' six p-values, neurons x 6, each input trials x neurons."""
    p = np.column_stack([
        _sign_rank(tr_rate - bsl),
        _sign_rank(stim_rate - bsl),
        _sign_rank(mv_rate - mv_bsl),
        _rank_sum(mv_rate[hR], mv_rate[hL]),
        _sign_rank(peri_rate - mv_bsl),
        _rank_sum(rew_rate, bsl),
    ])
    return p
