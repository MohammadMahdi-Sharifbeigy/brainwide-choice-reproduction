"""Movement detection, trial sets and contrast matching."""
import numpy as np

from .config import CONTRASTS, MIN_MATCHED_TRIALS, WIN_COUNT_MOVE, WIN_NO_MOVE


def find_moves(stim, fb_time, move_on, move_type):
    """First movement of each trial, following the authors' findMoveTimes.

    Returns first_time, first_type (NaN when no movement came before feedback) and has_none. has_none keeps their
    quirk: it holds trials whose first movement came before feedback but outside the window, so a fully still trial
    is not in it.
    """
    n = len(stim)
    first_time, first_type = np.full(n, np.nan), np.full(n, np.nan)
    has_none = np.zeros(n, bool)
    idx = np.searchsorted(move_on, stim - 0.1, side="right")
    for t in range(n):
        if idx[t] >= len(move_on):
            continue
        mt = move_on[idx[t]]
        if mt < fb_time[t]:
            first_time[t] = mt - stim[t]
            first_type[t] = move_type[idx[t]]
            has_none[t] = first_time[t] < WIN_NO_MOVE[0] or first_time[t] > WIN_NO_MOVE[1]
    return first_time, first_type, has_none


def overlaps_movement(start, stop, move_on, move_off):
    """True for each [start, stop] window that any detected movement touches."""
    i = np.searchsorted(move_on, stop, side="right")  # moves that began by the window's end
    latest_off = np.maximum.accumulate(move_off) if len(move_off) else move_off
    out = np.zeros(len(start), bool)
    has = i > 0
    out[has] = latest_off[i[has] - 1] >= start[has]
    return out


def build_trials(session):
    """All per-trial arrays that later stages need, in one dict."""
    tr = session.trials()
    on, off, kind = session.moves()
    stim, fb_time = tr["stim"], tr["fb_time"]
    first_time, first_type, has_none = find_moves(stim, fb_time, on, kind)
    in_win = (first_time > WIN_COUNT_MOVE[0]) & (first_time < WIN_COUNT_MOVE[1])
    has_left, has_right = in_win & (first_type == 1), in_win & (first_type == 2)
    still = ~overlaps_movement(stim + WIN_NO_MOVE[0], stim + WIN_NO_MOVE[1], on, off)
    inc = tr["included"]
    tr.update(
        first_time=first_time, first_type=first_type, has_left=has_left, has_right=has_right, has_none=has_none,
        still=still,
        kernel_trial=inc & (has_left | has_right | has_none),
        move_abs=stim + first_time,
        go=inc & (has_left | has_right),
        nogo=inc & still,
        miss=inc & still & ((tr["cl"] > 0) | (tr["cr"] > 0)),
        correct_contra=inc & has_right & (tr["fb"] == 1) & (tr["cr"] > 0),
        correct_ipsi=inc & has_left & (tr["fb"] == 1) & (tr["cl"] > 0),
        miss_contra=inc & still & (tr["cr"] > 0),
        miss_ipsi=inc & still & (tr["cl"] > 0) & (tr["cr"] == 0),
    )
    pa = session.passive()
    pa["still"] = ~overlaps_movement(pa["times"] + WIN_NO_MOVE[0], pa["times"] + WIN_NO_MOVE[1], on, off)
    tr.update(passive_times=pa["times"], passive_cl=pa["cl"], passive_cr=pa["cr"], passive_still=pa["still"])
    return tr


def pseudo_onsets(tr, rng):
    """Movement time per trial, drawing one for trials without a movement from real onsets, with replacement."""
    out = tr["first_time"].copy()
    real = tr["first_time"][(tr["has_left"] | tr["has_right"]) & tr["included"]]
    no_move = ~(tr["has_left"] | tr["has_right"])
    out[no_move] = rng.choice(real, no_move.sum(), replace=True)
    return out


def match_contrast(groups, rng, contrasts=CONTRASTS, min_trials=MIN_MATCHED_TRIALS):
    """Equal trials per contra contrast across groups.

    groups maps a name to (mask, contrast array). Returns name -> indices, or None when under min_trials remain.
    """
    picked = {k: [] for k in groups}
    for c in contrasts:
        pools = {k: np.flatnonzero(m & (con == c)) for k, (m, con) in groups.items()}
        n = min(len(p) for p in pools.values())
        for k, p in pools.items():
            picked[k].append(rng.permutation(p)[:n])
    picked = {k: np.sort(np.concatenate(v)).astype(int) for k, v in picked.items()}
    return picked if len(next(iter(picked.values()))) >= min_trials else None


def matched_sets(tr, rng):
    """Three-way matched trials of Fig 2d, f, h, Fig 5a-b and Ext 6b, or None for a dropped session."""
    return match_contrast({
        "go": (tr["correct_contra"], tr["cr"]),
        "miss": (tr["miss_contra"], tr["cr"]),
        "passive": (tr["passive_still"] & (tr["passive_cr"] > 0), tr["passive_cr"]),
    }, rng)


def includable(mask_a, mask_b, cond):
    """Mask of trials in conditions that hold at least one trial of both groups."""
    keep = np.zeros(len(cond), bool)
    for c in np.unique(cond):
        m = cond == c
        if (mask_a & m).any() and (mask_b & m).any():
            keep |= m & (mask_a | mask_b)
    return keep


def condition_ids(tr):
    return (np.round(tr["cl"] * 100).astype(int) * 1000 + np.round(tr["cr"] * 100).astype(int))
