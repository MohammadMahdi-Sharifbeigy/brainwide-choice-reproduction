"""Panels shared by several figures."""
import matplotlib.pyplot as plt
import numpy as np

from steinmetz import atlas
from steinmetz.config import ANATOMY_ORDER, DT, LATENCY_ORDER
from steinmetz.regression import build_design
from steinmetz.results import PSTH_T
from steinmetz.style import CONTRA, IPSI, MISS_CONTRA_RASTER, RASTER_BG, REGION_COLORS, clean, nice_top, no_data, rotated_region_labels, shade

# rows of Fig 2a-c: (name, y0, y1) in the paper's pixels
STIM_ROWS = (("Contra", 58, 83, "contra_correct"), ("Ipsi", 87, 123, "ipsi_correct"), ("Contra", 138, 156, "contra_miss"),
             ("Ipsi", 163, 187, "ipsi_miss"), ("Contra", 203, 238, "contra_passive"), ("Ipsi", 245, 265, "ipsi_passive"))
MOVE_ROWS = (("Contra", 329, 355, "contra_correct"), ("Ipsi", 362, 397, "ipsi_correct"))


def row_trials(b):
    """Stimulus alone on one side, per raster row."""
    cc, ci = (b["cr"] > 0) & (b["cl"] == 0), (b["cl"] > 0) & (b["cr"] == 0)
    return {
        "contra_correct": (b["stim"][b["correct_contra"] & cc], b["first_time"][b["correct_contra"] & cc]),
        "ipsi_correct": (b["stim"][b["correct_ipsi"] & ci], b["first_time"][b["correct_ipsi"] & ci]),
        "contra_miss": (b["stim"][b["miss_contra"] & cc], None),
        "ipsi_miss": (b["stim"][b["miss_ipsi"]], None),
        "contra_passive": (b["passive_times"][b["passive_still"] & (b["passive_cr"] > 0) & (b["passive_cl"] == 0)], None),
        "ipsi_passive": (b["passive_times"][b["passive_still"] & (b["passive_cl"] > 0) & (b["passive_cr"] == 0)], None),
    }


ROW_COLOR = {"contra_correct": CONTRA, "ipsi_correct": IPSI, "contra_miss": MISS_CONTRA_RASTER, "ipsi_miss": IPSI, "contra_passive": "#999999", "ipsi_passive": "#000000"}


def _raster(ax, spikes, events, marks, win, color, align_move=False):
    ax.set_facecolor(RASTER_BG)
    order = np.argsort(marks) if marks is not None else np.arange(len(events))
    for i, k in enumerate(order):
        t0 = events[k] + (marks[k] if align_move else 0)
        s = spikes[np.searchsorted(spikes, t0 + win[0]):np.searchsorted(spikes, t0 + win[1])] - t0
        ax.plot(s, np.full(len(s), i), ls="", marker="s", ms=1.0, mew=0, color=color)
    if marks is not None:  # drawn last so the black dots sit on top, as in the paper
        ax.plot(-marks[order] if align_move else marks[order], np.arange(len(order)), "o", ms=2.3, color="k", mew=0)
    ax.set_xlim(*win)
    ax.set_ylim(-1, max(len(events), 1))
    ax.axvline(0, color="#444444", lw=0.5, ls=(0, (3, 2)))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]), ax.set_yticks([])


def raster_column(cv, res, session, cluster, x0, x1, title, title_color, letter=None, labels=False):
    """One column of Fig 2a-c. Row heights follow the trial counts, at one scale per column, as in the paper. Returns group centers."""
    if letter:
        cv.letter(letter, x0 - 108 if labels else x0 - 8, 28)
    cv.text((x0 + x1) / 2 + 12, 33, title, fontsize=9, color=title_color, fontweight="bold")
    if session is None:
        a = cv.ax((x0, 58, x1, 397))
        clean(a, False, False)
        no_data(a)
        return {}
    b = res.base(session)
    spikes = res.spike_times(session, cluster)
    rows = row_trials(b)
    gaps = (4, 13, 4, 15, 4)
    n = [max(len(rows[k][0]), 1) for *_, k in STIM_ROWS]
    px = (268 - 52 - sum(gaps)) / sum(n)
    y, centers = 52.0, {}
    for i, (label, _, _, key) in enumerate(STIM_ROWS):
        ev, marks = rows[key]
        h = n[i] * px
        a = cv.ax((x0, y, x1, y + h))
        _raster(a, spikes, ev, marks, (-0.2, 0.4), ROW_COLOR[key])
        if labels:
            cv.text(x0 - 26, y + h / 2, label, fontsize=7)
        centers[key] = y + h / 2
        y += h + (gaps[i] if i < 5 else 0)
    ax = cv.ax((x0, 274, x1, 279))
    clean(ax, False)
    ax.set_xlim(-0.2, 0.4), ax.set_xticks([-0.2, 0, 0.2, 0.4]), ax.set_xticklabels(["-0.2", "0", "0.2", "0.4"])
    if labels:
        ax.set_xlabel("Time from stimulus (s)")
    n2 = [max(len(rows[k][0]), 1) for *_, k in MOVE_ROWS]
    px2 = (400 - 327 - 6) / sum(n2)
    y = 327.0
    for i, (label, _, _, key) in enumerate(MOVE_ROWS):
        ev, marks = rows[key]
        h = n2[i] * px2
        a = cv.ax((x0, y, x1, y + h))
        _raster(a, spikes, ev, marks, (-0.4, 0.2), ROW_COLOR[key], align_move=True)
        if labels:
            cv.text(x0 - 26, y + h / 2, label, fontsize=7)
        centers["move_" + key] = y + h / 2
        y += h + 6
    ax = cv.ax((x0, 403, x1, 408))
    clean(ax, False)
    ax.set_xlim(-0.4, 0.2), ax.set_xticks([-0.4, -0.2, 0, 0.2]), ax.set_xticklabels(["-0.4", "-0.2", "0", "0.2"])
    if labels:
        ax.set_xlabel("Time from move (s)")
    return centers


def choice_trace(ax, res, session, cluster, win=(-0.2, 0.05), sd=0.02, legend=False):
    """Fig 4e panel: mean rate aligned to movement for contra and ipsi choices, shaded by s.e.m. across trials."""
    from steinmetz.rates import aligned_rates, smooth

    ax.set_xlim(*win)
    ax.set_xticks([-0.2, -0.1, 0]), ax.set_xticklabels(["-0.2", "-0.1", "0"])
    if session is None:
        ax.set_yticks([])
        no_data(ax)
        return
    b = res.base(session)
    st, clu = res.spikes(session)
    top = 0
    for mask, color in ((b["has_right"] & b["included"], CONTRA), (b["has_left"] & b["included"], IPSI)):
        if mask.sum() < 2:
            continue
        r, t = aligned_rates(st, clu, np.array([cluster]), b["move_abs"][mask], (win[0] - 0.15, win[1]), sd)
        r = r[:, :, 0]
        m, e = smooth(r.mean(0), 0.03), smooth(r.std(0, ddof=1) / np.sqrt(len(r)), 0.03)
        keep = t >= win[0]
        shade(ax, t[keep], m[keep], e[keep], color, lw=1.1, alpha=0.32)
        top = max(top, (m + e)[keep].max())
    ax.axvline(0, color="k", lw=0.6, ls=(0, (3, 2)))
    limit = nice_top(top)
    ax.set_ylim(0, limit), ax.set_yticks([0, limit]), ax.set_yticklabels(["0", f"{limit:g}"])


def region_heat(cv, res, key, box_heat, box_bars, zkey, matched, regions=LATENCY_ORDER, xmax=0.3, labels=False, tick_kw=None):
    """One column of Fig 2d-h: normalized rate per region and % of neurons with |z| > 4."""
    from steinmetz.config import SIGNIFICANT_Z

    x, r = res.normalized(key, "high", matched_only=matched)
    use = (PSTH_T >= 0) & (PSTH_T <= xmax)
    heat = np.full((len(regions), use.sum()), np.nan)
    for i, name in enumerate(regions):
        if (r == name).any():
            heat[i] = np.nanmean(x[r == name][:, use], 0)
    a = cv.ax(box_heat)
    a.imshow(np.nan_to_num(heat), aspect="auto", cmap="gray_r", vmin=0.7, vmax=4, extent=(0, xmax, len(regions), 0), interpolation="bilinear")
    a.axvline(0, color="k", lw=0.6, ls=(0, (3, 2)))
    for v in (0.1, 0.2):
        a.axvline(v, color="#bbbbbb", lw=0.5)
    for y in range(2, len(regions), 2):
        a.axhline(y, color="#dddddd", lw=0.35)
    a.set_yticks([])
    a.set_xticks([0, 0.1, 0.2]), a.set_xticklabels(["0", "0.1", "0.2"])
    clean(a, False)

    n = res.neurons()
    pct = np.zeros(len(regions))
    if len(n):
        sub = n[n["matched"]] if matched else n
        for i, name in enumerate(regions):
            z = sub.loc[sub["region"] == name, zkey]
            pct[i] = 100 * (np.abs(z) > SIGNIFICANT_Z).mean() if len(z) else 0
    b = cv.ax(box_bars)
    b.barh(np.arange(len(regions)) + 0.5, pct, height=0.75, color="#555555")
    b.set_ylim(len(regions), 0), b.set_xlim(0, 30), b.set_xticks([0, 20]), b.set_yticks([])
    b.xaxis.tick_top(), b.spines["top"].set_visible(True), b.spines["bottom"].set_visible(False)
    for v in (10, 20):
        b.axvline(v, color="#cccccc", lw=0.4)
    return a


def region_name_columns(cv, x_left, x_right, y0, y1, order=LATENCY_ORDER, fontsize=6.3, axis_x=None):
    """Alternating two columns of colored region names, with a tick line out to the axis, as in Fig 2d and Fig 5b."""
    from matplotlib.lines import Line2D

    step = (y1 - y0) / len(order)
    for i, name in enumerate(order):
        x = x_right if i % 2 == 0 else x_left
        y = y0 + (i + 0.5) * step
        cv.text(x, y, name, color=REGION_COLORS[name], fontweight="bold", fontsize=fontsize, ha="right")
        if axis_x is not None:
            cv.fig.add_artist(Line2D([(x + 1.5) / cv.w, axis_x / cv.w], [1 - y / cv.h] * 2, color="#555555", lw=0.35))


def region_bars(ax, table, column, order=ANATOMY_ORDER, ymax=None, min_n=5):
    """Bars per region as % of neurons; empty when fewer than min_n pass."""
    pct, count = [], []
    for name in order:
        t = table[table["region"] == name]
        k = (t[column] > 0.02).sum() if len(t) else 0
        count.append(k)
        pct.append(100 * k / len(t) if len(t) else 0)
    for i, (p, k, name) in enumerate(zip(pct, count, order)):
        c = REGION_COLORS[name]
        if k < min_n:
            ax.bar(i, p, 0.75, facecolor="white", edgecolor=c, lw=0.8)
        else:
            ax.bar(i, p, 0.75, color=c)
    ax.set_xlim(-0.8, len(order) - 0.2)
    if ymax:
        ax.set_ylim(0, ymax)
    return pct, count


def brain_maps(cv, box_top, box_sag, colors):
    """Top-down and sagittal region maps. colors maps a region name to RGBA. Other tissue is outlined in grey. Returns both axes."""
    axes = []
    for box, name in ((box_top, "top"), (box_sag, "sagittal")):
        a = cv.ax(box)
        axes.append(a)
        v = atlas.view(name)
        a.imshow(atlas.colored(v, colors), aspect="equal", interpolation="nearest")
        tissue_edges = atlas.outlines(atlas.tissue(name)) & (v < 0)
        a.contour(tissue_edges, levels=[0.5], colors="#b5b5b5", linewidths=0.14)
        a.contour(v > -2, levels=[0.5], colors="#8f8f8f", linewidths=0.35)
        a.contour(atlas.outlines(v) & (v >= -1), levels=[0.5], colors="#222222", linewidths=0.14)
        a.set_axis_off()
        if name == "top":
            a.set_xlim(-2, 314)  # the left hemisphere and a little past the midline, as in the paper
    return axes


def trial_means(res, s, trace, trials_mask, event, win=(-0.05, 0.2)):
    """Mean and s.e. of a regression-window trace aligned to an event, using samples inside trial windows."""
    b = res.base(s)
    _, rows, trnum, *_ = build_design(b)
    keep = np.flatnonzero(b["kernel_trial"])
    bins = np.arange(int(round((win[1] - win[0]) / DT)))
    out = np.full((len(keep), len(bins)), np.nan)
    tc = (rows + 0.5) * DT
    for k, tr in enumerate(keep):
        if not trials_mask[tr]:
            continue
        m = trnum == k
        idx = np.floor((tc[m] - (event[tr] + win[0])) / DT).astype(int)
        ok = (idx >= 0) & (idx < len(bins))
        out[k, idx[ok]] = trace[m][ok]
    out = out[np.isfinite(out).any(1)]
    t = win[0] + (bins + 0.5) * DT
    if len(out) == 0:
        return t, np.full(len(bins), np.nan), np.zeros(len(bins))
    n = np.maximum(np.isfinite(out).sum(0), 1)
    return t, np.nanmean(out, 0), np.nanstd(out, 0) / np.sqrt(n)


def anatomy_labels(ax):
    rotated_region_labels(ax, list(ANATOMY_ORDER), fontsize=6)
