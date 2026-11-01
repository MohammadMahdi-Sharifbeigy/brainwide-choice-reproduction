"""Figures 1 to 5, in the paper's panel layout. Boxes are pixels of the paper's PNG."""
import colorsys

import numpy as np

from steinmetz import atlas, decoding, store
from steinmetz.config import ANATOMY_ORDER, FOREBRAIN, FRONTAL, LATENCY_ORDER, MIDBRAIN, STRIATUM, SIGNIFICANT_Z
from steinmetz.data import Session, subject_of
from steinmetz.pipeline import JPECC_WIN
from steinmetz.regression import move_direction
from steinmetz.results import PSTH_T
from steinmetz.rates import smooth
from steinmetz.style import (BAR_RED, CHOICE_CMAP, CONTRA, ENGAGE, GO, HIST_BLUE, IPSI, JPECC_CMAP, MISS, PASSIVE, REGION_COLORS, STIM_CONTRA, STIM_IPSI, Canvas, clean, nice_top,
                             no_data, rotated_region_labels, shade)

from .common import anatomy_labels, brain_maps, choice_trace, raster_column, region_bars, region_heat, region_name_columns, trial_means


def _ex(chosen, key):
    e = chosen.get(key)
    return (e["session"], e["cluster"]) if e else (None, None)


# ---- Figure 1

SUBJECT_COLORS = ("#e6443c", "#f08a24", "#d4c021", "#7cc242", "#3fb56a", "#2bb5b0", "#3f7cc4", "#6a4ea0", "#a24ea0", "#e0438a")
EXAMPLE_TRIAL_SESSION = "Theiler_2017-10-11"  # holds six of the seven regions of the paper's Fig 1j
NOT_SHOWN = ("PAG", "LS", "LGd", "MD", "ILA", "BLA")
TOP_VIEW = ("OLF", "PL", "MOs", "ACA", "MOp", "SSp", "VISa", "VISam", "VISrl", "VISp", "VISpm", "VISl", "RSP", "SCs")
I_LABELS = {  # region: (x, y, ha), pixels of the paper's Fig 1i
    "OLF": (150, 431, "right"), "PL": (150, 448, "right"), "MOs": (145, 464, "right"), "ACA": (125, 480, "right"), "MOp": (125, 496, "right"),
    "SSp": (110, 512, "right"), "VISa": (100, 529, "right"), "VISam": (100, 545, "right"), "VISrl": (95, 561, "right"), "VISp": (95, 577, "right"),
    "VISpm": (100, 593, "right"), "VISl": (100, 609, "right"), "RSP": (110, 625, "right"), "SCs": (110, 641, "right"),
    "RT": (296, 480, "center"), "LD": (332, 472, "center"), "LP": (352, 458, "center"), "CA3": (382, 450, "center"), "DG": (412, 442, "center"),
    "CA1": (446, 435, "center"), "SUB": (480, 435, "center"), "POST": (530, 437, "center"), "SCm": (570, 450, "center"), "APN": (582, 465, "center"),
    "POL": (590, 490, "center"), "ORB": (282, 510, "center"), "CP": (302, 605, "center"), "GPe": (308, 619, "center"), "ACB": (320, 633, "center"),
    "VPL": (335, 645, "center"), "PO": (340, 658, "center"), "VPM": (400, 660, "center"), "ZI": (445, 648, "center"), "MG": (482, 636, "center"),
    "SNr": (506, 642, "center"), "MRN": (552, 655, "center"), "PAG": (607, 532, "center"), "LS": (607, 550, "center"), "LGd": (607, 568, "center"),
    "MD": (607, 586, "center"), "ILA": (607, 604, "center"), "BLA": (607, 622, "center"),
}


def fig1(res):
    from matplotlib.patches import ConnectionPatch

    cv = Canvas(res.cfg, "fig1", "fig1.png", scale=1.1)
    for s, x, y in (("a", 28, 30), ("b", 28, 283), ("c", 253, 30), ("d", 253, 283), ("e", 362, 30), ("f", 362, 155), ("g", 362, 285), ("h", 497, 30), ("i", 28, 438), ("j", 652, 33)):
        cv.letter(s, x, y)
    cv.slot((40, 30, 248, 270), "Task schematic")
    cv.slot((42, 288, 247, 412), "Trial timeline")
    cv.slot((372, 36, 486, 142), "Setup")
    cv.slot((372, 148, 486, 272), "Subject photo")
    cv.slot((372, 282, 486, 402), "Histology")
    b = res.stage("behavior.npz")
    for key, y0, y1 in (("left", 37, 100), ("nogo", 114, 180), ("right", 191, 255)):
        a = cv.ax((285, y0, 350, y1))
        if f"map_{key}" in b:
            a.imshow(b[f"map_{key}"], origin="lower", cmap=CHOICE_CMAP, vmin=0, vmax=100, aspect="auto")
        a.set_xticks([]), a.set_yticks([])
        clean(a, False, False)
        if key == "right":
            a.set_xticks(range(4)), a.set_xticklabels(["0", "25", "50", "100"], fontsize=5.5)
            a.set_yticks(range(4)), a.set_yticklabels(["0", "25", "50", "100"], fontsize=5.5)
            a.tick_params(length=0)
            a.set_xlabel("Right contrast"), a.set_ylabel("Left contrast")
    c = cv.ax((272, 52, 277, 103))
    c.imshow(np.linspace(100, 0, 256)[:, None], cmap=CHOICE_CMAP, aspect="auto", extent=(0, 1, 0, 100))
    c.set_xticks([]), c.yaxis.tick_left(), c.set_yticks([0, 100]), c.set_ylabel("% choice", labelpad=1, fontsize=6)
    c.spines["left"].set_visible(False), c.spines["bottom"].set_visible(False), c.tick_params(length=0, labelsize=6)
    d = cv.ax((284, 295, 350, 385))
    if "rt_mean" in b:
        pos = {0.25: 0, 0.5: 1, 1.0: 2}
        for two, color, name, off in ((1.0, "k", "two\nstimuli", 7), (0.0, "#1f7fc4", "one\nstimulus", -9)):
            m = b["rt_two"] == two
            x = [pos[v] for v in b["rt_chosen"][m]]
            d.errorbar(x, b["rt_mean"][m], b["rt_sem"][m], color=color, marker="o", ms=2.3, lw=0.8, capsize=1)
            if len(x):
                d.text(x[-1], b["rt_mean"][m][-1] + off, name, color=color, ha="right", fontsize=5.8, linespacing=0.9)
    d.set_xticks([0, 1, 2]), d.set_xticklabels(["25", "50", "100"])
    d.set_xlabel("Chosen stimulus\ncontrast"), d.set_ylabel("Reaction time (ms)")
    _probe_tracks(cv, res)
    n = res.neurons()
    counts = n.groupby("region").size().to_dict() if len(n) else {}
    light = {r: (*[0.55 * v + 0.45 for v in REGION_COLORS[r]], 1.0) for r in ANATOMY_ORDER}
    top_ax, sag_ax = brain_maps(cv, (115, 435, 232, 660), (258, 440, 553, 660), light)
    for r, (x, y, ha) in I_LABELS.items():
        cv.text(x, y, f"{r} ({counts.get(r, 0):,})", color=REGION_COLORS[r], ha=ha, fontsize=5.8, fontweight="bold" if r in ("VISp", "RSP") else None)
        if r in NOT_SHOWN:
            continue
        view, ax = ("top", top_ax) if r in TOP_VIEW else ("sagittal", sag_ax)
        rows, cols = np.nonzero(atlas.view(view) == ANATOMY_ORDER.index(r))
        if len(rows):
            start = (x + (3 if ha == "right" else 0)) / cv.w, 1 - (y + (0 if ha == "right" else 4)) / cv.h
            cv.fig.add_artist(ConnectionPatch(start, (np.median(cols), np.median(rows)), coordsA=cv.fig.transFigure, coordsB=ax.transData, color=REGION_COLORS[r], lw=0.45))
    cv.text(308, 668, "1.7 mm ML", style="italic", fontsize=5.8), cv.text(607, 516, "not shown:", style="italic", fontsize=5.8)
    _example_trial(cv, res)
    cv.text(718, 35, "Example trial", fontsize=8)
    return cv.finish()


def _probe_tracks(cv, res):
    views = (("Sagittal", (500, 50, 635, 130), 2, 0, 1), ("Horizontal", (500, 150, 635, 255), 1, 0, 2), ("Coronal", (510, 275, 620, 355), 0, 2, 1))
    subjects = sorted({subject_of(s) for s in res.sessions})
    for name, box, proj, hor, ver in views:
        a = cv.ax(box)
        sil = atlas.silhouette(proj)
        img = sil if proj == 1 else sil.T if proj == 2 else sil.T
        if proj == 1:
            a.contour(img.T, levels=[0.5], colors="k", linewidths=0.6)
        else:
            a.contour(img, levels=[0.5], colors="k", linewidths=0.6)
        for s in res.sessions:
            ses = Session(res.cfg.data, s)
            ccf = ses.channel_ccf() / 25
            pr = ses.a("channels.probe")
            c = SUBJECT_COLORS[subjects.index(subject_of(s)) % 10]
            for p in np.unique(pr):
                x = ccf[pr == p]
                a.plot(x[[0, -1], hor], x[[0, -1], ver], color=c, lw=0.5)
        a.set_aspect("equal")
        a.invert_yaxis()  # dorsal up in the side views, left up in the horizontal view
        if proj != 0:
            a.invert_xaxis()  # anterior to the right, as the paper's axes show
        a.set_axis_off()
        cv.text(box[2] - 25, box[1] - 7, name, style="italic", fontsize=6.5)
        cv.text(box[0] + 22, box[3] - 18, {"Sagittal": "D\n↑ → A", "Horizontal": "L\n↑ → A", "Coronal": "D\nL ← ↑"}[name], fontsize=5.5)
    for i, s in enumerate(subjects[:10]):
        cv.text(508 + i * 10, 385, str(i + 1), color=SUBJECT_COLORS[i], fontweight="bold", fontsize=6.5)
    cv.text(560, 371, "Mouse", fontsize=6.5)


def _example_trial(cv, res):
    a = cv.ax((690, 75, 830, 670))
    clean(a, False, False)
    s = EXAMPLE_TRIAL_SESSION if EXAMPLE_TRIAL_SESSION in res.sessions else None
    n = res.neurons()
    if s is None and len(n):
        s = n.groupby("session").size().idxmax()
    if s is None:
        no_data(a)
        return
    b = res.base(s)
    trial = np.flatnonzero(b["correct_contra"] | b["correct_ipsi"])
    if len(trial) == 0:
        no_data(a)
        return
    t = trial[len(trial) // 2]
    st, clu = res.spikes(s)
    t0 = b["stim"][t] - 0.6
    w = 2.8
    lo, hi = np.searchsorted(st, t0), np.searchsorted(st, t0 + w)
    ids = np.flatnonzero(b["n_base"])
    # each probe from the surface down, with a gap between probes
    probes = np.unique(b["n_probe"][ids])
    order, y, spans = [], 0.0, {}
    row = {}
    for p in probes:
        q = ids[b["n_probe"][ids] == p]
        q = q[np.argsort(-b["n_depth"][q])]
        for k in q:
            row[int(k)] = y
            y += 1
        y += 12
    tt, cc = st[lo:hi] - t0, clu[lo:hi]
    rows = np.array([row.get(int(k), -1) for k in cc])
    keep = rows >= 0
    a.plot(tt[keep], rows[keep], ls="", marker="|", ms=1.5, mew=0.35, color="#3a6fa5")
    a.set_xlim(0, w), a.set_ylim(y, -1)
    for name, ev in (("Stim", b["stim"][t]), ("Move", b["move_abs"][t]), ("Tone cue", b["cue"][t]), ("Reward", b["fb_time"][t])):
        x = ev - t0
        a.axvline(x, color="#999999", lw=0.7)
        cv.text(690 + 140 * x / w, 62, name, rotation=62, fontsize=5.8)
    cv.fig.lines.append(__import__("matplotlib").lines.Line2D([(690 + 140 * (w - 0.6) / w) / cv.w, (690 + 140 * (w - 0.1) / w) / cv.w], [1 - 68 / cv.h] * 2, color="k", lw=1.5, transform=cv.fig.transFigure, figure=cv.fig))
    cv.text(808, 58, "0.5 s", fontsize=6)
    region_of = b["n_region"]
    last = -40.0
    for r in ("VISp", "SUB", "MOs", "ACA", "PL", "ILA"):
        m = [row[k] for k in row if region_of[k] == r]
        if m:
            py = max(75 + 595 * np.median(m) / y, last + 36)  # keep neighbouring labels apart
            last = py
            cv.text(662, py, r, rotation=90, style="italic", fontsize=6)
    strip = cv.ax((674, 75, 680, 670))
    strip.set_xlim(0, 1), strip.set_ylim(y, -1), strip.set_axis_off()
    y_ = 0.0
    for p in probes:
        n_p = (b["n_probe"][ids] == p).sum()
        strip.fill_between([0, 1], y_, y_ + n_p, facecolor="#dddddd", hatch="......", edgecolor="#888888", lw=0)
        strip.plot([0.5, 0.5], [y_ + n_p, y_ + n_p + 6], "k", lw=0.6)
        y_ += n_p + 12
    cv.text(650, 96, "200 µm", rotation=90, fontsize=4.8)


# ---- Figure 2

def fig2(res, chosen):
    cv = Canvas(res.cfg, "fig2", "fig2.png", scale=1.05)
    centers = {}
    for x0, x1, key, label, region, letter, labels in ((122, 287, "fig2a", "SUB neuron", "SUB", "a", True), (359, 523, "fig2b", "VISpm neuron", "VISpm", "b", False),
                                                        (604, 769, "fig2c", "ZI neuron", "ZI", "c", False)):
        s, c = _ex(chosen, key)
        got = raster_column(cv, res, s, c, x0, x1, label, REGION_COLORS[region], letter, labels)
        centers = centers or got
    if centers:
        for t, k1, k2 in (("Correct\nchoices", "contra_correct", "ipsi_correct"), ("Misses", "contra_miss", "ipsi_miss"), ("Passive", "contra_passive", "ipsi_passive"),
                          ("Correct\nchoices", "move_contra_correct", "move_ipsi_correct")):
            cv.text(38, (centers[k1] + centers[k2]) / 2, t, style="italic", fontsize=7)
    panels = ((93, 187, 188, 220, "contra_correct", True, "Correct\nchoices", "Contra", "d"), (239, 333, 334, 364, "ipsi_correct", False, "Correct\nchoices", "Ipsi", "e"),
              (381, 475, 480, 505, "contra_miss", True, "Misses", "Contra", "f"), (528, 622, 624, 655, "ipsi_miss", False, "Misses", "Ipsi", "g"),
              (672, 766, 769, 800, "contra_passive", True, "Passive", "Contra", "h"))
    for x0, x1, b0, b1, key, matched, head, side, letter in panels:
        a = region_heat(cv, res, f"rate_{key}", (x0, 500, x1, 748), (b0, 500, b1, 748), f"z_{key}", matched)
        cv.letter(letter, x0 - 80 if letter == "d" else x0 - 2, 460)
        cv.text((x0 + x1) / 2, 472, head, style="italic", fontsize=6.5)
        cv.text((x0 + x1) / 2, 494, side, fontsize=6.5)
        if letter == "d":
            a.set_xlabel("Time from stimulus (s)")
            cv.text(205, 482, "% neurons", fontsize=6)
    region_name_columns(cv, 52, 88, 505, 748, axis_x=93)
    bar = cv.ax((667, 792, 756, 800))
    bar.imshow(np.linspace(0.7, 4, 200)[None], cmap="gray_r", aspect="auto", extent=(0.7, 4, 0, 1))
    bar.set_yticks([]), bar.set_xticks([1, 2, 3, 4]), clean(bar, False)
    bar.set_xlabel("Avg. norm. firing rate")
    return cv.finish()


# ---- Figure 3

def _groups_win(kind):
    return (-0.05, 0.2) if kind == "stim" else (-0.15, 0.0)


def _groups(b, kind):
    if kind == "stim":
        return [(b["cr"] > 0, STIM_CONTRA), (b["cl"] > 0, STIM_IPSI)], b["stim"], (-0.05, 0.2)
    return [(b["has_right"], CONTRA), (b["has_left"], IPSI)], b["move_abs"], (-0.15, 0.0)


def model_panel(ax, res, s, c, kind, without=None, win=None, both=False):
    """Data with s.e. plus the cross-validated model (solid), the model without a kernel (dashed), or both."""
    if s is None:
        ax.set_xlim(*(win or _groups_win(kind)))
        ax.set_yticks([])
        no_data(ax)
        return
    b = res.base(s)
    tr = res.model_traces(s, c)
    groups, event, default = _groups(b, kind)
    win = win or default
    tidy = lambda v: smooth(np.nan_to_num(v, nan=np.nanmean(v)), 0.015) if np.isfinite(v).any() else v
    for mask, color in groups:
        t, m, e = trial_means(res, s, tr["F"], mask, event, win)
        shade(ax, t, tidy(m), e, color, lw=0.45, alpha=0.28)
        if without and f"no_{without}" in tr:
            _, p, _ = trial_means(res, s, tr[f"no_{without}"], mask, event, win)
            ax.plot(t, tidy(p), color=color, lw=1.1, ls=(0, (2, 1.5)))
        if not without or both:
            _, p, _ = trial_means(res, s, tr["cv"], mask, event, win)
            ax.plot(t, tidy(p), color=color, lw=1.1)
    ax.axvline(0, color="k", lw=0.5, ls=(0, (3, 2)))
    ax.set_xlim(*win)
    limit = nice_top(ax.get_ylim()[1] / 1.05)
    ax.set_ylim(0, limit), ax.set_yticks([0, limit]), ax.set_yticklabels(["0", f"{limit:g}"])


def _pct_colors(pct, base, floor=2, top=5):
    out = {}
    h, s, v = colorsys.rgb_to_hsv(*base)
    for r, p in pct.items():
        if p < floor:
            out[r] = (0.78, 0.78, 0.78, 1)
        else:
            f = min((p - floor) / (top - floor), 1)
            out[r] = (*colorsys.hsv_to_rgb(h, s * (0.5 + 0.5 * f), v), 1)
    return out


def pct_legend(cv, box, base, ticks=((0, 0), (0.4, 2), (1, 5)), floor=0.4):
    """Gradient key for a brain map: grey below the first step, then the color, with ticks above."""
    a = cv.ax(box)
    grad = np.ones((1, 200, 3))
    h, s, v = colorsys.rgb_to_hsv(*base)
    for i, f in enumerate(np.linspace(0, 1, 200)):
        grad[0, i] = (0.78, 0.78, 0.78) if f < floor else colorsys.hsv_to_rgb(h, s * (0.5 + 0.5 * (f - floor) / (1 - floor)), v)
    a.imshow(grad, aspect="auto", extent=(0, 1, 0, 1))
    a.set_yticks([]), a.set_xticks([t[0] for t in ticks]), a.set_xticklabels([f"{t[1]:g}" for t in ticks])
    a.xaxis.tick_top()
    for s_ in a.spines.values():
        s_.set_visible(True), s_.set_linewidth(0.6)
    return a


def fig3(res, chosen):
    from matplotlib.lines import Line2D

    cv = Canvas(res.cfg, "fig3", "fig3.png", scale=1.5)
    t = res.regression()
    vis, sub = _ex(chosen, "fig2b"), _ex(chosen, "fig2a")
    cv.letter("a", 38, 22), cv.letter("b", 345, 22)
    cv.text(180, 24, "VISpm neuron", fontsize=7.4, color=REGION_COLORS["VISpm"], fontweight="bold")
    cv.text(490, 24, "SUB neuron", fontsize=7.4, color=REGION_COLORS["SUB"], fontweight="bold")
    for (s, c), xs in ((vis, ((72, 187), (212, 325))), (sub, ((362, 473), (505, 615)))):
        top1, top2 = cv.ax((xs[0][0], 50, xs[0][1], 135)), cv.ax((xs[1][0], 50, xs[1][1], 135))
        low1, low2 = cv.ax((xs[0][0], 165, xs[0][1], 245)), cv.ax((xs[1][0], 165, xs[1][1], 245))
        model_panel(top1, res, s, c, "stim"), model_panel(top2, res, s, c, "move")
        model_panel(low1, res, s, c, "stim", "vision_contra"), model_panel(low2, res, s, c, "move", "action")
        for a in (top1, top2):
            a.set_xticks([])
        low1.set_xticks([0, 0.1, 0.2]), low1.set_xticklabels(["0", "0.1", "0.2"]), low1.set_xlabel("Time from stimulus (s)")
        low2.set_xticks([-0.1, 0]), low2.set_xticklabels(["-0.1", "0"]), low2.set_xlabel("Time from move (s)")
        cv.text(xs[0][0] + 28, 158, "Model without\nVision kernel", style="italic", fontsize=5.6)
        cv.text(xs[1][0] + 28, 158, "Model without\nAction kernel", style="italic", fontsize=5.6)
    cv.text(50, 95, "Firing rate (sp/s)", rotation=90, fontsize=5.8)
    cv.text(97, 36, "Stimulus:", fontsize=5.6), cv.text(88, 46, "Contra", fontsize=5.6, color=STIM_CONTRA), cv.text(122, 46, "Ipsi", fontsize=5.6, color=STIM_IPSI)
    cv.text(232, 36, "Choice:", fontsize=5.6), cv.text(226, 46, "Contra", fontsize=5.6, color=CONTRA), cv.text(260, 46, "Ipsi", fontsize=5.6, color=IPSI)

    cv.letter("c", 43, 298), cv.letter("d", 43, 404), cv.letter("e", 43, 516), cv.letter("f", 43, 662), cv.letter("g", 343, 662)
    a = cv.ax((100, 310, 590, 380))
    for i, r in enumerate(ANATOMY_ORDER):
        v = 100 * t.loc[t["region"] == r, "ve"].to_numpy() if len(t) else np.zeros(0)
        if len(v):
            q = np.percentile(v, [10, 25, 50, 75, 90])
            a.plot([i, i], q[[0, 4]], color=REGION_COLORS[r], lw=0.7)
            a.plot([i, i], q[[1, 3]], color=REGION_COLORS[r], lw=2.1)
            a.plot(i, q[2], ".", color="w", ms=1.3)
    a.set_xlim(-0.8, 41.8), a.set_ylim(0, 20), a.set_xticks([]), a.set_yticks([0, 10, 20]), a.set_ylabel("% C.V. var. expl.")
    cv.text(345, 302, "Full model C.V. variance explained", fontsize=6)
    k = cv.ax((525, 297, 585, 302))
    k.plot([0, 1], [0.5, 0.5], "k", lw=0.5), k.plot([0.25, 0.75], [0.5, 0.5], "k", lw=2), k.plot(0.5, 0.5, ".", color="w", ms=1.2)
    k.set_xlim(-0.05, 1.05), k.set_axis_off()
    cv.text(555, 283, "percentile:\n10 25 50 75 90", fontsize=3.8)
    vis_pct, act_pct = None, None
    for box, col, title, y in (((100, 415, 590, 498), "vision_contra", "Visual encoding", 422), ((100, 530, 590, 613), "action", "Action encoding", 534)):
        b = cv.ax(box)
        pct, _ = region_bars(b, t, col, ymax=30)
        b.set_ylabel("% neurons"), b.set_yticks([0, 10, 20, 30]), b.set_xticks([])
        cv.text(348, y, title, fontsize=6)
        if col == "vision_contra":
            vis_pct = dict(zip(ANATOMY_ORDER, pct))
        else:
            act_pct = dict(zip(ANATOMY_ORDER, pct))
            anatomy_labels(b)
    purple, green = (0.58, 0.44, 0.74), (0.27, 0.6, 0.42)
    brain_maps(cv, (42, 690, 147, 822), (140, 690, 335, 822), _pct_colors(vis_pct, purple))
    brain_maps(cv, (330, 690, 425, 822), (420, 690, 610, 822), _pct_colors(act_pct, green))
    cv.text(135, 668, "Visual encoding", fontsize=6.4), cv.text(420, 668, "Action encoding", fontsize=6.4)
    pct_legend(cv, (240, 685, 325, 691), purple), pct_legend(cv, (510, 685, 597, 691), green)
    cv.text(283, 668, "% neurons", fontsize=4.4), cv.text(553, 668, "% neurons", fontsize=4.4)
    return cv.finish()


# ---- Figure 4

def _decoding_curves(res):
    out = {}
    for s in res.sessions:
        p = res.cfg.results / "decoding" / f"{s}.npz"
        if p.exists():
            for k, v in store.load(p).items():
                key, r = k.split("|")
                out.setdefault(key, {}).setdefault(r, []).append(v)
    return out


def _jpecc_mean(res, ev, a, c):
    arr = []
    for s in res.sessions:
        p = res.cfg.results / "jpecc" / f"{s}.npz"
        if p.exists():
            z = store.load(p)
            if f"r|{ev}|{a}|{c}" in z:
                arr.append(z[f"r|{ev}|{a}|{c}"])
    return np.array(arr)


def jpecc_panel(ax, r, ev, cmap_v=0.5):
    w = JPECC_WIN[ev]
    im = ax.imshow(r, origin="lower", extent=(*w, *w), cmap=JPECC_CMAP, vmin=-cmap_v, vmax=cmap_v, aspect="equal", interpolation="nearest")
    ax.plot(w, w, "w", lw=1.2)
    ax.axhline(0, color="w", lw=1.2), ax.axvline(0, color="w", lw=1.2)
    return im


def fig4(res, chosen):
    cv = Canvas(res.cfg, "fig4", "fig4.png", scale=1.15)
    for s, x, y in (("a", 27, 28), ("b", 283, 28), ("c", 27, 225), ("d", 193, 225), ("e", 440, 203), ("f", 27, 422), ("g", 241, 422), ("h", 462, 441)):
        cv.letter(s, x, y)
    slots = (("fig4a_vis", "VISp", (62, 55, 146, 110)), ("fig2c", "ZI", (168, 55, 250, 110)), ("fig4a_mrn", "MRN", (62, 125, 146, 183)), ("fig4a_cp", "CP", (168, 125, 250, 183)))
    for key, r, box in slots:
        s, c = _ex(chosen, key)
        a = cv.ax(box)
        model_panel(a, res, s, c, "move", "choice", (-0.25, 0.0), both=True)
        a.set_xticks([-0.2, -0.1, 0] if box[1] > 100 else [])
        if box[1] > 100:
            a.set_xticklabels(["-0.2", "-0.1", "0"])
        cv.text(box[0] + 16, box[1] + 2, r, color=REGION_COLORS[r], fontweight="bold", fontsize=7.4)
    cv.text(105, 205, "Time from move (s)", fontsize=6.2), cv.text(36, 150, "Firing rate (sp/s)", rotation=90, fontsize=6.2)
    cv.text(84, 28, "Data", fontsize=5.6), cv.text(125, 28, "Model", fontsize=5.6), cv.text(190, 28, "Model without\nChoice kernel", fontsize=5.6)
    cv.text(92, 74, "Choice:\nContra", fontsize=5.4, color=CONTRA), cv.text(125, 100, "Ipsi", fontsize=5.4, color=IPSI)
    t = res.regression()
    b = cv.ax((313, 65, 805, 148))
    region_bars(b, t, "choice", ymax=10)
    b.set_ylabel("% neurons"), b.set_yticks([0, 5, 10])
    anatomy_labels(b)
    cv.text(535, 72, "Choice encoding", fontsize=7)

    curves = _decoding_curves(res).get("choice", {})
    c_ax = cv.ax((50, 240, 172, 368))
    off = decoding.OFFSETS["move"]
    groups, mats = [], []
    for name, regions in (("Frontal", FRONTAL), ("Midbrain", MIDBRAIN), ("CP", STRIATUM)):
        for r in regions:
            for v in curves.get(r, []):
                mats.append(v), groups.append(name)
    if mats:
        norm = decoding.normalize_groups(np.array(mats), groups, -1)
        groups = np.array(groups)
        details = {"Frontal": ("(MOs, PL, MOp)", 0.97), "Midbrain": ("(MRN, SCm,\nZI, SNr)", 0.74), "CP": ("", 0.52)}
        for name, color in (("Frontal", "#d0a050"), ("Midbrain", "#9e3a3a"), ("CP", "#5a3ea0")):
            m = groups == name
            if m.any():
                n = np.isfinite(norm[m]).sum(0)
                shade(c_ax, off, smooth(np.nan_to_num(np.nanmean(norm[m], 0)), 0.01), np.nanstd(norm[m], 0) / np.sqrt(np.maximum(n, 1)), color, alpha=0.2)
                c_ax.text(0.03, details[name][1], f"{name} (n={m.sum()})\n  {details[name][0]}".rstrip(), color=color, transform=c_ax.transAxes, fontsize=5.5, va="top", linespacing=0.95)
    c_ax.axvline(0, color="k", lw=0.6, ls=(0, (3, 2)))
    c_ax.set_xlim(-0.15, 0), c_ax.set_ylim(-0.25, 1.9), c_ax.set_yticks([0, 1])
    c_ax.set_xticks([-0.15, -0.1, -0.05, 0]), c_ax.set_xticklabels(["-0.15", "-0.1", "-0.05", "0"])
    c_ax.set_xlabel("Time from move (s)"), c_ax.set_ylabel("Normalized choice decoding")

    for (a_name, c_name), box, title, ylab in ((("VIS", "Frontal"), (207, 258, 310, 361), "jPECC: VIS vs Frontal", "VIS"), (("MB", "Frontal"), (338, 258, 441, 361), "Midbrain vs Frontal", "Midbrain")):
        a = cv.ax(box)
        arr = _jpecc_mean(res, "move", a_name, c_name)
        if len(arr):
            jpecc_panel(a, arr.mean(0), "move")
            a.text(0.05, 0.9, f"n={len(arr)}", color="w", transform=a.transAxes, fontsize=5.6)
        else:
            no_data(a)
        a.set_xticks([-0.3, 0]), a.set_yticks([-0.3, 0]), a.set_xticklabels(["-0.3", "0"]), a.set_yticklabels(["-0.3", "0"])
        a.set_ylabel(ylab, labelpad=1)
        cv.text((box[0] + box[2]) / 2, 247, title, fontsize=5.8)
    cv.text(258, 382, "Frontal, time from move (s)", fontsize=5.8)
    bar = cv.ax((380, 388, 433, 397))
    bar.imshow(np.linspace(-0.5, 0.5, 200)[None], cmap=JPECC_CMAP, aspect="auto", extent=(-0.5, 0.5, 0, 1))
    bar.set_yticks([]), bar.set_xticks([-0.5, 0, 0.5]), bar.set_xticklabels(["-0.5", "0", "0.5"]), clean(bar, False)
    cv.text(406, 381, "C.V. corr coeff", fontsize=5.6)

    rows = ((222, 295), (320, 390))
    cols = ((475, 552), (569, 650), (662, 740), (751, 828))
    names = ("MRN", "MRN", "SCm", "SCm", "MOs", "MOs", "CP", "CP")
    for i, name in enumerate(names):
        y, x = rows[i // 4], cols[i % 4]
        a = cv.ax((x[0], y[0], x[1], y[1]))
        s, c = _ex(chosen, f"fig4e_{i + 1}")
        choice_trace(a, res, s, c)
        cv.text((x[0] + x[1]) / 2, y[0] - 9, name, color=REGION_COLORS[name], fontweight="bold", fontsize=7.4)
        if i < 4:
            a.set_xticklabels([])
    cv.text(450, 275, "Firing rate (sp/s)", rotation=90, fontsize=6.2), cv.text(508, 412, "Time from move (s)", fontsize=6.2)
    cv.text(497, 228, "Choice:", fontsize=5.4), cv.text(490, 239, "Contra", fontsize=5.4, color=CONTRA), cv.text(516, 239, "Ipsi", fontsize=5.4, color=IPSI)

    for box, regions, color, title, sub in (((70, 450, 222, 600), MIDBRAIN, "#e0443c", "Midbrain", "(MRN, SCm, SNr, ZI)"), ((263, 450, 414, 600), FOREBRAIN, "#e0a040", "Forebrain", "(CP, MOs, MOp, PL)")):
        a = cv.ax(box)
        if len(t):
            sel = t[t["region"].isin(regions) & t["ve"].ge(0.02)]
            sig = sel["choice"] > 0.02
            a.scatter(sel.loc[~sig, "ipsi"].clip(-50, 150), sel.loc[~sig, "contra"].clip(-50, 150), s=2, color=color, alpha=0.15, lw=0)
            a.scatter(sel.loc[sig, "ipsi"].clip(-50, 150), sel.loc[sig, "contra"].clip(-50, 150), s=5, color=color, lw=0)
        a.plot([-50, 150], [-50, 150], "k", lw=0.6, ls=(0, (3, 2)))
        a.axhline(0, color="k", lw=0.6, ls=(0, (3, 2))), a.axvline(0, color="k", lw=0.6, ls=(0, (3, 2)))
        a.set_xlim(-50, 150), a.set_ylim(-50, 150), a.set_xticks(range(-50, 151, 50)), a.set_yticks(range(-50, 151, 50))
        cv.text((box[0] + box[2]) / 2, 423, title, fontsize=7), cv.text((box[0] + box[2]) / 2, 437, sub, fontsize=5.8)
    cv.text(148, 627, "Pre-move ipsilateral (sp/s)", fontsize=6.2), cv.text(36, 525, "Pre-move\ncontralateral (sp/s)", rotation=90, fontsize=6.2)

    pct = {}
    if len(t):
        for r in ANATOMY_ORDER:
            s = t[t["region"] == r]
            pct[r] = 100 * (s["choice"] > 0.02).mean() if len(s) else 0
    colors = {r: (0.78, 0.78, 0.78, 1) for r in ANATOMY_ORDER}
    unilateral, bilateral = (0.86, 0.3, 0.33), (0.87, 0.72, 0.47)
    for r, p in pct.items():
        if p >= 1:
            colors[r] = (*(unilateral if r in MIDBRAIN else bilateral), 1)
    brain_maps(cv, (480, 462, 590, 625), (585, 462, 812, 625), colors)
    cv.text(572, 455, "Choice encoding", fontsize=7), cv.text(755, 421, "% neurons", fontsize=5.4)
    cv.text(678, 447, "Bilateral", color=bilateral, fontsize=5.8), cv.text(675, 461, "Unilateral", color=unilateral, fontsize=5.8)
    pct_legend(cv, (707, 443, 805, 450), bilateral, ((0, 0), (0.33, 1), (0.66, 2)), 0.33)
    uni = pct_legend(cv, (707, 455, 805, 462), unilateral, ((0, 0), (0.33, 1), (0.66, 2)), 0.33)
    uni.set_xticks([])
    return cv.finish()


# ---- Figure 5

def rate_panel(cv, res, box, region, conds, ylab=False):
    """Population mean with s.e. across neurons, for each (key, color)."""
    a = cv.ax(box)
    any_data = False
    for key, color in conds:
        got = res.region_rate(f"rate_{key}", region)
        if got:
            m, e, _ = got
            shade(a, PSTH_T, m, e, color, lw=1.1, alpha=0.4)
            any_data = True
    if not any_data:
        no_data(a)
    a.axvline(0, color="k", lw=0.5, ls=(0, (3, 2)))
    a.set_xlim(-0.06, 0.3), a.set_xticks([0, 0.1, 0.2]), a.set_xticklabels(["0", "0.1", "0.2"])
    if any_data:
        top = a.get_ylim()[1] / 1.05
        limit = nice_top(top)
        step = limit / 2 if limit / 2 >= top * 0.3 else limit
        a.set_ylim(0, max(top * 1.05, limit if top > limit * 0.8 else top * 1.05))
        ticks = np.arange(0, a.get_ylim()[1] + 1e-9, step)
        a.set_yticks(ticks), a.set_yticklabels([f"{v:g}" for v in ticks])
    return a


def fig5(res):
    cv = Canvas(res.cfg, "fig5", "fig5.png", scale=1.35)
    for s, x, y in (("a", 18, 21), ("b", 18, 366), ("c", 233, 366), ("d", 498, 27), ("e", 613, 366), ("f", 613, 614)):
        cv.letter(s, x, y)
    cols = ((55, 150), (172, 270), (285, 380), (404, 500))
    rows = ((45, 137), (183, 276))
    regions = ("VISp", "VISa", "CP", "GPe", "SCm", "MRN", "CA1", "MOs")
    for i, r in enumerate(regions):
        y, x = rows[i // 4], cols[i % 4]
        rate_panel(cv, res, (x[0], y[0], x[1], y[1]), r, (("contra_miss", MISS), ("contra_passive", PASSIVE)))
        cv.text((x[0] + x[1]) / 2, y[0] - 10, r, color=REGION_COLORS[r], fontweight="bold", fontsize=9)
    cv.text(22, 90, "Firing rate (sp/s)", rotation=90, fontsize=5.6)
    cv.text(108, 317, "Time from\nstimulus onset (s)", fontsize=5.8)
    cv.text(222, 308, "Task, miss trial", color=MISS, fontsize=5.8), cv.text(241, 325, "Passive presentation", color=PASSIVE, fontsize=5.8)

    n = res.neurons()
    diff = np.zeros(len(LATENCY_ORDER))
    if len(n):
        m = n[n["matched"]]
        for i, r in enumerate(LATENCY_ORDER):
            s = m[m["region"] == r]
            if len(s):
                diff[i] = 100 * ((np.abs(s["z_contra_miss"]) > SIGNIFICANT_Z).mean() - (np.abs(s["z_contra_passive"]) > SIGNIFICANT_Z).mean())
    a = cv.ax((143, 415, 218, 752))
    a.barh(np.arange(42) + 0.5, diff, height=0.7, color=[BAR_RED if v > 0 else "#888888" for v in diff])
    a.set_ylim(42, 0), a.set_xlim(-1, 4), a.set_xticks([0, 2, 4]), a.set_yticks([]), a.xaxis.tick_top()
    a.spines["bottom"].set_visible(False), a.spines["top"].set_visible(True)
    cv.text(120, 373, "Δ % activated neurons,\nmiss − passive trials", fontsize=7)
    region_name_columns(cv, 65, 118, 415, 752, fontsize=6, axis_x=143)

    e = res.stage("engagement.npz")
    colors = {}
    if "region_names" in e:
        for r, v in zip(e["region_names"].tolist(), e["task_passive"]):
            colors[str(r)] = ENGAGE((np.clip(v, -1, 1) + 1) / 2)
    brain_maps(cv, (232, 420, 352, 620), (352, 420, 600, 620), colors)
    cv.text(420, 398, "Engagement Correlates", fontsize=7.2)
    bar = cv.ax((350, 630, 492, 641))
    bar.imshow(np.linspace(-1, 1, 200)[None], cmap=ENGAGE, aspect="auto", extent=(-1, 1, 0, 1))
    bar.set_yticks([]), bar.set_xticks([-1, 0, 1]), clean(bar, False)
    bar.set_xlabel("Δ Pre-stimulus firing rate,\nTask − Passive (sp/s)")

    d = cv.ax((545, 65, 818, 288))
    if "region_names" in e:
        from scipy.stats import pearsonr
        names, x, y, size = [str(v) for v in e["region_names"]], e["task_passive"], e["go_miss"], e["region_n"]
        lo, hi = size.min(), max(size.max(), size.min() + 1)
        for r, a_, b_, s_ in zip(names, x, y, size):
            if np.isfinite(b_):
                d.text(a_, b_, r, color=REGION_COLORS[r], ha="center", va="center", fontsize=4 + 7 * (s_ - lo) / (hi - lo), fontweight="bold")
        ok = np.isfinite(y)
        if ok.sum() > 2:
            r_, p_ = pearsonr(x[ok], y[ok])
            d.text(0.03, 0.93, f"r = {r_:.2f}, p = {p_:.3f}", transform=d.transAxes, fontsize=7)
        d.set_xlim(min(-1.3, x.min() * 1.1), max(1.7, x.max() * 1.1))
        d.set_ylim(min(-0.6, np.nanmin(y) * 1.1), max(1.0, np.nanmax(y) * 1.1))
    d.axhline(0, color="k", lw=0.6, ls=(0, (3, 2))), d.axvline(0, color="k", lw=0.6, ls=(0, (3, 2)))
    d.set_xlabel("Δ Pre-stimulus firing rate,\nTask − Passive (sp/s)"), d.set_ylabel("Δ Pre-stimulus firing rate, Go − Miss (sp/s)", fontsize=7)

    ax = cv.ax((660, 395, 790, 530))
    tab = e.get("trial_table", np.zeros((0, 7)))
    if len(tab) > 20:
        _engagement_curve(ax, tab[:, 0], tab[:, 1])
    ax.set_xlim(-5, 5), ax.set_ylim(0, 1), ax.set_xticks([-5, 0, 5]), ax.set_yticks([0, 1])
    ax.set_xlabel("Engagement index\n(z-score)"), ax.set_ylabel("P(Go)")
    f = cv.ax((677, 627, 770, 705))
    if len(e.get("ei_diff", [])):
        from scipy.stats import ttest_1samp

        f.hist(e["ei_diff"], bins=12, color=HIST_BLUE)
        f.plot(np.mean(e["ei_diff"]), f.get_ylim()[1], "kv", ms=3, clip_on=False)
        p_ = ttest_1samp(e["ei_diff"], 0).pvalue
        f.text(0.98, 0.92, "p<10$^{-4}$" if p_ < 1e-4 else f"p={p_:.2g}", transform=f.transAxes, ha="right", fontsize=5.4)
    f.axvline(0, color="k", lw=0.6, ls=(0, (3, 2)))
    f.set_xlabel("Δ Engagement index (a.u.)\nGo − Miss trials"), f.set_ylabel("n recordings")
    return cv.finish()


def _engagement_curve(ax, x, go, color="k", point="r"):
    import statsmodels.api as sm

    ok = np.isfinite(x)
    x, go = x[ok], go[ok]
    r = sm.GLM(go, sm.add_constant(x), family=sm.families.Binomial()).fit()
    grid = np.linspace(x.min(), x.max(), 50)
    X = sm.add_constant(grid)
    eta, se = X @ r.params, np.sqrt(np.einsum("ij,jk,ik->i", X, r.cov_params(), X))
    f = lambda v: 1 / (1 + np.exp(-v))
    ax.fill_between(grid, f(eta - 1.96 * se), f(eta + 1.96 * se), color="#999999", lw=0)
    ax.plot(grid, f(eta), color=color, lw=1.5)
    edges = np.quantile(x, np.linspace(0, 1, 11))
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x <= b)
        if m.sum() > 1:
            p = go[m].mean()
            ax.errorbar(x[m].mean(), p, np.sqrt(p * (1 - p) / m.sum()), color=point, marker="o", ms=2, lw=1)
    return r
