"""Extended Data Figures 1 to 9, in the paper's panel layout. Boxes are pixels of the paper's PNG."""
import numpy as np

from steinmetz import atlas, behavior, cccp, decoding, engagement, regression as rr, store
from steinmetz.config import ANATOMY_ORDER, DT, LATENCY_ORDER
from steinmetz.data import Session, subject_of
from steinmetz.rates import aligned_rates
from steinmetz.report import focality_interval
from steinmetz.results import NESTED, PSTH_T
from steinmetz.rates import smooth
from steinmetz.style import (BLUE_WHITE_RED, CONTRA, GO, GROUP_COLORS, HIST_BLUE, HIST_ORANGE, IPSI, JPECC_CMAP, MISS, PARULA, PASSIVE, PURPLE_HEAT, REGION_COLORS, Canvas, clean,
                             nice_top, no_data, rotated_region_labels, shade)

from .common import region_name_columns, trial_means
from .main_figs import _decoding_curves, _ex, _engagement_curve, _jpecc_mean, jpecc_panel, rate_panel

EXT2_PROBE = ("Hench_2017-06-15", 1)  # the only probe through all the regions of the paper's example
EXT8_SESSION = "Radnitz_2017-01-08"  # has the 187 trials of the paper's example
EXT9_SESSION = "Muller_2017-01-08"  # 141 trials against the paper's 140
PAPER_SUBJECTS = ("Cori", "Forssmann", "Hench", "Lederberg", "Muller", "Radnitz", "Richards", "Tatum", "Moniz", "Theiler")
BLOCKS = ((57, 39), (253, 39), (452, 39), (57, 192), (253, 192), (452, 192), (57, 345), (253, 345), (452, 345), (57, 497))


def _pedestal_block(cv, x, y, data, params, four_rows=True):
    cl, cr, ch = data[:, 0], data[:, 1], data[:, 2]
    rows = ((0, 26), (31, 58), (61, 86), (90, 117))
    cols = ((0, 53), (57, 112), (113, 165))
    for i, ped in enumerate((0.0, 0.25, 0.5, 1.0)):
        m = np.minimum(cl, cr) == ped
        d = cr[m] - cl[m]
        grid = np.linspace(-(1 - ped), 1 - ped, 41) if ped < 1 else np.array([0.0])
        pl, pr, pn = behavior.model_probs(params, ped + np.maximum(-grid, 0), ped + np.maximum(grid, 0))
        for j, (ch_v, curve, color) in enumerate(((1, pl, "#3f7cc4"), (0, pn, "k"), (-1, pr, CONTRA))):
            a = cv.ax((x + cols[j][0], y + rows[i][0], x + cols[j][1], y + rows[i][1]))
            a.plot(100 * grid, curve, color=color, lw=0.9)
            for v in np.unique(d):
                k, n = (ch[m][d == v] == ch_v).sum(), (d == v).sum()
                lo, hi = behavior.binomial_ci(k, n)
                a.plot([100 * v] * 2, [lo, hi], color=color, lw=0.6)
                a.plot(100 * v, k / n, "o", color=color, ms=1.2)
            a.set_xlim(-105, 105), a.set_ylim(0, 1.02), a.set_yticks([]), a.set_xticks([])
            clean(a, j == 0 and False, False)
            a.spines["bottom"].set_visible(True), a.spines["bottom"].set_color("#cccccc")
            if i == 3 and j == 0:
                a.set_xticks([-100, 0, 100]), a.set_yticks([0, 0.5, 1]), a.spines["left"].set_visible(True)
                a.tick_params(labelsize=5)


def _non_repeat_trials(res, b):
    """The paper's Ext 1 leaves out repeat trials: included trials with repNum 1 reproduce its trial counts. Returns subject -> data, params."""
    rows = {}
    for s in res.sessions:
        ses = Session(res.cfg.data, s)
        if not (ses.path / "trials.repNum.npy").exists():
            return {str(k): b[f"data|{k}"] for k in b["subjects"]}, {str(k): b[f"params|{k}"] for k in b["subjects"]}
        q = res.base(s)
        m = q["included"] & (ses.a("trials.repNum") == 1)
        rows.setdefault(subject_of(s), []).append(np.c_[q["cl"][m], q["cr"][m], q["choice"][m], q["fb"][m]])
    data = {k: np.concatenate(v) for k, v in rows.items()}
    return data, {k: behavior.fit_psychometric(v[:, 0], v[:, 1], v[:, 2]) for k, v in data.items()}


def ext1(res):
    from matplotlib.lines import Line2D

    cv = Canvas(res.cfg, "ext1", "extended-data-fig1.png", scale=1.15)
    b = res.stage("behavior.npz")
    all_data, all_params = _non_repeat_trials(res, b)
    subj = [s for s in PAPER_SUBJECTS if s in all_data] + [s for s in all_data if s not in PAPER_SUBJECTS]
    for i, s in enumerate(subj[:10]):
        x, y = BLOCKS[i]
        data = all_data[s]
        n_ses = sum(subject_of(q) == s for q in res.sessions)
        cv.letter("abcdefghij"[i], x - 22, y - 16)
        cv.text(x + 85, y - 14, f"{s} ({len(data)} trials, {n_ses} session{'s' if n_ses > 1 else ''})", fontsize=5.5, fontweight="bold")
        _pedestal_block(cv, x, y, data, all_params[s])
    x, y = BLOCKS[0]
    cv.text(x - 3, y - 4, "pedestal", style="italic", fontsize=4.8)
    for t, yy in (("0%", y + 18), ("25%", y + 48), ("50%", y + 78), ("100%", y + 110)):
        cv.text(x + 6, yy, t, style="italic", fontsize=4.8)
    cv.text(x + 34, y + 2, "Left\nchoice", color=IPSI, fontsize=4.8), cv.text(x + 85, y + 3, "NoGo", fontsize=4.8), cv.text(x + 140, y + 2, "Right\nchoice", color=CONTRA, fontsize=4.8)
    cv.text(38, 110 + 15, "Probability", rotation=90, fontsize=4.8), cv.text(x + 26, y + 135, "Δ Contrast (%)", fontsize=4.8)
    cv.letter("k", 263, 487), cv.letter("l", 440, 487), cv.letter("m", 35, 645), cv.letter("n", 35, 725), cv.letter("o", 347, 645), cv.letter("p", 480, 645), cv.letter("q", 480, 735)

    k = cv.ax((289, 508, 390, 598))
    grid = np.linspace(-1, 1, 101)
    for s in subj:
        pl, pr, _ = behavior.model_probs(all_params[s], np.maximum(-grid, 0), np.maximum(grid, 0))
        k.plot(100 * grid, pl, color=IPSI, lw=0.9), k.plot(100 * grid, pr, color=CONTRA, lw=0.9)
    k.set_ylim(0, 1), k.set_xlim(-100, 110), k.set_yticks([0, 0.5, 1]), k.set_xlabel("Contrast (%)"), k.set_ylabel("Probability choose left")
    k.text(0.98, 0.97, "pedestal 0%", transform=k.transAxes, style="italic", ha="right", va="top", fontsize=5)
    cv.text(340, 490, "Combined subjects", fontweight="bold", fontsize=5.5)
    l = cv.ax((480, 510, 600, 598))
    for i, s in enumerate(subj):
        d = all_data[s]
        m = ((d[:, 0] == 1) & (d[:, 1] == 0)) | ((d[:, 0] == 0) & (d[:, 1] == 1))
        n, c = m.sum(), (d[m, 3] == 1).sum()
        if n:
            lo, hi = behavior.binomial_ci(c, n)
            l.plot([i, i], [lo, hi], "k", lw=0.9), l.plot(i, c / n, "ko", ms=2)
    l.axhline(1 / 3, color="#888888", ls="--", lw=0.6), l.set_ylim(0, 1.02), l.set_yticks([0, 0.5, 1]), l.set_xlim(-0.5, max(len(subj) - 0.5, 1))
    l.text(0.98, 0.26, "chance = 33%", transform=l.transAxes, ha="right", style="italic", fontsize=5)
    l.set_xticks(range(len(subj))), l.set_xticklabels(subj, rotation=45, ha="right", fontsize=5), l.set_ylabel("Probability correct")
    cv.text(532, 498, "Performance at 100% contrast", fontweight="bold", fontsize=5.5)

    s = res.sessions[0]
    ses = Session(res.cfg.data, s)
    t, mm = behavior.wheel_mm(ses)
    on, off, kind = ses.moves()
    colors = {1: IPSI, 2: CONTRA}
    # the 28 s stretch with the most left and right turns, like the paper's example
    turns = on[(kind == 1) | (kind == 2)]
    start = max(turns[:-1], key=lambda u: ((turns > u) & (turns < u + 28)).sum()) - 1 if len(turns) > 1 else on[0]
    pos, vel = cv.ax((66, 655, 325, 715)), cv.ax((66, 735, 325, 795))
    m = (t >= start) & (t <= start + 28)
    step = max(int(0.02 / np.median(np.diff(t[:1000]))), 1)
    tt, xx = t[m][::step], mm[m][::step]
    pos.plot(tt, xx - xx[0], color="#cccccc", lw=0.6)
    v = np.gradient(xx, tt)
    vel.plot(tt, v, color="#cccccc", lw=0.6)
    for a, z, ty in zip(on, off, kind):
        if start <= a <= start + 28:
            w = (tt >= a) & (tt <= z)
            pos.plot(tt[w], (xx - xx[0])[w], color=colors.get(ty, "k"), lw=0.9)
            vel.plot(tt[w], v[w], color=colors.get(ty, "k"), lw=0.9)
            if w.any():
                pos.plot(tt[w][0], (xx - xx[0])[w][0], "o", color="#2fb34a", ms=1.8, mew=0), pos.plot(tt[w][-1], (xx - xx[0])[w][-1], "o", color="#e0403c", ms=1.8, mew=0)
    pos.set_ylabel("Position (mm)"), vel.set_ylabel("Velocity (mm/s)"), vel.set_xlabel("Time (s)")
    pos.text(0.8, 0.2, "← Detected onset", color="#2fb34a", transform=pos.transAxes, fontsize=5)
    cv.text(195, 647, "Example segment of rotary encoder data", fontsize=6)
    o, p = cv.ax((383, 658, 462, 795)), cv.ax((510, 660, 610, 733))
    use = np.flatnonzero((kind == 1) | (kind == 2))[:60]
    w = np.arange(-0.1, 0.5, 0.01)
    for i in use:
        y = np.interp(on[i] + w, t, mm) - np.interp(on[i], t, mm)
        o.plot(w, y, color=colors[kind[i]], lw=0.7)
        zoom = (w >= -0.1) & (w <= 0.03)
        p.plot(w[zoom], y[zoom], color=colors[kind[i]], lw=0.5)
    o.axvline(0, color="#888888", ls="--", lw=0.6), p.axvline(0, color="#888888", ls="--", lw=0.6)
    o.set_xlim(-0.1, 0.5), o.set_ylim(-100, 100), o.set_yticks([-100, -50, 0, 50, 100]), o.set_xticks([0, 0.2, 0.4]), o.set_xticklabels(["0", "0.2", "0.4"])
    o.set_xlabel("Time from detected onset (s)"), o.set_ylabel("Position (mm)"), p.set_ylabel("Position (mm)"), p.set_xticks([]), p.set_ylim(-2, 2), p.set_xlim(-0.1, 0.03)
    cv.text(422, 645, "Examples aligned to\nmovement onset", fontsize=5.5), cv.text(560, 645, "Examples aligned to\nmovement onset, zoom", fontsize=5.5, color="#555555")
    q = cv.ax((510, 738, 610, 796))
    acc = np.array([store.load(res.cfg.results / "wheel" / f"{x}.npz")["acc"] for x in res.sessions])
    off_t = np.arange(-0.1, 0.0301, 0.005)
    shade(q, off_t, acc.mean(0), acc.std(0), "k", alpha=0.45)
    q.axhline(0.5, color="k", ls="--", lw=0.6), q.axvline(0, color="#888888", ls="--", lw=0.6), q.set_ylim(0, 1.02), q.set_xlim(-0.1, 0.03), q.set_xticks([-0.1, -0.05, 0]), q.set_xticklabels(["-0.1", "-0.05", "0"])
    q.set_xlabel("Time from detected onset (s)"), q.set_ylabel("Decoding\nperformance")
    return cv.finish()


def ext2(res):
    cv = Canvas(res.cfg, "ext2", "extended-data-fig2.png", scale=1.1)
    cv.slot((15, 8, 285, 200), "Histology and atlas images")
    cv.slot((270, 8, 628, 200), "Probe track fit")
    cv.slot((418, 204, 610, 590), "Area assignment with histology")
    cv.slot((15, 597, 505, 815), "Alignment cross-validation")
    cv.letter("c", 22, 208), cv.letter("d", 365, 208)
    cv.text(190, 208, "Step 3: Use physiological signatures to find depth and scaling along that vector", fontsize=5.8)
    a = cv.ax((52, 228, 360, 574))
    d = cv.ax((371, 228, 414, 574))
    session, probe = EXT2_PROBE
    if session not in res.sessions:
        session = res.sessions[0]
    ses = Session(res.cfg.data, session)
    keep_clu = ses.a("clusters.probes") == probe
    st, clu = res.spikes(session)
    sel = keep_clu[clu]
    tt, depth = st[sel], ses.a("spikes.depths")[sel]
    tm = tt < 2200
    h, xe, ye = np.histogram2d(tt[tm], depth[tm], bins=(440, 190), range=((0, 2200), (0, 3800)))
    a.imshow(h.T, origin="lower", aspect="auto", extent=(0, 2200, 0, 3800), cmap="gray_r", vmin=0, vmax=np.percentile(h, 99), interpolation="nearest")
    a.set_xlabel("Time (s)"), a.set_yticks(range(0, 4000, 500)), a.set_xticks(range(0, 2500, 500))
    labels = ses.channel_labels()[ses.a("channels.probe") == probe]
    site_y = ses.raw("channels.sitePositions")[ses.a("channels.probe") == probe, 1]
    names = {"Surface\nof brain": None, "Bottom of\ncortex": ("VISa", "low"), "CA1\npyramidal\nlayer": ("CA1", "mid"), "Hippocampus/\nthalamus\nborder": ("LP", "high"), "Border\nbetween\nnuclei": ("LD", "low")}
    for text, spec in names.items():
        if spec is None:
            y = site_y[labels != "root"].max() if (labels != "root").any() else site_y.max()
        else:
            ys = site_y[labels == spec[0]]
            if not len(ys):
                continue
            y = {"low": ys.min(), "mid": np.median(ys), "high": ys.max()}[spec[1]]
        py = 228 + 346 * (1 - y / 3800)
        cv.text(46, py, text, style="italic", fontsize=5, ha="right")
        a.annotate("", xy=(0, y), xytext=(-120, y), arrowprops=dict(arrowstyle="-|>", color="#f39c32", lw=0.8), annotation_clip=False)

    b = res.base(session)
    pos, times = ses.raw("sparseNoise.positions"), ses.a("sparseNoise.times")
    el, el_i = np.unique(pos[:, 0], return_inverse=True)
    az, az_i = np.unique(pos[:, 1], return_inverse=True)
    edges = np.arange(0, 3800 + 80, 80)
    maps = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sp = np.sort(tt[(depth >= lo) & (depth < hi)])
        c = np.searchsorted(sp, times + 0.2) - np.searchsorted(sp, times)
        m, n_ev = np.zeros((len(el), len(az))), np.zeros((len(el), len(az)))
        np.add.at(m, (el_i, az_i), c)
        np.add.at(n_ev, (el_i, az_i), 1)
        r = m / np.maximum(n_ev, 1)
        maps.append((r - r.min()) / max(r.max() - r.min(), 1e-9))
    d.imshow(np.concatenate(maps[::-1]), aspect="auto", cmap=PARULA, interpolation="nearest")
    for x in (0.333, 0.667):
        d.axvline(x * len(az) - 0.5, color="k", lw=0.5)
    d.set_axis_off()
    return cv.finish()


def ext3(res):
    cv = Canvas(res.cfg, "ext3", "extended-data-fig3.png")
    cv.slot((12, 6, 618, 808), "DiI tracks in aligned histology")
    return cv.finish()


def _contrast_block(cv, res, s, c, x0, x1, title, region):
    if s is None:
        return
    b = res.base(s)
    st, clu = res.spikes(s)
    spikes = st[clu == c]
    ses = Session(res.cfg.data, s)
    cv.text((x0 + x1) / 2 + 8, 26, title, style="italic", fontsize=8.4, color=REGION_COLORS[region])
    cv.text(x0 + 16, 46, "Waveform", fontsize=5.6), cv.text(x0 + 84, 44, "Neuron\nlocation", fontsize=5.6)
    wave = ses.raw("clusters.templateWaveforms")[c]
    chans = ses.raw("clusters.templateWaveformChans")[c].astype(int)
    on_probe = ses.a("channels.probe") == ses.a("clusters.probes")[c]
    sites = ses.raw("channels.sitePositions")[on_probe]
    near = chans < len(sites)
    centre = sites[chans[near][0]]
    ax = cv.ax((x0 - 12, 54, x0 + 48, 118))
    scale = np.abs(wave).max()
    for k, ch in enumerate(chans[near]):
        dx, dy = sites[ch] - centre
        if abs(dy) <= 90:
            ax.plot(dx / 16 * 45 + np.arange(wave.shape[0]) * 0.5, dy / 20 * 18 + wave[:, np.flatnonzero(near)[k]] / scale * 20, "k", lw=0.45)
    ax.set_axis_off()
    ins = cv.ax((x0 + 54, 54, x0 + 106, 120))
    xyz = ses.channel_ccf()[ses.peak_channel()[c]] / 25
    sl = atlas.volume()[int(np.clip(xyz[0], 0, 527))]
    ins.contour((atlas.outlines(sl != 0) | atlas.outlines(sl)).astype(float), levels=[0.5], colors="k", linewidths=0.12)
    ins.plot(xyz[2], xyz[1], "s", color="#e0403c", ms=2.2, mew=0), ins.invert_yaxis(), ins.set_xlim(0, 240), ins.set_axis_off()
    levels = (1.0, 0.5, 0.25, 0.0)
    greys = ("0.05", "0.3", "0.55", "0.72")
    ra = cv.ax((x0, 130, x1, 190))
    bar = cv.ax((x0 - 12, 130, x0 - 5, 190))
    row, start = 0, []
    for lv, col in zip(levels, greys):
        ev = b["stim"][b["included"] & (b["cr"] == lv)]
        start.append((row, row + len(ev), col))
        for e in ev:
            sp = spikes[np.searchsorted(spikes, e - 0.05):np.searchsorted(spikes, e + 0.2)] - e
            ra.plot(sp, np.full(len(sp), row), ls="", marker="s", ms=0.55, mew=0, color=col)
            row += 1
    for a_, z_, col in start:
        bar.axhspan(a_, z_, color=col, lw=0)
    bar.set_ylim(row, 0), bar.set_xlim(0, 1), bar.set_axis_off()
    if x0 < 100:
        for (a_, z_, _), t in zip(start, ("High", "Med.", "Low", "Zero")):
            cv.text(x0 - 15, 130 + 60 * (a_ + z_) / 2 / row, t, ha="right", fontsize=5.4, color={"High": "k", "Med.": "#444444", "Low": "#777777", "Zero": "#999999"}[t])
        cv.text(x0 - 15, 124, "Contrast", fontsize=5.4, ha="right")
    ra.axvline(0, color="k", ls=":", lw=0.5), ra.set_xlim(-0.05, 0.2), ra.set_ylim(row, -1), ra.set_axis_off()
    pa = cv.ax((x0, 205, x1, 268))
    for lv, col in zip(levels, greys):
        ev = b["stim"][b["included"] & (b["cr"] == lv)]
        if len(ev) > 1:
            r, t = aligned_rates(st, clu, np.array([c]), ev, (-0.05, 0.2), 0.03)
            shade(pa, t, r[:, :, 0].mean(0), r[:, :, 0].std(0) / np.sqrt(len(r)), col, lw=1.1, alpha=0.35)
    limit = nice_top(pa.get_ylim()[1] / 1.05)
    pa.axvline(0, color="k", ls=":", lw=0.5), pa.set_xlim(-0.05, 0.2), pa.set_ylim(0, limit), pa.set_xticks([0, 0.1, 0.2]), pa.set_xticklabels(["0", "0.1", "0.2"])
    pa.set_xlabel("Time from stim onset (s)"), pa.set_ylabel("Firing rate (sp/s)")


def ext4(res, chosen):
    cv = Canvas(res.cfg, "ext4", "extended-data-fig4.png", scale=1.15)
    cv.letter("a", 28, 16), cv.letter("b", 25, 300), cv.letter("c", 403, 8), cv.letter("d", 514, 8), cv.letter("e", 596, 8), cv.letter("f", 408, 618)
    s, c = _ex(chosen, "ext4_visp")
    _contrast_block(cv, res, s, c, 72, 175, "VISp example", "VISp")
    s, c = _ex(chosen, "ext4_visam")
    _contrast_block(cv, res, s, c, 225, 328, "VISam example", "VISam")

    x, r = res.normalized("ext4_even", "high")
    odd, _ = res.normalized("ext4_odd", "high")
    keep_t = PSTH_T >= -0.05
    tb = PSTH_T[keep_t]
    x, odd = x[:, keep_t], odd[:, keep_t]
    order = [("VISp", "VISl", "VISpm", "VISam", "VISrl", "VISa", "RSP"), ("ACA", "MOs", "PL", "ILA", "ORB", "MOp", "SSp"), ("DG", "CA3", "CA1", "POST", "SUB", "OLF", "BLA"),
             ("CP", "GPe", "SNr", "ACB", "LS"), ("LGd", "LP", "LD", "POL", "MD", "VPL", "PO", "VPM", "RT", "MG"), ("SCs", "SCm", "MRN", "APN", "PAG", "ZI")]
    columns = ((31, 45, 80), (93, 107, 143), (155, 168, 205), (218, 232, 268), (281, 294, 330), (344, 358, 393))
    cv.text(190, 310, "Stimulus onset aligned", fontsize=6.2)
    for (sx, hx0, hx1), regions in zip(columns, order):
        rows = []
        y = 323.0
        counts = [(r == name).sum() for name in regions]
        total = max(sum(counts), 1)
        for name, n in zip(regions, counts):
            h = 452 * n / total
            if n:
                strip = cv.ax((sx, y, sx + 7, y + h))
                strip.axvspan(0, 1, color=REGION_COLORS[name]), strip.set_axis_off()
                cv.text(sx - 6, y + h / 2, name, rotation=90, color=REGION_COLORS[name], fontsize=5)
                m = r == name
                q, o = x[m], np.nan_to_num(odd[m])
                # latency: first time the odd-trial rate reaches half of its peak after the stimulus
                after = np.maximum(o, 0) * (tb >= 0)
                peak = after.max(1, keepdims=True)
                lat = (after >= 0.5 * np.maximum(peak, 1e-9)).argmax(1)
                rows.append(q[np.argsort(lat)])
            y += h
        a = cv.ax((hx0, 323, hx1, 775))
        if rows:
            a.imshow(np.nan_to_num(np.concatenate(rows)), aspect="auto", cmap="gray_r", vmin=0.7, vmax=4, extent=(-0.05, 0.3, 1, 0), interpolation="antialiased", interpolation_stage="data")
        a.axvline(0, color="#666666", ls=(0, (4, 3)), lw=0.6), a.set_yticks([]), a.set_xticks([0, 0.2]), a.set_xticklabels(["0", "0.2"])
        clean(a, False)
    bar = cv.ax((310, 303, 380, 308))
    bar.imshow(np.linspace(0.7, 4, 100)[None], cmap="gray_r", aspect="auto", extent=(0.7, 4, 0, 1)), bar.set_yticks([]), bar.set_xticks([1, 4]), clean(bar, False)
    cv.text(340, 297, "Average normalized firing rate", fontsize=5, color="#666666")
    cv.text(75, 790, "Time from stim (s)", fontsize=5.8)

    from matplotlib.ticker import FixedLocator

    t_m, t_r = -0.2 + DT / 2 + np.arange(60) * DT, -0.1 + DT / 2 + np.arange(60) * DT
    stacks = (("ext4_stim", (438, 48, 513, 578), PSTH_T, (-0.03, 0.27), "Time from\nstim (s)", [0, 0.1, 0.2], "Stimulus onset\naligned"),
              ("ext4_move", (520, 48, 595, 578), t_m, (-0.2, 0.05), "Time from\nmove (s)", [-0.2, -0.1, 0], "Movement\nonset aligned"),
              ("ext4_reward", (602, 48, 677, 578), t_r, (-0.05, 0.16), "Time from\nreward (s)", [0, 0.1], "Reward\ndelivery aligned"))
    for key, box, t, xl, lab, ticks, head in stacks:
        a = cv.ax(box)
        q, reg = res.normalized(key, "high")
        for i, name in enumerate(ANATOMY_ORDER):
            d = q[reg == name] if len(q) else np.zeros((0, len(t)))
            if len(d):
                n = min(d.shape[1], len(t))
                shade(a, t[:n], np.nanmean(d, 0)[:n] * 0.8 - i, np.nanstd(d, 0)[:n] * 0.8 / np.sqrt(len(d)), REGION_COLORS[name], lw=1.0, alpha=0.4)
        a.axvline(0, color="#666666", ls=(0, (4, 3)), lw=0.6), a.set_ylim(-42, 3), a.set_xlim(*xl), a.set_yticks([]), a.spines["left"].set_visible(False), a.set_xlabel(lab)
        a.set_xticks(ticks), a.set_xticklabels([f"{v:g}" for v in ticks])
        cv.text((box[0] + box[2]) / 2, 20, head, fontsize=6.2)
    for i, name in enumerate(ANATOMY_ORDER):
        cv.text(436, 48 + 530 * (3 + i) / 45, name, color=REGION_COLORS[name], fontsize=5, ha="right")
    groups = (("Visual cortex", 0, 5), ("Frontal/somatomotor\ncortex", 7, 13), ("Hippocampus", 14, 18), ("Basal ganglia", 21, 25), ("Thalamus", 26, 35), ("Midbrain", 36, 40))
    for name, a_, b_ in groups:
        color = GROUP_COLORS.get(name.replace("\ncortex", " cortex"), "#555555")
        cv.text(407, 48 + 530 * (3 + (a_ + b_) / 2) / 45, name, rotation=90, color=color, fontsize=5)
    sb = cv.ax((445, 40, 447, 58))
    sb.axvspan(0, 1, color="k"), sb.set_axis_off()
    cv.text(472, 49, "2 sp/s\n(norm.)", fontsize=4.6)

    f = cv.ax((480, 620, 640, 775))
    t = res.regression()
    if len(t):
        for i, col in enumerate(("action", "vision_contra", "choice")):
            est, lo, hi = focality_interval(t, col, res.cfg.seed)
            f.plot([i, i], [lo, hi], "k", lw=1.2), f.plot(i, est, "ko", ms=3)
    f.axhline(1 / 42, color="#888888", ls="--", lw=0.7), f.set_ylim(0, 0.105), f.set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
    f.text(0.97, 0.17, "uniform distribution", transform=f.transAxes, style="italic", ha="right", fontsize=5)
    f.set_xlim(-0.3, 2.4), f.set_xticks([0, 1, 2]), f.set_xticklabels(["Action", "Contra Vision", "Choice"]), f.set_ylabel("Focality index")
    return cv.finish()


def _density_scatter(ax, x, y, lim, size):
    ok = np.isfinite(x) & np.isfinite(y) & (x >= lim[0]) & (x <= lim[1]) & (y >= lim[0]) & (y <= lim[1])
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return
    h, xe, ye = np.histogram2d(x, y, bins=60, range=[lim, lim])
    from scipy.ndimage import gaussian_filter

    d = gaussian_filter(h, 2.0)
    i = np.clip(np.searchsorted(xe, x) - 1, 0, 59)
    j = np.clip(np.searchsorted(ye, y) - 1, 0, 59)
    z = d[i, j]
    o = np.argsort(z)
    ax.scatter(x[o], y[o], c=z[o], s=size, cmap=PARULA, lw=0)


def ext5(res, chosen):
    from matplotlib.patches import FancyArrowPatch, Rectangle

    cv = Canvas(res.cfg, "ext5", "extended-data-fig5.png", scale=1.15)
    for s, x, y in (("a", 18, 20), ("b", 18, 124), ("c", 422, 20), ("d", 18, 200), ("e", 18, 398), ("f", 360, 398), ("g", 20, 678), ("h", 398, 670)):
        cv.letter(s, x, y)
    cv.slot((16, 120, 432, 196), "Model cartoons")
    s, c = _ex(chosen, "ext5")
    a = cv.ax((52, 22, 400, 98))
    cc = cv.ax((470, 36, 600, 160))
    if s:
        b = res.base(s)
        P, rows, trnum, *_ = rr.build_design(b)
        d = store.load(res.cfg.results / "regression" / "data" / f"{s}.npz")
        i = int(np.flatnonzero(d["ids"] == c)[0])
        K = res.kernels(s)["K"][:, i]
        keep = np.flatnonzero(b["kernel_trial"])
        moved = (b["has_left"] | b["has_right"])[keep]
        first = next((k for k in range(len(keep) - 4) if moved[k:k + 4].all()), 0)
        add = d["means"][i] if "means" in d else 0
        pieces = {"Smoothed spiking": (d["F"][:, i] + add, "#6fb543"), "Full model": (P @ K + add, "k"), "Vision kernels": (P[:, :540] @ K[:540], "#2a78c4"),
                  "Action kernel": (P[:, 540:595] @ K[540:595], "#f0b429"), "Choice kernel": (P[:, 595:] @ K[595:], "#8e44ad")}
        x0 = 0.0
        for k in range(first, first + 4):
            m = trnum == k
            x = x0 + np.arange(m.sum()) * DT
            for name, (v, col) in pieces.items():
                a.plot(x, v[m], color=col, lw=0.85 if name != "Smoothed spiking" else 0.7, label=name if k == first else None)
            a.axvline(x0, color="#555555", lw=0.8)
            x0 = x[-1] + DT * 4
        a.legend(fontsize=4.8, loc="upper right", labelcolor="linecolor", handlelength=0, handletextpad=0, labelspacing=0.1)
        a.text(0.62, 0.9, "(trial onsets)", style="italic", transform=a.transAxes, fontsize=5), a.set_xlim(0, x0), a.set_ylabel("Firing rate (sp/s)"), a.set_xlabel("Time (s)")
        a.set_ylim(bottom=min(0, a.get_ylim()[0]))
        act, cho = K[540:595], K[595:]
        t = -0.25 + np.arange(55) * DT
        # the paper plots the sum and the difference of the two responses, which is twice each kernel
        cc.plot(t, act + cho, color="#2a78c4", label="Left response", lw=1.3), cc.plot(t, act - cho, color=CONTRA, label="Right response", lw=1.3)
        cc.plot(t, 2 * act, "k", label="Action kernel (L+R)", lw=1.3), cc.plot(t, 2 * cho, color="#bbbbbb", label="Choice kernel (L-R)", lw=1.3)
        cc.axhline(0, color="k", ls="--", lw=0.5), cc.legend(fontsize=4.8, loc="upper left", handlelength=1.2), cc.set_xlabel("Time from move onset (s)"), cc.set_ylabel("Response (sp/s)")
        cc.set_xticks([-0.2, -0.1, 0]), cc.set_xticklabels(["-0.2", "-0.1", "0"])
        cv.text(530, 22, "Example Action/Choice\nkernel decomposition", fontsize=5.4)

        def pick(cl, cr):
            k = np.flatnonzero((b["cl"][keep] == cl) & (b["cr"][keep] == cr) & moved)
            return k[0] if len(k) else None

        three = [q for q in (pick(1, 0), pick(0, 0.5), pick(1, 0.5)) if q is not None]
        if len(three) < 3:
            three = [int(q) for q in np.flatnonzero(moved)[:3]]
        idx = np.concatenate([np.flatnonzero(trnum == q)[:75] for q in three])
        X = P[idx]
        rows_, cols_, vals = __import__("scipy.sparse", fromlist=["find"]).find(X)
        tox = cv.ax((76, 212, 240, 380))
        tox.plot(rows_[vals > 0], cols_[vals > 0], ls="", marker="s", ms=0.4, mew=0, color="#e84a4a", rasterized=True), tox.plot(rows_[vals < 0], cols_[vals < 0], ls="", marker="s", ms=0.4, mew=0, color="#4a68c4", rasterized=True)
        tox.set_xlim(0, len(idx)), tox.set_ylim(650, 0)
        cos = cv.ax((252, 212, 416, 380))
        C = rr.cosine_design(P, trnum)[idx]
        cos.imshow(C.T, aspect="auto", cmap=BLUE_WHITE_RED, vmin=-1.5, vmax=1.5, interpolation="nearest")
        names = ("Left low", "Left med.", "Left high", "Right low", "Right med.", "Right high", "Action", "Choice")
        bounds = [i * 90 for i in range(7)] + [595, 650]
        cos_bounds = [i * 18 for i in range(7)] + [119, 130]
        for ax_, bd in ((tox, bounds), (cos, cos_bounds)):
            for y in bd[1:-1]:
                ax_.axhline(y - (0.5 if ax_ is cos else 0), color="#777777", lw=0.6)
            ax_.set_xticks([]), ax_.set_yticks([])
            for sp in ax_.spines.values():
                sp.set_visible(True), sp.set_color("#999999")
        for name, lo, hi in zip(names, bounds[:-1], bounds[1:]):
            cv.text(74, 212 + 168 * (lo + hi) / 2 / 650, name, ha="right", fontsize=4.8, color={"Left high": "k", "Action": "k", "Choice": "k"}.get(name, "#777777"))
        for i_, name in enumerate(("Trial 1", "Trial 2", "Trial 3")):
            for x_ in (76, 252, 446):
                cv.text(x_ + 164 * (i_ + 0.5) / 3, 388, name, fontsize=4.8)
        B = np.load(res.cfg.results / "regression" / "real" / "basis_full.npy")
        rrm = cv.ax((446, 212, 610, 380))
        z = (X @ B)[:, :50].T
        z = z / np.maximum(np.abs(z).max(1, keepdims=True), 1e-9)
        rrm.imshow(z, aspect="auto", cmap=BLUE_WHITE_RED, vmin=-0.45, vmax=0.45, interpolation="nearest"), rrm.set_xticks([]), rrm.set_yticks([0, 49]), rrm.set_yticklabels(["1", "50"]), rrm.set_ylabel("Predictor Index")
        cv.text(158, 204, "Toeplitz Model: X", fontsize=5.2), cv.text(334, 204, "Cosine Model: X", fontsize=5.2), cv.text(528, 204, "Reduced Rank Model: X*b", fontsize=5.2)

    cmp_, included = {}, []
    for x in res.sessions:
        p = res.cfg.results / "regression" / "compare" / f"{x}.npz"
        if p.exists():
            for k, v in store.load(p).items():
                cmp_.setdefault(k, []).append(v)
            included.append(res.kernels(x)["ve"] >= 0.02)
    cmp_ = {k: 100 * np.concatenate(v) for k, v in cmp_.items()}
    included = np.concatenate(included) if included else np.zeros(0, bool)
    for name, y0 in (("toeplitz", 405), ("cosine", 540)):
        for (x0, x1), lim, size in (((78, 195), (-30, 80), 0.5), ((232, 348), (-5, 20), 3.2)):
            ax = cv.ax((x0, y0, x1, y0 + 115))
            if cmp_:
                _density_scatter(ax, cmp_["reduced_test"], cmp_[f"{name}_test"], lim, size)
            ax.plot(lim, lim, "r--", lw=0.6), ax.axhline(0, color="r", ls="--", lw=0.6), ax.axvline(0, color="r", ls="--", lw=0.6)
            ax.set_xlim(*lim), ax.set_ylim(*lim)
            if x0 == 78:
                ax.set_ylabel(f"% Cross-validated\nVariance Explained,\n{name.capitalize()} model", fontsize=5.6)
                ax.add_patch(Rectangle((-5, -5), 25, 25, fc="none", ec="#999999", lw=0.8))
                ax.set_xticks([-20, 0, 20, 40, 60, 80]), ax.set_yticks([-20, 0, 20, 40, 60, 80])
            else:
                ax.set_xticks([0, 20]), ax.set_yticks([0, 20])
        cv.fig.add_artist(FancyArrowPatch(((165) / cv.w, 1 - (y0 + 80) / cv.h), (228 / cv.w, 1 - (y0 + 55) / cv.h), transform=cv.fig.transFigure, color="#999999", lw=0.9, arrowstyle="-|>", mutation_scale=7))
    cv.text(135, 668, "% C.V. Var. Expl., Reduced Rank model", fontsize=5.2)
    for name, label, (y0, y1), last in (("reduced", "Reduced rank model", (415, 477), False), ("toeplitz", "Toeplitz model", (494, 556), False), ("cosine", "Cosine model", (573, 635), True)):
        ax = cv.ax((400, y0, 583, y1))
        if cmp_:
            tr_, te = cmp_[f"{name}_train"][included], cmp_[f"{name}_test"][included]
            over = (tr_ - te) / np.where(tr_ > 0, tr_, np.nan)
            over = over[np.isfinite(over) & (over >= -1) & (over <= 1)]
            ax.hist(over, bins=np.linspace(-1, 1, 41), color=HIST_BLUE, edgecolor="k", lw=0.3)
            ax.axvline(np.median(over), color="r", lw=0.8)
            ax.set_title(f"{label}: median={np.median(over):.2f}", fontsize=5.4, pad=1)
        ax.axvline(0, color="k", ls="--", lw=0.7), ax.set_xlim(-1, 1), ax.set_xticks([-1, 0, 1] if last else []), ax.set_ylabel("Number of\nneurons", fontsize=5.2)
    cv.text(490, 650, "Proportion of overfit explained variance", fontsize=5.2)

    cur = _decoding_curves(res)
    panels = ((78, 113, "vision_contra", "stim", "Vision Contra"), (132, 167, "vision_ipsi", "stim", "Vision Ipsi"), (195, 233, "action", "move", "Action"), (257, 290, "choice", "move", "Choice"))
    heats = {}
    for _, _, key, align, _ in panels:
        heats[key] = np.full((42, len(decoding.OFFSETS[align])), np.nan)
        for i, r in enumerate(LATENCY_ORDER):
            v = cur.get(key, {}).get(r, [])
            if v:
                heats[key][i] = np.nanmean(v, 0)
    finite = np.concatenate([h[np.isfinite(h)] for h in heats.values()])
    top = np.percentile(np.abs(finite), 98) if len(finite) else 1.0  # the paper's projection units are not stated
    for x0, x1, key, align, title in panels:
        ax = cv.ax((x0, 680, x1, 795))
        off = decoding.OFFSETS[align]
        white_under = PURPLE_HEAT.with_extremes(under="white")
        ax.imshow(np.nan_to_num(heats[key]), aspect="auto", cmap=white_under, vmin=top / 3, vmax=top, extent=(off[0], off[-1], 42, 0), interpolation="nearest")
        ax.axvline(0, color="#666666", ls=(0, (3, 2)), lw=0.6), ax.set_yticks([])
        ax.set_xticks([0, 0.2] if align == "stim" else [-0.1, 0]), ax.set_xticklabels(["0", "0.2"] if align == "stim" else ["-0.1", "0"])
        clean(ax, False)
        cv.text(x1 + 4, 738, title, rotation=-90 if align == "stim" else 90, fontsize=5.6)
    region_name_columns(cv, 54, 74, 682, 793, fontsize=2.9, axis_x=76)
    region_name_columns(cv, 318, 340, 682, 793, fontsize=2.9)
    cv.text(120, 812, "Time from stim (s)", fontsize=5.2), cv.text(240, 812, "Time from move (s)", fontsize=5.2)
    bar = cv.ax((345, 705, 350, 750))
    bar.imshow(np.linspace(1, 0, 100)[:, None], cmap=PURPLE_HEAT, aspect="auto"), bar.set_axis_off()
    cv.text(362, 777, "C.V. residual\npopulation\ndecoding", fontsize=4.6)

    h = cv.ax((445, 670, 565, 795))
    t = res.regression()
    if len(t):
        bins = np.arange(-5, 20.5, 0.5)
        h.hist(100 * t["choice"].dropna(), bins=bins, color=HIST_BLUE)
        h.hist(100 * res.shuffle_ve(), bins=bins, color=HIST_ORANGE, alpha=0.9)
    h.text(0.62, 0.62, "Original", color=HIST_BLUE, transform=h.transAxes, fontsize=5.6), h.text(0.62, 0.54, "Shuffle", color=HIST_ORANGE, transform=h.transAxes, fontsize=5.6)
    h.text(0.32, 0.93, "2% threshold", style="italic", transform=h.transAxes, fontsize=5.4)
    h.axvline(2, color="k", ls="--", lw=0.8), h.set_xlim(-5, 20), h.set_ylim(0, 100), h.set_xlabel("C.V. % var. expl."), h.set_ylabel("# neurons")
    return cv.finish()


EXT6_A = (("ACA", "MOs", "PL", "ILA", "ORB", "MOp", "SSp"), ("CP", "GPe", "SNr", "ACB", "LS", "SCs", "SCm"), ("MRN", "APN", "PAG", "VISp", "VISl", "VISpm", "VISam"),
          ("VISrl", "VISa", "LGd", "LP", "LD", "POL", "MD"), ("VPL", "PO", "VPM", "RT", "MG", "DG", "CA3"), ("CA1", "SUB", "POST", "RSP", "ZI", "OLF", "BLA"))
EXT6_B = (("SCs", "SCm", "MRN", "APN", "PAG", "ZI", "PO", "VPM"), ("CP", "GPe", "SNr", "ACB", "LS", "RT", "MG", "BLA"), ("DG", "CA3", "CA1", "SUB", "POST", "POL", "MD", "OLF"),
          ("ACA", "MOs", "PL", "ILA", "ORB", "MOp", "SSp", "VPL", "RSP"), ("VISp", "VISl", "VISpm", "VISam", "VISrl", "VISa", "LGd", "LP", "LD"))


def ext6(res):
    cv = Canvas(res.cfg, "ext6", "extended-data-fig6.png", scale=1.25)
    cv.letter("a", 15, 18), cv.letter("b", 15, 433)
    t = res.regression()
    cv.text(70, 22, "All neurons, full model", color="#aaaaaa", fontsize=5.8), cv.text(150, 22, "Included neurons:", style="italic", fontsize=5.8)
    for x, s, c in ((215, "full model", "k"), (258, "VisionContra", "#1f7fc4"), (305, "VisionIpsi", GO), (336, "Action", "#f0b429"), (365, "Choice", "#8e44ad")):
        cv.text(x, 22, s, color=c, fontsize=5.8)
    ys = (46, 111, 176, 242, 306, 370)
    series = (("ve_all", "#999999"), ("ve", "k"), ("vision_contra", "#1f7fc4"), ("vision_ipsi", GO), ("action", "#f0b429"), ("choice", "#8e44ad"))
    for row, y in zip(EXT6_A, ys):
        for j, r in enumerate(row):
            x = 36 + j * 76
            a = cv.ax((x, y, x + 64, y + 39))
            s = t[t["region"] == r] if len(t) else t
            inc = s[s["ve"] >= 0.02] if len(s) else s
            for i, (col, color) in enumerate(series):
                v = 100 * (s["ve"] if col == "ve_all" else inc[col] if col != "ve" else inc["ve"]).dropna().to_numpy() if len(s) else np.zeros(0)
                if len(v):
                    q = np.percentile(v, [10, 25, 50, 75, 90])
                    a.plot([i, i], q[[0, 4]], color=color, lw=0.5), a.plot([i, i], q[[1, 3]], color=color, lw=1.6)
            a.set_xlim(-1, 6), a.set_ylim(0, 25), a.set_yticks([0, 25] if j == 0 else []), a.set_xticks([]), clean(a, j == 0)
            a.spines["bottom"].set_visible(True)
            cv.text(x + 32, y - 7, f"{r} ({len(s)}, {len(inc)})", color=REGION_COLORS[r], fontweight="bold", fontsize=5.3)
    cv.text(20, 66, "C.V. % var expl", rotation=90, fontsize=5.5)
    conds = (("contra_correct", GO), ("contra_miss", MISS), ("contra_passive", PASSIVE))
    for row, y in zip(EXT6_B, (437, 512, 588, 663, 738)):
        for j, r in enumerate(row):
            x = 37 + j * 64.3
            a = rate_panel(cv, res, (x, y, x + 49, y + 56), r, conds)
            a.tick_params(labelsize=5)
            cv.text(x + 26, y - 4, r, color=REGION_COLORS[r], fontweight="bold", fontsize=6)
    cv.text(592, 450, "Task, Go trial", color=GO, fontsize=5.8), cv.text(592, 462, "Task, Miss trial", color=MISS, fontsize=5.8), cv.text(592, 478, "Passive\npresentation", color=PASSIVE, fontsize=5.8)
    cv.text(70, 812, "Time from stimulus onset (s)", fontsize=6), cv.text(20, 766, "Firing rate (sp/s)", rotation=90, fontsize=5.5)
    return cv.finish()


def _cccp_table(res):
    """stat -> event -> region -> (sessions' significance arrays stacked as neurons x bins)."""
    out = {}
    for s in res.sessions:
        z = store.load(res.cfg.results / "cccp" / f"{s}.npz")
        if len(z["ids"]) == 0:
            continue
        reg = res.base(s)["n_region"][z["ids"]]
        for k in z:
            if k.endswith("_p") and k[:2] in ("cp", "dp"):
                stat, ev, _ = k.split("_")
                sig = (z[k] < 0.025) | (z[k] > 0.975)
                for r in np.unique(reg):
                    out.setdefault(stat, {}).setdefault(ev, {}).setdefault(r, []).append(sig[:, reg == r].T)
    return {a: {b: {r: np.concatenate(v) for r, v in c.items()} for b, c in d.items()} for a, d in out.items()}


def _centers(ev):
    """Center of each 100 ms count box, so bins sit exactly on 0 and 0.2 s."""
    w = cccp.WINDOWS[ev]
    return w[0] + np.arange(int(round((w[1] - w[0]) / cccp.STEP)) + 1) * cccp.STEP


def ext7(res):
    cv = Canvas(res.cfg, "ext7", "extended-data-fig7.png", scale=1.1)
    cv.letter("a", 18, 22), cv.letter("b", 385, 22), cv.letter("c", 474, 22), cv.letter("d", 546, 22), cv.letter("e", 380, 490), cv.letter("f", 380, 652)
    table = _cccp_table(res)
    groups = ((("cp", "stim", 28, 65), ("cp", "move", 69, 105), ("dp", "stim", 112, 148), ("dp", "move", 152, 188)), (("cp", "stim", 204, 240), ("cp", "move", 245, 281), ("dp", "stim", 288, 324), ("dp", "move", 328, 365)))
    left = ANATOMY_ORDER[::2]
    right = ANATOMY_ORDER[1::2]
    for regions, cols in zip((left, right), groups):
        for i, r in enumerate(regions):
            y = 50 + i * 35.4
            for stat, ev, x0, x1 in cols:
                a = cv.ax((x0, y, x1, y + 29))
                v = table.get(stat, {}).get(ev, {}).get(r)
                if v is not None and len(v):
                    p = v.mean(0)
                    shade(a, _centers(ev)[: len(p)], 100 * p, 196 * np.sqrt(p * (1 - p) / len(v)), REGION_COLORS[r], lw=0.9, alpha=0.4)
                a.axhline(5, color="k", ls="--", lw=0.4), a.set_xticks([]), a.set_yticks([]), clean(a, False, True)
                a.set_ylim(0, 15 if stat == "cp" else 50), a.axvline(0, color="#888888", ls=(0, (3, 2)), lw=0.4)
            cv.text((cols[2][2] + cols[3][3]) / 2, y + 2, r, color=REGION_COLORS[r], fontsize=6)
    for x, s in ((64, "Choice prob."), (148, "Detect prob."), (241, "Choice prob."), (326, "Detect prob.")):
        cv.text(x, 26, s, fontsize=5.2)
    for x0, x1 in ((46, 87), (130, 170), (222, 263), (306, 347)):
        cv.text(x0 - 2, 34, "Stim-aligned", fontsize=4.8), cv.text(x1 + 2, 40, "Move-aligned", fontsize=4.8)
    cv.text(30, 42, "15", fontsize=5), cv.text(112, 42, "50", fontsize=5), cv.text(16, 56, "% neurons", rotation=90, fontsize=5)
    cv.text(52, 815, "Time from\nstim onset (s)", fontsize=4.8), cv.text(95, 815, "Time from\nmove onset (s)", fontsize=4.8)

    bars = (("dp", "move", (410, 73, 455, 480), 0.0, "DP\n-50 to 50 ms\nfrom move onset", 50), ("cp", "move", (485, 73, 528, 480), 0.0, "CP\n-50 to 50 ms\nfrom move onset", 25),
            ("cp", "move", (552, 73, 595, 480), 0.2, "CP\n+150 to 250 ms\nfrom move onset", 25))
    for stat, ev, box, center, title, xmax in bars:
        a = cv.ax(box)
        t = _centers(ev)
        for i, r in enumerate(ANATOMY_ORDER):
            v = table.get(stat, {}).get(ev, {}).get(r)
            if v is not None and len(v):
                j = int(np.argmin(np.abs(t - center)))
                p = v[:, j].mean()
                lo, hi = behavior.binomial_ci(v[:, j].sum(), len(v))
                a.barh(i, 100 * p, 0.8, color=REGION_COLORS[r])
                a.plot([100 * lo, 100 * hi], [i, i], "k", lw=0.7)
                if lo > 0.05:
                    a.text(100 * hi + 0.5, i + 0.15, "*", va="center", fontsize=8)
        a.axvline(5, color="k", ls="--", lw=0.5), a.set_ylim(41.5, -0.5), a.set_xlim(0, xmax), a.set_yticks([]), a.xaxis.tick_top()
        a.spines["top"].set_visible(True), a.spines["bottom"].set_visible(False)
        cv.text((box[0] + box[2]) / 2, 35, title, fontsize=5)
        cv.text((box[0] + box[2]) / 2, 59, "% neurons", fontsize=4.8)
    for i, r in enumerate(ANATOMY_ORDER):
        cv.text(406, 73 + 407 * (i + 0.5) / 42, r, color=REGION_COLORS[r], fontsize=5.4, ha="right", rotation=-30, rotation_mode="anchor")

    e = cv.ax((408, 490, 585, 627))
    for r, col in (("MOs", REGION_COLORS["MOs"]), ("SNr", REGION_COLORS["SNr"]), ("SCm", REGION_COLORS["SCm"]), ("MRN", REGION_COLORS["MRN"])):
        v = table.get("cp", {}).get("move", {}).get(r)
        if v is not None and len(v):
            p = v.mean(0)
            shade(e, _centers("move")[: len(p)], 100 * p, 196 * np.sqrt(p * (1 - p) / len(v)), col, alpha=0.3)
            e.text(0.03, {"MOs": 0.97, "SNr": 0.89, "SCm": 0.81, "MRN": 0.73}[r], r, color=col, transform=e.transAxes, va="top", fontsize=5.6)
    e.axhline(5, color="#888888", ls="--", lw=0.6), e.axvline(0, color="#888888", ls="--", lw=0.6), e.set_xlim(-0.3, 0.25), e.set_ylim(0, 30)
    e.set_xlabel("Time from move onset (s)"), e.set_ylabel("% neurons with significant CP")
    e.text(0.97, 0.13, "chance", style="italic", transform=e.transAxes, ha="right", fontsize=5.2)

    f = cv.ax((405, 655, 530, 795))
    pre = {}
    for s in res.sessions:
        z = store.load(res.cfg.results / "cccp" / f"{s}.npz")
        if "dp_pre_val" in z:
            reg = res.base(s)["n_region"][z["ids"]]
            for r in np.unique(reg):
                pre.setdefault(r, []).append(z["dp_pre_val"][reg == r])
    g = res.stage("engagement.npz")
    if "region_names" in g:
        gm = dict(zip([str(x) for x in g["region_names"]], g["go_miss"]))
        for r, v in pre.items():
            if r in gm and np.isfinite(gm[r]):
                f.text(gm[r], np.mean(np.concatenate(v)) - 0.5, r, color=REGION_COLORS[r], fontsize=4, ha="center")
        vals = [(gm[r], np.mean(np.concatenate(v)) - 0.5) for r, v in pre.items() if r in gm and np.isfinite(gm[r])]
        if vals:
            a_, b_ = np.array(vals).T
            f.set_xlim(min(a_.min(), -0.2) * 1.2, max(a_.max(), 1) * 1.1), f.set_ylim(min(b_.min(), -0.02) * 1.2, max(b_.max(), 0.02) * 1.2)
            if len(a_) > 2:
                from scipy.stats import pearsonr

                r_, p_ = pearsonr(a_, b_)
                f.text(0.45, 0.2, f"r = {r_:.2f}, p = {p_:.2e}", transform=f.transAxes, fontsize=5)
    from matplotlib.patches import FancyArrowPatch

    cv.text(562, 680, "Higher rates on\nGo trials", fontsize=5), cv.text(562, 775, "Higher rates on\nNoGo trials", fontsize=5)
    cv.fig.add_artist(FancyArrowPatch((562 / cv.w, 1 - 700 / cv.h), (562 / cv.w, 1 - 752 / cv.h), transform=cv.fig.transFigure, color="k", lw=1.6, arrowstyle="<|-|>", mutation_scale=9))
    f.axhline(0, color="#888888", ls="--", lw=0.6), f.axvline(0, color="#888888", ls="--", lw=0.6)
    f.set_xlabel("Go-Miss Firing rate (sp/s)"), f.set_ylabel("Pre-stimulus Detect Probability - 0.5")
    return cv.finish()


def _cca_once(X, Y, lam, nd):
    """Ridge CCA on one split, as in jpecc. Returns the top canonical weights."""
    ax_ = np.vstack([X, lam * np.eye(nd), np.zeros((nd, nd))])
    ay_ = np.vstack([Y, np.zeros((nd, nd)), lam * np.eye(nd)])
    qx, rx = np.linalg.qr(ax_ - ax_.mean(0))
    qy, ry = np.linalg.qr(ay_ - ay_.mean(0))
    l, _, mh = np.linalg.svd(qx.T @ qy, full_matrices=False)
    return np.linalg.solve(rx, l[:, 0]), np.linalg.solve(ry, mh[0])


def column_stats(values, min_n=3):
    """Mean and s.e. per column, only where min_n values exist. The asymmetry index has none in its first and last 50 ms."""
    n = np.isfinite(values).sum(0)
    ok = n >= min_n
    mean, se = np.full(values.shape[1], np.nan), np.full(values.shape[1], np.nan)
    mean[ok] = np.nanmean(values[:, ok], 0)
    se[ok] = np.nanstd(values[:, ok], 0) / np.sqrt(n[ok])
    return mean, se, ok


def _lead_panel(cv, res, ev, pair, box, up, down, show_x):
    from scipy.stats import ttest_1samp
    from steinmetz.jpecc import asymmetry

    a = cv.ax(box)
    arr = _jpecc_mean(res, ev, *pair)
    w = (-0.1, 0.3) if ev == "stim" else (-0.3, 0.1)
    if len(arr):
        idx = np.array([asymmetry(np.array(r, float)) for r in arr])
        t = w[0] + 0.005 + np.arange(idx.shape[1]) * 0.01
        m, se, ok = column_stats(idx)
        shade(a, t[ok], m[ok], 2 * se[ok], "k", alpha=0.5, lw=0.9)
        p = np.array([ttest_1samp(col[np.isfinite(col)], 0).pvalue if f else 1.0 for col, f in zip(idx.T, ok)])
        for mask, y, color in (((p < 0.05) & (m > 0), 0.31, up[1]), ((p < 0.05) & (m < 0), 0.27, down[1])):
            edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
            for s_, e_ in zip(edges[::2], edges[1::2]):
                if e_ - s_ >= 2:
                    a.plot(t[[s_, e_ - 1]], [y, y], color=color, lw=2, solid_capstyle="butt")
    a.axhline(0, color="#888888", ls="--", lw=0.5), a.set_xlim(*((0.05, 0.25) if ev == "stim" else (-0.2, 0.05))), a.set_ylim(-0.28, 0.35), a.set_yticks([-0.2, 0, 0.2]), a.set_yticklabels(["-0.2", "0", "0.2"])
    a.tick_params(labelsize=5.2)
    ticks = [0.1, 0.2] if ev == "stim" else [-0.2, -0.1, 0]
    a.set_xticks(ticks), a.set_xticklabels([f"{v:g}" for v in ticks])
    a.text(0.72, 0.84, f"{up[0]} leads ↑", color=up[1], transform=a.transAxes, fontsize=5.4), a.text(0.05, 0.08, f"{down[0]} leads ↓", color=down[1], transform=a.transAxes, fontsize=5.4)
    return a


def ext8(res):
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    from steinmetz.jpecc import pc_scores

    cv = Canvas(res.cfg, "ext8", "extended-data-fig8.png", scale=1.15)
    for s, x, y in (("a", 24, 22), ("b", 24, 354), ("c", 24, 585), ("d", 368, 347), ("e", 368, 582)):
        cv.letter(s, x, y)
    pairs = ((("VIS", "Frontal"), "VIS vs Frontal", "VIS", "Frontal"), (("VIS", "MB"), "VIS vs MB", "VIS", "MB"), (("MB", "Frontal"), "MB vs Frontal", "MB", "Frontal"))
    session = EXT8_SESSION if EXT8_SESSION in res.sessions else None
    example = store.load(res.cfg.results / "jpecc" / f"{session}.npz") if session else None
    if example is None or len(example) < 12:
        example = None
        for s in res.sessions:
            z = store.load(res.cfg.results / "jpecc" / f"{s}.npz")
            if len(z) >= 12:
                example, session = z, s
                break
    cm = JPECC_CMAP.with_extremes(bad="#888888")
    for row, (ev, y0, y1) in enumerate((("stim", 378, 448), ("move", 483, 553))):
        for (pair, title, ya, xa), (x0, x1) in zip(pairs, ((52, 126), (162, 234), (270, 344))):
            a = cv.ax((x0, y0, x1, y1))
            key = f"r|{ev}|{pair[0]}|{pair[1]}"
            if example is not None and key in example:
                r = np.array(example[key], float)
                r[example[key.replace("r|", "p|")] > 0.05] = np.nan
                jpecc_panel(a, r, ev)
                a.images[0].set_cmap(cm)
            else:
                no_data(a)
            lim = (0, 0.3) if ev == "stim" else (-0.3, 0)
            a.set_xticks(lim), a.set_yticks(lim), a.set_xticklabels([f"{v:g}" for v in lim]), a.set_yticklabels([f"{v:g}" for v in lim]), a.tick_params(labelsize=5)
            a.set_ylabel(ya, labelpad=-4, fontsize=5.4), a.set_xlabel(xa, labelpad=0, fontsize=5.4)
            if row == 0:
                cv.text((x0 + x1) / 2, y0 - 9, title, fontsize=6.4)
    cv.text(195, 354, "Example session", fontweight="bold", fontsize=7.2)
    for y in (378, 483):
        bar = cv.ax((352, y + 18, 358, y + 66))
        bar.imshow(np.linspace(0.5, -0.5, 100)[:, None], cmap=JPECC_CMAP, aspect="auto", extent=(0, 1, -0.5, 0.5)), bar.set_xticks([]), bar.yaxis.tick_right(), bar.set_yticks([-0.5, 0, 0.5]), bar.set_yticklabels(["-0.5", "0", "0.5"])
        bar.tick_params(labelsize=4.6), clean(bar, False, False), bar.set_yticks([-0.5, 0, 0.5]), cv.text(355, y + 10, "r", fontsize=6)
    for row, (ev, y0, y1) in enumerate((("stim", 617, 690), ("move", 723, 797))):
        for (pair, _, ya, xa), (x0, x1) in zip(pairs, ((46, 120), (156, 229), (266, 340))):
            a = cv.ax((x0, y0, x1, y1))
            arr = _jpecc_mean(res, ev, *pair)
            if len(arr):
                jpecc_panel(a, arr.mean(0), ev)
                if row == 0:
                    cv.text((x0 + x1) / 2, y0 - 15, f"n={len(arr)} sessions", fontsize=6.2)
            else:
                no_data(a)
            lim = (0, 0.3) if ev == "stim" else (-0.3, 0)
            a.set_xticks(lim), a.set_yticks(lim), a.set_xticklabels([f"{v:g}" for v in lim]), a.set_yticklabels([f"{v:g}" for v in lim]), a.tick_params(labelsize=5)
            a.set_ylabel(ya, labelpad=-4, fontsize=5.4), a.set_xlabel(xa, labelpad=0, fontsize=5.4)
    for y in (617, 723):
        bar = cv.ax((352, y + 18, 358, y + 66))
        bar.imshow(np.linspace(0.5, -0.5, 100)[:, None], cmap=JPECC_CMAP, aspect="auto", extent=(0, 1, -0.5, 0.5)), bar.set_xticks([]), bar.yaxis.tick_right(), bar.set_yticks([-0.5, 0, 0.5]), bar.set_yticklabels(["-0.5", "0", "0.5"])
        bar.tick_params(labelsize=4.6), clean(bar, False, False), bar.set_yticks([-0.5, 0, 0.5]), cv.text(355, y + 10, "r", fontsize=6)
    mb, vis, fr = ("MB", REGION_COLORS["MRN"]), ("VIS", REGION_COLORS["VISpm"]), ("Frontal", "#a67c3d")
    rows = (("stim", ((352, 412), (422, 482), (495, 555))), ("move", ((585, 645), (660, 720), (735, 795))))
    for ev, ys in rows:
        for (pair, up, down), (y0, y1) in zip((((("MB", "Frontal")), mb, fr), (("VIS", "Frontal"), vis, fr), (("VIS", "MB"), vis, mb)), ys):
            a = _lead_panel(cv, res, ev, pair, (408, y0, 553, y1), up, down, True)
            a.set_ylabel("jPECC asymmetry\n(-50 to +50ms)", fontsize=4.4)
    cv.text(480, 563, "Time from stim (s)", fontsize=5.6), cv.text(480, 815, "Time from move (s)", fontsize=5.6)

    # worked example from the example session
    n = res.neurons()
    if session is None or not len(n):
        return cv.finish()
    b = res.base(session)
    st, clu = res.spikes(session)
    events = b["stim"][b["kernel_trial"]]
    mats = []
    arrow = lambda p, q: cv.fig.add_artist(FancyArrowPatch((p[0] / cv.w, 1 - p[1] / cv.h), (q[0] / cv.w, 1 - q[1] / cv.h), transform=cv.fig.transFigure, color="#444444", lw=1.5, arrowstyle="-|>", mutation_scale=8))
    for r, t, box, pc_box, tr_y in (("VISp", 0.1, (53, 45, 116, 155), (200, 45, 236, 160), 102), ("MOs", 0.15, (56, 210, 138, 318), (200, 210, 236, 318), 268)):
        ids = n[(n["session"] == session) & (n["region"] == r)]["cluster"].to_numpy()
        a = cv.ax(box)
        if len(ids) == 0:
            no_data(a)
            continue
        q, _ = aligned_rates(st, clu, ids, events, (t - 0.01, t), 0.025, dt=0.01)
        base, _ = aligned_rates(st, clu, ids, events, (-0.2, 0.0), 0.025, dt=0.01)
        mat = (q[:, 0] / (base.mean((0, 1)) + 1)).astype(float)
        mats.append(mat)
        a.imshow(mat, aspect="auto", cmap=PARULA, vmin=0, vmax=10, interpolation="nearest"), a.set_xticks([0, mat.shape[1] - 1]), a.set_xticklabels(["1", str(mat.shape[1])]), a.set_yticks([0, len(mat) - 1]), a.set_yticklabels(["1", str(len(mat))])
        a.tick_params(labelsize=5, length=0), a.set_ylabel("Trials", labelpad=-6, fontsize=5.4), a.set_xlabel("Neurons", labelpad=-4, fontsize=5.4)
        pc = cv.ax(pc_box)
        scores = pc_scores(mat, min(10, mat.shape[1]))
        pc.imshow(scores, aspect="auto", cmap=BLUE_WHITE_RED, vmin=-20, vmax=20, interpolation="nearest"), pc.set_yticks([]), pc.set_xticks([0, scores.shape[1] - 1]), pc.set_xticklabels(["1", str(scores.shape[1])])
        pc.tick_params(labelsize=5, length=0), pc.set_xlabel("PC Dimensions", labelpad=-3, fontsize=5)
        cv.text((box[0] + box[2]) / 2 + 12, box[1] - 22, f"{r} firing rates,\nt = {t} s after stim onset", fontsize=5.6, color="#555555")
        arrow((box[2] + 30, tr_y), (pc_box[0] - 6, tr_y)), cv.text(box[2] + 44, tr_y - 11, "PCA", fontsize=6)
    cv.text(350, 170, "Canonical\ncorrelation\nanalysis", fontsize=6)
    arrow((232, 90), (318, 158)), arrow((240, 262), (318, 185)), arrow((345, 196), (345, 238))
    cv.text(346, 268, "A, B such that\nXA is maximally\ncorrelated with YB", fontsize=5.2, color="#555555")
    cv.text(262, 92, "Train set (X)", fontsize=5.2, color="#555555"), cv.text(262, 148, "Test set (X$_t$)", fontsize=5.2, color="#555555")
    cv.text(262, 262, "Train set (Y)", fontsize=5.2, color="#555555"), cv.text(262, 308, "Test set (Y$_t$)", fontsize=5.2, color="#555555")
    if len(mats) == 2:
        nd = min(10, *[m.shape[1] for m in mats])
        X, Y = pc_scores(mats[0], nd), pc_scores(mats[1], nd)
        n_tr = int(0.9 * len(X))
        order_ = np.random.default_rng(res.cfg.seed).permutation(len(X))
        tr_i, te_i = order_[:n_tr], order_[n_tr:]
        wa, wb = _cca_once(X[tr_i], Y[tr_i], 0.5, nd)
        xa, yb, xta, ytb = X[tr_i] @ wa, Y[tr_i] @ wb, X[te_i] @ wa, Y[te_i] @ wb
        scale = np.std(xa) or 1
        sc2 = np.std(yb) or 1
        _strip(cv, xa / scale, (428, 45, 436, 160), "XA", "Train set trials", n_tr)
        _strip(cv, yb / sc2, (428, 210, 436, 318), "YB", "Train set trials", n_tr)
        _strip(cv, xta / scale, (508, 45, 516, 160), "X$_t$A", "Test set trials", len(te_i))
        _strip(cv, ytb / sc2, (508, 210, 516, 318), "Y$_t$B", "Test set trials", len(te_i))
        r_tr = np.corrcoef(xa, yb)[0, 1]
        r_te = np.corrcoef(xta, ytb)[0, 1]
        cv.text(460, 187, f"r = {r_tr:.2f}", fontsize=7), cv.text(555, 182, f"r = {r_te:.2f}\n(one test fold)", fontsize=6)
        arrow((442, 118), (468, 172)), arrow((442, 265), (468, 200)), arrow((522, 105), (548, 155)), arrow((522, 275), (548, 208))
    return cv.finish()


def _strip(cv, values, box, title, label, n):
    a = cv.ax(box)
    a.imshow(values[:, None], aspect="auto", cmap=BLUE_WHITE_RED, vmin=-2, vmax=2, interpolation="nearest")
    a.set_xticks([]), a.set_yticks([0, n - 1]), a.set_yticklabels(["1", str(n)]), a.tick_params(labelsize=4.8, length=0)
    for s in a.spines.values():
        s.set_visible(False)
    a.set_ylabel(label, fontsize=5, labelpad=-5)
    cv.text((box[0] + box[2]) / 2, box[1] - 7, title, fontsize=5.4, color="#777777")


def _hist(ax, tab, mask, bins):
    miss = tab[:, 1] == 0
    go = tab[:, 1] == 1
    ax.hist(tab[mask & miss, 0], bins=bins, color=HIST_BLUE, label="Miss")
    ax.hist(tab[mask & go, 0], bins=bins, color=HIST_ORANGE, alpha=0.85, label="Go")
    ax.set_xlim(-6, 6), ax.set_ylabel("Trials"), ax.set_xticks([])


def ext9(res, chosen):
    from scipy.stats import ttest_ind, ttest_rel

    cv = Canvas(res.cfg, "ext9", "extended-data-fig9.png", scale=1.2)
    for s, x, y in (("a", 30, 31), ("b", 30, 163), ("c", 30, 243), ("d", 30, 321), ("e", 30, 395), ("f", 30, 468), ("g", 360, 163), ("h", 360, 312), ("i", 30, 557), ("j", 460, 557), ("k", 30, 693)):
        cv.letter(s, x, y)
    e = res.stage("engagement.npz")
    groups = ((("CP", "GPe", "SNr", "ACB", "LS", "SCs", "SCm", "MRN", "APN", "PAG", "ZI", "DG", "CA3", "CA1", "SUB", "POST"), (50, 38, 372, 76)),
              (("VISp", "VISl", "VISpm", "VISam", "VISrl", "VISa", "RSP", "ACA", "MOs", "PL", "ILA", "ORB", "MOp", "SSp"), (50, 100, 372, 150)),
              (("LGd", "LP", "LD", "POL", "MD", "VPL", "PO", "VPM", "RT", "MG", "OLF", "BLA"), (380, 52, 617, 125)))
    for regions, box, ylim, ticks in zip(*zip(*groups), ((0, 0.4), (-0.4, 0.2), (-0.4, 0.4)), ((0, 0.2, 0.4), (-0.4, -0.2, 0), (-0.4, -0.2, 0, 0.2, 0.4))):
        a = cv.ax(box)
        for i, r in enumerate(regions):
            v = e.get(f"anova|{r}")
            if v is not None:
                a.bar(i, v[2], 0.75, color=REGION_COLORS[r])
                a.plot([i, i], [v[2] - v[3], v[2] + v[3]], "k", lw=0.6)
                if v[1] < 0.05 / 42:
                    a.text(i, v[2] + np.sign(v[2]) * (v[3] + 0.02), "*", ha="center", va="center")
        a.axhline(0, color="#888888", lw=0.6), a.set_xlim(-0.6, len(regions) - 0.4), a.set_ylim(*ylim), a.set_yticks(ticks), a.set_yticklabels([f"{v:g}" for v in ticks]), a.tick_params(labelsize=5)
        rotated_region_labels(a, list(regions), 4.8)
        a.spines["bottom"].set_visible(False), a.tick_params(axis="x", length=0)
    for text, x, y, r_ in (("Basal ganglia", 92, 34, "ACB"), ("Midbrain", 186, 34, "MRN"), ("Hippocampus", 300, 34, "CA1"), ("Visual cortex", 88, 96, "VISam"), ("Frontal/somatomotor cortex", 262, 96, "SSp"), ("Thalamus", 420, 46, "PO"), ("Other", 596, 46, "OLF")):
        cv.text(x, y, text, color=REGION_COLORS[r_], fontsize=5)
    cv.text(200, 24, "Δ Pre-stimulus firing rate (normalized), Task-Passive", fontsize=5)
    cv.text(38, 95, "Δ normalized rate", rotation=90, fontsize=4.8)

    tab = e.get("trial_table", np.zeros((0, 7)))
    bins = np.linspace(-6, 6, 100)
    if len(tab):
        go = tab[:, 1] == 1
        z, prev, iti = tab[:, 0], tab[:, 4], tab[:, 5]
        med = np.nanmedian(iti)
        adj = z.copy()
        for v in (0, 1):
            adj[prev == v] -= np.nanmean(z[prev == v])
        cases = (("All matched contrast trials", np.ones(len(z), bool), tab, (54, 185, 358, 237)), ("Trials following rewards only", prev == 1, tab, (54, 258, 358, 305)),
                 ("All trials with previous reward effect removed", np.isfinite(prev), np.c_[adj, tab[:, 1:]], (54, 330, 358, 383)), ("Trials following short ITIs", iti <= med, tab, (54, 405, 358, 455)),
                 ("Trials following long ITIs", iti > med, tab, (54, 478, 358, 527)))
        for title, m, t_, box in cases:
            a = cv.ax(box)
            _hist(a, t_, m, bins)
            p = ttest_ind(t_[m & go, 0], t_[m & ~go, 0], nan_policy="omit").pvalue if (m & go).any() and (m & ~go).any() else np.nan
            cv.text(205, box[1] - 7, f"{title} (p={p:.3g})", fontsize=4.8)
        g = cv.ax((400, 180, 570, 275))
        pz = tab[:, 2]
        g.hist(pz[prev == 1], bins=np.linspace(-5, 5, 80), density=True, color="#e8807a", label="Previous trial rewarded")
        g.hist(pz[prev == 0], bins=np.linspace(-5, 5, 80), density=True, color="#555555", alpha=0.7, label="Previous trial no reward")
        g.legend(fontsize=4.6, loc="upper right"), g.set_xlim(-5, 5), g.set_xlabel("Pupil area (z-score)"), g.set_ylabel("Probability density")
        cv.text(485, 168, "Pupil area vs. previous trial reward", fontsize=5)

        ok = np.isfinite(prev) & np.isfinite(iti) & np.isfinite(tab[:, 2]) & np.isfinite(tab[:, 3])
        y = go[ok].astype(float)
        cols = {"ei": tab[ok, 0], "pupil": tab[ok, 2], "pupil2": tab[ok, 2] ** 2, "motion": tab[ok, 3], "motion2": tab[ok, 3] ** 2, "prev": prev[ok], "iti": iti[ok]}
        names = list(cols)
        if len(y) > 30 and 0 < y.mean() < 1:
            big = engagement.fit_glm(y, cols)
            small = engagement.fit_glm(y, {k: v for k, v in cols.items() if k != "ei"})
            pv = engagement.deviance_test(small, big, 1)
            spec = (("ei", None, (53, 561, 124, 665), "Engagement index\n(z-score)", tab[:, 0], f"p={pv:.1e}"), ("pupil", "pupil2", (135, 561, 207, 665), "Pupil area (z-score)", tab[:, 2], ""),
                    ("motion", "motion2", (218, 561, 290, 665), "Video motion energy\n(z-score)", tab[:, 3], ""), ("prev", None, (300, 561, 372, 665), "Previous trial reward?", prev, ""),
                    ("iti", None, (385, 561, 462, 665), "Inter-trial interval (s)", iti, ""))
            means = {k: v.mean() for k, v in cols.items()}
            for which, sq, box, lab, raw, note in spec:
                a = cv.ax(box)
                grid = np.linspace(np.nanpercentile(raw, 1), np.nanpercentile(raw, 99), 40)
                X = np.ones((40, len(names) + 1))
                for j, k in enumerate(names):
                    X[:, j + 1] = means[k]
                X[:, names.index(which) + 1] = grid
                if sq:
                    X[:, names.index(sq) + 1] = grid**2
                eta = X @ big.params
                se = np.sqrt(np.einsum("ij,jk,ik->i", X, big.cov_params(), X))
                inv = lambda v: 1 / (1 + np.exp(-v))
                a.fill_between(grid, inv(eta - 1.96 * se), inv(eta + 1.96 * se), color="#999999", lw=0)
                a.plot(grid, inv(eta), "k", lw=1.3)
                r_ = raw[ok]
                for lo, hi in zip(*(lambda q: (q[:-1], q[1:]))(np.nanquantile(r_, np.linspace(0, 1, 8)))):
                    m = (r_ >= lo) & (r_ <= hi)
                    if m.sum() > 2:
                        p = y[m].mean()
                        a.errorbar(r_[m].mean(), p, 1.96 * np.sqrt(p * (1 - p) / m.sum()), color="r", marker="o", ms=2, lw=0.8)
                a.set_ylim(0, 1), a.set_yticks([0, 0.5, 1] if which == "ei" else []), a.set_xlabel(lab, fontsize=4.8), a.tick_params(labelsize=4.8)
                if which in ("ei", "pupil", "motion"):
                    a.set_xlim(-5, 5), a.set_xticks([-5, 0, 5])
                elif which == "prev":
                    a.set_xlim(-0.1, 1.1), a.set_xticks([0, 1]), a.set_xticklabels(["No", "Yes"])
                else:
                    a.set_xlim(0, 7), a.set_xticks([0, 5])
                if which == "ei":
                    a.set_ylabel("P(Go)", fontsize=5.2)
                if note:
                    a.text(0.08, 0.92, note, transform=a.transAxes, fontsize=5.5)
            j = cv.ax((474, 562, 556, 666))
            for v, color in ((1, "#e0403c"), (0, "#4a60b0")):
                m = prev == v
                if m.sum() > 10 and 0 < tab[m, 1].mean() < 1:
                    _engagement_curve(j, tab[m, 0], tab[m, 1], color=color, point=color)
            j.set_ylim(0, 1), j.set_xlim(-5, 5), j.set_xlabel("Engagement index\n(z-score)", fontsize=4.8), j.set_yticks([]), j.tick_params(labelsize=4.8)
            j.text(0.05, 0.97, "Previous trial\nreward", color="#e0403c", transform=j.transAxes, va="top", fontsize=4.8), j.text(0.6, 0.12, "Previous\ntrial no reward", color="#4a60b0", transform=j.transAxes, fontsize=4.8)

    h = cv.ax((400, 318, 580, 400))
    m = cv.ax((400, 437, 580, 522))
    n = res.neurons()
    if len(n):
        s = EXT9_SESSION if EXT9_SESSION in res.sessions else n.groupby("session").size().idxmax()
        ses = Session(res.cfg.data, s)
        t, v = ses.series(("face", "motionEnergy"))
        ev = res.base(s)["stim"][res.base(s)["included"]]
        grid = np.arange(-1, 1, 0.025)
        mat = np.array([np.interp(e + grid, t, v) for e in ev])
        h.imshow(mat, aspect="auto", cmap=PARULA, extent=(-1, 1, len(ev), 0), vmin=0, vmax=np.percentile(mat, 99), interpolation="nearest"), h.set_xticks([]), h.set_ylabel("Trial")
        shade(m, grid, mat.mean(0), mat.std(0) / np.sqrt(len(mat)), "#e0403c")
        for a in (h, m):
            a.axvline(0, color="r", lw=0.6), a.axvline(-0.2, color="r", ls="--", lw=0.6)
    m.set_xlim(-1, 1), m.set_ylim(0, 5), m.set_xlabel("Time from stim onset (s)"), m.set_ylabel("Mean motion energy (a.u.)")
    cv.text(490, 311, "Video motion energy", fontsize=5)
    cv.slot((502, 408, 578, 486), "Video frame")

    vc = e.get("vector_corr", np.zeros((0, 5)))
    for i, (title, x0, lab) in enumerate((("Pupil area", 56, "[Large-small pupil]"), ("Video motion energy", 200, "[Large-small motion]"), ("Previous trial reward", 345, "[Reward-NoReward]"), ("Inter-trial interval", 483, "[Long-short ITI]"))):
        a = cv.ax((x0, 710, x0 + 94, 803))
        if len(vc):
            ok = np.isfinite(vc[:, 0]) & np.isfinite(vc[:, i + 1])
            a.plot(vc[ok, 0], vc[ok, i + 1], ".", color="#1f7fc4", ms=3)
            if ok.sum() > 2:
                a.text(0.6, 0.06, f"p={ttest_rel(vc[ok, 0], vc[ok, i + 1]).pvalue:.3f}", transform=a.transAxes, fontsize=5.5)
        a.plot([-1, 1], [-1, 1], "k--", lw=0.5), a.set_xlim(-1, 1), a.set_ylim(-1, 1), a.set_xticks([-1, 0, 1]), a.set_yticks([-1, 0, 1]), a.tick_params(labelsize=5)
        a.set_xlabel("corr([Task-Passive], [Go-Miss])", fontsize=5), a.set_ylabel(f"corr([Active-passive],\n{lab})", fontsize=5)
        cv.text(x0 + 47, 703, title, fontsize=5)
    return cv.finish()
