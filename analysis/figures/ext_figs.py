"""Extended Data Figures 1 to 9, in the paper's panel layout. Boxes are pixels of the paper's PNG."""
import numpy as np

from steinmetz import atlas, behavior, cccp, decoding, engagement, regression as rr, store
from steinmetz.config import ANATOMY_ORDER, DT, LATENCY_ORDER
from steinmetz.data import Session, subject_of
from steinmetz.rates import aligned_rates
from steinmetz.report import focality_interval
from steinmetz.results import NESTED, PSTH_T
from steinmetz.style import (BLUE_WHITE_RED, CONTRA, GO, IPSI, JPECC_CMAP, MISS, PASSIVE, PURPLE_HEAT, REGION_COLORS, Canvas, clean, no_data,
                             rotated_region_labels, shade)

from .common import region_name_columns, trial_means
from .main_figs import _decoding_curves, _ex, _engagement_curve, _jpecc_mean, jpecc_panel, rate_panel

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


def ext1(res):
    cv = Canvas(res.cfg, "ext1", "extended-data-fig1.png")
    b = res.stage("behavior.npz")
    subj = [str(s) for s in b.get("subjects", [])]
    subj = [s for s in PAPER_SUBJECTS if s in subj] + [s for s in subj if s not in PAPER_SUBJECTS]
    for i, s in enumerate(subj[:10]):
        x, y = BLOCKS[i]
        data = b[f"data|{s}"]
        cv.letter("abcdefghij"[i], x - 22, y - 16)
        n_ses = sum(subject_of(q) == s for q in res.sessions)
        cv.text(x + 85, y - 14, f"{s} ({len(data)} trials, {n_ses} session{'s' if n_ses > 1 else ''})", fontsize=6.3, fontweight="bold")
        _pedestal_block(cv, x, y, data, b[f"params|{s}"])
    cv.letter("k", 263, 487), cv.letter("l", 440, 487), cv.letter("m", 35, 645), cv.letter("n", 35, 725), cv.letter("o", 347, 645), cv.letter("p", 480, 645), cv.letter("q", 480, 735)

    k = cv.ax((289, 508, 390, 598))
    grid = np.linspace(-1, 1, 101)
    for s in subj:
        pl, pr, _ = behavior.model_probs(b[f"params|{s}"], np.maximum(-grid, 0), np.maximum(grid, 0))
        k.plot(100 * grid, pl, color="#3f7cc4", lw=0.9), k.plot(100 * grid, pr, color=CONTRA, lw=0.9)
    k.set_yticks([0, 0.5, 1]), k.set_xlabel("Contrast (%)"), k.set_ylabel("Probability choose left")
    cv.text(340, 490, "Combined subjects", fontweight="bold", fontsize=6.3)
    l = cv.ax((480, 510, 600, 598))
    for i, s in enumerate(subj):
        d = b[f"data|{s}"]
        m = ((d[:, 0] == 1) & (d[:, 1] == 0)) | ((d[:, 0] == 0) & (d[:, 1] == 1))
        n, c = m.sum(), (d[m, 3] == 1).sum()
        if n:
            lo, hi = behavior.binomial_ci(c, n)
            l.plot([i, i], [lo, hi], "k", lw=0.9), l.plot(i, c / n, "ko", ms=2)
    l.axhline(1 / 3, color="#888888", ls="--", lw=0.6), l.set_ylim(0, 1.02), l.set_yticks([0, 0.5, 1]), l.set_xlim(-0.5, max(len(subj) - 0.5, 1))
    l.set_xticks(range(len(subj))), l.set_xticklabels(subj, rotation=45, ha="right"), l.set_ylabel("Probability correct")
    cv.text(532, 498, "Performance at 100% contrast", fontweight="bold", fontsize=6.3)

    s = res.sessions[0]
    ses = Session(res.cfg.data, s)
    t, mm = behavior.wheel_mm(ses)
    on, off, kind = ses.moves()
    start = on[len(on) // 2]
    colors = {1: "#3f7cc4", 2: CONTRA}
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
    pos.set_ylabel("Position (mm)"), vel.set_ylabel("Velocity (mm/s)"), vel.set_xlabel("Time (s)")
    cv.text(195, 647, "Example segment of rotary encoder data", fontsize=7)
    o, p = cv.ax((383, 658, 462, 795)), cv.ax((510, 660, 610, 733))
    use = np.flatnonzero((kind == 1) | (kind == 2))[:60]
    w = np.arange(-0.1, 0.5, 0.01)
    for i in use:
        y = np.interp(on[i] + w, t, mm) - np.interp(on[i], t, mm)
        o.plot(w, y, color=colors[kind[i]], lw=0.7)
        zoom = (w >= -0.1) & (w <= 0.05)
        p.plot(w[zoom], y[zoom], color=colors[kind[i]], lw=0.5)
    o.axvline(0, color="#888888", ls="--", lw=0.6), p.axvline(0, color="#888888", ls="--", lw=0.6)
    o.set_xlabel("Time from detected onset (s)"), o.set_ylabel("Position (mm)"), p.set_ylabel("Position (mm)"), p.set_xticks([])
    q = cv.ax((510, 738, 610, 796))
    acc = np.array([store.load(res.cfg.results / "wheel" / f"{x}.npz")["acc"] for x in res.sessions])
    off_t = np.arange(-0.1, 0.0301, 0.005)
    shade(q, off_t, acc.mean(0), acc.std(0), "k")
    q.axhline(0.5, color="k", ls="--", lw=0.6), q.axvline(0, color="#888888", ls="--", lw=0.6), q.set_ylim(0, 1.02), q.set_xlim(-0.1, 0.03)
    q.set_xlabel("Time from detected onset (s)"), q.set_ylabel("Decoding\nperformance")
    return cv.finish()


def ext2(res):
    cv = Canvas(res.cfg, "ext2", "extended-data-fig2.png")
    cv.letter("a", 22, 18), cv.letter("b", 280, 14), cv.letter("c", 22, 208), cv.letter("d", 365, 208), cv.letter("e", 428, 208), cv.letter("f", 22, 609)
    cv.slot((22, 50, 282, 195), "Histology and atlas images")
    cv.slot((298, 50, 610, 170), "3D probe track fit (histology)")
    cv.slot((36, 225, 362, 575), "Electrophysiology landmarks for alignment")
    cv.slot((430, 225, 580, 575), "Area assignment with histology")
    cv.slot((180, 610, 490, 800), "Alignment cross-validation (histology)")
    a = cv.ax((371, 228, 414, 574))
    n = res.neurons()
    if len(n):
        s = n.groupby("session").size().idxmax()
        ses = Session(res.cfg.data, s)
        b = res.base(s)
        st, clu = res.spikes(s)
        pos, times = ses.raw("sparseNoise.positions"), ses.a("sparseNoise.times")
        probe = b["n_probe"][clu] == b["n_probe"][clu].min()
        d = b["n_depth"][clu[probe]]
        tt = st[probe]
        el, el_i = np.unique(pos[:, 0], return_inverse=True)
        az, az_i = np.unique(pos[:, 1], return_inverse=True)
        edges = np.arange(0, np.nanmax(d) + 80, 80)
        maps = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sp = tt[(d >= lo) & (d < hi)]
            m = np.zeros((len(el), len(az)))
            n_ev = np.zeros((len(el), len(az)))
            c = np.searchsorted(sp, times + 0.2) - np.searchsorted(sp, times)
            np.add.at(m, (el_i, az_i), c)
            np.add.at(n_ev, (el_i, az_i), 1)
            r = m / np.maximum(n_ev, 1)
            maps.append((r - r.min()) / max(r.max() - r.min(), 1e-9))
        a.imshow(np.concatenate(maps[::-1]), aspect="auto", cmap="viridis", interpolation="nearest")
    a.set_axis_off()
    return cv.finish()


def ext3(res):
    cv = Canvas(res.cfg, "ext3", "extended-data-fig3.png")
    cv.slot((30, 30, 600, 790), "DiI tracks in aligned histology (not in the dataset)")
    return cv.finish()


def _contrast_block(cv, res, s, c, x0, x1, title, region):
    if s is None:
        return
    b = res.base(s)
    st, clu = res.spikes(s)
    spikes = st[clu == c]
    cv.text((x0 + x1) / 2, 28, title, style="italic", fontsize=9, color=REGION_COLORS[region])
    cv.text(x0 + 15, 46, "Waveform", fontsize=6), cv.text(x0 + 80, 46, "Neuron\nlocation", fontsize=6)
    w = Session(res.cfg.data, s).raw("clusters.templateWaveforms")[c]
    ax = cv.ax((x0 - 10, 55, x0 + 45, 120))
    ax.plot(w[:, : min(8, w.shape[1])] + np.arange(min(8, w.shape[1])) * (np.abs(w).max() * 1.2), "k", lw=0.5)
    ax.set_axis_off()
    ins = cv.ax((x0 + 55, 55, x0 + 105, 120))
    xyz = Session(res.cfg.data, s).channel_ccf()[Session(res.cfg.data, s).peak_channel()[c]] / 25
    sl = atlas.volume()[int(np.clip(xyz[0], 0, 527))]
    ins.contour(atlas.outlines(sl), levels=[0.5], colors="k", linewidths=0.15)
    ins.plot(xyz[2], xyz[1], "r.", ms=3), ins.invert_yaxis(), ins.set_xlim(0, 228), ins.set_axis_off()
    ra = cv.ax((x0, 130, x1, 190))
    row = 0
    levels = (1.0, 0.5, 0.25, 0.0)
    shades = ("0.1", "0.35", "0.6", "0.75")
    for lv, col in zip(levels, shades):
        ev = b["stim"][b["included"] & (b["cr"] == lv)]
        for e in ev:
            sp = spikes[np.searchsorted(spikes, e - 0.05):np.searchsorted(spikes, e + 0.2)] - e
            ra.plot(sp, np.full(len(sp), row), ls="", marker="s", ms=0.5, mew=0, color=col)
            row += 1
    ra.axvline(0, color="k", ls=":", lw=0.5), ra.set_xlim(-0.05, 0.2), ra.set_ylim(row, -1), ra.set_axis_off()
    pa = cv.ax((x0, 210, x1, 270))
    for lv, col in zip(levels, shades):
        ev = b["stim"][b["included"] & (b["cr"] == lv)]
        if len(ev) > 1:
            r, t = aligned_rates(st, clu, np.array([c]), ev, (-0.05, 0.2), 0.03)
            shade(pa, t, r[:, :, 0].mean(0), r[:, :, 0].std(0) / np.sqrt(len(r)), col)
    pa.axvline(0, color="k", ls=":", lw=0.5), pa.set_xlim(-0.05, 0.2), pa.set_xticks([0, 0.1, 0.2]), pa.set_xticklabels(["0", "0.1", "0.2"])
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
    x, odd = x[:, keep_t], odd[:, keep_t]
    order = [("VISp", "VISl", "VISpm", "VISam", "VISrl", "VISa", "RSP"), ("ACA", "MOs", "PL", "ILA", "ORB", "MOp", "SSp"), ("DG", "CA3", "CA1", "POST", "SUB", "OLF", "BLA"),
             ("CP", "GPe", "SNr", "ACB", "LS"), ("LGd", "LP", "LD", "POL", "MD", "VPL", "PO", "VPM", "RT", "MG"), ("SCs", "SCm", "MRN", "APN", "PAG", "ZI")]
    columns = ((31, 45, 80), (93, 107, 143), (155, 168, 205), (218, 232, 268), (281, 294, 330), (344, 358, 393))
    cv.text(190, 310, "Stimulus onset aligned", fontsize=7)
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
                cv.text(sx - 6, y + h / 2, name, rotation=90, color=REGION_COLORS[name], fontsize=5.5)
                m = r == name
                q, o = x[m], odd[m]
                lat = np.nanargmax(o, 1)
                rows.append(q[np.argsort(lat)])
            y += h
        a = cv.ax((hx0, 323, hx1, 775))
        if rows:
            a.imshow(np.nan_to_num(np.concatenate(rows)), aspect="auto", cmap="gray_r", vmin=0.7, vmax=4, extent=(-0.05, 0.3, 1, 0), interpolation="nearest")
        a.axvline(0, color="#666666", ls=(0, (4, 3)), lw=0.6), a.set_yticks([]), a.set_xticks([0, 0.2]), a.set_xticklabels(["0", "0.2"])
        clean(a, False)
    bar = cv.ax((310, 303, 380, 308))
    bar.imshow(np.linspace(0, 1, 100)[None], cmap="gray_r", aspect="auto"), bar.set_axis_off()
    cv.text(340, 297, "Average normalized firing rate", fontsize=5.5, color="#666666")

    stacks = (("ext4_stim", (450, 40, 530, 570), PSTH_T, 0.3, "Time from\nstim (s)"), ("ext4_move", (530, 40, 605, 570), np.arange(-0.2, 0.1, DT)[:60], 0.1, "Time from\nmove (s)"),
              ("ext4_reward", (605, 40, 680, 570), np.arange(-0.1, 0.2, DT)[:60], 0.2, "Time from\nreward (s)"))
    for key, box, t, _, lab in stacks:
        a = cv.ax(box)
        a.set_xticks({"ext4_stim": [0, 0.1, 0.2], "ext4_move": [-0.2, -0.1, 0], "ext4_reward": [0, 0.1]}[key])
        q, reg = res.normalized(key, "responsive")
        for i, name in enumerate(ANATOMY_ORDER):
            d = q[reg == name] if len(q) else np.zeros((0, len(t)))
            if len(d):
                n = min(d.shape[1], len(t))
                shade(a, t[:n], np.nanmean(d, 0)[:n] * 0.5 - i * 1.0, np.nanstd(d, 0)[:n] * 0.5 / np.sqrt(len(d)), REGION_COLORS[name], lw=0.9)
        a.axvline(0, color="#666666", ls=(0, (4, 3)), lw=0.6), a.set_ylim(-42, 3), a.set_yticks([]), a.spines["left"].set_visible(False), a.set_xlabel(lab)
        a.set_xticklabels([f"{v:g}" for v in a.get_xticks()])
    for i, name in enumerate(ANATOMY_ORDER):
        cv.text(447, 40 + 530 * (3 + i) / 45, name, color=REGION_COLORS[name], fontsize=5, ha="right")

    f = cv.ax((480, 620, 640, 775))
    t = res.regression()
    if len(t):
        for i, col in enumerate(("action", "vision_contra", "choice")):
            est, lo, hi = focality_interval(t, col, res.cfg.seed)
            f.plot([i, i], [lo, hi], "k", lw=1.2), f.plot(i, est, "ko", ms=3)
    f.axhline(1 / 42, color="#888888", ls="--", lw=0.7)
    f.set_xlim(-0.3, 2.4), f.set_xticks([0, 1, 2]), f.set_xticklabels(["Action", "Contra Vision", "Choice"]), f.set_ylabel("Focality index")
    return cv.finish()


def ext5(res, chosen):
    cv = Canvas(res.cfg, "ext5", "extended-data-fig5.png")
    for s, x, y in (("a", 18, 20), ("b", 18, 124), ("c", 422, 20), ("d", 18, 200), ("e", 18, 398), ("f", 360, 398), ("g", 20, 678), ("h", 398, 670)):
        cv.letter(s, x, y)
    cv.slot((30, 125, 415, 195), "Model cartoons")
    s, c = _ex(chosen, "fig4a_mrn")
    a = cv.ax((52, 20, 400, 100))
    cc = cv.ax((470, 35, 600, 160))
    if s:
        b = res.base(s)
        P, rows, trnum, *_ = rr.build_design(b)
        d = store.load(res.cfg.results / "regression" / "data" / f"{s}.npz")
        i = int(np.flatnonzero(d["ids"] == c)[0])
        K = res.kernels(s)["K"][:, i]
        m = trnum < 5
        parts = {"Vision kernels": (P[:, :540] @ K[:540], "#2a78c4"), "Action kernel": (P[:, 540:595] @ K[540:595], "#f0b429"), "Choice kernel": (P[:, 595:] @ K[595:], "#8e44ad")}
        x = np.arange(m.sum()) * DT
        add = d["means"][i]
        a.plot(x, d["F"][m, i] + add, color="#6fb543", lw=0.7, label="Smoothed spiking")
        a.plot(x, (P @ K)[m] + add, "k", lw=0.8, label="Full model")
        for k, (v, col) in parts.items():
            a.plot(x, v[m], color=col, lw=0.8, label=k)
        a.legend(fontsize=5, loc="upper right", ncol=3), a.set_ylabel("Firing rate (sp/s)"), a.set_xlabel("Time (s)")
        act, cho = K[540:595], K[595:]
        t = -0.25 + np.arange(55) * DT
        cc.plot(t, act + cho, "#3f7cc4", label="Left response"), cc.plot(t, act - cho, CONTRA, label="Right response")
        cc.plot(t, act, "k", label="Action kernel"), cc.plot(t, cho, "#bbbbbb", label="Choice kernel")
        cc.legend(fontsize=5), cc.set_xlabel("Time from move onset (s)"), cc.set_ylabel("Response (sp/s)")

        X = P[np.isin(trnum, [0, 1, 2])]
        tox = cv.ax((76, 212, 240, 380))
        tox.imshow(X.toarray().T, aspect="auto", cmap=BLUE_WHITE_RED, vmin=-1, vmax=1, interpolation="nearest"), tox.set_xticks([]), tox.set_yticks([])
        cos = cv.ax((252, 212, 416, 380))
        cos.imshow(rr.cosine_design(P, trnum)[np.isin(trnum, [0, 1, 2])].T, aspect="auto", cmap=BLUE_WHITE_RED, vmin=-2, vmax=2, interpolation="nearest")
        cos.set_xticks([]), cos.set_yticks([])
        B = np.load(res.cfg.results / "regression" / "real" / "basis_full.npy")
        rrm = cv.ax((446, 212, 610, 380))
        z = (X @ B)[:, :50].T
        rrm.imshow(z, aspect="auto", cmap=BLUE_WHITE_RED, vmin=-np.nanstd(z) * 2, vmax=np.nanstd(z) * 2, interpolation="nearest"), rrm.set_xticks([]), rrm.set_yticks([])
        cv.text(158, 204, "Toeplitz Model: X", fontsize=6), cv.text(334, 204, "Cosine Model: X", fontsize=6), cv.text(528, 204, "Reduced Rank Model: X*b", fontsize=6)

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
        for (x0, x1), lim in (((78, 195), (-30, 80)), ((232, 348), (-5, 20))):
            ax = cv.ax((x0, y0, x1, y0 + 115))
            if cmp_:
                ax.hist2d(cmp_["reduced_test"], cmp_[f"{name}_test"], bins=60, range=[lim, lim], cmap="viridis", cmin=1)
            ax.plot(lim, lim, "r--", lw=0.6), ax.axhline(0, color="r", ls="--", lw=0.6), ax.axvline(0, color="r", ls="--", lw=0.6)
            ax.set_xlim(*lim), ax.set_ylim(*lim)
            if x0 == 78:
                ax.set_ylabel(f"% C.V. var. expl.,\n{name.capitalize()} model")
    cv.text(140, 672, "% C.V. Var. Expl., Reduced Rank model", fontsize=6)
    for name, label, (y0, y1) in (("reduced", "Reduced rank model", (410, 477)), ("toeplitz", "Toeplitz model", (490, 556)), ("cosine", "Cosine model", (570, 635))):
        ax = cv.ax((400, y0, 583, y1))
        if cmp_:
            tr_, te = cmp_[f"{name}_train"][included], cmp_[f"{name}_test"][included]
            over = ((tr_ - te) / np.where(tr_ > 0, tr_, np.nan))
            over = over[np.isfinite(over)]
            ax.hist(np.clip(over, -1, 1), bins=np.linspace(-1, 1, 41), color="#5aa5dc", edgecolor="k", lw=0.3)
            ax.axvline(np.median(over), color="r", lw=0.8)
            ax.set_title(f"{label}: median={np.median(over):.2f}", fontsize=6, pad=1)
        ax.axvline(0, color="k", ls="--", lw=0.7), ax.set_xlim(-1, 1), ax.set_ylabel("Number of\nneurons")
    cv.text(490, 648, "Proportion of overfit explained variance", fontsize=6)

    cur = _decoding_curves(res)
    panels = ((78, 113, "vision_contra", "stim"), (132, 167, "vision_ipsi", "stim"), (195, 233, "action", "move"), (257, 290, "choice", "move"))
    heats = {}
    for _, _, key, align in panels:
        heats[key] = np.full((42, len(decoding.OFFSETS[align])), np.nan)
        for i, r in enumerate(LATENCY_ORDER):
            v = cur.get(key, {}).get(r, [])
            if v:
                heats[key][i] = np.nanmean(v, 0)
    finite = np.concatenate([h[np.isfinite(h)] for h in heats.values()])
    top = np.percentile(np.abs(finite), 98) if len(finite) else 1.0  # the paper's projection units are not stated
    for x0, x1, key, align in panels:
        ax = cv.ax((x0, 680, x1, 795))
        off = decoding.OFFSETS[align]
        ax.imshow(np.nan_to_num(heats[key]), aspect="auto", cmap=PURPLE_HEAT, vmin=top / 3, vmax=top, extent=(off[0], off[-1], 42, 0), interpolation="nearest")
        ax.axvline(0, color="#666666", ls=(0, (3, 2)), lw=0.6), ax.set_yticks([])
        ax.set_xticks([0, 0.2] if align == "stim" else [-0.1, 0]), ax.set_xticklabels(["0", "0.2"] if align == "stim" else ["-0.1", "0"])
        clean(ax, False)
    cv.text(180, 812, "Time from stim (s)   /   Time from move (s)", fontsize=6)
    bar = cv.ax((332, 710, 337, 755))
    bar.imshow(np.linspace(0.15, 0.05, 100)[:, None], cmap=PURPLE_HEAT, aspect="auto", vmin=0.05, vmax=0.15), bar.set_axis_off()

    h = cv.ax((445, 670, 565, 795))
    t = res.regression()
    if len(t):
        bins = np.arange(-5, 20.5, 0.5)
        h.hist(100 * t["choice"].dropna(), bins=bins, color="#5aa5dc", label="Original")
        h.hist(100 * res.shuffle_ve(), bins=bins, color="#c8704e", alpha=0.8, label="Shuffle")
        h.legend(fontsize=5.5)
    h.axvline(2, color="k", ls="--", lw=0.8), h.set_xlim(-5, 20), h.set_xlabel("C.V. % var. expl."), h.set_ylabel("# neurons")
    return cv.finish()


EXT6_A = (("ACA", "MOs", "PL", "ILA", "ORB", "MOp", "SSp"), ("CP", "GPe", "SNr", "ACB", "LS", "SCs", "SCm"), ("MRN", "APN", "PAG", "VISp", "VISl", "VISpm", "VISam"),
          ("VISrl", "VISa", "LGd", "LP", "LD", "POL", "MD"), ("VPL", "PO", "VPM", "RT", "MG", "DG", "CA3"), ("CA1", "SUB", "POST", "RSP", "ZI", "OLF", "BLA"))
EXT6_B = (("SCs", "SCm", "MRN", "APN", "PAG", "ZI", "PO", "VPM"), ("CP", "GPe", "SNr", "ACB", "LS", "RT", "MG", "BLA"), ("DG", "CA3", "CA1", "SUB", "POST", "POL", "MD", "OLF"),
          ("ACA", "MOs", "PL", "ILA", "ORB", "MOp", "SSp", "VPL", "RSP"), ("VISp", "VISl", "VISpm", "VISam", "VISrl", "VISa", "LGd", "LP", "LD"))


def ext6(res):
    cv = Canvas(res.cfg, "ext6", "extended-data-fig6.png", scale=1.15)
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
    cv = Canvas(res.cfg, "ext7", "extended-data-fig7.png")
    cv.letter("a", 18, 22), cv.letter("b", 385, 22), cv.letter("c", 474, 22), cv.letter("d", 546, 22), cv.letter("e", 380, 490), cv.letter("f", 380, 652)
    table = _cccp_table(res)
    groups = ((("cp", "stim", 28, 65), ("cp", "move", 69, 105), ("dp", "stim", 112, 148), ("dp", "move", 152, 188)), (("cp", "stim", 204, 240), ("cp", "move", 245, 281), ("dp", "stim", 288, 324), ("dp", "move", 328, 365)))
    left = ANATOMY_ORDER[::2]
    right = ANATOMY_ORDER[1::2]
    for regions, cols in zip((left, right), groups):
        for i, r in enumerate(regions):
            y = 55 + i * 35.4
            for stat, ev, x0, x1 in cols:
                a = cv.ax((x0, y, x1, y + 22))
                v = table.get(stat, {}).get(ev, {}).get(r)
                if v is not None and len(v):
                    p = v.mean(0)
                    shade(a, _centers(ev)[: len(p)], 100 * p, 100 * np.sqrt(p * (1 - p) / len(v)), REGION_COLORS[r])
                a.axhline(5, color="k", ls="--", lw=0.4), a.set_xticks([]), a.set_yticks([]), clean(a, False, True)
                a.set_ylim(0, 15 if stat == "cp" else 50), a.axvline(0, color="#888888", ls=(0, (3, 2)), lw=0.4)
            cv.text((cols[2][2] + cols[3][3]) / 2, y - 4, r, color=REGION_COLORS[r], fontsize=6)
    for x, s in ((64, "Choice prob."), (148, "Detect prob."), (241, "Choice prob."), (326, "Detect prob.")):
        cv.text(x, 26, s, fontsize=5.8)

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
                    a.text(100 * hi + 1, i, "*", va="center", fontsize=6, clip_on=True)
        a.axvline(5, color="k", ls="--", lw=0.5), a.set_ylim(41.5, -0.5), a.set_xlim(0, xmax), a.set_yticks([]), a.xaxis.tick_top()
        a.spines["top"].set_visible(True), a.spines["bottom"].set_visible(False)
        cv.text((box[0] + box[2]) / 2, 35, title, fontsize=5.5)
    for i, r in enumerate(ANATOMY_ORDER):
        cv.text(406, 73 + 407 * (i + 0.5) / 42, r, color=REGION_COLORS[r], fontsize=5, ha="right")

    e = cv.ax((408, 490, 585, 627))
    for r, col in (("MOs", REGION_COLORS["MOs"]), ("SNr", REGION_COLORS["SNr"]), ("SCm", REGION_COLORS["SCm"]), ("MRN", REGION_COLORS["MRN"])):
        v = table.get("cp", {}).get("move", {}).get(r)
        if v is not None and len(v):
            p = v.mean(0)
            shade(e, _centers("move")[: len(p)], 100 * p, 196 * np.sqrt(p * (1 - p) / len(v)), col, alpha=0.3, label=r)
    e.axhline(5, color="#888888", ls="--", lw=0.6), e.axvline(0, color="#888888", ls="--", lw=0.6), e.set_xlim(-0.3, 0.25), e.set_ylim(0, 30)
    e.set_xlabel("Time from move onset (s)"), e.set_ylabel("% neurons with significant CP")
    if e.get_legend_handles_labels()[0]:
        e.legend(fontsize=5.5)

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
    f.axhline(0, color="#888888", ls="--", lw=0.6), f.axvline(0, color="#888888", ls="--", lw=0.6)
    f.set_xlabel("Go-Miss Firing rate (sp/s)"), f.set_ylabel("Pre-stimulus Detect Probability - 0.5")
    return cv.finish()


def ext8(res):
    from steinmetz.jpecc import asymmetry, pc_scores

    cv = Canvas(res.cfg, "ext8", "extended-data-fig8.png", scale=1.1)
    for s, x, y in (("a", 24, 22), ("b", 24, 354), ("c", 24, 585), ("d", 368, 347), ("e", 368, 582)):
        cv.letter(s, x, y)
    pairs = ((("VIS", "Frontal"), "VIS vs Frontal"), (("VIS", "MB"), "VIS vs MB"), (("MB", "Frontal"), "MB vs Frontal"))
    example = None
    for s in res.sessions:
        z = store.load(res.cfg.results / "jpecc" / f"{s}.npz")
        if len(z) >= 12:
            example = z
            break
    for row, (ev, y0, y1) in enumerate((("stim", 378, 448), ("move", 483, 553))):
        for (pair, title), (x0, x1) in zip(pairs, ((52, 126), (162, 234), (270, 344))):
            a = cv.ax((x0, y0, x1, y1))
            key = f"r|{ev}|{pair[0]}|{pair[1]}"
            if example is not None and key in example:
                r = np.array(example[key], float)
                r[example[key.replace("r|", "p|")] > 0.05] = np.nan
                cm = JPECC_CMAP.copy()
                cm.set_bad("#888888")
                jpecc_panel(a, r, ev)
                a.images[0].set_cmap(cm)
            else:
                no_data(a)
            a.set_xticks([]), a.set_yticks([])
            if row == 0:
                cv.text((x0 + x1) / 2, y0 - 9, title, fontsize=7)
    cv.text(195, 354, "Example session", fontweight="bold", fontsize=9)
    for row, (ev, y0, y1) in enumerate((("stim", 617, 690), ("move", 723, 797))):
        for (pair, _), (x0, x1) in zip(pairs, ((46, 120), (156, 229), (266, 340))):
            a = cv.ax((x0, y0, x1, y1))
            arr = _jpecc_mean(res, ev, *pair)
            if len(arr):
                jpecc_panel(a, arr.mean(0), ev)
                if row == 0:
                    cv.text((x0 + x1) / 2, y0 - 15, f"n={len(arr)} sessions", fontsize=7)
            else:
                no_data(a)
            a.set_xticks([]), a.set_yticks([])
    rows = (("stim", ((352, 412), (422, 482), (495, 555))), ("move", ((585, 645), (660, 720), (735, 795))))
    labels = (("MB", "Frontal"), ("VIS", "Frontal"), ("VIS", "MB"))
    for ev, ys in rows:
        for (a_name, c_name), (y0, y1) in zip((("MB", "Frontal"), ("VIS", "Frontal"), ("VIS", "MB")), ys):
            a = cv.ax((408, y0, 553, y1))
            arr = _jpecc_mean(res, ev, a_name, c_name)
            w = (-0.1, 0.3) if ev == "stim" else (-0.3, 0.1)
            if len(arr):
                idx = np.array([asymmetry(np.array(r, float)) for r in arr])
                t = w[0] + 0.005 + np.arange(idx.shape[1]) * 0.01
                n = np.isfinite(idx).sum(0)
                m, se = np.nanmean(idx, 0), np.nanstd(idx, 0) / np.sqrt(np.maximum(n, 1))
                shade(a, t, m, 2 * se, "k", alpha=0.5)
            a.axhline(0, color="#888888", ls="--", lw=0.5), a.set_xlim(*((0.05, 0.25) if ev == "stim" else (-0.2, 0.05))), a.set_ylim(-0.3, 0.35)
            a.set_ylabel("Asymmetry", fontsize=5), a.tick_params(labelsize=5)
            cv.text(525, y0 + 8, f"{a_name} leads", fontsize=6), cv.text(440, y1 - 8, f"{c_name} leads", fontsize=6)

    n = res.neurons()
    if example is not None and len(n):
        s = n.groupby("session").size().idxmax()
        b = res.base(s)
        st, clu = res.spikes(s)
        panels = (("VISp", 0.1, (53, 45, 116, 155)), ("MOs", 0.15, (56, 210, 138, 318)))
        for r, t, box in panels:
            ids = n[(n["session"] == s) & (n["region"] == r)]["cluster"].to_numpy()
            a = cv.ax(box)
            if len(ids) == 0:
                no_data(a)
                continue
            q, tb = aligned_rates(st, clu, ids, b["stim"][b["kernel_trial"]], (t - 0.01, t), 0.025, dt=0.01)
            mat = q[:, 0]
            a.imshow(mat, aspect="auto", cmap="viridis", interpolation="nearest"), a.set_xticks([]), a.set_yticks([])
            a.set_ylabel("Trials")
            pc = cv.ax((200, box[1], 236, box[3]))
            pc.imshow(pc_scores(mat.astype(float), min(10, mat.shape[1])), aspect="auto", cmap=BLUE_WHITE_RED, interpolation="nearest"), pc.set_axis_off()
            cv.text((box[0] + box[2]) / 2, box[1] - 22, f"{r} firing rates, t = {t} s after stim onset", fontsize=6.3)
        cv.text(165, 90, "PCA →", fontsize=7), cv.text(165, 265, "PCA →", fontsize=7)
        cv.text(345, 170, "Canonical\ncorrelation\nanalysis", fontsize=7)
    return cv.finish()


def _hist(ax, tab, mask, bins):
    miss = tab[:, 1] == 0
    go = tab[:, 1] == 1
    ax.hist(tab[mask & miss, 0], bins=bins, color="#5aa5dc", label="Miss")
    ax.hist(tab[mask & go, 0], bins=bins, color="#d9805e", alpha=0.85, label="Go")
    ax.set_xlim(-6, 6), ax.set_ylabel("Trials"), ax.set_xticks([])


def ext9(res, chosen):
    from scipy.stats import ttest_ind, ttest_rel

    cv = Canvas(res.cfg, "ext9", "extended-data-fig9.png")
    for s, x, y in (("a", 30, 31), ("b", 30, 163), ("c", 30, 243), ("d", 30, 321), ("e", 30, 395), ("f", 30, 468), ("g", 360, 163), ("h", 360, 312), ("i", 30, 557), ("j", 460, 557), ("k", 30, 693)):
        cv.letter(s, x, y)
    e = res.stage("engagement.npz")
    groups = ((("CP", "GPe", "SNr", "ACB", "LS", "SCs", "SCm", "MRN", "APN", "PAG", "ZI", "DG", "CA3", "CA1", "SUB", "POST"), (50, 38, 372, 76)),
              (("VISp", "VISl", "VISpm", "VISam", "VISrl", "VISa", "RSP", "ACA", "MOs", "PL", "ILA", "ORB", "MOp", "SSp"), (50, 100, 372, 150)),
              (("LGd", "LP", "LD", "POL", "MD", "VPL", "PO", "VPM", "RT", "MG", "OLF", "BLA"), (380, 52, 617, 125)))
    for regions, box in groups:
        a = cv.ax(box)
        for i, r in enumerate(regions):
            v = e.get(f"anova|{r}")
            if v is not None:
                a.bar(i, v[2], 0.75, color=REGION_COLORS[r])
                a.plot([i, i], [v[2] - v[3], v[2] + v[3]], "k", lw=0.6)
                if v[1] < 0.05 / 42:
                    a.text(i, v[2] + np.sign(v[2]) * (v[3] + 0.02), "*", ha="center", va="center")
        a.axhline(0, color="#888888", lw=0.6), a.set_xlim(-0.6, len(regions) - 0.4), a.set_ylim(-0.4, 0.4), a.set_yticks([-0.4, 0, 0.4])
        rotated_region_labels(a, list(regions), 5.5)
    cv.text(200, 31, "Δ Pre-stimulus firing rate (normalized), Task-Passive", fontsize=6)

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
            cv.text(205, box[1] - 7, f"{title} (p={p:.3g})", fontsize=6)
        g = cv.ax((400, 180, 570, 275))
        pz = tab[:, 2]
        g.hist(pz[prev == 1], bins=60, density=True, color="#e8807a", label="Previous trial rewarded")
        g.hist(pz[prev == 0], bins=60, density=True, color="#555555", alpha=0.7, label="Previous trial no reward")
        g.legend(fontsize=5), g.set_xlabel("Pupil area (z-score)"), g.set_ylabel("Probability density")
        cv.text(485, 168, "Pupil area vs. previous trial reward", fontsize=6)

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
                a.set_ylim(0, 1), a.set_yticks([0, 0.5, 1] if which == "ei" else []), a.set_xlabel(lab, fontsize=5.8), a.tick_params(labelsize=5)
                if note:
                    a.text(0.08, 0.92, note, transform=a.transAxes, fontsize=5.5)
            j = cv.ax((474, 562, 556, 666))
            for v, color in ((1, "#e0403c"), (0, "#4a60b0")):
                m = prev == v
                if m.sum() > 10 and 0 < tab[m, 1].mean() < 1:
                    _engagement_curve(j, tab[m, 0], tab[m, 1], color=color, point=color)
            j.set_ylim(0, 1), j.set_xlim(-5, 5), j.set_xlabel("Engagement index\n(z-score)", fontsize=5.8)

    h = cv.ax((400, 318, 580, 400))
    m = cv.ax((400, 437, 580, 522))
    n = res.neurons()
    if len(n):
        s = n.groupby("session").size().idxmax()
        ses = Session(res.cfg.data, s)
        t, v = ses.series(("face", "motionEnergy"))
        ev = res.base(s)["stim"][res.base(s)["included"]]
        grid = np.arange(-1, 1, 0.025)
        mat = np.array([np.interp(e + grid, t, v) for e in ev])
        h.imshow(mat, aspect="auto", cmap="viridis", extent=(-1, 1, len(ev), 0), interpolation="nearest"), h.set_xticks([]), h.set_ylabel("Trial")
        shade(m, grid, mat.mean(0), mat.std(0) / np.sqrt(len(mat)), "#e0403c")
        for a in (h, m):
            a.axvline(0, color="r", lw=0.6), a.axvline(-0.2, color="r", ls="--", lw=0.6)
    m.set_xlim(-1, 1), m.set_xlabel("Time from stim onset (s)"), m.set_ylabel("Mean motion energy (a.u.)")
    cv.text(490, 311, "Video motion energy", fontsize=6)
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
        cv.text(x0 + 47, 703, title, fontsize=6)
    return cv.finish()
