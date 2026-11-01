"""Fake sessions written in the real file format, for tests only. Every figure made from them is watermarked."""
import csv
from pathlib import Path

import numpy as np

from .config import ANATOMY_ORDER

VISUAL = {"VISp", "VISl", "VISpm", "VISam", "VISrl", "VISa", "SCs", "LGd", "LP", "LD"}
CHOICE = {"MOs", "PL", "MOp", "CP", "SNr", "SCm", "MRN", "ZI", "LP"}
SUPPRESSED = {"VISp", "VISl", "VISpm", "VISam", "VISrl", "VISa", "RSP", "ACA", "MOs", "PL", "ILA", "ORB", "MOp", "SSp", "LGd", "LP", "LD"}


def _task(rng, n):
    levels = np.array([0, 0.25, 0.5, 1.0])
    cl, cr = rng.choice(levels, n), rng.choice(levels, n)
    stim = 60 + np.cumsum(rng.uniform(3.2, 4.2, n))
    engaged = rng.random(n) < 0.72
    has_stim = (cl > 0) | (cr > 0)
    go = engaged & has_stim & (rng.random(n) < 0.9)
    right = np.where(cr != cl, cr > cl, rng.random(n) < 0.5)
    wrong = rng.random(n) < 0.08
    right = right ^ wrong
    ft = rng.uniform(0.15, 0.38, n)
    cue = stim + rng.uniform(0.5, 1.2, n)
    fb_time = np.where(go, np.maximum(cue, stim + ft + 0.4), cue + 1.5)
    choice = np.where(go, np.where(right, -1.0, 1.0), 0.0)
    correct = np.where(cr != cl, (cr > cl) == right, True)
    fb = np.where(go, np.where(correct, 1.0, -1.0), np.where(has_stim, -1.0, 1.0))
    included = np.r_[np.ones(n - 12, bool), np.zeros(12, bool)]
    late = ~go & (rng.random(n) < 0.3)  # a late small movement, so has_none is populated
    return dict(cl=cl, cr=cr, stim=stim, engaged=engaged, go=go, right=right, ft=ft, cue=cue, fb_time=fb_time, choice=choice,
                fb=fb, included=included, late=late)


def _tuning(rng, regions):
    n = len(regions)
    u = dict(base=rng.uniform(1.5, 12, n), vis=np.zeros(n), act=rng.uniform(0, 22, n) * (rng.random(n) < 0.6), cho=np.zeros(n),
             eng=np.zeros(n), ctx=np.zeros(n))
    for i, r in enumerate(regions):
        if r in VISUAL and rng.random() < 0.55:
            u["vis"][i] = rng.uniform(8, 35)
        if r in CHOICE and rng.random() < 0.25:
            u["cho"][i] = rng.uniform(8, 25) * (1 if r in {"SNr", "SCm", "MRN", "ZI"} or rng.random() < 0.6 else -1)
        sign = -1 if r in SUPPRESSED else 1
        u["eng"][i] = sign * rng.uniform(0, 1.2)
        u["ctx"][i] = sign * rng.uniform(0, 1.5)
    return u


def _spikes(rng, u, tk, passive_t, passive_cr, t_end):
    """Rates on a trial grid, background elsewhere. Contra stimuli drive `vis`; movement drives `act` and `cho` (positive prefers contra)."""
    dt = 0.005
    grid = np.arange(-0.5, 1.5, dt)
    all_t, all_c = [], []
    b = u["base"]
    n = len(b)
    for s, cr, eng, go, right, ft in zip(tk["stim"], tk["cr"], tk["engaged"], tk["go"], tk["right"], tk["ft"]):
        r = np.tile(b + u["eng"] * (1.0 if eng else 0.0), (len(grid), 1))
        if cr > 0:
            r += np.exp(-(((grid - 0.09) / 0.03) ** 2))[:, None] * (u["vis"] * cr**0.5)[None]
        if go:
            ramp = np.exp(-(((grid - ft - 0.05) / 0.08) ** 2))[:, None]
            r += ramp * u["act"][None]
            r += ramp * (u["cho"] * (1 if right else -1))[None]
        k = rng.poisson(np.maximum(r, 0.05) * dt)
        i, c = np.nonzero(k)
        i, c = np.repeat(i, k[i, c]), np.repeat(c, k[i, c])
        all_t.append(s + grid[i] + rng.random(len(i)) * dt)
        all_c.append(c)
    for s, cr in zip(passive_t, passive_cr):
        g = grid[(grid > -0.4) & (grid < 0.6)]
        r = np.tile(np.maximum(b - u["ctx"], 0.05), (len(g), 1))
        if cr > 0:
            r += 0.7 * np.exp(-(((g - 0.09) / 0.03) ** 2))[:, None] * (u["vis"] * cr**0.5)[None]
        k = rng.poisson(r * dt)
        i, c = np.nonzero(k)
        i, c = np.repeat(i, k[i, c]), np.repeat(c, k[i, c])
        all_t.append(s + g[i] + rng.random(len(i)) * dt)
        all_c.append(c)
    # background outside those windows, at the resting rate
    counts = rng.poisson(b * t_end * 0.45)
    bg_t = rng.random(counts.sum()) * t_end
    bg_c = np.repeat(np.arange(n), counts)
    busy = np.sort(np.r_[tk["stim"] - 0.5, passive_t - 0.4])
    idx = np.searchsorted(busy, bg_t) - 1
    keep = (idx < 0) | (bg_t - busy[np.maximum(idx, 0)] > 2.0)
    all_t.append(bg_t[keep])
    all_c.append(bg_c[keep])
    t, c = np.concatenate(all_t), np.concatenate(all_c)
    o = np.argsort(t)
    return t[o], c[o]


def write_session(folder, seed, n_trials=240, per_region=15):
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save = lambda name, a: np.save(folder / f"{name}.npy", np.asarray(a))
    tk = _task(rng, n_trials)
    t_end_task = tk["fb_time"][-1] + 5
    pc = np.tile([0.25, 0.5, 1.0, 0, 0, 0], 16)
    pl = np.tile([0, 0, 0, 0.25, 0.5, 1.0], 16)
    o = rng.permutation(len(pc))
    pc, pl = pc[o], pl[o]
    pt = t_end_task + 30 + np.arange(len(pc)) * 1.5
    t_end = pt[-1] + 60

    regions = [r for r in ANATOMY_ORDER for _ in range(per_region)] + ["root"] * 30
    n = len(regions)
    u = _tuning(rng, regions)
    st, sc = _spikes(rng, u, tk, pt, pc, t_end)

    rows = [{"ccf_ap": 4000 + 4 * i + rng.normal() * 5, "ccf_dv": 1000 + 8 * i, "ccf_lr": 4000 + i, "allen_ontology": regions[i]} for i in range(n)]
    with open(folder / "channels.brainLocation.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    save("channels.probe", (np.arange(n) >= n // 2).astype(float)[:, None])
    save("channels.site", np.arange(n)[:, None])
    save("channels.sitePositions", np.c_[np.zeros(n), np.arange(n) * 20.0])
    ann = np.full(n, 2.0)
    ann[rng.choice(n, 10, replace=False)] = 1
    save("clusters._phy_annotation", ann[:, None])
    save("clusters.peakChannel", (np.arange(n) + 1.0)[:, None])
    save("clusters.probes", (np.arange(n) >= n // 2).astype(float)[:, None])
    save("clusters.depths", (np.arange(n) * 20.0)[:, None])
    wf = np.zeros((n, 82, 8))
    wf[:, :, 0] = -np.exp(-(((np.arange(82) - 30) / 5.0) ** 2))
    save("clusters.templateWaveforms", wf)
    save("clusters.templateWaveformChans", np.tile(np.arange(8.0), (n, 1)))
    save("spikes.times", st[:, None])
    save("spikes.clusters", sc.astype(np.int32)[:, None])

    save("trials.visualStim_times", tk["stim"][:, None])
    save("trials.goCue_times", tk["cue"][:, None])
    save("trials.response_times", tk["fb_time"][:, None])
    save("trials.feedback_times", tk["fb_time"][:, None])
    save("trials.feedbackType", tk["fb"][:, None])
    save("trials.response_choice", tk["choice"][:, None])
    save("trials.visualStim_contrastLeft", tk["cl"][:, None])
    save("trials.visualStim_contrastRight", tk["cr"][:, None])
    save("trials.included", tk["included"][:, None])
    save("trials.intervals", np.c_[tk["stim"] - 0.5, tk["fb_time"] + 1])

    on = [tk["stim"][tk["go"]] + tk["ft"][tk["go"]], tk["stim"][tk["late"]] + 0.6]
    kind = [np.where(tk["right"][tk["go"]], 2.0, 1.0), np.zeros(tk["late"].sum())]
    gone = rng.random(n_trials) < 0.25  # incidental movements between trials
    on.append(tk["stim"][gone] - 1.2)
    kind.append(np.zeros(gone.sum()))
    pa = rng.random(len(pt)) < 0.05
    on.append(pt[pa] + 0.2)
    kind.append(np.zeros(pa.sum()))
    on, kind = np.concatenate(on), np.concatenate(kind)
    o = np.argsort(on)
    on, kind = on[o], kind[o]
    save("wheelMoves.intervals", np.c_[on, on + 0.25])
    save("wheelMoves.type", kind[:, None])
    wt = np.arange(0, t_end, 0.01)
    v = np.zeros_like(wt)
    for a, k in zip(on, kind):
        m = (wt >= a) & (wt < a + 0.25)
        v[m] = {1.0: 8.0, 2.0: -8.0, 0.0: 1.0}[float(k)]
    save("wheel.position", np.cumsum(v)[:, None])
    save("wheel.timestamps", np.array([[0, 0.0], [len(wt) - 1, wt[-1]]]))

    save("passiveVisual.times", pt[:, None])
    save("passiveVisual.contrastLeft", pl[:, None])
    save("passiveVisual.contrastRight", pc[:, None])

    for name, field, rate in (("eye", "area", 30), ("face", "motionEnergy", 30)):
        t = np.arange(0, t_end, 1 / rate)
        x = np.convolve(rng.normal(size=len(t)), np.ones(30) / 30, "same") * 3
        for s, e in zip(tk["stim"], tk["engaged"]):
            m = (t > s - 0.5) & (t < s)
            x[m] += (0.5 if e else -0.5) * (1 if name == "eye" else -0.3)
        save(f"{name}.{field}", x[:, None])
        save(f"{name}.timestamps", np.c_[np.arange(len(t)), t])

    pos = np.c_[rng.choice(np.arange(-32.4, 33, 8), 600), rng.choice(np.arange(-128, 129, 8), 600)]
    save("sparseNoise.positions", pos)
    save("sparseNoise.times", (t_end - 50 + np.sort(rng.random(600)) * 45)[:, None])


def make_dataset(root, names=("Alpha_2020-01-01", "Alpha_2020-01-02", "Beta_2020-01-01", "Gamma_2020-01-01"), **kw):
    for i, s in enumerate(names):
        write_session(Path(root) / s, 100 + i, **kw)
    return list(names)
