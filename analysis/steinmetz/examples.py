"""Example neurons. The paper does not identify its examples, so each slot is filled by the real neuron closest to the published one.

Curves are digitized from the paper's own panels, and every neuron in the region is scored against them in absolute sp/s. Kernel
requirements keep the choice meaningful: a slot shown as a choice neuron can only be filled by a neuron that passes the nested Choice test.
`select_examples.py` writes results/example_neurons.json, which can be edited by hand. Scores go to results/example_scores.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import regression, store
from .config import ANALYSIS, ASSIGNMENT_REGIONS, DT, VE_THRESHOLD
from .rates import aligned_rates, smooth

HIRES = ANALYSIS / "resources" / "paper"
PALETTE = json.loads((Path(__file__).parent / "paper_palette.json").read_text())

# a panel of the paper: figure, plot box in PNG pixels, the data value at x0 and x1, and at the box's bottom and top
#   slot: region, panel, kernels that must pass 2%, kernels that must not, sign of preference (True contra, False ipsi, None any)
SLOTS = {
    "fig2a": ("SUB", ("fig3", 508, 62, 614, 126, -.145, 0, 0, 100), ("action",), ("vision_contra", "choice"), None),
    "fig2b": ("VISpm", ("fig3", 212, 62, 325, 131, -.145, 0, 0, 50), ("vision_contra",), (), None),
    "fig2c": ("ZI", ("fig4", 168, 56, 248, 108, -.25, 0, 0, 100), ("choice",), (), True),
    "fig4a_vis": ("VISp", ("fig4", 62, 65, 145, 107, -.25, 0, 0, 40), ("vision_contra",), ("choice",), None),
    "fig4a_mrn": ("MRN", ("fig4", 62, 128, 145, 180, -.25, 0, 0, 200), ("choice",), (), True),
    "fig4a_cp": ("CP", ("fig4", 168, 130, 248, 181, -.25, 0, 0, 60), ("choice",), (), True),
    "fig4e_1": ("MRN", ("fig4", 473, 222, 554, 292, -.2, .05, 0, 60), ("choice",), (), True),
    "fig4e_2": ("MRN", ("fig4", 572, 222, 646, 292, -.2, .05, 0, 200), ("choice",), (), True),
    "fig4e_3": ("SCm", ("fig4", 663, 222, 737, 292, -.2, .05, 0, 80), ("choice",), (), True),
    "fig4e_4": ("SCm", ("fig4", 755, 222, 829, 292, -.2, .05, 0, 80), ("choice",), (), True),
    "fig4e_5": ("MOs", ("fig4", 473, 317, 554, 386, -.2, .05, 0, 100), ("choice",), (), True),
    "fig4e_6": ("MOs", ("fig4", 572, 317, 646, 386, -.2, .05, 0, 20), ("choice",), (), False),
    "fig4e_7": ("CP", ("fig4", 663, 317, 737, 386, -.2, .05, 0, 40), ("choice",), (), True),
    "fig4e_8": ("CP", ("fig4", 755, 317, 829, 386, -.2, .05, 0, 60), ("choice",), (), False),
}
# stimulus-aligned targets, read from Ext 4a: (region, peak sp/s at 100% contrast, baseline sp/s)
CONTRAST_SLOTS = {"ext4_visp": ("VISp", 90.0, 2.0), "ext4_visam": ("VISam", 55.0, 15.0)}
EXT5_TARGET = (12.0, 25.0)  # Left and Right response at movement onset, read from Ext 5c
# raster row heights of Fig 2a-c in pixels, which follow each session's trial counts
ROW_HEIGHTS = {"fig2a": (25, 33, 19, 25, 33, 19), "fig2b": (28, 36, 28, 20, 32, 24), "fig2c": (36, 36, 15, 12, 37, 24)}
ROW_WEIGHT = 1.0
ASSIGN_MIN_RATE = 8.0  # sp/s around movement, so a panel is not just sparse spikes
ASSIGN_MIN_TRIALS = 15  # trials per side, so a panel has rows or an average to read


def hires_rgb(key):
    return np.asarray(Image.open(HIRES / f"{key}.png").convert("RGB")).astype(float) / 255


def digitize(panel, cache={}):
    """Contra (orange) and ipsi (blue) line of a paper panel, as times and two value arrays."""
    key, x0, y0, x1, y1, t0, t1, v0, v1 = panel
    im = cache.setdefault(key, hires_rgb(key))
    m = PALETTE["maps"][key]
    sc = m["scale"]
    a, b = int((x0 + m["ox"]) * sc), int((x1 + m["ox"]) * sc)
    c, d = int((y0 + m["oy"]) * sc), int((y1 + m["oy"]) * sc)
    p = im[c:d, a:b]
    orange = (p[..., 0] > .8) & (p[..., 1] > .35) & (p[..., 1] < .66) & (p[..., 2] < .35)
    blue = (p[..., 2] > .55) & (p[..., 0] < .42) & (p[..., 2] - p[..., 0] > .3)
    times = np.arange(t0 + 0.0025, t1, DT)
    cols = (times - t0) / (t1 - t0) * (b - a - 1)

    def line(mask):
        ys = np.array([np.median(np.flatnonzero(mask[:, j])) if mask[:, j].any() else np.nan for j in range(mask.shape[1])])
        return np.interp(cols, np.arange(len(ys)), ys, left=np.nan, right=np.nan)

    to_value = lambda y: v1 + (v0 - v1) * y / (d - c - 1)
    return times, to_value(line(orange)), to_value(line(blue))


def move_curves(res, s, clusters, times):
    """Contra and ipsi rate per neuron, with the paper's display smoothing, on the given times."""
    b = res.base(s)
    st, clu = res.spikes(s)
    out = []
    win = (times[0] - DT / 2, times[-1] + DT / 2)
    for mask in (b["has_right"] & b["included"], b["has_left"] & b["included"]):
        if mask.sum() < 3:
            return None
        r, t = aligned_rates(st, clu, clusters, b["move_abs"][mask], win, 0.02)
        out.append(smooth(r.mean(0), 0.03, axis=0)[-len(times):])
    return out


def _candidates(t, region, need, avoid, contra):
    q = t[t["region"] == region]
    strict = q[q["ve"] >= VE_THRESHOLD]
    for k in need:
        strict = strict[strict[k] > VE_THRESHOLD]
    for k in avoid:
        strict = strict[~(strict[k] > VE_THRESHOLD)]
    if contra is not None:
        strict = strict[strict["prefers_contra"] == contra]
    if len(strict):
        return strict, True
    return q, False


def score_slot(res, table, slot, used):
    region, panel, need, avoid, contra = SLOTS[slot]
    times, c_t, i_t = digitize(panel)
    pool, strict = _candidates(table, region, need, avoid, contra)
    peak = max(np.nanmax(c_t), np.nanmax(i_t))
    rows = []
    for s, g in pool.groupby("session"):
        ids = g["cluster"].to_numpy()
        cur = move_curves(res, s, ids, times)
        if cur is None:
            continue
        c_r, i_r = cur
        err = np.sqrt((np.nanmean((c_r - c_t[:, None]) ** 2, 0) + np.nanmean((i_r - i_t[:, None]) ** 2, 0)) / 2) / peak
        rows += [dict(slot=slot, region=region, session=s, cluster=int(k), error=float(e), passes_kernels=strict) for k, e in zip(ids, err) if (s, int(k)) not in used]
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    if slot in ROW_HEIGHTS:
        target = np.array(ROW_HEIGHTS[slot], float)
        target = target / target.sum()
        shares = {s: np.array(row_counts(res, s), float) for s in d["session"].unique()}
        d["row_error"] = [float(np.abs(shares[s] / max(shares[s].sum(), 1) - target).sum()) for s in d["session"]]
        d["curve_error"] = d["error"]
        d["error"] = d["curve_error"] + ROW_WEIGHT * d["row_error"]
    return d.sort_values("error")


def contrast_slot(res, table, slot, used):
    region, peak, base = CONTRAST_SLOTS[slot]
    pool, strict = _candidates(table, region, ("vision_contra",), (), None)
    rows = []
    for s, g in pool.groupby("session"):
        b = res.base(s)
        ev = b["stim"][b["included"] & (b["cr"] == 1)]
        if len(ev) < 5:
            continue
        ids = g["cluster"].to_numpy()
        st, clu = res.spikes(s)
        r, t = aligned_rates(st, clu, ids, ev, (-0.05, 0.2), 0.03)
        m = r.mean(0)
        pk, bs = m[(t > 0.05) & (t < 0.15)].max(0), m[t < 0].mean(0)
        err = (np.abs(pk - peak) + np.abs(bs - base)) / peak
        rows += [dict(slot=slot, region=region, session=s, cluster=int(k), error=float(e), passes_kernels=strict) for k, e in zip(ids, err) if (s, int(k)) not in used]
    d = pd.DataFrame(rows)
    return d if d.empty else d.sort_values("error")


def ext5_slot(res, table):
    """The neuron whose Left and Right responses at movement onset match Ext 5c, with no pre-movement drift."""
    left_t, right_t = EXT5_TARGET
    rows = []
    q = table[(table["ve"] >= VE_THRESHOLD) & (table["choice"] > VE_THRESHOLD)]
    for s, g in q.groupby("session"):
        K = res.kernels(s)["K"][:, g["index"].to_numpy()]
        contra, ipsi = regression.rates_at_move(K, 0.02)
        early_c, early_i = regression.rates_at_move(K, -0.2)
        err = (np.abs(ipsi - left_t) + np.abs(contra - right_t) + np.abs(early_c) + np.abs(early_i)) / right_t
        rows += [dict(slot="ext5", region=r, session=s, cluster=int(k), error=float(e), passes_kernels=True) for r, k, e in zip(g["region"], g["cluster"], err)]
    d = pd.DataFrame(rows)
    return d if d.empty else d.sort_values("error")


def row_counts(res, s):
    b = res.base(s)
    alone_c, alone_i = (b["cr"] > 0) & (b["cl"] == 0), (b["cl"] > 0) & (b["cr"] == 0)
    return [(b["correct_contra"] & alone_c).sum(), (b["correct_ipsi"] & alone_i).sum(), (b["miss_contra"] & alone_c).sum(), b["miss_ipsi"].sum(),
            (b["passive_still"] & (b["passive_cr"] > 0) & (b["passive_cl"] == 0)).sum(), (b["passive_still"] & (b["passive_cl"] > 0) & (b["passive_cr"] == 0)).sum()]


def _raster_rule(res, table, region):
    """Assignment rasters: most reliable contra stimulus response, in a session with enough trials in every row."""
    from .results import PSTH_T

    n = res.neurons()
    rows = []
    for s in n[n["region"] == region]["session"].unique():
        c = row_counts(res, s)
        if min(c[:2]) < ASSIGN_MIN_TRIALS or min(c) < 5:
            continue
        b, p = res.base(s), store.load(res.cfg.results / "psth" / f"{s}.npz")
        keep = b["n_region"][p["ids"]] == region
        m = (PSTH_T >= 0.05) & (PSTH_T <= 0.15)
        key = "rate_contra_correct" if bool(b["matched"]) else "ext4_stim"
        drive = np.nanmean(p[key][m], 0) - p["baseline"]
        z = np.abs(p["z_contra_correct"]) if bool(b["matched"]) else np.zeros(len(keep))
        rows.append(pd.DataFrame({"session": s, "cluster": p["ids"][keep], "score": (drive / np.sqrt(p["baseline"] + 1) + z / 10)[keep]}))
    return pd.concat(rows).sort_values("score", ascending=False) if rows else pd.DataFrame(columns=["session", "cluster", "score"])


def select(res):
    """Fill every slot and write the choice and its scores."""
    t = res.regression()
    chosen, scores, used = {}, [], set()
    if not len(t):
        store.save_json(res.cfg.results / "example_neurons.json", chosen)
        return chosen
    for slot in SLOTS:
        d = score_slot(res, t, slot, used)
        if len(d):
            first = d.iloc[0]
            chosen[slot] = {"session": first["session"], "cluster": int(first["cluster"])}
            used.add((first["session"], int(first["cluster"])))
            scores.append(d.head(5))
    for slot in CONTRAST_SLOTS:
        d = contrast_slot(res, t, slot, used)
        if len(d):
            chosen[slot] = {"session": d.iloc[0]["session"], "cluster": int(d.iloc[0]["cluster"])}
            scores.append(d.head(5))
    d = ext5_slot(res, t)
    if len(d):
        chosen["ext5"] = {"session": d.iloc[0]["session"], "cluster": int(d.iloc[0]["cluster"])}
        scores.append(d.head(5))
    for region in ASSIGNMENT_REGIONS:
        top = _raster_rule(res, t, region).head(5)
        if len(top):
            chosen[f"assign_raster_{region}"] = {"session": top.iloc[0]["session"], "cluster": int(top.iloc[0]["cluster"])}
            scores.append(top.assign(slot=f"assign_raster_{region}", region=region, error=-top["score"]).drop(columns="score"))
        passed = t[(t["region"] == region) & (t["ve"] >= VE_THRESHOLD)]
        window = np.arange(-0.2, 0.05, DT) + DT / 2
        for label, contra in (("contra", True), ("ipsi", False)):
            ranked = passed[passed["prefers_contra"] == contra].sort_values("choice", ascending=False)
            keep = []
            for _, q in ranked.iterrows():  # highest Choice variance explained among neurons that fire enough to show a curve
                b = res.base(q["session"])
                choices = min((b["has_left"] & b["included"]).sum(), (b["has_right"] & b["included"]).sum())
                cur = move_curves(res, q["session"], np.array([int(q["cluster"])]), window)
                if cur is not None and max(cur[0].max(), cur[1].max()) >= ASSIGN_MIN_RATE and choices >= ASSIGN_MIN_TRIALS:
                    keep.append(q)
                if len(keep) == 5:
                    break
            if keep:
                best = pd.DataFrame(keep)
                chosen[f"assign_4e_{region}_{label}"] = {"session": best.iloc[0]["session"], "cluster": int(best.iloc[0]["cluster"]), "choice_ve": float(best.iloc[0]["choice"])}
                scores.append(best.assign(slot=f"assign_4e_{region}_{label}", error=-best["choice"])[["slot", "region", "session", "cluster", "error"]])
    store.save_json(res.cfg.results / "example_neurons.json", chosen)
    (pd.concat(scores) if scores else pd.DataFrame(columns=["slot", "region", "session", "cluster", "error"])).to_csv(res.cfg.results / "example_scores.csv", index=False)
    return chosen


def choose(res):
    path = res.cfg.results / "example_neurons.json"
    if path.exists():
        return store.load_json(path)
    return select(res)
