"""Example neurons, chosen by written rules. Edit results/example_neurons.json by hand to swap one; the file is kept if present."""
import numpy as np
import pandas as pd

from . import store
from .config import ASSIGNMENT_REGIONS

# slot -> (region, scoring column, keep only prefers_contra True/False/None)
SLOTS = {
    "fig2a": ("SUB", "action", None), "fig2b": ("VISpm", "vision_contra", None), "fig2c": ("ZI", "choice", True),
    "fig4a_vis": ("VISp", "vision_contra", None), "fig4a_mrn": ("MRN", "choice", None), "fig4a_cp": ("CP", "choice", None),
    "fig4e_1": ("MRN", "choice", True), "fig4e_2": ("MRN", "choice", True), "fig4e_3": ("SCm", "choice", True), "fig4e_4": ("SCm", "choice", True),
    "fig4e_5": ("MOs", "choice", True), "fig4e_6": ("MOs", "choice", False), "fig4e_7": ("CP", "choice", True), "fig4e_8": ("CP", "choice", False),
    "ext4_visp": ("VISp", "vision_contra", None), "ext4_visam": ("VISam", "vision_contra", None),
}


def rows_available(res, s, min_trials=1):
    b = res.base(s)
    alone_c, alone_i = (b["cr"] > 0) & (b["cl"] == 0), (b["cl"] > 0) & (b["cr"] == 0)
    counts = [(b["correct_contra"] & alone_c).sum(), (b["correct_ipsi"] & alone_i).sum(), (b["miss_contra"] & alone_c).sum(), b["miss_ipsi"].sum(),
              (b["passive_still"] & (b["passive_cr"] > 0) & (b["passive_cl"] == 0)).sum(), (b["passive_still"] & (b["passive_cl"] > 0) & (b["passive_cr"] == 0)).sum()]
    return min(counts) >= min_trials


def _top(table, region, column, contra, n, skip=()):
    t = table[(table["region"] == region) & table[column].notna()]
    if contra is not None:
        t = t[t["prefers_contra"] == contra]
    t = t.sort_values(column, ascending=False)
    t = t[[(a, b) not in skip for a, b in zip(t["session"], t["cluster"])]]
    return t.head(n)


def _stim_score(res, region):
    """Assignment raster rule: highest contra stimulus response from 0.05 to 0.15 s, among neurons with a trial in every row."""
    from .results import PSTH_T
    n = res.neurons()
    rows = []
    for s in n[n["region"] == region]["session"].unique():
        if not rows_available(res, s):
            continue
        p = store.load(res.cfg.results / "psth" / f"{s}.npz")
        m = (PSTH_T >= 0.05) & (PSTH_T <= 0.15)
        key = "rate_contra_correct" if bool(res.base(s)["matched"]) else "ext4_stim"
        score = np.nanmean(p[key][m], 0) - p["baseline"]
        keep = res.base(s)["n_region"][p["ids"]] == region
        rows.append(pd.DataFrame({"session": s, "cluster": p["ids"][keep], "score": score[keep]}))
    return pd.concat(rows).sort_values("score", ascending=False) if rows else pd.DataFrame(columns=["session", "cluster", "score"])


def choose(res):
    path = res.cfg.results / "example_neurons.json"
    if path.exists():
        return store.load_json(path)
    t = res.regression()
    chosen, cands = {}, []
    used = set()
    for slot, (region, col, contra) in SLOTS.items():
        top = _top(t, region, col, contra, 5, used) if len(t) else pd.DataFrame()
        for rank, (_, r) in enumerate(top.iterrows()):
            cands.append((slot, rank, r["session"], int(r["cluster"]), float(r[col])))
        if len(top):
            first = top.iloc[0]
            chosen[slot] = {"session": first["session"], "cluster": int(first["cluster"])}
            used.add((first["session"], first["cluster"]))
    for region in ASSIGNMENT_REGIONS:
        top = _stim_score(res, region).head(5)
        for rank, (_, r) in enumerate(top.iterrows()):
            cands.append((f"assign_raster_{region}", rank, r["session"], int(r["cluster"]), float(r["score"])))
        if len(top):
            chosen[f"assign_raster_{region}"] = {"session": top.iloc[0]["session"], "cluster": int(top.iloc[0]["cluster"])}
        if len(t):
            passed = t[t["ve"] >= 0.02]
            for label, contra in (("contra", True), ("ipsi", False)):
                best = _top(passed, region, "choice", contra, 5)
                for rank, (_, r) in enumerate(best.iterrows()):
                    cands.append((f"assign_4e_{region}_{label}", rank, r["session"], int(r["cluster"]), float(r["choice"])))
                if len(best):
                    chosen[f"assign_4e_{region}_{label}"] = {"session": best.iloc[0]["session"], "cluster": int(best.iloc[0]["cluster"]),
                                                             "choice_ve": float(best.iloc[0]["choice"])}
    store.save_json(path, chosen)
    pd.DataFrame(cands, columns=["slot", "rank", "session", "cluster", "score"]).to_csv(res.cfg.results / "example_candidates.csv", index=False)
    return chosen
