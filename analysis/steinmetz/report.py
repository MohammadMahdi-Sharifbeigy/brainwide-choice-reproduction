"""report_vs_paper.md: the paper's numbers beside ours, with every known departure."""
import numpy as np
from scipy.stats import norm, pearsonr

from . import store
from .config import (ANATOMY_ORDER, FOREBRAIN, INCLUDED_TRIALS, LOCALIZED_NEURONS, MIDBRAIN, SIGNIFICANT_Z, TABLE1_RESPONSIVE, TABLE1_TOTAL, TOTAL_NEURONS,
                     VE_THRESHOLD)
from .engagement import nested_anova  # noqa: F401

DEPARTURES = (
    "Elastic net solved at lambda directly; the authors interpolate along a glmnet path.",
    "Example neurons are chosen by rule; the paper does not identify its examples.",
    "Released region labels differ from the paper's table per region (table below).",
    "Pupil panels use all 39 sessions; the paper's five rejected recordings are not named.",
    "Fig 2d-h and Fig 5b use two-sided |z|>4 as in the Methods; captions read as one-sided. One-sided counts are below.",
    "Contrast-matching Go group is correct contra trials, which gives the paper's 34 sessions; the Methods text says Left or Right.",
    "Fig 4c normalizes by group, reading the Methods' 'area' as the three groups.",
    "Decoding regularization and folds are not stated: L1 logistic regression, C from (0.01, 0.1, 1), 5 outer and 3 inner folds.",
    "ccCP counts use a 100 ms box stepped by 10 ms; the time-course window is not stated.",
    "Ext 4b latency is the peak on odd-numbered trials; its definition is not stated.",
    "Ext 9e-f and 9k use within-session median splits; the paper gives labels only.",
    "Fig 5c-d and Ext 7f use -0.2 to 0 s; their windows are not stated.",
    "The 34 engagement sessions are taken as the contrast-matched sessions; the paper does not say which 34.",
    "Fig 1d ANOVA uses chosen contrast, stimulus count and subject; its factors are not stated.",
    "Passive trials in matching require no movement from -0.05 to 0.4 s; this does not change the session count.",
    "DP applies the >=10 includable-trials rule itself, so it keeps more sessions than CP.",
    "Ext 5f uses neurons passing the 2% full-model cut, matching the neuron counts implied by its histograms.",
    "Ext 5g's color scale is set from the data, because the paper's projection units are not stated.",
    "Released region labels are kept: labels read from the atlas at channel positions were far further from Extended Data Table 1.",
    "Region colors approximate the authors' myCopper ramp, which is not in their repository.",
)


def focality(p):
    p = np.asarray(p, float)
    s = p.sum()
    return (p**2).sum() / s**2 if s > 0 else np.nan


def _proportions(region_codes, hits, n_regions):
    n = np.bincount(region_codes, minlength=n_regions)
    k = np.bincount(region_codes, weights=hits, minlength=n_regions)
    return np.divide(k, n, out=np.zeros(n_regions), where=n > 0)


def focality_interval(table, column, seed, n_boot=1000):
    """Point estimate and bias-corrected normal 95% interval, resampling neurons within region."""
    t = table[table["region"].isin(ANATOMY_ORDER)]
    codes = t["region"].map({r: i for i, r in enumerate(ANATOMY_ORDER)}).to_numpy(int)
    hits = (t[column].fillna(0).to_numpy() > VE_THRESHOLD).astype(float)
    if hits.sum() == 0:
        return np.nan, np.nan, np.nan  # no selective neurons, so focality is undefined
    est = focality(_proportions(codes, hits, 42))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    idx_by = [np.flatnonzero(codes == i) for i in range(42)]
    for b in range(n_boot):
        p = np.zeros(42)
        for i, idx in enumerate(idx_by):
            if len(idx):
                p[i] = hits[rng.choice(idx, len(idx))].mean()
        boot[b] = focality(p)
    z = norm.ppf(0.975)
    corrected = 2 * est - np.nanmean(boot)
    se = np.nanstd(boot, ddof=1)
    return est, corrected - z * se, corrected + z * se


def _row(name, paper, ours):
    return f"| {name} | {paper} | {ours} |"


def write(res):
    cfg = res.cfg
    n, t = res.neurons(), res.regression()
    base_total = sum(int(res.base(s)["n_base"].sum()) for s in res.sessions)
    trials_total = sum(int(res.base(s)["included"].sum()) for s in res.sessions)
    lines = ["# Paper against this run", ""]
    if cfg.synthetic:
        lines += ["**SYNTHETIC DATA. These numbers say nothing about the paper.**", ""]
    lines += [f"Sessions in this run: {len(res.sessions)} of 39.", "", "| Quantity | Paper | Ours |", "|---|---|---|"]
    lines.append(_row("Included trials", INCLUDED_TRIALS, trials_total))
    lines.append(_row("Neurons", TOTAL_NEURONS, base_total))
    lines.append(_row("Localized neurons", LOCALIZED_NEURONS, len(n)))
    if len(n):
        resp = int(n["responsive"].sum())
        lines.append(_row("Modulated neurons", "13,466 (60.0%)", f"{resp:,} ({100 * resp / len(n):.1f}%)"))
    matched = [s for s in res.sessions if bool(res.base(s)["matched"])]
    lines.append(_row("Sessions with contrast-matched trials", "34 of 39", f"{len(matched)} of {len(res.sessions)}"))
    z = [store.load(cfg.results / "cccp" / f"{s}.npz") for s in res.sessions]
    cp_ok = [s for s, q in zip(res.sessions, z) if bool(q["cp_ok"])]
    dp_ok = [s for s, q in zip(res.sessions, z) if bool(q["dp_ok"])]
    lines.append(_row("Sessions in CP analysis", "33 of 39", f"{len(cp_ok)} of {len(res.sessions)}"))
    lines.append(_row("Sessions in DP analysis", "not stated", f"{len(dp_ok)} of {len(res.sessions)}"))
    if len(t):
        shuffle = res.shuffle_ve()
        if len(shuffle):
            lines.append(_row("Shuffle false positives", "0.33%", f"{100 * (shuffle > VE_THRESHOLD).mean():.2f}%"))
        sel = t[(t["choice"] > VE_THRESHOLD)]
        mid, fore = sel[sel["region"].isin(MIDBRAIN)], sel[sel["region"].isin(FOREBRAIN)]
        if len(mid):
            lines.append(_row("Midbrain choice neurons preferring contra", "53/54", f"{int(mid['prefers_contra'].sum())}/{len(mid)}"))
            below = (mid[["contra", "ipsi"]].min(axis=1) < 0).sum()
            lines.append(_row("Midbrain suppressed for non-preferred choice", "29/54", f"{int(below)}/{len(mid)}"))
        if len(fore):
            lines.append(_row("Forebrain choice neurons preferring ipsi", "19/48", f"{int((~fore['prefers_contra'].astype(bool)).sum())}/{len(fore)}"))
            below = (fore[["contra", "ipsi"]].min(axis=1) < 0).sum()
            lines.append(_row("Forebrain suppressed for non-preferred choice", "10/48", f"{int(below)}/{len(fore)}"))
        foc = {c: focality_interval(t, c, cfg.seed)[0] for c in ("action", "vision_contra", "choice")}
        lines.append(_row("Focality: action, contra vision, choice", "0.040, 0.069, 0.079", ", ".join(f"{foc[c]:.3f}" for c in foc)))
    e = res.stage("engagement.npz")
    if "region_names" in e:
        ok = np.isfinite(e["go_miss"])
        if ok.sum() > 2:
            r, p = pearsonr(e["task_passive"][ok], e["go_miss"][ok])
            lines.append(_row("Fig 5d correlation", "r = 0.46, p = 0.002", f"r = {r:.2f}, p = {p:.3g}"))
        if len(e["ei_diff"]):
            lines.append(_row("Mean engagement index difference", "8.42", f"{e['ei_diff'].mean():.2f} (n = {len(e['ei_diff'])})"))
    pre = {}
    for s, q in zip(res.sessions, z):
        if "dp_pre_val" in q:
            reg = res.base(s)["n_region"][q["ids"]]
            for r_ in np.unique(reg):
                pre.setdefault(r_, []).append((s, q["dp_pre_val"][reg == r_]))
    if "region_names" in e and pre:
        gm = dict(zip([str(x) for x in e["region_names"]], e["go_miss"]))
        tp = dict(zip([str(x) for x in e["region_names"]], e["task_passive"]))
        for label, keep in (("DP-qualified", set(dp_ok)), ("33-session CP subset", set(dp_ok) & set(cp_ok))):
            names = [r_ for r_ in pre if r_ in gm and np.isfinite(gm[r_]) and any(s in keep for s, _ in pre[r_])]
            if len(names) > 2:
                y = [np.mean(np.concatenate([v for s, v in pre[r_] if s in keep])) - 0.5 for r_ in names]
                lines.append(_row(f"Ext 7f with Go-Miss, {label} sessions ({len(keep)})", "r = 0.69", f"r = {pearsonr([gm[r_] for r_ in names], y)[0]:.2f}"))
                lines.append(_row(f"Ext 7f with Task-Passive, {label} sessions", "r = 0.48", f"r = {pearsonr([tp[r_] for r_ in names], y)[0]:.2f}"))
    lines += ["", "## Sessions left out", "", f"Contrast matching: {', '.join(s for s in res.sessions if s not in matched) or 'none'}",
              f"CP: {', '.join(s for s in res.sessions if s not in cp_ok) or 'none'}", f"DP: {', '.join(s for s in res.sessions if s not in dp_ok) or 'none'}"]
    lines += ["", "## Two-sided and one-sided significance (Fig 2d-h, Fig 5b)", "", "| Condition | |z|>4 | z>4 |", "|---|---|---|"]
    if len(n):
        for k in ("contra_correct", "ipsi_correct", "contra_miss", "ipsi_miss", "contra_passive"):
            lines.append(f"| {k} | {(np.abs(n[f'z_{k}']) > SIGNIFICANT_Z).mean() * 100:.1f}% | {(n[f'z_{k}'] > SIGNIFICANT_Z).mean() * 100:.1f}% |")
    lines += ["", "## Extended Data Table 1", "", "| Region | Paper total | Ours | Paper responsive | Ours |", "|---|---|---|---|---|"]
    for r in sorted(TABLE1_TOTAL):
        s = n[n["region"] == r] if len(n) else n
        lines.append(f"| {r} | {TABLE1_TOTAL[r]} | {len(s)} | {TABLE1_RESPONSIVE[r]} | {int(s['responsive'].sum()) if len(s) else 0} |")
    lines += ["", "## Known departures", ""] + [f"- {d}" for d in DEPARTURES]
    out = cfg.output / "report_vs_paper.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf8")
    return out
