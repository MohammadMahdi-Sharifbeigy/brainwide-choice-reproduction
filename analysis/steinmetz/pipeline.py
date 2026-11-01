"""Stages. Each writes results atomically and skips work already done."""
import json
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import psutil
from tqdm.auto import tqdm

from . import behavior, cccp, decoding, engagement, jpecc, regression, store, trials
from .config import (ALPHA_RESPONSIVE, ANOVA_WIN, BONFERRONI_REGION, DT, ENGAGEMENT_WIN, FRONTAL, MIDBRAIN, MIN_INCLUDABLE_TRIALS,
                     MIN_RATE, SIGNIFICANT_Z, VIS)
from .data import Session, list_sessions, subject_of
from .rates import aligned_counts, aligned_rates, significance_z, six_tests, smooth, window_mean
from .regions import kept_regions, neuron_regions, region_counts

PSTH_WIN = (-0.1, 0.3)


# ---- running

def auto_workers(peak_gb, cores=None):
    """Workers that fit in free RAM, leaving 1 GB spare: at most cores - 1, at least 1."""
    cores = cores or os.cpu_count()
    free = psutil.virtual_memory().available / 1e9
    return max(1, min(cores - 1, int((free - 1) // max(peak_gb, 0.05))))


def scaled_peak(smoke_peak_gb, smoke_spikes, largest_spikes, overhead_gb=0.25):
    """Worst-case worker memory: the part above the interpreter's own use grows with the session's spikes."""
    return overhead_gb + max(smoke_peak_gb - overhead_gb, 0) * largest_spikes / smoke_spikes


class Runner:
    """Parallel map plus timing and peak memory per stage."""

    def __init__(self, cfg, workers):
        self.cfg, self.workers = cfg, workers
        self.pool = ProcessPoolExecutor(workers) if workers > 1 else None
        self.timing = {}
        self.current = ""

    def _label(self, fn, jobs):
        """Bar text: the stage, its step, and the kernel a nested step is testing."""
        step = fn.__name__.strip("_").replace("_", " ")
        key = jobs[0][3] if jobs and len(jobs[0]) > 3 else ""
        return " | ".join(p for p in (self.current, step, key) if p)

    def pmap(self, fn, jobs):
        jobs = list(jobs)
        bar = tqdm(total=len(jobs), desc=self._label(fn, jobs), unit="session", position=1, leave=False)
        try:
            if self.pool is None or len(jobs) <= 1:
                out = []
                for j in jobs:
                    out.append(fn(j))
                    bar.update()
                return out
            futures = [self.pool.submit(fn, j) for j in jobs]
            for f in as_completed(futures):
                f.result()
                bar.update()
            return [f.result() for f in futures]
        finally:
            bar.close()

    def stage(self, name, fn, *args):
        self.current = name
        peak = [0.0]
        stop = threading.Event()

        def watch():
            me = psutil.Process()
            while not stop.wait(0.2):
                total = 0
                for p in [me, *me.children(recursive=True)]:
                    try:
                        total += p.memory_info().rss
                    except psutil.Error:
                        pass
                peak[0] = max(peak[0], total)

        th = threading.Thread(target=watch, daemon=True)
        th.start()
        t0 = time.perf_counter()
        fn(*args)
        stop.set()
        th.join()
        self.timing[name] = {"seconds": round(time.perf_counter() - t0, 2), "peak_gb": round(peak[0] / 1e9, 3)}
        tqdm.write(f"{name}: {self.timing[name]['seconds']} s, peak {self.timing[name]['peak_gb']} GB")
        self._save_timing()

    def _save_timing(self):
        """Keep the largest figures seen, so a resumed run that skips work cannot erase the real measurements."""
        path = self.cfg.results / "timing.json"
        old = store.load_json(path)["stages"] if path.exists() else {}
        merged = {**old}
        for k, v in self.timing.items():
            merged[k] = {f: max(v[f], old.get(k, v)[f]) for f in v}
        store.save_json(path, {"workers": self.workers, "stages": merged})

    def close(self):
        if self.pool:
            self.pool.shutdown()


def _exists(path):
    return path.exists()


# ---- base

def base_stage(job):
    cfg, s = job
    out = cfg.results / "base" / f"{s}.npz"
    if _exists(out):
        return
    ses = Session(cfg.data, s)
    tr = trials.build_trials(ses)
    st, clu = ses.spikes()
    ann = ses.a("clusters._phy_annotation")
    raw, region = neuron_regions(ses)

    keep = tr["kernel_trial"]
    ev = tr["stim"][keep]
    lo, hi = np.searchsorted(st, ev), np.searchsorted(st, ev + 0.4)
    n = np.zeros(len(ann))
    for a, b in zip(lo, hi):
        n += np.bincount(clu[a:b], minlength=len(ann))[: len(ann)]
    rate = n / (0.4 * max(len(ev), 1))
    is_base = (ann >= 2) & (rate >= MIN_RATE)

    good = np.flatnonzero(is_base)
    p = np.full((len(ann), 6), np.nan)
    if len(good):
        r, t = aligned_rates(st, clu, good, ev, (-0.2, 0.4), 0.025)
        moved = (tr["has_left"] | tr["has_right"]) & keep
        m, tm = aligned_rates(st, clu, good, tr["move_abs"][moved], (-0.1, 0.2), 0.025)
        rew = tr["included"] & (tr["choice"] == 0) & (tr["fb"] == 1)
        w, tw = aligned_rates(st, clu, good, tr["fb_time"][rew], (0.0, 0.15), 0.025)
        bsl = window_mean(r, t, -0.2, 0)
        p[good] = six_tests(window_mean(r, t, 0, 0.4), bsl, window_mean(r, t, 0.05, 0.15), window_mean(m, tm, -0.1, 0.05),
                            bsl[moved[keep]], tr["has_left"][moved], tr["has_right"][moved], window_mean(m, tm, -0.05, 0.2),
                            w.mean(1) if rew.any() else np.zeros((0, len(good))))

    if cfg.neuron_lists is not None:
        incl = np.load(cfg.neuron_lists / f"{s}_inclNeurons.npy").ravel().astype(bool)
    else:
        incl = is_base & ~np.all(p[:, :4] > 0.05, axis=1)  # same NaN behaviour as the authors' all(p>0.05)

    m = trials.matched_sets(tr, store.session_rng(cfg.seed, s, 10))
    extra = {"matched": m is not None}
    for k in ("go", "miss", "passive"):
        extra[f"match_{k}"] = m[k] if m else np.zeros(0, int)
    store.save(out, **{k: v for k, v in tr.items()}, **extra, ann=ann, n_rate=rate, n_base=is_base, n_raw=raw, n_region=region,
               n_p=p, n_incl=incl, n_probe=ses.a("clusters.probes"), n_depth=ses.a("clusters.depths"))


def regions_stage(cfg, sessions):
    out = cfg.results / "regions.json"
    if _exists(out):
        return
    per = {}
    for s in sessions:
        b = store.load(cfg.results / "base" / f"{s}.npz")
        per[s] = region_counts(b["n_region"], b["n_base"])
    kept = kept_regions(per, {s: subject_of(s) for s in sessions}, cfg.min_subjects)
    store.save_json(out, {"kept": {s: sorted(v) for s, v in kept.items()}})


def neuron_sets(cfg, s, b=None):
    """Masks over clusters for the sets of the design's Section 8.2."""
    b = b if b is not None else store.load(cfg.results / "base" / f"{s}.npz")
    kept = store.load_json(cfg.results / "regions.json")["kept"][s]
    in_kept = np.isin(b["n_region"], kept)
    all_ = b["n_base"] & in_kept
    return {"all": all_, "responsive": all_ & np.any(b["n_p"] < ALPHA_RESPONSIVE, axis=1), "regression": b["n_incl"] & in_kept}


def regression_data_stage(job):
    cfg, s = job
    out = cfg.results / "regression" / "data" / f"{s}.npz"
    if _exists(out):
        return
    b = store.load(cfg.results / "base" / f"{s}.npz")
    ids = np.flatnonzero(neuron_sets(cfg, s, b)["regression"])
    st, clu = Session(cfg.data, s).spikes()
    _, rows, *_ = regression.build_design(b)
    F, means = regression.target_rates(st, clu, ids, b, rows)
    store.save(out, F=F, ids=ids, means=means)


# ---- rates

def _psth(st, clu, ids, ev):
    if len(ev) == 0:
        return np.full((int(round((PSTH_WIN[1] - PSTH_WIN[0]) / DT)), len(ids)), np.nan, np.float32)
    r, t = aligned_rates(st, clu, ids, ev, (PSTH_WIN[0] - 0.15, PSTH_WIN[1]), 0.02)
    return smooth(r.mean(0), 0.03, axis=0)[t > PSTH_WIN[0]].astype(np.float32)


def _z(st, clu, ids, ev):
    if len(ev) < 2:
        return np.zeros(len(ids), np.float32)
    c, t = aligned_counts(st, clu, ids, ev, (-0.1, 0.25))
    post = c[:, t >= 0].sum(1) / 0.25
    return significance_z(post, c[:, t < 0].sum(1) / 0.1).astype(np.float32)


def psth_stage(job):
    cfg, s = job
    out = cfg.results / "psth" / f"{s}.npz"
    if _exists(out):
        return
    b = store.load(cfg.results / "base" / f"{s}.npz")
    ids = np.flatnonzero(neuron_sets(cfg, s, b)["all"])
    if len(ids) == 0:
        store.save(out, ids=ids)
        return
    st, clu = Session(cfg.data, s).spikes()
    conds = {"contra_correct": b["stim"][b["match_go"]], "ipsi_correct": b["stim"][b["correct_ipsi"]],
             "contra_miss": b["stim"][b["match_miss"]], "ipsi_miss": b["stim"][b["miss_ipsi"]],
             "contra_passive": b["passive_times"][b["match_passive"]]}
    res = {"ids": ids, "matched": b["matched"]}
    for k, ev in conds.items():
        res[f"rate_{k}"] = _psth(st, clu, ids, ev)
        res[f"z_{k}"] = _z(st, clu, ids, ev)
    r, t = aligned_rates(st, clu, ids, b["stim"][b["included"]], (-0.02, 0.0), 0.02)
    res["baseline"] = r.mean((0, 1))

    # Ext 4
    both = b["included"] & (b["cr"] > 0) & (b["has_left"] | b["has_right"])
    idx = np.flatnonzero(both)
    res["ext4_stim"] = _psth(st, clu, ids, b["stim"][both])
    res["ext4_odd"] = _psth(st, clu, ids, b["stim"][idx[::2]])
    res["ext4_even"] = _psth(st, clu, ids, b["stim"][idx[1::2]])
    moved = both
    rm, tm = aligned_rates(st, clu, ids, b["move_abs"][moved], (-0.3, 0.1), 0.02) if moved.any() else (None, None)
    res["ext4_move"] = smooth(rm.mean(0), 0.03, axis=0)[tm > -0.2] if rm is not None else np.zeros((0, len(ids)))
    rew = b["included"] & (b["choice"] == 0) & (b["fb"] == 1) & (b["cl"] == 0) & (b["cr"] == 0)
    if rew.any():
        rr, tr_ = aligned_rates(st, clu, ids, b["fb_time"][rew], (-0.2, 0.2), 0.02)
        res["ext4_reward"] = smooth(rr.mean(0), 0.03, axis=0)[tr_ > -0.1]
    else:
        res["ext4_reward"] = np.full((60, len(ids)), np.nan)

    # sliding rank-sum for "highly activated" neurons: stimulus and movement against neither
    from scipy.stats import mannwhitneyu

    neither = b["included"] & b["still"] & (b["cl"] == 0) & (b["cr"] == 0)
    hit = np.zeros(len(ids), bool)
    if both.sum() > 3 and neither.sum() > 3:
        ca, t = aligned_counts(st, clu, ids, b["stim"][both], (-0.05, 0.4))
        cb, _ = aligned_counts(st, clu, ids, b["stim"][neither], (-0.05, 0.4))
        sa, sb = cccp.sliding_counts(ca, 20), cccp.sliding_counts(cb, 20)
        ps = np.ones((sa.shape[1], len(ids)))
        for j in range(sa.shape[1]):
            use = np.any(sa[:, j] != 0, 0) | np.any(sb[:, j] != 0, 0)
            if use.any():
                ps[j, use] = mannwhitneyu(sa[:, j][:, use], sb[:, j][:, use], axis=0, method="asymptotic").pvalue
        lo = ps < 1e-4
        hit = (lo[:-2] & lo[1:-1] & lo[2:]).any(0)
    res["highly_active"] = hit
    store.save(out, **res)


# ---- decoding

DECODE = {"vision_contra": "stim", "vision_ipsi": "stim", "action": "move", "choice": "move"}


def decoding_stage(job):
    cfg, s = job
    out = cfg.results / "decoding" / f"{s}.npz"
    if _exists(out):
        return
    b = store.load(cfg.results / "base" / f"{s}.npz")
    data = store.load(cfg.results / "regression" / "data" / f"{s}.npz")
    res = {}
    inc = regression._included(cfg, s, "real") if (cfg.results / "regression/real/full" / f"{s}.npz").exists() else np.zeros(0, bool)
    if inc.any():
        _, rows, trnum, *_ = regression.build_design(b)
        region = b["n_region"][data["ids"]][inc]
        keep = b["kernel_trial"]
        move = decoding_moves(b, store.session_rng(cfg.seed, s, 20))
        cl, cr = b["cl"][keep], b["cr"][keep]
        moved = (b["has_left"] | b["has_right"])[keep]
        right = b["has_right"][keep]
        for key, align in DECODE.items():
            path = cfg.results / "regression" / "real" / f"resid_{key}" / f"{s}.npz"
            resid = store.load(path)["resid"]
            events = b["stim"][keep] if align == "stim" else move
            off = decoding.OFFSETS[align]
            tensor = decoding.event_tensor(resid, rows, trnum, events, off)
            labels = {"vision_contra": cr > 0, "vision_ipsi": cl > 0, "action": moved, "choice": right}[key]
            use = moved if key == "choice" else np.ones(len(labels), bool)
            for r in np.unique(region):
                c = decoding.decode_curve(tensor[use][:, :, region == r], labels[use], off, decoding.TRAIN_WIN[align])
                if c is not None:
                    res[f"{key}|{r}"] = c
    store.save(out, **res)


def decoding_moves(b, rng):
    """Absolute movement time for kept trials; trials without a movement get a pseudo onset."""
    return (b["stim"] + trials.pseudo_onsets(b, rng))[b["kernel_trial"]]


# ---- jPECC

JPECC_GROUPS = {"VIS": VIS, "MB": MIDBRAIN, "Frontal": FRONTAL}
JPECC_PAIRS = (("VIS", "Frontal"), ("VIS", "MB"), ("MB", "Frontal"))
JPECC_WIN = {"stim": (-0.1, 0.3), "move": (-0.3, 0.1)}


def jpecc_stage(job):
    cfg, s = job
    out = cfg.results / "jpecc" / f"{s}.npz"
    if _exists(out):
        return
    b = store.load(cfg.results / "base" / f"{s}.npz")
    mask = neuron_sets(cfg, s, b)["all"]
    groups = {k: np.flatnonzero(mask & np.isin(b["n_region"], v)) for k, v in JPECC_GROUPS.items()}
    res = {}
    if sum(len(v) >= 5 for v in groups.values()) >= 2:
        st, clu = Session(cfg.data, s).spikes()
        ids = np.concatenate([v for v in groups.values()])
        stim = b["stim"][b["kernel_trial"]]
        lo = aligned_rates(st, clu, ids, stim, (-0.2, 0.0), 0.025, dt=0.01)[0].mean((0, 1))
        moved = b["included"] & (b["has_left"] | b["has_right"])
        for ev, times in (("stim", stim), ("move", b["move_abs"][moved])):
            r = aligned_rates(st, clu, ids, times, JPECC_WIN[ev], 0.025, dt=0.01)[0] / (lo + 1)
            pos = {k: np.isin(ids, v) for k, v in groups.items()}
            for a, c in JPECC_PAIRS:
                if len(groups[a]) >= 5 and len(groups[c]) >= 5:
                    rr, pp = jpecc.jpecc(r[:, :, pos[a]], r[:, :, pos[c]])
                    res[f"r|{ev}|{a}|{c}"] = rr.astype(np.float32)
                    res[f"p|{ev}|{a}|{c}"] = pp.astype(np.float32)
    store.save(out, **res)


# ---- ccCP

def cccp_stage(job):
    cfg, s = job
    out = cfg.results / "cccp" / f"{s}.npz"
    if _exists(out):
        return
    b = store.load(cfg.results / "base" / f"{s}.npz")
    ids = np.flatnonzero(neuron_sets(cfg, s, b)["all"])
    cond = trials.condition_ids(b)
    left, right, nogo = b["included"] & b["has_left"], b["included"] & b["has_right"], b["nogo"]
    go = left | right
    use_cp, use_dp = trials.includable(left, right, cond), trials.includable(go, nogo, cond)
    res = {"ids": ids, "cp_ok": use_cp.sum() >= MIN_INCLUDABLE_TRIALS, "dp_ok": use_dp.sum() >= MIN_INCLUDABLE_TRIALS,
           "cp_trials": use_cp.sum(), "dp_trials": use_dp.sum()}
    if len(ids):
        st, clu = Session(cfg.data, s).spikes()
        move = b["stim"] + trials.pseudo_onsets(b, store.session_rng(cfg.seed, s, 30))
        for stat, mask, a, ok in (("cp", use_cp, left, res["cp_ok"]), ("dp", use_dp, go, res["dp_ok"])):
            if not ok:
                continue
            plan = cccp.plan_shuffles(a[mask], cond[mask], store.session_rng(cfg.seed, s, 31 if stat == "cp" else 32))
            for ev, times in (("stim", b["stim"]), ("move", move)):
                win = cccp.WINDOWS[ev]
                c, _ = aligned_counts(st, clu, ids, times[mask], (win[0] - 0.05, win[1] + 0.05), dt=cccp.STEP)
                val, p = cccp.scan(cccp.sliding_counts(c, int(round(cccp.WIDTH / cccp.STEP))), plan)
                res[f"{stat}_{ev}_val"], res[f"{stat}_{ev}_p"] = val, p
            c, _ = aligned_counts(st, clu, ids, b["stim"][mask], (-0.2, 0.0), dt=0.2)
            res[f"{stat}_pre_val"] = cccp.scan(c, plan)[0][0]
    store.save(out, **res)


# ---- engagement

def _pre(st, clu, ids, ev, win):
    if len(ev) == 0:
        return np.zeros((0, len(ids)), np.float32)
    return aligned_counts(st, clu, ids, ev, win, dt=win[1] - win[0])[0][:, 0] / (win[1] - win[0])


def _window_signal(t, v, ev, win):
    a, z = np.searchsorted(t, ev + win[0]), np.searchsorted(t, ev + win[1])
    return np.array([v[i:j].mean() if j > i else np.nan for i, j in zip(a, z)])


def engagement_session(job):
    cfg, s = job
    out = cfg.results / "engagement" / f"{s}.npz"
    if _exists(out):
        return
    b = store.load(cfg.results / "base" / f"{s}.npz")
    ids = np.flatnonzero(neuron_sets(cfg, s, b)["all"])
    if len(ids) == 0:
        store.save(out, ids=ids)
        return
    ses = Session(cfg.data, s)
    st, clu = ses.spikes()
    on, off, _ = ses.moves()
    task = b["stim"][b["included"]]
    ok_t = ~trials.overlaps_movement(task + ANOVA_WIN[0], task + ANOVA_WIN[1], on, off)
    pas = b["passive_times"]
    ok_p = ~trials.overlaps_movement(pas + ANOVA_WIN[0], pas + ANOVA_WIN[1], on, off)
    log_task = np.log10(_pre(st, clu, ids, task[ok_t], ANOVA_WIN).mean(0) + 1)
    log_pas = np.log10(_pre(st, clu, ids, pas[ok_p], ANOVA_WIN).mean(0) + 1)
    f_task = _pre(st, clu, ids, b["stim"], ENGAGEMENT_WIN)
    f_pas = _pre(st, clu, ids, pas, ENGAGEMENT_WIN)
    pt, pv = ses.series(("eye", "area"))
    mt, mv = ses.series(("face", "motionEnergy"))
    store.save(out, ids=ids, region=b["n_region"][ids], log_task=log_task, log_passive=log_pas, f_task=f_task, f_passive=f_pas,
               pupil=_window_signal(pt, pv, b["stim"], ENGAGEMENT_WIN), motion=_window_signal(mt, mv, b["stim"], ENGAGEMENT_WIN))


def engagement_pool(cfg, sessions):
    out = cfg.results / "engagement.npz"
    if _exists(out):
        return
    res = {}
    ses = {}
    for s in sessions:
        e = store.load(cfg.results / "engagement" / f"{s}.npz")
        if len(e["ids"]):
            ses[s] = (e, store.load(cfg.results / "base" / f"{s}.npz"))
    regions = sorted({r for e, _ in ses.values() for r in e["region"].tolist()})
    sig = []
    for r in regions:
        task = np.concatenate([e["log_task"][e["region"] == r] for e, _ in ses.values()])
        pas = np.concatenate([e["log_passive"][e["region"] == r] for e, _ in ses.values()])
        f, p = engagement.nested_anova(task, pas)
        d = task - pas
        res[f"anova|{r}"] = np.array([f, p, d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d)])
        if p < BONFERRONI_REGION:
            sig.append(r)
    res["significant"] = np.array(sig)

    rows = []
    per_ei, per_vec = {}, {}
    tp, gm, nn = {}, {}, {}
    for s, (e, b) in ses.items():
        rng = store.session_rng(cfg.seed, s, 40)
        go, miss = engagement.match_go_miss(b, rng)
        f, fp = e["f_task"], e["f_passive"]
        act = f[b["included"]].mean(0)
        pas = fp.mean(0)
        for r in np.unique(e["region"]):
            m = e["region"] == r
            tp.setdefault(r, []).append((act - pas)[m])
            if len(go):
                gm.setdefault(r, []).append((f[go].mean(0) - f[miss].mean(0))[m])
        if not bool(b["matched"]) or len(go) == 0:
            continue
        m = np.isin(e["region"], sig)
        if not m.any():
            continue
        x = engagement.engagement_vector(f[b["included"]][:, m], fp[:, m])
        ei = f[:, m] @ x
        iti, prev = engagement.previous_trial(b)
        per_ei[s] = ei[go].mean() - ei[miss].mean()
        used = np.r_[go, miss]
        z = engagement.zscore(ei[used])
        pup, mot = engagement.zscore(e["pupil"][used]), engagement.zscore(e["motion"][used])
        for tri, a, c, d in zip(used, z, pup, mot):
            rows.append((a, float(tr_go(b, tri)), c, d, prev[tri], iti[tri], len(per_ei)))
        fa = f[b["included"]]
        inc = b["included"]
        vec_eng = act[m] - pas[m]
        vec_gm = (f[go].mean(0) - f[miss].mean(0))[m]
        others = {"pupil": engagement.split_difference(fa[:, m], e["pupil"][inc]), "motion": engagement.split_difference(fa[:, m], e["motion"][inc]),
                  "iti": engagement.split_difference(fa[:, m], iti[inc])}
        pr = prev[inc]
        others["reward"] = fa[:, m][pr == 1].mean(0) - fa[:, m][pr == 0].mean(0) if (pr == 0).any() and (pr == 1).any() else vec_eng * 0
        cc = lambda u, v: float(np.corrcoef(u, v)[0, 1]) if u.std() > 0 and v.std() > 0 else np.nan
        per_vec[s] = [cc(vec_eng, vec_gm)] + [cc(vec_eng, others[k]) for k in ("pupil", "motion", "reward", "iti")]
    res["ei_diff"] = np.array(list(per_ei.values()))
    res["ei_sessions"] = np.array(list(per_ei.keys()))
    res["trial_table"] = np.array(rows, float) if rows else np.zeros((0, 7))
    res["vector_corr"] = np.array(list(per_vec.values())) if per_vec else np.zeros((0, 5))
    names = sorted(tp)
    res["region_names"] = np.array(names)
    res["task_passive"] = np.array([np.concatenate(tp[r]).mean() for r in names])
    res["go_miss"] = np.array([np.concatenate(gm[r]).mean() if r in gm else np.nan for r in names])
    res["region_n"] = np.array([len(np.concatenate(tp[r])) for r in names])
    store.save(out, **res)


def tr_go(b, t):
    return b["go"][t]


# ---- behavior

def wheel_stage(job):
    cfg, s = job
    out = cfg.results / "wheel" / f"{s}.npz"
    if _exists(out):
        return
    ses = Session(cfg.data, s)
    t, mm = behavior.wheel_mm(ses)
    on, _, kind = ses.moves()
    store.save(out, acc=behavior.direction_decoding(t, mm, on, kind))


def behavior_pool(cfg, sessions):
    out = cfg.results / "behavior.npz"
    if _exists(out):
        return
    import pandas as pd

    tabs, per_subj = [], {}
    res = {}
    for s in sessions:
        b = store.load(cfg.results / "base" / f"{s}.npz")
        m = b["included"]
        per_subj.setdefault(subject_of(s), []).append((b["cl"][m], b["cr"][m], b["choice"][m], b["fb"][m]))
        tabs.append(behavior.rt_table(b, s, subject_of(s)))
    rt = pd.concat(tabs)
    summary = behavior.rt_summary(rt)
    res["rt_chosen"], res["rt_two"], res["rt_mean"], res["rt_sem"] = [summary[k].to_numpy(float) for k in ("chosen", "two", "mean", "sem")]
    if rt["two"].nunique() > 1 and rt["chosen"].nunique() > 1 and rt["subject"].nunique() > 1:
        res["rt_anova_p"] = np.array([behavior.rt_anova(rt).get("C(chosen)", np.nan)])
    maps = {"left": [], "nogo": [], "right": []}
    subj = sorted(per_subj)
    res["subjects"] = np.array(subj)
    for k in subj:
        cl, cr, ch, fb = [np.concatenate(x) for x in zip(*per_subj[k])]
        mp = behavior.choice_map(cl, cr, ch)
        for name in maps:
            maps[name].append(mp[name])
        res[f"params|{k}"] = behavior.fit_psychometric(cl, cr, ch)
        res[f"data|{k}"] = np.c_[cl, cr, ch, fb]
    for name, v in maps.items():
        res[f"map_{name}"] = np.nanmean(v, axis=0)
    store.save(out, **res)


# ---- pooled regression wrapper

def regression_stage(cfg, runner, sessions):
    regression.run_pooled(cfg, sessions, runner.pmap, "real")


def shuffle_stage(cfg, runner, sessions):
    regression.run_pooled(cfg, sessions, runner.pmap, "shuffle", tests=("choice",))


def steps(cfg, sessions, r):
    """The stages in order, as name -> function, for `run` and for the notebook."""
    jobs = [(cfg, s) for s in sessions]
    per_session = lambda fn: (lambda: r.pmap(fn, jobs))
    return dict([
        ("base", per_session(base_stage)),
        ("regions", lambda: regions_stage(cfg, sessions)),
        ("regression_data", per_session(regression_data_stage)),
        ("regression", lambda: regression_stage(cfg, r, sessions)),
        ("shuffle", lambda: shuffle_stage(cfg, r, sessions)),
        ("compare", per_session(regression.compare_session)),
        ("psth", per_session(psth_stage)),
        ("decoding", per_session(decoding_stage)),
        ("jpecc", per_session(jpecc_stage)),
        ("cccp", per_session(cccp_stage)),
        ("engagement_sessions", per_session(engagement_session)),
        ("engagement", lambda: engagement_pool(cfg, sessions)),
        ("wheel", per_session(wheel_stage)),
        ("behavior", lambda: behavior_pool(cfg, sessions)),
    ])


def run(cfg, sessions=None, workers=1, skip=()):
    """Everything up to the figures."""
    sessions = sessions or list_sessions(cfg.data)
    cfg.results.mkdir(parents=True, exist_ok=True)
    r = Runner(cfg, workers)
    todo = {k: v for k, v in steps(cfg, sessions, r).items() if k not in skip}
    try:
        for name, fn in tqdm(todo.items(), desc="analysis", unit="stage", position=0):
            r.stage(name, fn)
    finally:
        r.close()
    return r.timing
