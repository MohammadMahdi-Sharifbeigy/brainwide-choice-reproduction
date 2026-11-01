"""Read saved stage outputs into tables for figures and the report."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import regression, store
from .config import DT, VE_THRESHOLD
from .data import Session, subject_of
from .pipeline import PSTH_WIN, neuron_sets

PSTH_T = PSTH_WIN[0] + DT / 2 + np.arange(int(round((PSTH_WIN[1] - PSTH_WIN[0]) / DT))) * DT
NESTED = tuple(regression.TESTS)


class Results:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sessions = sorted(p.stem for p in (cfg.results / "base").glob("*.npz"))
        self.kept = store.load_json(cfg.results / "regions.json")["kept"]

    @lru_cache(maxsize=4)
    def base(self, s):
        return store.load(self.cfg.results / "base" / f"{s}.npz")

    @lru_cache(maxsize=2)
    def spikes(self, s):
        return Session(self.cfg.data, s).spikes()

    def spike_times(self, s, cluster):
        st, clu = self.spikes(s)
        return st[clu == cluster]

    @lru_cache(maxsize=1)
    def neurons(self):
        """All neurons in kept regions, with z-scores per condition."""
        rows = []
        for s in self.sessions:
            b = self.base(s)
            sets = neuron_sets(self.cfg, s, b)
            ids = np.flatnonzero(sets["all"])
            if len(ids) == 0:
                continue
            p = store.load(self.cfg.results / "psth" / f"{s}.npz")
            d = {"session": s, "subject": subject_of(s), "cluster": ids, "region": b["n_region"][ids], "responsive": sets["responsive"][ids],
                 "high": p["highly_active"], "matched": bool(b["matched"])}
            d.update({k: p[k] for k in p if k.startswith("z_")})
            rows.append(pd.DataFrame(d))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["region", "session", "cluster"])

    def stacked(self, key, mask_col=None, matched_only=False):
        """Per-neuron arrays stacked across sessions: returns rows x time and the regions."""
        arrs, regs = [], []
        for s in self.sessions:
            p = store.load(self.cfg.results / "psth" / f"{s}.npz")
            if len(p["ids"]) == 0 or key not in p or p[key].size == 0:
                continue
            b = self.base(s)
            if matched_only and not bool(b["matched"]):
                continue
            ids = p["ids"]
            use = np.ones(len(ids), bool)
            if mask_col == "responsive":
                use = neuron_sets(self.cfg, s, b)["responsive"][ids]
            elif mask_col == "high":
                use = p["highly_active"]
            arrs.append(p[key][:, use].T if p[key].ndim == 2 else p[key][use])
            regs.append(b["n_region"][ids][use])
        if not arrs:
            return np.zeros((0, 1)), np.array([], dtype=str)
        return np.concatenate(arrs), np.concatenate(regs)

    def region_rate(self, key, region, mask_col=None, matched_only=True):
        x, r = self.stacked(key, mask_col, matched_only)
        x = x[r == region]
        if len(x) == 0:
            return None
        return np.nanmean(x, 0), np.nanstd(x, 0) / np.sqrt(len(x)), len(x)

    def normalized(self, key, mask_col="responsive", matched_only=False):
        x, r = self.stacked(key, mask_col, matched_only)
        b, _ = self.stacked("baseline", mask_col, matched_only)
        return (x - b[:, None]) / (b[:, None] + 0.5), r

    @lru_cache(maxsize=1)
    def regression(self):
        """Regression neurons with variance explained, nested results and movement-onset rates."""
        rows = []
        root = self.cfg.results / "regression" / "real"
        for s in self.sessions:
            f = root / "full" / f"{s}.npz"
            if not f.exists():
                continue
            full = store.load(f)
            ids = store.load(self.cfg.results / "regression" / "data" / f"{s}.npz")["ids"]
            if len(ids) == 0:
                continue
            b = self.base(s)
            d = pd.DataFrame({"session": s, "cluster": ids, "region": b["n_region"][ids], "ve": full["ve"], "index": np.arange(len(ids))})
            inc = full["ve"] >= VE_THRESHOLD
            for key in NESTED:
                v = np.full(len(ids), np.nan)
                n = store.load(root / f"nested_{key}" / f"{s}.npz")["ve"]
                v[inc] = n
                d[key] = v
            contra, ipsi = regression.rates_at_move(full["K"])
            d["contra"], d["ipsi"] = contra, ipsi
            d["prefers_contra"] = regression.prefers_contra(full["K"])
            rows.append(d)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["region", "ve", *NESTED, "session", "cluster", "contra", "ipsi", "prefers_contra", "index"])

    def shuffle_ve(self):
        out = []
        for s in self.sessions:
            p = self.cfg.results / "regression" / "shuffle" / "nested_choice" / f"{s}.npz"
            if p.exists():
                out.append(store.load(p)["ve"])
        return np.concatenate(out) if out else np.zeros(0)

    def kernels(self, s):
        return store.load(self.cfg.results / "regression" / "real" / "full" / f"{s}.npz")

    def model_traces(self, s, cluster):
        """Data, cross-validated prediction and predictions without each kernel for one neuron."""
        d = store.load(self.cfg.results / "regression" / "data" / f"{s}.npz")
        i = int(np.flatnonzero(d["ids"] == cluster)[0])
        full = self.kernels(s)
        add = d["means"][i] if "means" in d else 0.0  # back to sp/s for plotting
        out = {"F": d["F"][:, i] + add, "cv": full["cvpred"][:, i] + add}
        inc = full["ve"] >= VE_THRESHOLD
        if inc[i]:
            j = int(np.cumsum(inc)[i] - 1)
            for key in NESTED:
                r = store.load(self.cfg.results / "regression" / "real" / f"resid_{key}" / f"{s}.npz")["resid"]
                out[f"no_{key}"] = d["F"][:, i] + add - r[:, j]
        return out

    def stage(self, name):
        p = self.cfg.results / name
        return store.load(p) if Path(p).exists() else {}
