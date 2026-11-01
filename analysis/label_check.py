"""Bounded region-label check from the design: are labels read from the atlas closer to Extended Data Table 1 than the released ones?

Reads each channel's CCF position in the atlas, applies the same rules to both label sets, and prints the error per set.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from steinmetz import atlas, trials  # noqa: E402
from steinmetz.config import Config, LOCALIZED_NEURONS, TABLE1_TOTAL  # noqa: E402
from steinmetz.data import Session, list_sessions, subject_of  # noqa: E402
from steinmetz.regions import atlas_labels, kept_regions, neuron_regions, region_counts, structure_tree  # noqa: E402


def counts(per, subjects):
    kept = kept_regions(per, subjects, 2)
    out = {}
    for s, regs in kept.items():
        for r in regs:
            out[r] = out.get(r, 0) + per[s][r]
    return out


def score(c):
    errors = {r: c.get(r, 0) / n - 1 for r, n in TABLE1_TOTAL.items()}
    return sum(abs(c.get(r, 0) - n) for r, n in TABLE1_TOTAL.items()), sum(abs(v) > 0.02 for v in errors.values()), errors


def main():
    cfg = Config()
    vol = atlas.volume()
    _, by_id = structure_tree()
    released, derived, subjects = {}, {}, {}
    for s in list_sessions(cfg.data):
        ses = Session(cfg.data, s)
        tr = trials.build_trials(ses)
        st, clu = ses.spikes()
        ev = tr["stim"][tr["kernel_trial"]]
        lo, hi = np.searchsorted(st, ev), np.searchsorted(st, ev + 0.4)
        ann = ses.a("clusters._phy_annotation")
        n = np.zeros(len(ann))
        for a, b in zip(lo, hi):
            n += np.bincount(clu[a:b], minlength=len(ann))[: len(ann)]
        base = (ann >= 2) & (n / (0.4 * len(ev)) >= 0.1)
        _, reg = neuron_regions(ses)
        released[s] = region_counts(reg, base)
        derived[s] = region_counts(atlas_labels(ses, vol, by_id)[ses.peak_channel()], base)
        subjects[s] = subject_of(s)
    for name, per in (("released", released), ("atlas", derived)):
        c = counts(per, subjects)
        total_err, off, err = score(c)
        print(f"{name}: total {sum(c.get(r, 0) for r in TABLE1_TOTAL)} vs {LOCALIZED_NEURONS}; sum |error| {total_err}; regions over 2% off: {off} of 42")
        print("   ", {r: f"{c.get(r, 0)} ({100 * err[r]:+.1f}%)" for r in ("VISp", "LGd", "LP", "ZI", "VPL", "MOs")})


if __name__ == "__main__":
    main()
