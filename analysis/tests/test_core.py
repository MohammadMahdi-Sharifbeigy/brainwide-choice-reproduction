"""Core tests: smoothing, trial rules and the facts checked against the real data."""
import numpy as np
import pytest

from steinmetz import rates, trials
from steinmetz.config import AUTHORS_REPO, INCLUDED_TRIALS, LOCALIZED_NEURONS, Config
from steinmetz.data import Session, list_sessions, subject_of
from steinmetz.regions import kept_regions, neuron_regions, region_counts, to_paper_region

cfg = Config()
real = pytest.mark.skipif(not cfg.data.exists(), reason="real data not present")
LISTS = AUTHORS_REPO / "behavior" / "inclusionData" / "trials"


def test_half_gaussian_is_causal_and_normalized():
    x = np.zeros(200)
    x[100] = 1
    y = rates.smooth(x, 0.025)
    assert y[:100].max() == 0
    assert y[100] == y.max()
    assert y.sum() == pytest.approx(1, abs=1e-6)


def test_first_move_and_quirk():
    stim = np.array([10.0, 20.0, 30.0])
    fb = np.array([11.0, 21.5, 31.0])
    on = np.array([10.2, 20.8, 35.0])
    ty = np.array([1, 2, 1])
    t, k, none = trials.find_moves(stim, fb, on, ty)
    assert t[0] == pytest.approx(0.2) and k[0] == 1
    assert none.tolist() == [False, True, False]  # third trial is still and, as in their code, not in hasNone
    assert np.isnan(t[2])


def test_overlap_catches_a_move_spanning_the_window():
    on, off = np.array([1.0, 5.0]), np.array([4.0, 6.0])
    out = trials.overlaps_movement(np.array([2.0, 4.1, 0.0]), np.array([3.0, 4.5, 0.5]), on, off)
    assert out.tolist() == [True, False, False]


def test_match_contrast_equal_counts_and_floor():
    rng = np.random.default_rng(0)
    con = np.array([0.25] * 10 + [0.5] * 6 + [1.0] * 4)
    a = np.ones(20, bool)
    b = np.zeros(20, bool)
    b[[0, 1, 10, 11, 12, 16, 17, 18, 19]] = True
    got = trials.match_contrast({"a": (a, con), "b": (b, con)}, rng, min_trials=9)
    assert len(got["a"]) == len(got["b"]) == 9
    assert sorted(con[got["a"]]) == sorted(con[got["b"]])
    assert trials.match_contrast({"a": (a, con), "b": (b, con)}, rng, min_trials=10) is None


def test_hierarchy_merge():
    assert to_paper_region("SCig") == "SCm"
    assert to_paper_region("SCsg") == "SCs"
    assert to_paper_region("LSr") == "LS"
    assert to_paper_region("ORBm") == "ORB"
    assert to_paper_region("VISp") == "VISp"
    assert to_paper_region("root") == ""
    assert to_paper_region("TH") == ""
    assert to_paper_region("not-a-label") == ""


def test_kept_regions_rules():
    per = {"a_1": {"X": 13, "Y": 12}, "b_1": {"X": 20}, "a_2": {"Z": 30}}
    got = kept_regions(per, {"a_1": "a", "b_1": "b", "a_2": "a"}, 2)
    assert got == {"a_1": {"X"}, "b_1": {"X"}, "a_2": set()}


@real
def test_authors_trial_lists_on_all_sessions():
    total = 0
    for s in list_sessions(cfg.data):
        tr = trials.build_trials(Session(cfg.data, s))
        for key, f in (("has_left", "hasLeft"), ("has_right", "hasRight"), ("has_none", "hasNone")):
            want = np.load(LISTS / f"{s}_{f}.npy").ravel().astype(bool)
            assert (tr[key] == want).all(), (s, key)
        total += tr["included"].sum()
    assert total == INCLUDED_TRIALS


@real
def test_sign_conventions():
    tr = trials.build_trials(Session(cfg.data, "Richards_2017-10-31"))
    rewarded = tr["fb"] == 1
    assert (tr["choice"][rewarded & (tr["cl"] > tr["cr"])] == 1).all()  # +1 is Left
    assert (tr["choice"][rewarded & (tr["cr"] > tr["cl"])] == -1).all()
    assert (tr["choice"][tr["has_left"]] == 1).mean() > 0.9  # wheel type 1 agrees with a Left choice
    assert (tr["choice"][tr["has_right"]] == -1).mean() > 0.9


@real
def test_session_rules_give_the_paper_counts():
    kept, under = [], []
    for s in list_sessions(cfg.data):
        tr = trials.build_trials(Session(cfg.data, s))
        if trials.matched_sets(tr, np.random.default_rng(0)) is not None:
            kept.append(s)
        left, right = tr["included"] & tr["has_left"], tr["included"] & tr["has_right"]
        if trials.includable(left, right, trials.condition_ids(tr)).sum() < 10:
            under.append(s)
    assert len(kept) == 34
    assert under == ["Cori_2016-12-14", "Moniz_2017-05-18", "Muller_2017-01-08", "Muller_2017-01-09",
                     "Radnitz_2017-01-11", "Tatum_2017-12-09"]


@real
def test_table1_total_within_two_percent():
    per, subjects = {}, {}
    for s in list_sessions(cfg.data):
        ses = Session(cfg.data, s)
        tr = trials.build_trials(ses)
        st, clu = ses.spikes()
        events = tr["stim"][tr["kernel_trial"]]
        lo, hi = np.searchsorted(st, events), np.searchsorted(st, events + 0.4)
        ann = ses.a("clusters._phy_annotation")
        n = np.zeros(len(ann))
        for a, b in zip(lo, hi):
            n += np.bincount(clu[a:b], minlength=len(ann))[: len(ann)]
        _, reg = neuron_regions(ses)
        base = (ann >= 2) & (n / (0.4 * len(events)) >= 0.1)
        per[s] = region_counts(reg, base)
        subjects[s] = subject_of(s)
    kept = kept_regions(per, subjects, 2)
    total = sum(per[s][r] for s in per for r in kept[s])
    assert abs(total / LOCALIZED_NEURONS - 1) < 0.02
