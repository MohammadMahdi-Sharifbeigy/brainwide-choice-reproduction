"""The whole pipeline on synthetic sessions written in the real file format."""
import numpy as np
import pytest

from steinmetz import examples, pipeline, regression, store
from steinmetz.config import Config
from steinmetz.data import Session
from steinmetz.results import Results
from steinmetz.trials import build_trials


def test_synthetic_trial_sets_are_populated(run):
    cfg, names, _ = run
    tr = build_trials(Session(cfg.data, names[0]))
    assert tr["has_left"].sum() > 20 and tr["has_right"].sum() > 20
    assert tr["has_none"].sum() > 3 and tr["miss"].sum() > 10


def test_every_stage_ran_and_reported_time(run):
    *_, timing = run
    assert {"base", "regression", "cccp", "engagement", "behavior"} <= set(timing)


def test_rerun_recomputes_nothing(run):
    cfg, names, _ = run
    t = pipeline.run(cfg, names, workers=1)
    assert sum(v["seconds"] for v in t.values()) < 20


def test_resume_keeps_the_measured_timing(run):
    cfg, names, timing = run
    saved = store.load_json(cfg.results / "timing.json")["stages"]
    assert saved["regression"]["seconds"] >= timing["regression"]["seconds"]
    pipeline.run(cfg, names, workers=1)
    after = store.load_json(cfg.results / "timing.json")["stages"]
    assert {k: v["seconds"] for k, v in after.items()} == {k: v["seconds"] for k, v in saved.items()}


def test_half_written_file_does_not_count(tmp_path):
    out = tmp_path / "x.npz"
    store.save(out, a=np.arange(3))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.npz"]


def test_choice_kernels_keep_the_planted_sign(run):
    cfg, names, _ = run
    hits = []
    for s in names:
        f = store.load(cfg.results / "regression/real/full" / f"{s}.npz")
        n = store.load(cfg.results / "regression/real/nested_choice" / f"{s}.npz")
        d = store.load(cfg.results / "regression/data" / f"{s}.npz")
        b = store.load(cfg.results / "base" / f"{s}.npz")
        inc = f["ve"] >= 0.02
        sel = np.flatnonzero(inc)[n["ve"] > 0.02]
        hits.append(np.isin(b["n_region"][d["ids"][sel]], ["SCm", "MRN", "SNr", "ZI"]) * regression.prefers_contra(f["K"][:, sel]))
    all_hits = np.concatenate(hits)
    assert all_hits.sum() >= 1


def test_peak_scales_with_the_largest_session():
    assert pipeline.scaled_peak(0.83, 4.7e6, 24.07e6) == pytest.approx(0.25 + 0.58 * 24.07 / 4.7)
    assert pipeline.scaled_peak(0.2, 1, 5) == pytest.approx(0.25)


def test_progress_label_names_stage_step_and_kernel():
    r = pipeline.Runner(Config(), 1)
    r.current = "regression"

    def _nested(job):
        return job

    assert r._label(_nested, [(0, 1, 2, "choice")]) == "regression | nested | choice"
    assert r.pmap(_nested, [(0, 1, 2, "choice")]) == [(0, 1, 2, "choice")]


def test_neuron_search_survives_a_region_with_no_candidates(run):
    cfg, *_ = run
    res = Results(cfg)
    table = res.regression()
    empty = table[table["region"] == "not-a-region"]
    assert examples.score_slot(res, empty, "fig2a", set()).empty
    assert examples.contrast_slot(res, empty, "ext4_visp", set()).empty
    assert examples.ext5_slot(res, empty).empty


def test_workers_shrink_to_fit_memory():
    assert pipeline.auto_workers(1e6) == 1
    assert 1 <= pipeline.auto_workers(0.5, cores=12) <= 11
