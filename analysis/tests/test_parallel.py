"""Worker processes must give the same numbers as one process, so a resumed or parallel run can be trusted."""
from dataclasses import replace

import numpy as np
import pytest

from steinmetz import pipeline, store, synthetic
from steinmetz.config import Config


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    root = tmp_path_factory.mktemp("par")
    names = synthetic.make_dataset(root / "data", names=("Alpha_2020-01-01", "Beta_2020-01-01"), n_trials=130, per_region=14)
    cfg = Config(data=root / "data", results=root / "serial", output=root / "o1", neuron_lists=None, synthetic=True)
    pipeline.run(cfg, names, workers=1)
    par = replace(cfg, results=root / "parallel", output=root / "o2")
    pipeline.run(par, names, workers=2)
    return cfg, par, names


@pytest.mark.parametrize("stage,key", [("base", "n_p"), ("regression/real/full", "ve"), ("regression/real/nested_choice", "ve"), ("cccp", "dp_move_p"), ("decoding", None)])
def test_parallel_matches_serial(both, stage, key):
    a, b, names = both
    for s in names:
        x, y = store.load(a.results / stage / f"{s}.npz"), store.load(b.results / stage / f"{s}.npz")
        assert sorted(x) == sorted(y)
        for k in [key] if key else list(x):
            assert np.allclose(x[k], y[k], rtol=1e-4, atol=1e-5, equal_nan=True), (s, k)
