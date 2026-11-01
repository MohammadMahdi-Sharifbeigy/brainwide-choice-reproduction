import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def run(tmp_path_factory):
    """The whole pipeline once, on synthetic sessions written in the real file format."""
    from steinmetz import pipeline, synthetic
    from steinmetz.config import Config

    root = tmp_path_factory.mktemp("syn")
    names = synthetic.make_dataset(root / "data")
    cfg = Config(data=root / "data", results=root / "results", output=root / "output", neuron_lists=None, synthetic=True)
    timing = pipeline.run(cfg, names, workers=1)
    return cfg, names, timing
