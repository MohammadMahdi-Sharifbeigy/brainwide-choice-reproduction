"""The notebook must keep covering the whole pipeline, and must not reintroduce an unsafe default."""
import json
import re
from pathlib import Path

import pytest

from steinmetz import pipeline
from steinmetz.config import Config

NB = Path(__file__).resolve().parents[1] / "full_analysis.ipynb"
FIGS = {f"fig{i}" for i in range(1, 6)} | {f"ext{i}" for i in range(1, 10)} | {"assignment", "report"}


@pytest.fixture(scope="module")
def source():
    nb = json.loads(NB.read_text(encoding="utf8"))
    return "".join("".join(c["source"]) + "\n" for c in nb["cells"] if c["cell_type"] == "code")


def test_notebook_runs_every_stage(source):
    wanted = set(pipeline.steps(Config(), [], pipeline.Runner(Config(), 1)))
    called = set()
    for args in re.findall(r"^run\((.*)\)$", source, re.M):
        called |= set(re.findall(r'"(\w+)"', args))
    assert called == wanted


def test_notebook_builds_every_figure(source):
    called = set()
    for args in re.findall(r"figure\((.*)\)$", source, re.M):
        called |= set(re.findall(r'"(\w+)"', args))
    called |= set(re.findall(r'make_all\(cfg, \{"(\w+)"\}\)', source))
    assert called == FIGS


def test_notebook_defaults_to_one_session(source):
    assert 'RUN_MODE = "session"' in source
