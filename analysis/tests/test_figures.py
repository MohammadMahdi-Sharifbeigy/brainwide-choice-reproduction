"""Every figure renders from synthetic results and is marked as synthetic."""
import numpy as np
import pytest
from PIL import Image

from figures.build import make_all
from steinmetz import store

FIGS = [f"fig{i}" for i in range(1, 6)] + [f"ext{i}" for i in range(1, 10)]
ASSIGN = ["assignment_fig2abc_rasters", "assignment_fig4e_choice", "assignment_fig5a_miss_passive", "assignment_ext6b_go_miss_passive"]


@pytest.fixture(scope="module")
def made(run):
    cfg, *_ = run
    return cfg, make_all(cfg)


def test_no_figure_failed(made):
    assert made[1] == {}


@pytest.mark.parametrize("name", FIGS)
def test_figure_and_overlay_exist(made, name):
    cfg, _ = made
    assert (cfg.output / "figures" / f"{name}.png").stat().st_size > 10000
    assert (cfg.output / "figures" / f"{name}.pdf").exists()
    assert (cfg.output / "overlays" / f"{name}.png").exists()


@pytest.mark.parametrize("name", ASSIGN)
def test_assignment_figures_exist(made, name):
    assert (made[0].output / "assignment" / f"{name}.png").stat().st_size > 10000


def test_synthetic_watermark_is_drawn(made):
    cfg, _ = made
    img = np.asarray(Image.open(cfg.output / "figures" / "ext3.png").convert("RGB")).astype(int)
    reddish = (img[..., 0] > img[..., 1] + 30) & (img[..., 0] > img[..., 2] + 30)
    assert reddish.mean() > 0.005


def test_report_says_synthetic_and_lists_departures(made):
    text = (made[0].output / "report_vs_paper.md").read_text(encoding="utf8")
    assert "SYNTHETIC DATA" in text
    assert "Known departures" in text
    assert "DP-qualified" in text and "33-session CP subset" in text


def test_examples_can_be_swapped_by_hand(made):
    cfg, _ = made
    assert "fig2a" in store.load_json(cfg.results / "example_neurons.json")
    assert (cfg.results / "example_candidates.csv").exists()
