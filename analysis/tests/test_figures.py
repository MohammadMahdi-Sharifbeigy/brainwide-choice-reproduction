"""Every figure renders from synthetic results and is marked as synthetic."""
import warnings

import numpy as np
import pytest
from PIL import Image

from figures.build import make_all
from figures.ext_figs import column_stats
from steinmetz import store
from steinmetz.config import Config
from steinmetz.style import Canvas

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
    assert (cfg.results / "example_scores.csv").exists()


def test_column_stats_skips_empty_bins_without_warning():
    values = np.array([[np.nan, 1.0, 2.0, np.nan], [np.nan, 3.0, 2.0, 5.0], [np.nan, 5.0, 2.0, 7.0], [np.nan, 7.0, np.nan, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mean, se, ok = column_stats(values)
    assert ok.tolist() == [False, True, True, False]
    assert np.isnan(mean[0]) and np.isnan(mean[3])
    assert mean[1] == pytest.approx(4) and se[1] == pytest.approx(np.std([1, 3, 5, 7]) / 2)
    assert mean[2] == 2 and se[2] == 0


def test_a_locked_output_does_not_stop_a_figure(tmp_path, capsys):
    cfg = Config(output=tmp_path, results=tmp_path, synthetic=True)
    (tmp_path / "overlays" / "locked.png").mkdir(parents=True)  # a path that cannot be written, like a file open in a viewer
    Canvas(cfg, "locked", "fig2.png").finish()
    assert (tmp_path / "figures" / "locked.png").exists()
    assert "could not write locked.png" in capsys.readouterr().out
