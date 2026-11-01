"""Paper styling: region colors, panels placed in the paper's pixel coordinates, watermark, layout overlays."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from .config import ANALYSIS, PAPER_FIGS

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 1.5, "pdf.fonttype": 42,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})

PALETTE = json.loads((Path(__file__).parent / "paper_palette.json").read_text())  # measured from the paper's figures
HIRES = ANALYSIS / "resources" / "paper"
_line = {k: tuple(v) for k, v in PALETTE["lines"].items()}

CONTRA, IPSI, MISS, GO = _line["contra"], _line["ipsi"], _line["miss"], _line["go"]
STIM_CONTRA, STIM_IPSI = "#8c4a1c", "#1f4e8c"
PASSIVE = "#8a8a8a"
MISS_CONTRA_RASTER = _line["miss_contra_raster"]
RASTER_BG = _line["raster_bg"]
BAR_RED = _line["bar_red"]
HIST_BLUE, HIST_ORANGE = _line["hist_blue"], _line["hist_orange"]
ENGAGE = LinearSegmentedColormap.from_list("eng", PALETTE["cmaps"]["engage"])
JPECC_CMAP = LinearSegmentedColormap.from_list("jp", PALETTE["cmaps"]["jpecc"])
CHOICE_CMAP = LinearSegmentedColormap.from_list("choice", PALETTE["cmaps"]["choice"])
PURPLE_HEAT = LinearSegmentedColormap.from_list("pur", PALETTE["cmaps"]["purple"])
BLUE_WHITE_RED = LinearSegmentedColormap.from_list("bwr2", ["#3b56a5", "#ffffff", "#e51a23"])
PARULA = LinearSegmentedColormap.from_list("parula", ["#352a87", "#0f5cdd", "#1481d6", "#06a4ca", "#2eb7a4", "#87bf77", "#d1bb59", "#fec832", "#f9fb0e"])
REGION_COLORS = {k: tuple(v) for k, v in PALETTE["regions"].items()}
GROUP_COLORS = {"Visual cortex": REGION_COLORS["VISam"], "Frontal/somatomotor cortex": REGION_COLORS["SSp"], "Hippocampus": REGION_COLORS["CA1"], "Basal ganglia": REGION_COLORS["ACB"],
                "Thalamus": REGION_COLORS["PO"], "Midbrain": REGION_COLORS["APN"]}


def nice_top(value):
    """The next round axis limit above value: 1, 2, 4, 5, 6 or 8 times a power of ten."""
    if not np.isfinite(value) or value <= 0:
        return 1.0
    base = 10 ** np.floor(np.log10(value))
    return float(next(m for m in (1, 2, 4, 5, 6, 8, 10) if m * base >= value * 0.999) * base)


def _write(save, path):
    """Save, or say which file is locked: Windows refuses to overwrite a file open in a viewer, and that must not stop a build."""
    try:
        save(path)
    except OSError:
        print(f"could not write {path.name}: close it in any viewer and rebuild")


class Canvas:
    """A figure the size of the paper's PNG, with axes placed in that PNG's pixels (x0, y0, x1, y1, y downward)."""

    INCHES_WIDE = 7.2

    def __init__(self, cfg, name, paper_png=None, scale=1.0, overlay=True, height=None):
        self.cfg, self.name, self.paper, self.scale, self.overlay = cfg, name, paper_png, scale, overlay
        self.key = (paper_png or "").replace(".png", "").replace("extended-data-fig", "ext")
        self.w, self.h = Image.open(PAPER_FIGS / paper_png).size if paper_png else (812, 620)
        self.h = height or self.h
        plt.rcParams["font.size"] = 6.5 * scale
        self.fig = plt.figure(figsize=(self.INCHES_WIDE, self.INCHES_WIDE * self.h / self.w), facecolor="white")
        self.boxes = []

    def ax(self, box, **kw):
        x0, y0, x1, y1 = box
        self.boxes.append(box)
        return self.fig.add_axes([x0 / self.w, 1 - y1 / self.h, (x1 - x0) / self.w, (y1 - y0) / self.h], **kw)

    def text(self, x, y, s, **kw):
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        kw["fontsize"] = kw.get("fontsize", 6.5) * self.scale
        return self.fig.text(x / self.w, 1 - y / self.h, s, **kw)

    def letter(self, s, x, y):
        self.text(x, y, s, fontweight="bold", fontsize=8, ha="left")

    def slot(self, box, label):
        """A panel that has no data behind it (photo, histology, cartoon): the paper's own panel, tagged as such."""
        if not self.paper or not (HIRES / f"{self.key}.png").exists():
            a = self.ax(box)
            a.set_xticks([]), a.set_yticks([])
            a.text(0.5, 0.5, label, transform=a.transAxes, ha="center", va="center", color="#999999", fontsize=5.5)
            return a
        m = PALETTE["maps"][self.key]
        to_hi = lambda v, o: int(round((v + o) * m["scale"]))
        crop = Image.open(HIRES / f"{self.key}.png").convert("RGB").crop((to_hi(box[0], m["ox"]), to_hi(box[1], m["oy"]), to_hi(box[2], m["ox"]), to_hi(box[3], m["oy"])))
        a = self.ax(box)
        a.imshow(np.asarray(crop), aspect="auto", interpolation="antialiased")
        a.set_axis_off()
        a.text(0.995, 0.004, "from the paper", transform=a.transAxes, ha="right", va="bottom", color="#777777", fontsize=3.6, style="italic",
               bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.8))
        return a

    def finish(self, subdir="figures"):
        if self.cfg.synthetic:
            self.fig.text(0.5, 0.5, "SYNTHETIC", color="red", alpha=0.18, fontsize=46, ha="center", va="center", rotation=30, fontweight="bold", zorder=100)
        out = self.cfg.output / subdir
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.name}.png"
        _write(lambda p: self.fig.savefig(p, dpi=300), path)
        _write(self.fig.savefig, out / f"{self.name}.pdf")
        plt.close(self.fig)
        if self.paper and self.overlay:
            self._overlay(path)
        return path

    def _overlay(self, path):
        """Ours blended over the paper's figure, for checking layout by eye."""
        out = self.cfg.output / "overlays"
        out.mkdir(parents=True, exist_ok=True)
        paper = Image.open(PAPER_FIGS / self.paper).convert("RGB").crop((0, 0, self.w, self.h))
        paper = paper.resize((paper.width * 2, paper.height * 2), Image.LANCZOS)
        ours = Image.open(path).convert("RGB").resize(paper.size, Image.LANCZOS)
        _write(Image.blend(paper, ours, 0.55).save, out / f"{self.name}.png")


def clean(ax, left=True, bottom=True):
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.set_yticks([])
    if not bottom:
        ax.set_xticks([])
    return ax


def no_data(ax, label="no neurons in this run"):
    ax.text(0.5, 0.5, label, transform=ax.transAxes, ha="center", va="center", color="#999999", fontsize=5)


def shade(ax, t, mean, sem, color, lw=1.0, alpha=0.3, **kw):
    ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=alpha, lw=0)
    ax.plot(t, mean, color=color, lw=lw, **kw)


def rotated_region_labels(ax, names, fontsize=6.5):
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=55, ha="right", rotation_mode="anchor", fontsize=fontsize)
    for lab, r in zip(ax.get_xticklabels(), names):
        lab.set_color(REGION_COLORS[r])
