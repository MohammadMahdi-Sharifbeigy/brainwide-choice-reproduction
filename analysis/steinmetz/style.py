"""Paper styling: region colors, panels placed in the paper's pixel coordinates, watermark, layout overlays."""
import colorsys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from .config import PAPER_FIGS

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 1.5, "pdf.fonttype": 42,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})

CONTRA, IPSI = "#f08a24", "#3f7cc4"
STIM_CONTRA, STIM_IPSI = "#8c4a1c", "#1f4e8c"
MISS, PASSIVE, GO = "#29abe2", "#8a8a8a", "#e8562a"
RASTER_BG = "#ebecee"
ENGAGE = LinearSegmentedColormap.from_list("eng", ["#3b56a5", "#e8e6f2", "#f4cdbd", "#c21f26"])
BLUE_WHITE_RED = LinearSegmentedColormap.from_list("bwr2", ["#3b56a5", "#ffffff", "#e51a23"])
JPECC_CMAP = LinearSegmentedColormap.from_list("jp", ["#29abe2", "#1a3a70", "#000000", "#c7141a", "#f68b1f", "#fff200"])
PURPLE_HEAT = LinearSegmentedColormap.from_list("pur", ["#ffffff", "#7b64b0", "#2a1a58", "#000000"])


def _ramp(hue, names, sat=(0.75, 0.42), val=(0.25, 0.97)):
    n = len(names)
    return {r: colorsys.hsv_to_rgb(hue, sat[0] + (sat[1] - sat[0]) * i / max(n - 1, 1), val[0] + (val[1] - val[0]) * i / max(n - 1, 1))
            for i, r in enumerate(names)}


# hues from the authors' 'gradient' region groups; the ramp stands in for their myCopper, which is not in the repository
REGION_COLORS = {
    **_ramp(0.1, ("ACA", "MOs", "PL", "ILA", "ORB", "MOp", "SSp")),
    **_ramp(0.72, ("CP", "GPe", "SNr", "ACB", "LS")),
    **_ramp(0.0, ("SCs", "SCm", "MRN", "APN", "PAG")),
    **_ramp(0.6, ("VISp", "VISl", "VISpm", "VISam", "VISrl", "VISa")),
    **_ramp(0.3, ("LGd", "LP", "LD", "POL", "MD", "VPL", "PO", "VPM", "RT", "MG")),
    **_ramp(0.89, ("DG", "CA3", "CA1", "SUB", "POST")),
    "RSP": (0, 0, 0), "ZI": (0.25, 0.25, 0.25), "OLF": (0.5, 0.5, 0.5), "BLA": (0.75, 0.75, 0.75),
}


class Canvas:
    """A figure the size of the paper's PNG, with axes placed in that PNG's pixels (x0, y0, x1, y1, y downward)."""

    INCHES_WIDE = 7.2

    def __init__(self, cfg, name, paper_png=None, scale=1.0, overlay=True, height=None):
        self.cfg, self.name, self.paper, self.scale, self.overlay = cfg, name, paper_png, scale, overlay
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
        """Labelled empty panel where the paper shows a photo, histology or a cartoon."""
        a = self.ax(box)
        a.set_xticks([]), a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True), sp.set_color("#bbbbbb"), sp.set_linestyle(":")
        a.text(0.5, 0.5, label, transform=a.transAxes, ha="center", va="center", color="#999999", fontsize=5.5, wrap=True)
        return a

    def finish(self, subdir="figures"):
        if self.cfg.synthetic:
            self.fig.text(0.5, 0.5, "SYNTHETIC", color="red", alpha=0.18, fontsize=46, ha="center", va="center", rotation=30, fontweight="bold", zorder=100)
        out = self.cfg.output / subdir
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.name}.png"
        self.fig.savefig(path, dpi=300)
        self.fig.savefig(out / f"{self.name}.pdf")
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
        Image.blend(paper, ours, 0.55).save(out / f"{self.name}.png")


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
