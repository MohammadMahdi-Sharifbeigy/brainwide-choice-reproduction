"""Assignment figures: the four plot types for VISp, LGd and LP, in the paper's panel style."""
from steinmetz.config import ASSIGNMENT_REGIONS, VE_THRESHOLD
from steinmetz.results import PSTH_T  # noqa: F401
from steinmetz.style import GO, MISS, PASSIVE, REGION_COLORS, Canvas

from .common import choice_trace, raster_column
from .main_figs import _ex, rate_panel

COLUMNS = ((122, 287), (359, 523), (604, 769))  # the paper's raster columns
PANELS_3 = ((60, 280), (320, 540), (580, 800))


def assignment_2(res, chosen):
    cv = Canvas(res.cfg, "assignment_fig2abc_rasters", "fig2.png", height=445)
    centers = {}
    for i, (r, (x0, x1)) in enumerate(zip(ASSIGNMENT_REGIONS, COLUMNS)):
        s, c = _ex(chosen, f"assign_raster_{r}")
        got = raster_column(cv, res, s, c, x0, x1, f"{r} neuron", REGION_COLORS[r], "abc"[i], labels=i == 0)
        centers = centers or got
    if centers:
        for t, k1, k2 in (("Correct\nchoices", "contra_correct", "ipsi_correct"), ("Misses", "contra_miss", "ipsi_miss"), ("Passive", "contra_passive", "ipsi_passive"),
                          ("Correct\nchoices", "move_contra_correct", "move_ipsi_correct")):
            cv.text(38, (centers[k1] + centers[k2]) / 2, t, style="italic", fontsize=7)
    return cv.finish("assignment")


def assignment_4e(res, chosen):
    cv = Canvas(res.cfg, "assignment_fig4e_choice", "fig4.png", overlay=False, height=560)
    rows = (("contra", 100, 255), ("ipsi", 305, 460))
    cols = ((120, 320), (360, 560), (600, 800))
    for label, y0, y1 in rows:
        for r, (x0, x1) in zip(ASSIGNMENT_REGIONS, cols):
            e = chosen.get(f"assign_4e_{r}_{label}")
            a = cv.ax((x0, y0, x1, y1))
            choice_trace(a, res, e["session"] if e else None, e["cluster"] if e else None)
            cv.text((x0 + x1) / 2, y0 - 22, r, color=REGION_COLORS[r], fontweight="bold", fontsize=10)
            if e:
                passed = "passes" if e["choice_ve"] > VE_THRESHOLD else "below"
                cv.text((x0 + x1) / 2, y0 - 8, f"Choice var. expl. {round(100 * e['choice_ve'], 1) + 0.0:.1f}% ({passed} 2%)", fontsize=6)
        cv.text(62, (y0 + y1) / 2, f"Strongest\n{label}\npreference", style="italic", fontsize=7)
    cv.text(30, 275, "Firing rate (sp/s)", rotation=90, fontsize=7)
    cv.text(460, 490, "Time from move (s)", fontsize=7)
    cv.text(140, 40, "Choice:", fontsize=7), cv.text(180, 40, "Contra", color="#f08a24", fontsize=7), cv.text(215, 40, "Ipsi", color="#3f7cc4", fontsize=7)
    cv.text(431, 540, "Separated curves can come from the stimulus, not choice: contra choices mostly follow contra stimuli.", fontsize=6, color="#555555")
    return cv.finish("assignment")


def assignment_5a(res):
    cv = Canvas(res.cfg, "assignment_fig5a_miss_passive", "fig5.png", overlay=False, height=500)
    for r, (x0, x1) in zip(ASSIGNMENT_REGIONS, PANELS_3):
        rate_panel(cv, res, (x0, 100, x1, 400), r, (("contra_miss", MISS), ("contra_passive", PASSIVE)))
        cv.text((x0 + x1) / 2, 80, r, color=REGION_COLORS[r], fontweight="bold", fontsize=11)
    cv.text(25, 250, "Firing rate (sp/s)", rotation=90, fontsize=8)
    cv.text(424, 445, "Time from stimulus onset (s)", fontsize=8)
    cv.text(350, 480, "Task, miss trial", color=MISS, fontsize=8), cv.text(500, 480, "Passive presentation", color=PASSIVE, fontsize=8)
    return cv.finish("assignment")


def assignment_6b(res):
    cv = Canvas(res.cfg, "assignment_ext6b_go_miss_passive", "extended-data-fig6.png", overlay=False, height=500)
    for r, (x0, x1) in zip(ASSIGNMENT_REGIONS, ((40, 195), (235, 390), (430, 585))):
        rate_panel(cv, res, (x0, 100, x1, 400), r, (("contra_correct", GO), ("contra_miss", MISS), ("contra_passive", PASSIVE)))
        cv.text((x0 + x1) / 2, 80, r, color=REGION_COLORS[r], fontweight="bold", fontsize=11)
    cv.text(15, 250, "Firing rate (sp/s)", rotation=90, fontsize=8)
    cv.text(314, 445, "Time from stimulus onset (s)", fontsize=8)
    cv.text(160, 480, "Task, Go trial", color=GO, fontsize=8), cv.text(310, 480, "Task, Miss trial", color=MISS, fontsize=8)
    cv.text(470, 480, "Passive presentation", color=PASSIVE, fontsize=8)
    return cv.finish("assignment")


def all_assignment(res, chosen):
    return [assignment_2(res, chosen), assignment_4e(res, chosen), assignment_5a(res), assignment_6b(res)]
