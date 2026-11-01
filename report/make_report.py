"""Build report.pdf: copy the figures, write the tables from saved results, run XeLaTeX twice. Runs no analysis."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / "analysis"
sys.path.insert(0, str(ANALYSIS))

from steinmetz import store  # noqa: E402
from steinmetz.config import ASSIGNMENT_REGIONS, TABLE1_TOTAL, Config  # noqa: E402
from steinmetz.results import Results  # noqa: E402

FIGURES = [f"fig{i}" for i in range(1, 6)] + [f"ext{i}" for i in range(1, 10)]
TITLES = {"fig1": "Figure 1", "fig2": "Figure 2", "fig3": "Figure 3", "fig4": "Figure 4", "fig5": "Figure 5", **{f"ext{i}": f"Extended Data Figure {i}" for i in range(1, 10)}}
ASSIGN = ("assignment_fig2abc_rasters", "assignment_fig4e_choice", "assignment_fig5a_miss_passive", "assignment_ext6b_go_miss_passive")

# Each panel: letter, agreement with the paper, note. Same/Close/Partial describe the result; "From the paper" has no data behind it.
SAME, CLOSE, PARTIAL, NODATA = "Same", "Close", "Partial", "From the paper"
PANELS = {
    "fig1": [
        ("a", NODATA, "Task cartoon."),
        ("b", NODATA, "Trial timeline cartoon."),
        ("c", CLOSE, "Same choice pattern for every contrast pair. Colormap measured from the paper."),
        ("d", CLOSE, "Reaction times of 200 to 240 ms, slower with two stimuli and with low contrast. ANOVA p = 2.6e-10 (paper p < 1e-4)."),
        ("e", NODATA, "Setup cartoon."),
        ("f", NODATA, "Photograph."),
        ("g", NODATA, "Histology."),
        ("h", CLOSE, "Probe tracks from each channel's CCF position, with the same spread. Mouse colors are approximate."),
        ("i", CLOSE, "Same map and labels. Counts are ours: 22,210 localized neurons against 22,458, with 23 of 42 regions more than 2\\% off."),
        ("j", PARTIAL, "A real trial, from a session holding six of the panel's seven regions. It has no TH label, so the layout differs."),
    ],
    "fig2": [
        ("a", CLOSE, "SUB neuron firing before movement to both sides, with no stimulus response. Row heights differ, so this is probably not the same cell."),
        ("b", CLOSE, "VISpm neuron with a contra stimulus burst near 0.1 s and later movement firing. Curves and row heights independently point to Cori\\_2016-12-17."),
        ("c", CLOSE, "ZI neuron firing before contra choices only."),
        ("d", CLOSE, "Visual areas first, then spread to most regions."),
        ("e", CLOSE, "Most regions active, with visual areas late."),
        ("f", CLOSE, "Activity stays in the visual pathway."),
        ("g", CLOSE, "Almost no activity."),
        ("h", PARTIAL, "Visual pathway only, as in the paper, but stronger than Miss in visual cortex where the paper's is weaker."),
    ],
    "fig3": [
        ("a", CLOSE, "Model fits and the loss of fit without the Vision kernel both match. Vision peak about 50 sp/s."),
        ("b", CLOSE, "Good fit is lost without the Action kernel, as in the paper. Peak rate is about 40 sp/s against 80."),
        ("c", SAME, "Same distribution across regions."),
        ("d", CLOSE, "Same regions. VISp 29.4\\% against about 30\\%."),
        ("e", CLOSE, "Broad action coding across regions. APN 23.4\\% against about 25\\%."),
        ("f", CLOSE, "Same map; the color steps are approximated."),
        ("g", CLOSE, "Same map; the color steps are approximated."),
    ],
    "fig4": [
        ("a", CLOSE, "VISp fits without Choice; ZI, MRN and CP need it."),
        ("b", CLOSE, "Same regions. ZI 3.8\\% against about 6\\%, SCm 2.6\\% against about 3\\%."),
        ("c", PARTIAL, "Similar time course and no time-by-group interaction (p = 1.0). However, the group main effect is significant here (p = 7e-4) and not in the paper. Frontal n = 26 against 29."),
        ("d", CLOSE, "VIS leads frontal, and midbrain and frontal show no lag. VIS panel uses n = 17 against 10."),
        ("e", CLOSE, "All eight panels show the same shape and scale as the paper. Error 4\\% to 26\\% of the panel's peak."),
        ("f", CLOSE, "45 of 46 midbrain choice neurons prefer contra, against 53 of 54."),
        ("g", CLOSE, "11 of 33 forebrain choice neurons prefer ipsi, against 19 of 48."),
        ("h", CLOSE, "Same bilateral and unilateral regions."),
    ],
    "fig5": [
        ("a", CLOSE, "Pre-stimulus Miss minus Passive has the paper's sign in all eight regions, though GPe and SCm are within their error. Contrast matched."),
        ("b", PARTIAL, "Mostly positive as in the paper, but several regions differ in size."),
        ("c", CLOSE, "Neocortex and visual thalamus below passive, subcortex above."),
        ("d", CLOSE, "Same positive relation, r = 0.58 against 0.46. SNr is highest here, VPM in the paper."),
        ("e", CLOSE, "P(Go) rises with the engagement index."),
        ("f", CLOSE, "Mean 9.15 against 8.42, with one outlying recording near 50."),
    ],
    "ext1": [
        ("a-j", SAME, "Non-repeat trials give the paper's trial counts for 9 of 10 subjects (Cori 536 against 526). Fits overlie the data as in the paper."),
        ("k", CLOSE, "Same family of fits."),
        ("l", CLOSE, "Same performance by subject."),
        ("m, n", CLOSE, "A real wheel segment, chosen automatically, so its shape differs."),
        ("o", CLOSE, "Same aligned turns."),
        ("p", PARTIAL, "Same onset, with more small pre-onset excursions than the paper."),
        ("q", CLOSE, "Direction decoding at chance before onset, rising through it."),
    ],
    "ext2": [
        ("a, b", NODATA, "Histology and 3D track fit."),
        ("c", CLOSE, "Real spike raster from the one probe passing through all the paper's regions. Landmark labels come from that probe's region borders."),
        ("d", CLOSE, "Real receptive-field maps from sparse noise on the same probe."),
        ("e", NODATA, "Histology."),
        ("f", NODATA, "Alignment landmarks are not in the dataset."),
    ],
    "ext3": [("all", NODATA, "Histology.")],
    "ext4": [
        ("a", CLOSE, "Contrast tuning, peak rates and rasters close to the paper. Error 4\\% and 13\\%."),
        ("b", PARTIAL, "Latency sorted, odd trials sorting even trials, with region strips in place. It is visibly denser than the paper's."),
        ("c-e", CLOSE, "Same shape per region. Visual areas first, wide spread by movement, and reward responses."),
        ("f", CLOSE, "0.040, 0.065 and 0.071 against 0.040, 0.069 and 0.079."),
    ],
    "ext5": [
        ("a", CLOSE, "Kernel decomposition of a real neuron on four trials."),
        ("b", NODATA, "Model cartoons."),
        ("c", CLOSE, "Left 12 and Right about 28 sp/s against 12 and 25."),
        ("d", CLOSE, "Same predictor structure."),
        ("e", CLOSE, "Reduced rank beats both other models."),
        ("f", CLOSE, "Medians 0.35, 0.46 and 0.56 against 0.37, 0.50 and 0.64."),
        ("g", PARTIAL, "Same regions decode. The color scale is set from the data, because the paper's units are not stated."),
        ("h", CLOSE, "0.50\\% false positives against 0.33\\%."),
    ],
    "ext6": [
        ("a", CLOSE, "Same variance explained per region."),
        ("b", CLOSE, "Same order of Go, Miss and Passive before the stimulus in nearly all regions."),
    ],
    "ext7": [
        ("a", CLOSE, "CP stays near chance in most regions, DP rises with movement."),
        ("b", CLOSE, "Same significant regions, broadly distributed."),
        ("c", CLOSE, "Choice probability around movement restricted to a similar set."),
        ("d", CLOSE, "Late choice signal more widespread."),
        ("e", CLOSE, "SNr reaches about 20\\%, others about 10\\%, as in the paper."),
        ("f", CLOSE, "r = 0.79 against 0.69 across all 39 sessions, and 0.81 on the 33 CP sessions."),
    ],
    "ext8": [
        ("a", CLOSE, "Worked example from Radnitz\\_2017-01-08, which has the paper's 187 trials. r = 0.72 train and 0.43 test, against 0.61 and 0.54."),
        ("b", PARTIAL, "Same session type and color scale, noisier than the paper's example."),
        ("c", CLOSE, "Same off-diagonal bias. n = 17, 11, 9 against 10, 5, 9."),
        ("d", CLOSE, "VIS leads near 0.1 s, as in the paper, with fewer significant bins."),
        ("e", PARTIAL, "VIS leads before movement, but significant in a much shorter stretch."),
    ],
    "ext9": [
        ("a", CLOSE, "22 regions pass p < 0.0012. Same signs by group."),
        ("b-f", CLOSE, "Go above Miss in every split. Our p-values are stronger than the paper's."),
        ("g", CLOSE, "Pupil larger after reward."),
        ("h", CLOSE, "Real motion energy from a 141-trial session; the photo is from the paper."),
        ("i", PARTIAL, "Same shapes. The engagement test is far stronger here, p = 1e-21 against 1.5e-8."),
        ("j", CLOSE, "Same separation by previous reward."),
        ("k", CLOSE, "Go-Miss vector closer to engagement. Some p-values differ."),
    ],
}


def tex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def panel_table(key):
    rows = [f"\\textbf{{{p}}} & {status} & {note} \\\\" for p, status, note in PANELS[key]]
    return "\\begin{longtable}{@{}p{1.1cm}p{2.5cm}p{11.8cm}@{}}\n\\toprule\nPanel & Agreement & What the comparison shows \\\\\n\\midrule\n\\endhead\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}\n"


def summary_table():
    counts = pd.Series([s for v in PANELS.values() for _, s, _ in v]).value_counts()
    rows = [(TITLES[k], *[sum(s == w for _, s, _ in v) for w in (SAME, CLOSE, PARTIAL, NODATA)]) for k, v in PANELS.items()]
    body = "\n".join(" & ".join(map(str, r)) + r" \\" for r in rows)
    total = [sum(r[i] for r in rows) for i in range(1, 5)]
    return ("\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\nFigure & Same & Close & Partial & From the paper \\\\\n\\midrule\n" + body + "\n\\midrule\nPanel groups & " + " & ".join(map(str, total)) + " \\\\\n\\bottomrule\n\\end{tabular}\n"), counts


def numbers_table(output):
    text = (output / "report_vs_paper.md").read_text(encoding="utf8").split("## Sessions left out")[0]
    rows = re.findall(r"^\| (.+?) \| (.+?) \| (.+?) \|$", text, re.M)[2:]
    return "\\begin{longtable}{@{}p{8.2cm}p{3.2cm}p{3.9cm}@{}}\n\\toprule\nQuantity & Paper & Full run \\\\\n\\midrule\n\\endhead\n" + "\n".join(f"{tex(a)} & {tex(b)} & {tex(c)} \\\\" for a, b, c in rows) + "\n\\bottomrule\n\\end{longtable}\n"


def table1(res):
    n = res.neurons()
    rows = sorted(TABLE1_TOTAL)
    cells = []
    for r in rows:
        ours = int((n["region"] == r).sum())
        cells.append(f"{r} & {TABLE1_TOTAL[r]} & {ours} & {100 * (ours / TABLE1_TOTAL[r] - 1):+.1f}")
    half = (len(cells) + 2) // 3
    lines = [" & ".join(cells[i + j * half] if i + j * half < len(cells) else "& & &"[:-0] for j in range(3)) + r" \\" for i in range(half)]
    return "\\begin{tabular}{@{}lrrr@{\\hspace{6mm}}lrrr@{\\hspace{6mm}}lrrr@{}}\n\\toprule\n" + " & ".join(["Region & Paper & Ours & \\%"] * 3) + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"


def neuron_table(cfg):
    d = pd.read_csv(cfg.results / "example_scores.csv")
    chosen = store.load_json(cfg.results / "example_neurons.json")
    label = {"fig2a": "Fig 2a, 3b", "fig2b": "Fig 2b, 3a", "fig2c": "Fig 2c, 4a (ZI)", "fig4a_vis": "Fig 4a VISp", "fig4a_mrn": "Fig 4a MRN", "fig4a_cp": "Fig 4a CP", "ext4_visp": "Ext 4a VISp", "ext4_visam": "Ext 4a VISam", "ext5": "Ext 5a, c"}
    label.update({f"fig4e_{i}": f"Fig 4e panel {i}" for i in range(1, 9)})
    lines = []
    for slot, name in label.items():
        q = d[d["slot"] == slot]
        if slot not in chosen or not len(q):
            continue
        e = chosen[slot]
        row = q[(q["session"] == e["session"]) & (q["cluster"] == e["cluster"])].iloc[0]
        err = row["curve_error"] if "curve_error" in row and pd.notna(row["curve_error"]) else row["error"]
        lines.append(f"{name} & {row['region']} & \\texttt{{{tex(e['session'])}}} & {e['cluster']} & {100 * err:.0f}\\% \\\\")
    return "\\begin{longtable}{@{}p{3.6cm}p{1.5cm}p{5.4cm}p{1.4cm}p{2cm}@{}}\n\\toprule\nPanel & Region & Session & Cluster & Curve error \\\\\n\\midrule\n\\endhead\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{longtable}\n"


def assignment_tables(res):
    n, t = res.neurons(), res.regression()
    e = store.load(res.cfg.results / "engagement.npz")
    names = [str(x) for x in e["region_names"]]
    rows = []
    for r in ASSIGNMENT_REGIONS:
        s, q = n[n["region"] == r], t[t["region"] == r]
        i = names.index(r)
        rows.append(f"{r} & {len(s)} & {s['session'].nunique()} & {int(s['responsive'].sum())} & {len(q)} & {int((q['ve'] >= .02).sum())} & {int((q['vision_contra'] > .02).sum())} & {int((q['action'] > .02).sum())} & {int((q['choice'] > .02).sum())} & {e['task_passive'][i]:+.2f} & {e['go_miss'][i]:+.2f} \\\\")
    head = "Region & Neurons & Sessions & Responsive & Regression & Pass 2\\% & Vision & Action & Choice & T$-$P & G$-$M"
    return "\\begin{tabular}{@{}lrrrrrrrrrr@{}}\n\\toprule\n" + head + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def assignment_neurons(cfg):
    chosen = store.load_json(cfg.results / "example_neurons.json")
    lines = []
    for r in ASSIGNMENT_REGIONS:
        for slot, name in ((f"assign_raster_{r}", "Raster"), (f"assign_4e_{r}_contra", "Contra preference"), (f"assign_4e_{r}_ipsi", "Ipsi preference")):
            e = chosen.get(slot)
            if e:
                extra = f"{100 * e['choice_ve']:.1f}\\%" if "choice_ve" in e else "-"
                lines.append(f"{r} & {name} & \\texttt{{{tex(e['session'])}}} & {e['cluster']} & {extra} \\\\")
    return "\\begin{tabular}{@{}llp{5.2cm}rr@{}}\n\\toprule\nRegion & Panel & Session & Cluster & Choice var. expl. \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"


def departures():
    from steinmetz.report import DEPARTURES

    return "\\begin{itemize}\n" + "\n".join(f"\\item {tex(d)}" for d in DEPARTURES) + "\n\\end{itemize}\n"


def comparison_pages():
    out = []
    for k in FIGURES:
        out.append(f"\\comparison{{{k}}}{{{TITLES[k]}}}\n\\input{{generated/panels_{k}.tex}}\n")
    return "\n".join(out)


def main():
    cfg = Config()
    res = Results(cfg)
    gen, figs = HERE / "generated", HERE / "figures"
    gen.mkdir(exist_ok=True), figs.mkdir(exist_ok=True)
    for k in FIGURES:
        shutil.copy(cfg.output / "figures" / f"{k}.png", figs / f"ours_{k}.png")
        Image.open(ANALYSIS / "resources" / "paper" / f"{k}.png").convert("RGB").save(figs / f"paper_{k}.jpg", quality=92)
        (gen / f"panels_{k}.tex").write_text(panel_table(k), encoding="utf8")
    for k in ASSIGN:
        shutil.copy(cfg.output / "assignment" / f"{k}.png", figs / f"{k}.png")
    table, counts = summary_table()
    (gen / "summary.tex").write_text(table, encoding="utf8")
    (gen / "numbers.tex").write_text(numbers_table(cfg.output), encoding="utf8")
    (gen / "table1.tex").write_text(table1(res), encoding="utf8")
    (gen / "neurons.tex").write_text(neuron_table(cfg), encoding="utf8")
    (gen / "assignment.tex").write_text(assignment_tables(res), encoding="utf8")
    (gen / "assignment_neurons.tex").write_text(assignment_neurons(cfg), encoding="utf8")
    (gen / "departures.tex").write_text(departures(), encoding="utf8")
    (gen / "comparisons.tex").write_text(comparison_pages(), encoding="utf8")
    print(counts.to_dict())
    tex_bin = shutil.which("xelatex") or r"C:\texlive\2025\bin\windows\xelatex.exe"
    for _ in range(3):
        r = subprocess.run([tex_bin, "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=HERE, capture_output=True, text=True, encoding="utf8", errors="replace")
        if r.returncode:
            print(r.stdout[-2500:])
            sys.exit(1)
    print("wrote", HERE / "report.pdf")


if __name__ == "__main__":
    main()
