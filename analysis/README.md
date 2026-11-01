# Paper reproduction

Reproduces the figures of the brain-wide choice paper (Nature 576:266, 2019) from `data/allData/`, plus the assignment figures for VISp, LGd and LP.
Design: `docs/2026-11-01-paper-reproduction-design.md`. Plan: `docs/2026-11-01-implementation-plan.md`.

## Setup after cloning

Clone with `--recurse-submodules` to get the authors' code, which holds the neuron and trial lists the regression uses.

The repository leaves out what is too large for git:

| Folder | How to restore it |
|---|---|
| `data/allData/` | Download the dataset (figshare, doi:10.6084/m9.figshare.9598406) and extract each session to its own folder |
| `env/` | `python -m venv env`, then install `analysis/requirements.txt` |
| `analysis/results/` | Rerun `run_all.py` |

## Run

From the `Task` folder:

```bash
env/Scripts/python.exe analysis/run_all.py
```

The run skips finished work, so run the same command again after an interruption. The worker count is set from free RAM when it starts; close other programs first
for a faster run, or set it with `--workers N`.

Other commands:

```bash
env/Scripts/python.exe analysis/run_all.py --session Richards_2017-10-31
```

```bash
env/Scripts/python.exe analysis/run_all.py --figures-only
```

```bash
env/Scripts/python.exe -m pytest analysis/tests -q
```

The run shows progress bars for stages, sessions and figures.

A one-session run writes to `results_smoke/` and `output_smoke/` and keeps regions seen in one mouse, so its numbers are not comparable with the paper.

## Notebook

`full_analysis.ipynb` explains and runs every analysis section by section, using the same code as `run_all.py`. Open it in VS Code with `env` as the kernel.
`RUN_MODE = "session"` runs `Moniz_2017-05-16`, which has VISp, LGd and LP, into `results_notebook/` and `output_notebook/`. `RUN_MODE = "full"` runs all 39 sessions and shares `results/` with `run_all.py`.

## Output

| Path | Contents |
|---|---|
| `output/figures/` | `fig1`-`fig5`, `ext1`-`ext9`, as PNG and PDF |
| `output/assignment/` | The four plot types for VISp, LGd and LP |
| `output/overlays/` | Each figure blended over the paper's, to check layout |
| `output/report_vs_paper.md` | Paper's numbers against ours, sessions left out, known departures |
| `results/` | Saved analysis results |

## Swapping an example neuron

`results/example_neurons.json` holds the chosen neuron per panel, and `results/example_candidates.csv` lists the top five for each. Edit the JSON, then rerun with `--figures-only`.
