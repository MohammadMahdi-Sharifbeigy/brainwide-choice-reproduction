# Paper Reproduction Implementation Plan

**Goal:** a Python pipeline in `analysis/` that reproduces the paper's figures and the VISp, LGd, LP assignment figures.

**Architecture:** per-session stages save to `results/`, pooled stages stream session summaries, and figures read only saved
results plus raw spikes for rasters. One entry point runs everything and skips finished work. The same code path runs on real data and on
synthetic sessions written in the real file format, so figure tests exercise the whole pipeline.

**Tech stack:** Python 3.13 in `env/`, numpy, scipy, scikit-learn, statsmodels, pynrrd, matplotlib, pytest.

**Spec:** `analysis/docs/2026-11-01-paper-reproduction-design.md`. Execution is inline, as the owner asked to proceed. No commits: the folder is not a git repository.

## Global constraints

- Contra = right stimulus and Right choice. `D_m` = +1 Left, -1 Right.
- Elastic net: `λₛ = 0.5/ys`, alpha 0.5, on `P·B`, weights `ys·b/xs`, intercept dropped.
- No figure from synthetic data leaves without the "SYNTHETIC" watermark.
- Builder runs tests and the single-session smoke run only.
- Paper citations use numeric indices only, per `CLAUDE.md`.

## Review focus

1. A region absent from a session, as VISp, LGd and LP are in the smoke session: figures must show a labelled empty panel, not crash.
2. An interrupted run: a half-written result file must not count as finished, so writes are atomic.
3. A neuron with very low variance: `ys` near zero must give zero weights, not NaN.
4. A session failing a rule (fewer than 10 matched trials): that stage skips it and records why.
5. Fewer workers than cores on low free RAM: must still finish, at one worker if needed.

## Files

| File | Responsibility |
|---|---|
| `run_all.py` | Entry point and stage order |
| `steinmetz/config.py` | Paths, constants, paper's reported numbers |
| `steinmetz/data.py` | Session reading |
| `steinmetz/trials.py` | Move detection, trial sets, contrast matching |
| `steinmetz/regions.py` | Allen hierarchy, region labels, neuron sets, label check |
| `steinmetz/rates.py` | Binning, half-Gaussian smoothing, aligned rates, significance |
| `steinmetz/regression.py` | Predictors, streamed basis, solver, full and nested models, shuffle, model comparison |
| `steinmetz/decoding.py` | Population decoding |
| `steinmetz/jpecc.py` | jPECC |
| `steinmetz/cccp.py` | ccCP and DP |
| `steinmetz/engagement.py` | Nested ANOVA, engagement index, GLM |
| `steinmetz/behavior.py` | Psychometrics, reaction times, wheel |
| `steinmetz/atlas.py` | Region maps |
| `steinmetz/pipeline.py` | Stages, atomic saves, worker cap |
| `steinmetz/report.py` | `report_vs_paper.md` |
| `steinmetz/style.py` | Colors, panel placement, watermark, layout overlays |
| `steinmetz/synthetic.py` | Fake sessions in the real file format |
| `figures/*.py` | One script per figure, plus assignment figures |
| `tests/` | Unit, synthetic and real-data fact tests |

## Tasks

Each task: write its tests, see them fail, implement, see them pass.

1. **Core:** `config`, `data`, `trials`, `regions`, `rates`. Tests: trial lists equal the authors' on all 39 sessions; session counts 34 and 6; sign conventions;
   hierarchy merge; Table 1 total within 2%; smoothing is causal; atomic save.
2. **Regression:** Tests: `fit_enet` matches a coordinate-descent reference; zero-variance neuron; streamed basis equals batch; predictors reproduce the
   authors' indexing; recovery of known kernels; contra/ipsi sign on a synthetic neuron; no-effect neuron not flagged; shuffle lowers Choice variance explained.
3. **Other analyses:** Tests: ccCP batched equals direct and the docstring trap; decoding recovers a planted axis; normalization keeps s.e.m. non-zero at t=0; jPECC shows a planted
   lag with correct sign; engagement vector recovers a planted direction; nested ANOVA equals a statsmodels reference; psychometric fit recovers parameters.
4. **Pipeline:** Stages, resume, worker cap, report. Test: full pipeline on synthetic sessions, then again with nothing recomputed.
5. **Figures:** all figures plus assignment figures. Test: every figure renders from synthetic results, carries the watermark and appears in a layout overlay on the paper's PNG; inspect by eye.
6. **Smoke run:** whole pipeline on `Richards_2017-10-31`, with timing and peak RAM per stage.
7. **Projection:** scale stage costs to all 39 sessions from real neuron and trial counts, then report.
