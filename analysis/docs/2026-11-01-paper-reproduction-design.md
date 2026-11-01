# Design: full reproduction of the brain-wide choice paper

Paper: "Distributed coding of choice, action, and engagement across the mouse brain",
Nature 576:266–273 (2019). Local copies: `EMS84617.md`, `EMS84617.pdf`, `fig*.png`, `extended-data-fig*.png`.
Authors' code (MATLAB, core analyses only): https://github.com/nsteinme/steinmetz-et-al-2019

Citations in this document use the paper's own reference numbers.

## 1. Intent

The owner has a course assignment: plot neural responses as in Fig 2a–c, Fig 4e, Fig 5a and Extended Data
Fig 6b, for VISp, LGd and LP. They also want the whole paper reproduced, not just those plots, so the
assignment sits on top of a complete, clean analysis pipeline.

The owner runs the full analysis themselves. The builder delivers code that is proven by tests and is
ready to run, not finished results.

## 2. Goals

1. **Assignment figures:** the four plot types redone for VISp, LGd and LP (Section 7).
2. **Paper figures:** every panel of Figs 1–5 and Extended Data Figs 1–9 that the data allows (Section 6).
3. **Exact look:** every figure copies the paper's panel layout and style: same panel positions, axes,
   tick ranges, colors, labels and fonts.
4. **Single entry point:** one command runs everything, resumes after interruption, and finishes in at most
   20 hours on the owner's computer.
5. **Comparison report:** the run writes a table of the paper's numbers against ours (Section 9).

## 3. Non-goals

- Supplementary Figs 1–4 and Supplementary Tables (not in the folder).
- Spike sorting, raw voltage data, DeepLabCut pupil tracking. Use the data's supplied `eye.area` and `face.motionEnergy`.
- A GUI, a data browser, or a slide deck.
- Redrawing photos, histology or cartoons.

## 4. Anti-goals

Any of these makes the work a failure even if it runs:

- Contra and ipsi swapped anywhere.
- A plot built from invented or fallback data without saying so. Synthetic data appears only in tests,
  and any figure made from it is watermarked "SYNTHETIC".
- A silent departure from the paper. Every known departure is listed in the comparison report.
- A number tuned to match the paper rather than computed from the stated method.
- A claim that results reproduce the paper before the owner's full run has been compared.
- Code that needs more memory than the computer has, or dies at hour 15 with nothing saved.
- Over-engineering. The project's `CLAUDE.md` applies: minimum code, no speculative options.

## 5. Constraints and verified facts

### Environment

- Windows 11, 12 cores, 15 GB RAM (about 2.5 GB free when idle-checked; the owner may free more), 47 GB free on E:.
- Python environment: the project's `env/` (Python 3.13.5, numpy 2.4.4, scipy 1.18.0, matplotlib 3.11.0, scikit-learn 1.9.0,
  pytest 9.1.1). Missing: statsmodels and pynrrd, to be installed into `env/` and listed in `analysis/requirements.txt`.
- Authors' code is available locally at `steinmetz-et-al-2019/` (commit 9a0bea8). Treat it as read-only reference.
- The folder is not a git repository.
- The builder runs tests only: synthetic data plus at most one small real session (`Richards_2017-10-31`).
  No full-dataset run.

### Data

Located at `data/allData/<session>/`: 39 session folders, 54–56 files each, 15.35 GB, already extracted.
`9598406.zip` is a compressed copy and is not used.

Facts checked against the real files:

| Fact | Result |
|---|---|
| Sessions | 39 |
| Sum of `trials.included` | 9,538, exactly the paper's Fig 1 count |
| Good units (`_phy_annotation >= 2`, rate >= 0.1 Hz, approximate) | 29,533 against the paper's 29,134 (1.4% high) |
| Contrast levels | 0, 0.25, 0.5, 1 in all 39 sessions |
| `response_choice` sign | **+1 = Left choice** (rewarded when left contrast is higher), **-1 = Right choice**, 0 = NoGo |
| `wheelMoves.type` | **1 = left turn, 2 = right turn**, 0 and 3 = other. Agrees with `response_choice` for most trials |
| `clusters.peakChannel` | **1-based**, global across probes: never 0 in any session, and equals the channel count in 8 sessions |
| Region labels | `channels.brainLocation.tsv` column `allen_ontology`. 73 raw labels against the paper's 42 regions |
| Region counts against Extended Data Table 1 | Total 22,280 against 22,458 (-0.8%), but 26 of 42 regions are more than 2% off. VISp is within 2%. LGd 788 against 811 (-2.8%) in 8 recordings against 10. LP 681 against 732 (-7.0%) in 9 against 11. Worst: ZI +19.5%, VPL +15.3%, MOs -12.5%. The neuron filter looks sound (CP matches exactly at 914), so the difference seems to lie in the labels. The 0-based and nearest-depth readings are no better |
| Pupil | Paper drops 5 of 39 recordings for poor eye video but does not name them. `eye.area` has no gaps in any session |
| All recordings | Left hemisphere, so **contra = right stimulus, contra choice = Right choice** |
| Passive stimuli | `passiveVisual.times`, `contrastLeft`, `contrastRight` |

Authors' published lists (`behavior/inclusionData/` in their repo, one set per session), compared on
`Richards_2017-10-31` and `Lederberg_2017-12-10`:

- `trials/*_hasLeft`, `*_hasRight`, `*_hasNone`: reproduced with **0 differences** using `findMoveTimes` logic.
- `neurons/*_inclNeurons`: total 18,248 neurons. This is the selection for Figs 3–5: rate > 0.1 Hz, `cgs > 1`,
  and any of the first four tests at p<0.05. A re-implementation agreed on 259 of 281 and 242 of 270 neurons,
  so it is close but not identical.

Two quirks in the authors' logic, both confirmed:

1. `hasNone` only contains trials whose first movement came **before feedback but outside** the window.
   Trials with no wheel movement at all before feedback are left out. It looks unintended and does not
   match the Methods text.
2. The neuron rule uses `all(p>0.05)`, which keeps neurons when a p-value is NaN. Their comment says this
   let about 1,000 non-responsive neurons in.

## 6. Figure inventory

Status: **data** = from the dataset; **atlas** = also needs the Allen atlas; **slot** = empty labelled panel that keeps
the layout, with no content.

| Figure | Panels | Status | Needs |
|---|---|---|---|
| Fig 1 | c, d, j | data | Behavior, one example trial |
| | h | data + atlas outline | Channel CCF coordinates |
| | i | atlas | Region map, neuron counts |
| | a, b, e, f, g | slot | Task cartoons, photos, histology |
| Fig 2 | a–c | data | Rasters |
| | d–h | data | Region colormaps, % significant bars |
| Fig 3 | a–e | data | Kernel regression |
| | f, g | atlas | Region maps |
| Fig 4 | a, b, e, f, g | data | Kernel regression, examples |
| | c | data | Population decoding |
| | d | data | jPECC |
| | h | atlas | Region map |
| Fig 5 | a, b, d, e, f | data | Population rates, engagement index |
| | c | atlas | Region map |
| Ext 1 | a–q | data | Psychometric fits, wheel analysis |
| Ext 2 | d | data | Receptive fields from `sparseNoise` |
| | a, b, c, e, f | slot | Histology and alignment |
| Ext 3 | all | slot | Histology |
| Ext 4 | a–f | data | Inset in a uses atlas |
| Ext 5 | a, c, d–h | data | Model comparison, decoding, shuffle. Panel c is the Action/Choice decomposition (the figure's letters differ from its legend) |
| | b | slot | Cartoon |
| Ext 6 | a, b | data | |
| Ext 7 | a–f | data | ccCP and detect probability |
| Ext 8 | a–e | data | jPECC, a includes one worked example |
| Ext 9 | a–k | data | Photo in h is a slot |

## 7. Assignment figures (VISp, LGd, LP)

Separate from the paper's figures, which keep the paper's regions. Same style. Output folder `output/assignment/`.

- **Fig 2a–c style:** three columns, one neuron per region. Rows as in the paper: Correct contra, Correct ipsi,
  Miss contra, Miss ipsi, Passive contra, Passive ipsi, all aligned to stimulus, then Correct contra and ipsi aligned to movement.
  Stimulus shown alone on the given side; black dots mark movement or stimulus, trials sorted by reaction time.
- **Fig 4e style:** 2 rows × 3 columns. Column per region. Candidates are Regression neurons in that region that passed the
  full-model 2% cut. Top row: highest nested Choice variance explained among contra-preferring neurons. Bottom row: the same among
  ipsi-preferring neurons. Preference comes from the sign of the mean Choice kernel from -0.1 to 0 s before movement. `D_m` is +1
  for Left (ipsi) and -1 for Right (contra), so a **negative** kernel (`K_D < 0`, Action − Choice > Action + Choice) means **contra
  preference** and a **positive** kernel means **ipsi preference**. A test on synthetic data must lock this sign. Trials shown are all contra and ipsi
  choice trials, as in the paper, with no stimulus matching. Each panel's caption gives the neuron's Choice variance explained and
  whether it passes the 2% threshold. A slot with no candidate stays empty and labelled. This figure needs the regression results.
- **Fig 5a style:** one row of three panels, Task Miss (cyan) against Passive (grey), contrast matched.
- **Ext Fig 6b style:** one row of three panels, Go (orange-red), Miss (cyan), Passive (grey), contrast matched.

Neurons are chosen by a written rule (highest contra stimulus response from 0.05 to 0.15 s for the raster, among neurons
with at least one trial in each row). Chosen neurons are stored in one table, `example_neurons`, by session and
cluster ID, so the owner can swap one by editing a line. A candidate sheet lists the top five per slot.

Expected results that must not be "fixed": VISp, LGd and LP have few or no choice-selective neurons, so in the Fig 4e style the
chosen neurons may fall under the 2% threshold. Contra and ipsi curves may still separate, because contra choices mostly follow contra
stimuli; that separation can be visual and is not by itself choice coding, as the paper shows for its VISp example in Fig 4a. Recorded
LGd neurons' receptive fields did not overlap the task stimuli, so LGd visual responses may be weak.

## 8. Analysis definitions

The Methods section is the primary source. Where it is silent or unclear, follow the authors' code. Where these
conflict, the decision below applies and the departure is listed in the comparison report.

### 8.1 Trials

Reimplement `findMoveTimes`. Bin times use `winCountMove = [0.125, 0.4]`, `winNoMove = [-0.05, 0.4]`.

Two trial sets, decided per analysis:

- **Kernel regression, decoding, Figs 3–4:** the authors' lists (`hasLeft`, `hasRight`, `hasNone`), so the regression
  matches theirs exactly.
- **Fig 2, Fig 5, Ext 6b, Ext 9, ccCP:** the Methods definition: Go = left or right first move in window; NoGo =
  no detected movement of any kind from -0.05 to 0.4 s. Includes the fully still trials that `hasNone` drops.
- Miss = NoGo with a stimulus present. Disengaged trials are removed through `trials.included`.

- Ext 4b: on the trials with contra stimuli and movement, latency for sorting comes from odd-numbered trials and the plotted activity from
  even-numbered trials. Latency's definition is not stated; the builder picks one and lists it as a departure.
- Ext 4e: only correct NoGo trials, meaning no stimulus and no wheel movement.

**Contrast matching** (Methods): applies to Fig 2d (Correct contra), Fig 2f (Miss contra), Fig 2h (Passive contra), Fig 5a–b and Ext 6b.
Take trials with contra contrast > 0; for each of low, medium, high count the trials in each of the three groups; draw the minimum
count at random from each; drop the session if fewer than 10 trials result. Fixed random seed. The Go group is **correct contra choice**
trials. The Methods text says "Left or Right", but that keeps 36 sessions; correct contra keeps exactly the paper's 34 (dropped:
Lederberg 12-05, 12-06, 12-07, Tatum 12-08, 12-09). Passive trials also require no detected movement from -0.05 to 0.4 s, which does
not change the session count.

### 8.2 Neurons and regions

- Base set: `_phy_annotation >= 2` and trial rate >= 0.1 Hz from 0 to 0.4 s.
- Region: `allen_ontology` at the neuron's peak channel (`peakChannel - 1`), merged to the paper's 42 regions (for example
  `LSr`, `LSc` into `LS`; `SCig` into `SCm`; `SCsg` into `SCs`; `ORBm` into `ORB`). The merge follows the Allen hierarchy and is
  applied by rule, never adjusted per region to hit a target. `root`, `TH`, `MB` and other non-specific labels are not localizable.
- Bounded label check: the authors' pipeline loaded the atlas volume when building its recordings, so their labels may have
  come from it. After the atlas is available, re-derive labels once from each channel's `ccf_ap`, `ccf_dv`, `ccf_lr`. Adopt that
  single rule only if it brings region counts closer to Table 1 overall. Otherwise keep the released labels and stop there.
  Result: atlas labels were far worse (total error 8,505 neurons against 790, all 42 regions over 2% off), so the released labels stay.
- Keep a region in a session only with >= 13 neurons there, and only if >= 2 subjects qualify.
- Six task tests reimplemented per Methods. Bonferroni alpha 0.0083.

Neuron set per analysis, each set defined on top of the base set and region rule:

| Set | Definition | Used for |
|---|---|---|
| **All** | Base set in a kept region (Table 1 "N total") | Fig 5a–f, Ext 6b, Ext 7, Ext 9, Fig 4d and Ext 8 jPECC |
| **Responsive** | Any of the six tests at 0.0083 (Table 1 "N responsive") | Fig 2d–h, Ext 4c–e |
| **Highly activated** | Sliding-window rule in the Methods (ranksum p<0.0001 in three consecutive bins) | Ext 4b |
| **Regression** | Authors' `inclNeurons` within a kept region | Fig 3, Fig 4a–c and f–g, Ext 5, Ext 6a |

The paper supports these: Fig 5d gives a neuron range of 130–1534, which matches Table 1's "N total", and Fig 2's legend says
responsive neurons, which matches Table 1's "N responsive".

### 8.3 Firing rates

Spikes binned at 5 ms, causal half-Gaussian. Regression sd 25 ms. Display: 20 ms, trial average, then 30 ms. Fig 2d–h
normalization: subtract baseline (-0.02 to 0 s), divide by baseline + 0.5 sp/s. Fig 2d, f and h use the contrast-matched trials of Section 8.1.

Significance for Fig 2d–h side bars and Fig 5b: mean rate from 0 to 0.25 s across the condition's trials, z-scored against trial-by-trial
baseline from -0.1 to 0 s, using the standard error across trials so that it equals the paper's t-test. Methods says **two-sided** (`z > 4
or z < -4`) while the captions say "more responsive" and "activated". Methods is the primary source, so the figures use `|z| > 4`.
The signed z is saved, and the one-sided count goes in the comparison report.

### 8.4 Kernel regression

Follow `kernelFitBasic`, `regFullPipeline`, `determineResidualsWithout`, `predictResidualsWith`, `CanonCor2all`:

- Six Vision kernels, -0.05 to 0.4 s, 90 lags. Action and Choice kernels, -0.25 to 0.025 s, 55 lags. Choice value +1 Left, -1 Right.
  650 predictors. Sample times are stimulus -0.05 to the earlier of stimulus +0.4 and movement +0.025.
- Baseline subtracted before fitting.
- Reduced-rank basis pooled across **all sessions** at once, streamed one session at a time so no multi-session `T × 650` or `T × N`
  matrix is held in memory. Per session keep `PᵀP` (650 × 650), `ΣP` and `T`, plus `ΣF` and `PᵀF` (650 × `N_s`). `CanonCor2all` uses a
  pooled mean for `CXX` but a per-session mean and `T_s - 1` for `CYX`, so the sums alone are not enough. Tested: this form matches
  the batch result to 5e-14, and bare sums without the means are badly wrong. Accumulate in float64. Keep 200 dims.
- Elastic net per neuron: alpha 0.5, lambda 0.5, fixed. The code sets `useCVG = false`, so it is `glmnet` at a fixed lambda, not `cvglmnet`.
  Must replicate glmnet exactly (`GLMnet.f` lines 775–806). `X` is the projected predictors `P·B` for that training fold, with `T` its sample
  count. Standardize each column by `xs` and `y` by `ys`, both with the 1/T standard deviation; minimize `1/(2T)·‖y/ys − X/xs·b‖² + λₛ[(1−α)/2·‖b‖²
  + α‖b‖₁]` with **`λₛ = 0.5/ys`**; return `β = ys·b/xs`. The penalty therefore differs per neuron, and a neuron with small `ys` can get all zero
  weights. Standardizing `y` while keeping λ = 0.5 is wrong (in a test: 1 non-zero weight instead of 11). They interpolate along a lambda path;
  we solve at λ directly, which is listed as a departure.
- One fit per fold on all columns; a rank `r` keeps the first `r` weights with no refit. The intercept is fitted but dropped from predictions.
- Full model: 5-fold by trial, ranks 1–50, best cross-validated variance explained (rank chosen on test data, as they do).
  Variance scored only on samples up to movement onset. Neurons under 2% are excluded.
- Nested test for each of Contra Vision, Ipsi Vision, Action, Choice: fit all other kernels (75 comps), get residuals, new
  reduced-rank fit on those residuals with just the test kernel (18 comps), 10-fold; score from 0.05 to 0.4 s after stimulus
  and -0.2 to 0 s before movement. Selective if > 2%.
- Choice shuffle (Ext 5h): relabel choices among Left and Right trials, rerun from the start.
- Model comparison (Ext 5e, f): Toeplitz and raised-cosine against reduced rank at fixed 18 comps.
- Fig 4f, g: predicted rates at movement onset from the kernels: contra = Action − Choice, ipsi = Action + Choice, because `D_m` = +1
  is Left (ipsi) and -1 is Right (contra).

### 8.5 Population decoding (Fig 4c, Ext 5g)

Residuals as above, L1 logistic regression, training windows per Methods, projection across time. For Action decoding, each trial
without movement gets a pseudo movement onset drawn at random, with replacement and a fixed seed, from the session's Left and Right
movement onset times, as in `findMoveTimes`.

Fig 4c normalization: divide each recording's trace by one number, the **mean across recordings within its group at choice time**
(t = 0), for frontal n=29, midbrain n=29, striatum n=5. Normalizing each recording to 1 would force the s.e.m. to zero at t = 0, but the
figure has a band there. "Area" in the Methods could mean single regions; the ANOVA uses it for these three groups, so groups are used.
Two-way ANOVA on time and group runs on these values.

Regularization strength and folds are not stated; the builder picks them by inner cross-validation and lists them as a departure.
The repo has no decoding code.

### 8.6 jPECC (Fig 4d, Ext 8)

Follow `jPECC.m`: groups VIS, Midbrain, Frontal; 10 ms bins, 25 ms causal half-Gaussian, divide by baseline + 1 sp/s, 10 PCs,
10 folds, ridge λ = 0.5 by the augmented-matrix method. Asymmetry index ±50 ms.

### 8.7 ccCP and detect probability (Ext 7)

Follow `choiceProbShuf`: combine Mann–Whitney numerators and denominators across the 16 stimulus conditions, 2,000 shuffles within
conditions. A condition with only Left or only Right choices contributes nothing. A session with fewer than 10 includable trials is
dropped. This gives the paper's count of 6 of 39. The paper does not name its six; ours are Cori 12-14, Moniz 05-18, Muller 01-08,
Muller 01-09, Radnitz 01-11, Tatum 12-09. In the paper, the six hold 20.8% of the MOs choice-selective neurons.

DP has its own session rule (owner's decision). The same >= 10 includable-trials threshold applies directly to DP, where a condition is
includable when it holds both Go and NoGo trials. More sessions qualify, because the CP drops come from mice turning only one way.
`report_vs_paper.md` logs both the DP-qualified session set and the 33-session CP subset, and gives Ext 7f on each.

Shuffle labels are `nA` positions drawn from `nA + nB` (the commented `randperm` line). The function's docstring says a permutation of
`1:nx`, which would not shuffle anything.

Windows, relative to movement onset unless noted:

| Panel | Statistic | Window |
|---|---|---|
| Ext 7a | CP and DP over time | -0.2 to 0.3 s from stimulus; -0.3 to 0.25 s from movement |
| Ext 7b | DP | -0.05 to 0.05 s |
| Ext 7c | CP | -0.05 to 0.05 s |
| Ext 7d | CP | 0.15 to 0.25 s |
| Ext 7e | CP for MOs, SNr, SCm, MRN | -0.3 to 0.25 s |
| Ext 7f | Pre-stimulus DP − 0.5 against Go − Miss rate per region | Not stated; -0.2 to 0 s from stimulus, matching Fig 5d. Paper: r = 0.69; r = 0.48 with Task − Passive |

Bin size and count window for the time courses are not stated; the builder picks them and lists them as a departure.

### 8.8 Engagement (Fig 5, Ext 9)

The repo has no engagement code, so the Methods is the only source. Two different pre-stimulus definitions are used:

| Analysis | Window | Rate | Trials |
|---|---|---|---|
| Nested ANOVA (Ext 9a), per region | -0.25 to -0.05 s | `log10(x + 1 sp/s)` per neuron | Excludes trials with wheel movement in the window |
| Engagement vector and index (Fig 5e–f, Ext 9b–k) | -0.2 to 0 s | Plain sp/s | All task and all passive trials for `x` |

- ANOVA factors are condition, session and neuron nested in session. Significant at p < 0.0012 (0.05/42). Ext 9a's "session and subject"
  wording conflicts with this; use the Methods. Ext 9a plots normalized differences, while Fig 5c–d and Ext 6b plot sp/s.
- `x_n = mean(f_active) − mean(f_passive)` from -0.2 to 0 s, only for neurons in regions significant by the ANOVA, unit L2 per session.
  Index per trial = `x · f_i`. Paper: 319 ± 32.5 neurons per session, n = 34. It does not say which 34; use the sessions kept by contrast
  matching, because that count matches, and list it as a departure.
- Go against Miss is contrast matched, using N = min(Go, NoGo) per contra contrast.
- Fig 5c–d windows are not stated; use -0.2 to 0 s so they match `x`.
- Logistic GLM with squared pupil and motion energy terms, deviance test.
- Pupil panels use all 39 sessions, because the paper's five rejected recordings are not named. No quality rule is invented, and this is
  listed as a departure.

**Not stated in the paper:** the short and long ITI split (Ext 9e–f) and the Ext 9k vectors (`Large − small pupil`, `Large − small motion`,
`Reward − NoReward`, `Long − short ITI`) have labels only. Builder's rule, listed as a departure: split at each session's median of the
pre-stimulus value for pupil, motion and ITI, and by previous-trial reward for reward. Ext 9d removes the reward effect by
subtracting the mean within each previous-reward condition, as its legend says.

### 8.9 Other

Fig 1d: reaction time (detected movement onset from stimulus onset) against chosen stimulus contrast (25, 50, 100%), as two curves: **one
stimulus** (other side absent) and **two stimuli** (competing stimulus present). Multi-way ANOVA, paper p < 10⁻⁴. ANOVA factors and error bars
are not stated; the builder uses chosen contrast and stimulus count as factors, with s.e.m. across sessions, and lists this as a departure.

Psychometric multinomial logistic fit per Extended Data Fig 1 legend. Focality index with bias-corrected bootstrap. Receptive fields per
Methods. Wheel-velocity direction decoding per Ext 1q.

### 8.10 Atlas

`annotation_25.nrrd` (4.0 MB, 25 µm) from the Allen Institute download server, plus their structure table. Owner approved this
download. Map views: top-down cortex, and sagittal at 1.7 mm ML. Allen CCF is not re-derived. The 10 µm volume is rejected because
it needs nearly 5 GB of RAM when read.

## 9. Validation

### Before handover (no big run)

- **Synthetic data:** generate spikes with known kernels, tuning and choice effects, and confirm each analysis recovers them,
  including contra/ipsi signs. A neuron with no effect must not be flagged.
- **Real-data facts:** sign conventions, trial counts, trial lists (all 39 sessions, cheap to check), and the Table 1 total of 22,458
  to within 2%. Per-region Table 1 differences are not a pass/fail test, because the released labels can't meet one: they are
  written to a table in the comparison report, and the figures for LGd and LP note their counts.
- **Session rules:** contrast matching keeps 34 sessions and the ccCP rule drops 6, matching the paper's counts, with our session names
  from Sections 8.1 and 8.7 fixed as a regression check.
- **Solver:** the elastic net matches a known glmnet-style answer, including a neuron whose `ys` is small enough to give all zero weights.
- **Basis:** streamed basis equals the batch `CanonCor2all` result on small data.
- **Shuffles:** batched ccCP equals the direct form on small data, and the Choice kernel sign is locked by a synthetic neuron.
- **Authors' equivalents:** where practical, small known-answer tests for ccCP, jPECC and the rank regression.
- **Layout:** render each figure from synthetic results over a faded copy of the paper's PNG and compare panel by panel.
- **Smoke run:** whole pipeline on `Richards_2017-10-31` with a time per stage.
- **Time projection:** scale stage timings to the full dataset. If the projection exceeds 20 hours, tell the owner first.

### On the owner's full run

`output/report_vs_paper.md` lists paper against ours, and every known departure. Numbers include: 29,134 neurons; 22,458 localized;
60.0% modulated, of which 74.3% up and 20.2% down; 0.33% shuffle false positives; 53/54 midbrain contra-preferring; 19/48 forebrain
ipsi-preferring; 29/54 and 10/48 suppressed; focality 0.040, 0.069, 0.079; Fig 5d r = 0.46; mean engagement difference 8.42; 34 and 33
sessions in the matched and ccCP analyses; 20.8% of MOs choice neurons outside ccCP; Ext 7f r = 0.69 and 0.48.

Exact matches are not expected for anything that depends on random splits or a different elastic-net solver.

## 10. Run design

- One command, for example `python analysis/run_all.py`, with stages: inclusion → rates → regression → nested → decoding → jPECC → ccCP → engagement → figures.
- Each stage saves to `analysis/results/` and skips completed work, per session where possible.
- Memory: one session in memory at a time per worker. Regression data stored as float32 on disk. Basis accumulation streams as in Section 8.4.
- Workers: set at start as `floor(free RAM / peak per-worker RAM)`, at most 11 and at least 1, with peak measured in the smoke run and
  free RAM read when the run starts (about 2.5 GB when last checked, so as few as one or two workers). BLAS threads are 1 per worker,
  to avoid swap thrashing.
- ccCP and DP shuffles are vectorized without per-shuffle Python loops or fancy indexing. Labels are drawn once per session and condition, stored as
  a `shuffles × trials` 0/1 matrix `M`, and reused across all neurons and bins. Then `numer = M @ ranks − nA(nA+1)/2`, in batches of about 100
  neurons. Tested on synthetic data with 1,000 neurons × 100 bins × 2,000 shuffles: 12 s and 0.16 GB, where plain indexing peaked at 17.6 GB.
  Sharing labels keeps each neuron's p-value valid, but shuffle noise is correlated across neurons, so % significant is not a set of
  independent tests. The repo's function allows this; whether the authors did it is unknown.
- The 20-hour limit is a design target, and all 2,000 shuffles are kept. If the time projection exceeds it, the owner is told first and
  nothing is cut without asking.
- Figures can be rebuilt from saved results alone.

## 11. Folder layout

Outline only; details left to the builder.

```
analysis/
  run_all.py, requirements.txt, README.md
  <package>/      data, trials, regions, rates, regression, decoding, jpecc, cccp, engagement, atlas, style
  figures/        one script per figure, plus assignment figures
  resources/      authors' lists and atlas, as a local copy with their license noted
  tests/
  results/        saved analysis outputs (not hand-edited)
  output/         figures (PNG and PDF), assignment/, report_vs_paper.md
  docs/
```

## 12. Decisions

### Owner's decisions

- "VISp, LGd, LP" applies to all four assignment plots.
- All paper figures are wanted as well as the assignment.
- "Same slides" means the paper's exact panel layout and style, not a slide deck.
- Allen atlas download approved.
- Full run at most 20 hours. The owner runs it; the builder runs tests only.
- Data is extracted to `data/allData/`.
- The `env/` environment and the `steinmetz-et-al-2019/` copy are the owner's and are used as they are, so libraries aren't reinstalled.
- Fig 4e assignment neurons are selected with the Choice kernel.
- DP applies the >= 10 includable-trials threshold on its own, with both session sets logged for Ext 7f.
- Design approved 2026-11-01.

### Builder's calls, open to change

- Python rather than MATLAB, because glmnet for MATLAB is hard to run on current systems.
- Empty labelled slots for panels that can't come from data.
- Ext 2d built from `sparseNoise`, though the rest of Ext 2 is a slot.
- Two trial definitions as in Section 8.1, rather than either one everywhere.
- Authors' published neuron lists used for Figs 3–5, including the roughly 1,000 extra neurons from their bug, to match the paper.
- 25 µm atlas.
- This document's location, since the folder is not a repository.
- Authors' inclusion lists are read from the local `steinmetz-et-al-2019/` copy, so no extra download is needed.
- statsmodels and pynrrd are installed into the owner's `env/`.
- Example neurons chosen by rule. The paper does not identify its examples, so Figs 2a–c, 3a–b, 4a, 4e show close matches, not
  necessarily the same cells.
- Released region labels are accepted if the bounded label check doesn't improve them, so LGd and LP keep slightly fewer
  neurons and recordings than the paper.
- Neuron sets per analysis as in Section 8.2.
- Pupil panels use all 39 sessions.
- Correct contra trials as the Go group in contrast matching, because that reproduces the paper's 34 sessions where its text would not.
- Two-sided `|z| > 4` in Fig 2d–h and Fig 5b, following the Methods over the captions.
- Fig 4c normalization by group, not by single region.
- Median splits for Ext 9e–f and 9k, -0.2 to 0 s for Fig 5c–d, the 34 engagement sessions, and the Fig 1d ANOVA factors, none of which the paper states.
- -0.1 to 0 s as the window for Choice kernel sign in the Fig 4e style plot.

### Left to the builder

Module structure, solver implementation (reuse of a precomputed Gram matrix per session and fold is expected), parallelism details,
synthetic data design, test organization, exact fonts available on Windows.
