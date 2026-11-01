# Paper against this run

Sessions in this run: 39 of 39.

| Quantity | Paper | Ours |
|---|---|---|
| Included trials | 9538 | 9538 |
| Neurons | 29134 | 29178 |
| Localized neurons | 22458 | 22210 |
| Modulated neurons | 13,466 (60.0%) | 12,972 (58.4%) |
| Sessions with contrast-matched trials | 34 of 39 | 34 of 39 |
| Sessions in CP analysis | 33 of 39 | 33 of 39 |
| Sessions in DP analysis | not stated | 39 of 39 |
| Shuffle false positives | 0.33% | 0.50% |
| Midbrain choice neurons preferring contra | 53/54 | 45/46 |
| Midbrain suppressed for non-preferred choice | 29/54 | 16/46 |
| Forebrain choice neurons preferring ipsi | 19/48 | 11/33 |
| Forebrain suppressed for non-preferred choice | 10/48 | 6/33 |
| Focality: action, contra vision, choice | 0.040, 0.069, 0.079 | 0.040, 0.065, 0.071 |
| Fig 5d correlation | r = 0.46, p = 0.002 | r = 0.58, p = 6.33e-05 |
| Mean engagement index difference | 8.42 | 9.15 (n = 34) |
| Ext 7f with Go-Miss, DP-qualified sessions (39) | r = 0.69 | r = 0.79 |
| Ext 7f with Task-Passive, DP-qualified sessions | r = 0.48 | r = 0.50 |
| Ext 7f with Go-Miss, 33-session CP subset sessions (33) | r = 0.69 | r = 0.81 |
| Ext 7f with Task-Passive, 33-session CP subset sessions | r = 0.48 | r = 0.47 |

## Sessions left out

Contrast matching: Lederberg_2017-12-05, Lederberg_2017-12-06, Lederberg_2017-12-07, Tatum_2017-12-08, Tatum_2017-12-09
CP: Cori_2016-12-14, Moniz_2017-05-18, Muller_2017-01-08, Muller_2017-01-09, Radnitz_2017-01-11, Tatum_2017-12-09
DP: none

## Two-sided and one-sided significance (Fig 2d-h, Fig 5b)

| Condition | |z|>4 | z>4 |
|---|---|---|
| contra_correct | 6.6% | 6.5% |
| ipsi_correct | 8.1% | 7.8% |
| contra_miss | 2.5% | 2.5% |
| ipsi_miss | 1.0% | 1.0% |
| contra_passive | 1.8% | 1.8% |

## Extended Data Table 1

| Region | Paper total | Ours | Paper responsive | Ours |
|---|---|---|---|---|
| ACA | 608 | 651 | 395 | 413 |
| ACB | 255 | 261 | 152 | 153 |
| APN | 235 | 236 | 177 | 179 |
| BLA | 261 | 266 | 107 | 110 |
| CA1 | 1129 | 1150 | 494 | 497 |
| CA3 | 438 | 430 | 243 | 219 |
| CP | 914 | 910 | 524 | 519 |
| DG | 711 | 702 | 336 | 329 |
| GPe | 274 | 281 | 146 | 147 |
| ILA | 338 | 349 | 192 | 192 |
| LD | 308 | 275 | 209 | 176 |
| LGd | 811 | 789 | 397 | 362 |
| LP | 732 | 680 | 393 | 344 |
| LS | 844 | 846 | 508 | 487 |
| MD | 381 | 388 | 244 | 248 |
| MG | 276 | 284 | 180 | 183 |
| MOp | 682 | 680 | 553 | 532 |
| MOs | 1534 | 1335 | 993 | 836 |
| MRN | 857 | 875 | 722 | 731 |
| OLF | 684 | 695 | 210 | 218 |
| ORB | 770 | 789 | 281 | 300 |
| PAG | 130 | 128 | 72 | 69 |
| PL | 728 | 730 | 438 | 432 |
| PO | 620 | 645 | 342 | 351 |
| POL | 190 | 186 | 132 | 120 |
| POST | 272 | 272 | 163 | 161 |
| RSP | 598 | 581 | 308 | 292 |
| RT | 160 | 152 | 111 | 111 |
| SCm | 997 | 992 | 738 | 728 |
| SCs | 317 | 281 | 192 | 158 |
| SNr | 274 | 273 | 201 | 202 |
| SSp | 461 | 448 | 296 | 280 |
| SUB | 669 | 661 | 494 | 477 |
| VISa | 285 | 288 | 207 | 203 |
| VISam | 805 | 804 | 501 | 482 |
| VISl | 403 | 382 | 248 | 223 |
| VISp | 923 | 917 | 649 | 616 |
| VISpm | 516 | 494 | 230 | 212 |
| VISrl | 236 | 220 | 85 | 66 |
| VPL | 255 | 294 | 202 | 226 |
| VPM | 357 | 327 | 235 | 206 |
| ZI | 220 | 263 | 167 | 182 |

## Known departures

- Elastic net solved at lambda directly; the authors interpolate along a glmnet path.
- Example neurons are the real neurons closest to the paper's panels; the paper does not identify its examples, so they need not be the same cells.
- Fig 2d-h heatmaps and Ext 4b-e use the Methods' highly activated neurons (sliding rank-sum); the captions say responsive.
- Ext 1 uses non-repeat trials (repNum 1), which reproduces the paper's trial counts for 9 of 10 subjects. Cori has 536 against 526.
- Panels with no data behind them (photos, histology, cartoons) are crops of the paper's own figure, tagged in the figure.
- jPECC averages use every session with both region groups: 17, 11 and 9 sessions against the paper's 10, 5 and 9. No neuron-count rule reproduced those numbers.
- Fig 1j uses a session with six of the seven regions of the paper's panel.
- Released region labels differ from the paper's table per region (table below).
- Pupil panels use all 39 sessions; the paper's five rejected recordings are not named.
- Fig 2d-h and Fig 5b use two-sided |z|>4 as in the Methods; captions read as one-sided. One-sided counts are below.
- Contrast-matching Go group is correct contra trials, which gives the paper's 34 sessions; the Methods text says Left or Right.
- Fig 4c normalizes by group, reading the Methods' 'area' as the three groups.
- Decoding regularization and folds are not stated: L1 logistic regression, C from (0.01, 0.1, 1), 5 outer and 3 inner folds.
- ccCP counts use a 100 ms box stepped by 10 ms; the time-course window is not stated.
- Ext 4b latency is when the odd-trial rate first reaches half its peak; its definition is not stated.
- Ext 9e-f and 9k use within-session median splits; the paper gives labels only.
- Fig 5c-d and Ext 7f use -0.2 to 0 s; their windows are not stated.
- The 34 engagement sessions are taken as the contrast-matched sessions; the paper does not say which 34.
- Fig 1d ANOVA uses chosen contrast, stimulus count and subject; its factors are not stated.
- Passive trials in matching require no movement from -0.05 to 0.4 s; this does not change the session count.
- DP applies the >=10 includable-trials rule itself, so it keeps more sessions than CP.
- Ext 5f uses neurons passing the 2% full-model cut, matching the neuron counts implied by its histograms.
- Ext 5g's color scale is set from the data, because the paper's projection units are not stated.
- Ext 5c plots the sum and difference of the Left and Right responses, as the paper's legend labels them: twice each kernel.
- Released region labels are kept: labels read from the atlas at channel positions were far further from Extended Data Table 1.
- Region, line and colormap colors are measured from the paper's figures. BLA-style pale regions and the brain-map color steps are approximate.
