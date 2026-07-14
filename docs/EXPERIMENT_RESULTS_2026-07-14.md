# Final fixed-holdout experiment results (2026-07-14)

This document records the completed Kaggle T4 x2 run that started on 2026-07-13 and finished
on 2026-07-14. The run used 10 development fights in five fixed, paired fight-level folds and
kept `Paddy Pimblett vs Michael Chandler` untouched until all architecture, task, epoch, and
smoothing choices were frozen.

The source archive is `mma_experiment_outputs.zip`. Extracted artifacts are under `outputs/`,
the executed Kaggle notebook is `outputs/kaggle_part4_background_executed.ipynb`, and the full
Kaggle console export is `outputs/logs/kaggle_version_console.log`. These large artifacts are
intentionally ignored by Git; this analysis is the version-controlled summary.

## Run integrity

- All five stages in `outputs/run_status.json` completed.
- `errors` is empty and the executed notebook's Papermill metadata reports no exception.
- The run covered 1,159 labeled fight clips from 11 fights. Model selection used 1,003 fight
  clips from the 10 development fights; the final holdout has 156 fight clips and 19 non-fight
  clips.
- The Python frozen-module and nbconvert `SyntaxWarning` messages in the console are environment
  warnings, not training failures.

## Frozen model selection

Choices were frozen from development out-of-fold (OOF) predictions before the holdout was
evaluated:

| Component | Frozen choice | Development OOF result | Final epochs |
|---|---|---:|---:|
| Gate | ResNet-18, threshold 0.4528 | AUC 0.956, 98.0% fight retention, 77.4% non-fight rejection | 4 |
| Phase | Multi-task R(2+1)D-18 | macro-F1 0.662, accuracy 0.724 | 8 |
| Pressure | Pressure-only R(2+1)D-18 | macro-F1 0.436, accuracy 0.498 | 10 |
| Phase smoothing | Raw predictions | macro-F1 0.662 | — |
| Pressure smoothing | Raw predictions | macro-F1 0.436 | — |

Three-window smoothing was correctly rejected. Probability averaging reduced phase macro-F1
from 0.662 to 0.591 and pressure macro-F1 from 0.436 to 0.395; majority voting reduced them to
0.640 and 0.403 respectively.

## Development experiment comparison

| Experiment | Phase macro-F1 | Phase accuracy | Pressure macro-F1 | Pressure accuracy |
|---|---:|---:|---:|---:|
| R(2+1)D multi-task | **0.662** | **0.724** | 0.404 | 0.462 |
| MC3 multi-task | 0.656 | 0.707 | 0.421 | 0.478 |
| R3D multi-task | 0.631 | 0.672 | 0.431 | 0.480 |
| R(2+1)D phase-only | 0.636 | 0.710 | — | — |
| R(2+1)D pressure-only | — | — | **0.436** | **0.498** |
| Pressure-only, no identity mask | — | — | 0.423 | 0.483 |
| Pressure-only, hierarchical head | — | — | 0.407 | 0.470 |
| Multi-task, learning rate 1e-4 | 0.650 | 0.711 | 0.406 | 0.466 |
| Multi-task, pressure weight 1.0 | 0.632 | 0.705 | 0.398 | 0.454 |

Interpretation:

- Multi-task learning helped phase: R(2+1)D improved by 2.7 macro-F1 points over its phase-only
  version. Separate pressure training improved pressure by 3.3 points over the same multi-task
  model, so the final pipeline appropriately uses two classifier checkpoints.
- The identity mask improved pressure by 1.4 macro-F1 points over the no-mask ablation. This is
  a positive but modest gain, not evidence that identity tracking is solved.
- The hierarchical pressure head, higher pressure loss weight, lower learning rate, and both
  temporal smoothing methods were negative results.
- R(2+1)D had the best phase score, but MC3 was only 0.6 points behind. Pressure differences
  among the strongest candidates were similarly small, so claims about backbone superiority
  should remain modest.

### Selected phase model by class (development OOF)

| Phase | F1 | Support |
|---|---:|---:|
| Striking | 0.809 | 601 |
| Grappling/Ground Work | 0.791 | 103 |
| Clinch | 0.631 | 56 |
| Transition/Takedown | 0.614 | 48 |
| Neutral/Measuring Distance | 0.468 | 195 |

The largest phase weakness is Neutral/Measuring Distance, especially its confusion with
Striking. The two rare action classes performed substantially better than their class frequency
alone would suggest.

### Selected pressure model by class (development OOF)

| Pressure | F1 | Support |
|---|---:|---:|
| Fighter 1 | 0.287 | 244 |
| Fighter 2 | 0.387 | 233 |
| Mutual | 0.636 | 526 |

The classifier relies heavily on the majority Mutual class and remains weak at distinguishing
which individual fighter is applying pressure.

## One-shot untouched-fight result

| Component | Accuracy | Macro-F1 | Additional result |
|---|---:|---:|---|
| Gate | 0.971 | — | 100% fight retention; 73.7% non-fight rejection; AUC 0.973 |
| Phase | 0.724 | 0.495 | all 156 fight clips passed the gate |
| Pressure | 0.353 | 0.327 | all 156 fight clips passed the gate |

The gate made no false rejection of a real-fight clip (156/156 retained) and rejected 14 of 19
non-fight clips. Phase accuracy matched development OOF accuracy, but macro-F1 fell because the
holdout distribution is unusual: it contains only one Clinch clip, which was missed. Holdout F1
was 0.804 for Striking, 0.837 for Grappling, 0.381 for Transition/Takedown, 0.452 for Neutral,
and 0.000 for the single Clinch example.

Pressure shows a real generalization problem: macro-F1 dropped from 0.436 development OOF to
0.327 on the holdout. Fighter 1 F1 was only 0.135, Fighter 2 F1 was 0.341, and Mutual F1 was
0.504. This should be reported as the main limitation rather than hidden behind overall pipeline
accuracy.

## Report-ready conclusion

The system successfully filters non-fight footage and recognizes broad fight phase, including
strong Striking and Grappling performance. Fight-level splitting and the untouched-fight test
make these estimates substantially more credible than a clip-random split. Pressure direction
is not yet reliable: identity masks provide a small development benefit, but individual-fighter
pressure remains difficult and degrades on the held-out fight. The most defensible final claim is
therefore a successful gate-and-phase pipeline with pressure classification presented as an
experimental, limited extension.
