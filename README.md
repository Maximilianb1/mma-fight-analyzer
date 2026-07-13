# MMA Fight Analyzer — Phase & Pressure Recognition in UFC Broadcasts

Final project for **046217 Deep Learning** (Technion, Spring 2026).
Maximilian Bershtman · Reut Yosefa Vitzner

Given a raw UFC broadcast video, the system splits it into 5-second windows, filters out
non-fight segments (replays, walkouts, breaks), tracks the two fighters, classifies **what is
happening** (fight phase) and **who is pushing the action** (pressure), and writes back an
annotated video with boxes and labels.

```
 raw fight video
      │  split into 5s windows
      ▼
 ┌─────────────┐  non-fight   ┌──────────────────────────┐
 │  GATE       │─────────────▶│ "NON-FIGHT SEGMENT" tag  │
 │  ResNet-18  │              └──────────────────────────┘
 └─────┬───────┘
       │ fight
       ▼
 ┌─────────────────────────┐   one-time user prompt:
 │ YOLOv8 person detection │   "which box is Fighter 1?"
 │ + IoU tracking          │──▶ identity anchored, then propagated
 │ + identity masks (±1)   │   across windows via appearance histograms
 └─────┬───────────────────┘
       │ RGB (3ch) + identity mask (1ch), 16 frames @ 112×112
       ▼
 ┌──────────────────────────────┐
 │ PHASE+PRESSURE model         │  two architectures compared:
 │  A: R(2+1)D-18 (Kinetics)    │  phase ∈ {Striking, Grappling/Ground,
 │  B: ResNet-18 + LSTM         │   Clinch, Transition/Takedown, Neutral}
 │  shared backbone, dual heads │  pressure ∈ {Fighter 1, Fighter 2, Mutual}
 └─────┬────────────────────────┘
       ▼
 annotated video (boxes + phase + pressure per 5s window)
```

## Repository layout

```
├── README.md
├── requirements.txt
├── data/
│   ├── fights_meta.csv      # per-fight shorts colors (identity anchor for training)
│   ├── raw/                 # downloaded dataset (gitignored)
│   └── cache/               # preprocessed clips (gitignored)
├── scripts/
│   ├── download_data.py     # fetch dataset ZIPs from Google Drive
│   ├── preprocess.py        # cache 16 frames + identity mask per clip (run once)
│   ├── train_gate.py        # fight/no-fight gate
│   ├── train_phase.py       # phase+pressure models, fight-level K-fold CV
│   ├── evaluate.py          # model comparison table + charts
│   └── infer.py             # full pipeline on a new video
├── src/mma/
│   ├── config.py            # labels, sizes, normalization stats
│   ├── data.py              # dataset discovery + PyTorch datasets + clip-consistent augmentation
│   ├── identity.py          # YOLO detection, shorts-color identity, temporal smoothing, masks
│   ├── models.py            # GateNet, R(2+1)D/R3D/MC3/LSTM heads + hierarchical pressure
│   ├── train_utils.py       # training loop (AMP, early stopping), grouped folds, metrics
│   ├── overlay.py           # drawing boxes/banners on output video
│   └── pipeline.py          # end-to-end inference
├── notebooks/
│   ├── colab_train.ipynb    # thin Colab wrapper that runs the scripts above
│   ├── kaggle_full_experiments.ipynb # complete T4x2 experiments + selection + demo
│   └── archive/             # earlier notebook-based experiments
└── tools/
    └── labeler.py           # Streamlit tool used to build the dataset
```

## Dataset

We built the dataset ourselves: 11 full UFC fights cut into 5-second clips and labeled with
a custom Streamlit tool ([tools/labeler.py](tools/labeler.py)) — ~1300 clips total, of which
~1160 are live-fight clips with phase + pressure labels and ~150 are marked *excluded*
(replays/walkouts/breaks; these train the gate). **Fighter 1** is always the fighter whose name
appears left of the timer in the broadcast overlay.

The dataset is hosted on Google Drive as one ZIP per fight. `scripts/download_data.py`
downloads and unpacks everything into `data/raw/`.

## Reproducing the experiments

```bash
pip install -r requirements.txt

# 1. data
python scripts/download_data.py                 # set the Drive folder ID inside first
python scripts/preprocess.py                    # one-time cache: frames + identity masks

# 2. gate: 5 development folds, aggregate OOF, then frozen holdout test
python scripts/train_gate.py --folds all
python scripts/train_gate.py --final

# 3. phase + pressure, 5-fold fight-level development CV
python scripts/train_phase.py --model r2plus1d
python scripts/train_phase.py --model lstm
python scripts/evaluate.py                      # comparison table + charts

# 4. ablations reported in the paper
python scripts/train_phase.py --model r2plus1d --no-mask        # no identity channel
python scripts/train_phase.py --model r2plus1d --task phase     # single-task: phase only
python scripts/train_phase.py --model r2plus1d --task pressure  # single-task: pressure only
python scripts/train_phase.py --model r2plus1d --lofo           # leave-one-fight-out CV

# 5. train on all 10 development fights, test the untouched fight, then demo
python scripts/train_phase.py --model r2plus1d --final
python scripts/infer.py --video path/to/fight.mp4 --f1-name "Fighter A" --f2-name "Fighter B"
```

On Colab open `notebooks/colab_train.ipynb`, which runs the same scripts on a GPU runtime.
Every fold checkpoints separately, so a disconnected session resumes with
`--folds i` for the missing folds.

For the complete submission experiment suite, open
`notebooks/kaggle_full_experiments.ipynb` on Kaggle with a T4 x2 accelerator. It runs the gate,
multi-task/single-task comparison, identity-mask ablation, hierarchical-pressure experiment,
R3D/MC3 baselines, temporal smoothing, a small tuning pilot, final full-data training, and an
end-to-end demo. Independent experiments are pinned to the two GPUs in parallel, and all
metrics, probabilities, plots, checkpoints, and logs are archived from `outputs/`.

### Method notes

- **Untouched final fight.** `Paddy Pimblett vs Michael Chandler` is excluded from model
  selection, early stopping, threshold selection, architecture comparison, ablations, and
  hyperparameter tuning. It contains all phase/pressure classes plus non-fight clips, so the
  frozen final pipeline can be evaluated once on a meaningful test fight.
- **Fight-level development cross-validation.** The other ten fights form five deterministic
  folds with exactly two validation fights and eight training fights each. Gate, phase,
  pressure, architectures, and ablations reuse the identical fight pairs; no fighter/arena
  appearance crosses from a validation fight into its training fold.
- **Selection versus testing.** OOF development predictions select models, epoch counts,
  temporal smoothing, and the gate threshold. Final models train on the ten development fights
  and are evaluated once on the untouched fight. The holdout is never folded back into tuning.
- **Identity channel.** The pressure task ("who is pushing the action") is only well-defined if
  the model knows who Fighter 1 is. We attach a 4th input channel: +1 over Fighter 1's box,
  −1 over Fighter 2's, 0 elsewhere, produced by YOLOv8 + per-fight shorts-color anchoring +
  temporal IoU smoothing.
- **Clip-consistent augmentation.** Crop/flip/color-jitter parameters are drawn once per clip
  and applied to all 16 frames *and* the identity mask, preserving temporal coherence.
- **Transfer learning.** Both classifiers start from pretrained weights (Kinetics-400 /
  ImageNet); the pretrained backbone uses a 10× lower learning rate than the new heads.

Full technical documentation — every tensor shape, decision rule, and pipeline stage for both
training and inference — is in [docs/TECHNICAL.md](docs/TECHNICAL.md) (or the rendered
[docs/technical.html](docs/technical.html) with pipeline diagrams). Design decisions, rationale,
and the experiment backlog are documented in [docs/DECISIONS.md](docs/DECISIONS.md).
The diagnosis and repair of the pressure task's corrupted identity supervision (including
negative results) is written up as a case study in
[docs/PRESSURE_INVESTIGATION.md](docs/PRESSURE_INVESTIGATION.md).

## Results

Historical out-of-fold results from the earlier 4-fold protocol (kept for comparison until the
new fixed-holdout experiment finishes):

| Model | Phase macro-F1 | Phase acc | Pressure macro-F1 | Pressure acc |
|---|---|---|---|---|
| R(2+1)D-18 | **0.680** | **0.729** | **0.390** | **0.446** |
| ResNet-18 + LSTM | 0.523 | 0.607 | 0.370 | 0.397 |

Per-class, per-fight breakdowns and confusion matrices: `outputs/phase/*_metrics.json`,
`confusion_*.png`, `model_comparison.png`. The diagnosis and repair of an identity-supervision
corruption that initially held pressure at chance is documented in
[docs/PRESSURE_INVESTIGATION.md](docs/PRESSURE_INVESTIGATION.md).

## Ethics

An ethics statement (stakeholders, implications, considerations) is included in the project
report, following the course template.
