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
│   ├── models.py            # GateNet, R(2+1)D-18 dual-head, ResNet-18+LSTM dual-head
│   ├── train_utils.py       # training loop (AMP, early stopping), grouped folds, metrics
│   ├── overlay.py           # drawing boxes/banners on output video
│   └── pipeline.py          # end-to-end inference
├── notebooks/
│   ├── colab_train.ipynb    # thin Colab wrapper that runs the scripts above
│   └── archive/             # earlier notebook-based experiments
└── tools/
    └── labeler.py           # Streamlit tool used to build the dataset
```

## Dataset

We built the dataset ourselves: 12 full UFC fights cut into 5-second clips and labeled with
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

# 2. gate (fight / non-fight)
python scripts/train_gate.py

# 3. phase + pressure, 4-fold fight-level CV, both architectures
python scripts/train_phase.py --model r2plus1d
python scripts/train_phase.py --model lstm
python scripts/evaluate.py                      # comparison table + charts

# 4. ablations reported in the paper
python scripts/train_phase.py --model r2plus1d --no-mask        # no identity channel
python scripts/train_phase.py --model r2plus1d --task phase     # single-task: phase only
python scripts/train_phase.py --model r2plus1d --task pressure  # single-task: pressure only
python scripts/train_phase.py --model r2plus1d --lofo           # leave-one-fight-out CV

# 5. train the deployment model on all fights + run the demo
python scripts/train_phase.py --model r2plus1d --final
python scripts/infer.py --video path/to/fight.mp4 --f1-name "Fighter A" --f2-name "Fighter B"
```

On Colab open `notebooks/colab_train.ipynb`, which runs the same scripts on a GPU runtime.
Every fold checkpoints separately, so a disconnected session resumes with
`--folds i` for the missing folds.

### Method notes

- **Fight-level cross-validation.** Folds are grouped by fight (`StratifiedGroupKFold`), so
  validation fights are never seen in training — clip-level splits would leak fighter/arena
  appearance and inflate scores.
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

*(to be filled after the final runs — see `outputs/phase/model_comparison.png`,
`confusion_*.png`, and `outputs/phase/*_metrics.json`)*

| Model | Phase macro-F1 | Phase acc | Pressure macro-F1 | Pressure acc |
|---|---|---|---|---|
| R(2+1)D-18 | – | – | – | – |
| ResNet-18 + LSTM | – | – | – | – |

## Ethics

An ethics statement (stakeholders, implications, considerations) is included in the project
report, following the course template.
