# Technical Documentation — MMA Fight Analyzer

This document explains **exactly** how the system works: every data transformation, every
tensor shape, every model, and every decision rule — at training time and at inference time.

Design rationale for each choice lives in [DECISIONS.md](DECISIONS.md); this file describes
*what the system does*, not *why we chose it*.

---

## 0. System at a glance

The system turns a raw UFC broadcast video into an annotated video where every 5-second
window is labeled with:

- **fight vs non-fight** (replays, walkouts, breaks are detected and skipped),
- **phase** — one of `Striking`, `Grappling/Ground Work`, `Clinch`, `Transition/Takedown`,
  `Neutral/Measuring Distance`,
- **pressure** — which fighter is pushing the action: `Fighter 1`, `Fighter 2`, or `Mutual`,
- **fighter bounding boxes** with persistent identities (F1 blue, F2 red).

Three learned models cooperate:

| Model | Backbone | Input | Output | Trained by |
|---|---|---|---|---|
| **Gate** | ResNet-18 (ImageNet) | 1 RGB frame `(3,112,112)` | 1 logit → P(non-fight) | `scripts/train_gate.py` |
| **Phase/Pressure A** | R(2+1)D-18 (Kinetics-400) | clip `(4,16,112,112)` | 5 phase logits + 3 pressure logits | `scripts/train_phase.py --model r2plus1d` |
| **Phase/Pressure B** | ResNet-18 + 2-layer LSTM | clip `(4,16,112,112)` | 5 phase logits + 3 pressure logits | `scripts/train_phase.py --model lstm` |

A and B are the two main architectures compared in the report. R(2+1)D was selected for
deployment: its multi-task checkpoint supplies phase, while a separately trained pressure-only
checkpoint supplies pressure. One non-learned **identity module** (YOLOv8 detection,
comparative shorts-color assignment, and temporal tracking) produces the 4th input channel that
tells the classifier *who is who*.

**Fighter 1 convention** (used consistently in labels, metadata, and the UI): the fighter whose
name appears **left of the timer** in the broadcast bottom overlay. Fighter 2 is on the right.

---

## 1. Dataset

Built with the Streamlit tool in `tools/labeler.py`: 11 full UFC fights were cut into
consecutive, non-overlapping 5-second clips (1280×720 @ 29.97 fps, ≈150 frames each).
For each clip the annotator either:

- assigned one **phase** label and one **pressure** label, or
- pressed **Skip/Exclude** → the clip is saved with an `_excluded` filename suffix and marked
  `excluded=True` (replay, walkout, round break, crowd shot…). Excluded clips carry no
  phase/pressure labels — they are the *positive class* for the gate model.

Totals: **1,315 clips, of which 1,159 are labeled fight clips and 156 are excluded.**

Label distribution (kept clips): Striking 56% · Neutral 18% · Grappling/Ground 16% ·
Transition/Takedown 5% · Clinch 5%. Pressure: Mutual 49% · Fighter 2 27% · Fighter 1 24%.
The imbalance is handled at training time (§5.3), not by resampling the data.

### 1.1 Raw data layout (`data/raw/`, created by `scripts/download_data.py`)

```
data/raw/<Fight Name>/
├── <Fight Name>_labels.csv
├── clip_0000_excluded.mp4      # excluded clip (suffix marks it)
├── clip_0002.mp4               # labeled fight clip
└── ...
```

CSV schema (one row per clip):

| column | type | meaning |
|---|---|---|
| `clip_index` | int | ordinal position in the fight |
| `start_time`,`end_time` | int (s) | source-video interval, always 5 s apart |
| `phase_label` | str/empty | one of the 5 phases; empty for excluded clips |
| `pressure_label` | str/empty | `Fighter 1`/`Fighter 2`/`Mutual`; empty for excluded |
| `excluded` | bool-ish | `True`/`False` — one file is Excel-mangled to `TRUE`/`FALSE`; the loader normalizes case |
| `saved_filename` | str | actual mp4 filename on disk |

`src/mma/data.py::discover_clips()` walks all fight folders (the CSV may sit inside the fight
folder or next to it), normalizes `excluded`, drops non-excluded rows with missing labels, and
returns one DataFrame with columns `fight, filename, clip_path, phase_label, pressure_label,
excluded`. Everything downstream consumes this frame.

### 1.2 Fight metadata (`data/fights_meta.csv`)

Per-fight shorts colors that anchor identity at *training* time:

```
fight,f1_color,f2_color,notes
Yoel Romero vs Paulo Costa,Red,Black,
```

Valid colors: `red, blue, black, white, green, gold, gray, orange, purple` (case-insensitive;
each maps to explicit HSV ranges, §3.2). If a fight is missing from the file its masks are all
zeros and the trainer warns — the phase task still works, pressure degrades (§3 explains why).

---

## 2. Preprocessing (`scripts/preprocess.py`) — run once

Converts every clip (kept **and** excluded) into a compact cache file so that training never
touches video again. Resumable: existing cache files are skipped.

For each clip:

1. **Decode 16 frames.** Read the mp4 sequentially, keep the frames whose indices are
   `round(linspace(0, n_frames-1, 16))` — i.e. one frame every ~0.33 s. Full resolution
   (1280×720), BGR.
2. **Detect fighters** (kept clips only, and only if the fight has colors in the meta file):
   YOLOv8-nano, person class only, confidence ≥ 0.35, per frame; detections sorted by box area,
   top 2 kept. Detection runs at **full resolution** — small/far fighters are found before any
   downscaling.
3. **Assign identity per frame** by shorts color (§3.3) → `(f1_box, f2_box)`, either may be None.
4. **Propagate identity** to ambiguous frames from temporal neighbors via IoU (§3.4).
5. **Rasterize the identity mask** at cache resolution `(128, 228)`: `+1` inside F1's box, `-1`
   inside F2's box, `0` elsewhere; where the two boxes overlap the region is set to `0`
   (ownership unknown). dtype `int8`.
6. **Resize frames** to height 128 (width 228 for 16:9), convert BGR→RGB, dtype `uint8`.
7. **Save** `data/cache/<fight>/<clip_stem>.npz` containing
   `frames: uint8 (16,128,228,3)` and `mask: int8 (16,128,228)`.
   ≈1 MB per clip, ≈1.5 GB for the full dataset.

Excluded clips get a zero mask (the gate doesn't use masks). At the end the script prints
per-fight **identity coverage** — the fraction of frames where both fighters were assigned; a
fight below 0.5 is flagged `<-- CHECK COLORS` (usually a wrong/ambiguous color in the meta CSV).

---

## 3. The identity module (`src/mma/identity.py`)

### 3.1 Why it exists

Pressure labels name a *person* ("Fighter 1 is pressuring"), but "Fighter 1" is broadcast
metadata (name left of the timer) — nothing in the action pixels encodes it. Two pixel-identical
clips can carry opposite pressure labels if the name order differs. Without identity input the
label is not a function of the input and no model can beat the base rate. The identity mask
turns "Fighter 1 pressuring" into the learnable pattern "the +1 region is advancing".

### 3.2 Shorts-color classification

A box's **shorts region** is rows 35–60% of box height, columns 15–85% of box width (trunk/shorts
area, avoids gloves and canvas). The region is converted to HSV; for each named color a pixel
mask is computed from fixed HSV ranges (e.g. `red = H∈[0,12]∪[168,180], S≥60, V≥60`;
`black = V≤70, S≤100`); the color with the highest matching-pixel ratio wins.

### 3.3 Clip-level pairing decision (`assign_identities`)

Detections are first linked into **tracks** across the 16 frames (greedy IoU linking, same
algorithm as the inference tracker); the two largest tracks are the fighter candidates
(a second track below 20% of the main track's area is discarded as crowd/referee junk).
Color evidence — the **non-skin** fraction of each shorts region matching each anchor color —
is averaged over each *whole track*, and the pairing is decided **once per clip**: assign the
combination (straight vs swapped) with the higher total evidence, requiring a minimum margin.
Single badly-lit frames get outvoted instead of getting a vote. Skin exclusion matters because
fighters are shirtless and skin hues overlap the red/orange/gold cloth ranges.

If the margin is too small, or only one usable track exists and its own evidence is ambiguous
(a merged fighter-pair box carries BOTH colors), the clip **abstains** — zero masks.
"Identity unknown" is harmless at training time; inverted identity is poison.

### 3.4 Temporal propagation

After the clip-level decision, frames not covered by the two main tracks are filled by
forward/backward sweeps: a neighbor frame's box is matched to the current frame's detections
by IoU (> 0.2 accepted — fighters move little in 0.33 s), and after matching one identity the
remaining detection becomes the other fighter. Still-unassigned frames contribute zero mask.

### 3.5 Guarantees and failure modes

The mask is **best-effort, not ground truth**: referees can steal a top-2 detection slot, both
fighters may wear similar shorts (see per-fight coverage stats), and clinches merge boxes (the
overlap region is zeroed on purpose). The classifier is trained with these imperfect masks and
learns to be robust to them; `--no-mask` (3-channel) training quantifies exactly how much the
channel contributes.

---

## 4. Training-time data pipeline (`src/mma/data.py`)

### 4.1 `PhaseClipDataset` — one item, step by step

```
npz  frames uint8 (16,128,228,3), mask int8 (16,128,228)
 →  frames float/255, permute        → (16,3,128,228), range [0,1]
 →  mask float, unsqueeze            → (16,1,128,228), values {-1,0,+1}
 →  augment_clip (train only, ONE param draw per clip, §4.2)
 →  frames normalized (per-backbone mean/std)
 →  frames permute → (3,16,112,112); mask permute → (1,16,112,112)
 →  concat → video (4,16,112,112)
 →  returns (video, phase_idx ∈ 0..4, pressure_idx ∈ 0..2)
```

Normalization stats: R(2+1)D uses Kinetics stats `mean=(0.432,0.395,0.376), std=(0.228,0.221,0.217)`;
the LSTM's ResNet encoder and the gate use ImageNet stats `mean=(0.485,0.456,0.406),
std=(0.229,0.224,0.225)`. The mask channel is never normalized.

### 4.2 Augmentation (train) — `augment_clip`

All parameters are sampled **once per clip** and applied identically to all 16 frames *and*
the mask, preserving temporal coherence and RGB↔mask alignment:

| transform | parameters | applied to mask? |
|---|---|---|
| RandomResizedCrop → 112×112 | scale (0.6, 1.0), ratio (4:3 … 16:9) | yes (nearest-neighbor) |
| Horizontal flip | p = 0.5 | yes |
| ColorJitter | brightness/contrast/saturation ×(0.7–1.3), hue ±0.03 | no (colors only) |

Wide crop ratios are deliberate: both train crops and the val transform map a wide region onto a
square, so the geometric distortion the model sees is consistent. Validation/inference:
plain resize of the full frame to 112×112 (content shrunk, never cropped away).

Horizontal flip is label-safe for pressure **because identity lives in the mask**: after the
flip the `+1` region still covers the same person, so "Fighter 1 pressuring" remains true.

### 4.3 `GateFrameDataset`

Each clip (kept *and* excluded) contributes 4 single frames (cache slots 0, 5, 10, 15).
Item = `(frame (3,112,112), label float)` with label `1.0` = excluded. Same augmentation
mechanics with a single frame and no mask; ImageNet normalization.

---

## 5. Models (`src/mma/models.py`)

### 5.1 Channel inflation (`_inflate_conv`)

Pretrained first convs expect 3 input channels; our clips have 4. A new conv is created with
identical geometry but `in_channels=4`; pretrained RGB weights are copied into channels 0–2 and
channel 3 (mask) weights are **zero-initialized**. At initialization the network output is
identical to the pretrained RGB model; mask weights grow only where gradient descent finds them
useful. Everything is then **fine-tuned** — no layer is frozen; the backbone uses a 10× lower
learning rate than newly initialized parts (§6.1).

### 5.2 Architectures

**Gate — `GateNet`** (≈11.2 M params): torchvision ResNet-18, final FC replaced by
`Linear(512→1)`. Input `(B,3,112,112)`, output `(B,)` logits.

**A — `R2Plus1DDual`** (≈31.3 M params): torchvision `r2plus1d_18` with Kinetics-400 weights.
Every 3D conv is factorized into a 2D spatial conv + 1D temporal conv. The stem's first conv is
inflated to 4 channels; the 400-way FC is replaced by `Identity`, exposing a 512-d clip
embedding. Input `(B,4,16,112,112)` → 512-d → heads.

**B — `ResNetLSTMDual`** (≈12.5 M params): each frame passes through an inflated
ImageNet ResNet-18 (`fc=Identity`) → per-frame 512-d features `(B,16,512)` → 2-layer LSTM,
hidden 256, dropout 0.3 → outputs **mean-pooled over time** → 256-d → heads. Space and time are
handled by separate modules, which is exactly the architectural contrast with model A.

**`DualHead`**: `Dropout(0.4 / 0.3)` → `Linear(feat→5)` for phase and `Linear(feat→3)` for
pressure. Either head can be disabled (`--task phase` / `--task pressure`) for the single-task
comparison; a missing head returns `None` and contributes nothing to the loss.

### 5.3 Checkpoint format

`torch.save({"state_dict": ..., "meta": {model_name, in_channels, with_phase, with_pressure,
phase_labels, pressure_labels}})` — `load_phase_model()` rebuilds the exact architecture from
`meta`, so inference needs only the `.pt` file.

---

## 6. Training procedure

### 6.1 Phase/pressure (`scripts/train_phase.py` → `train_utils.train_model`)

- **Loss** — `L = CE_phase + λ·CE_pressure`, λ = 0.5 (`--pressure-weight`). Both CE terms use
  label smoothing ε=0.05 and per-class weights `w_c = N/(K·n_c)` computed **on the training fold
  only** (Striking ≈ 0.36, Clinch ≈ 4.1 — a Clinch error costs ~11× a Striking error).
- **Optimizer** — AdamW, weight decay 1e-4, two parameter groups: pretrained backbone at
  `lr×0.1 = 3e-5`, everything new (heads, LSTM, inflated conv) at `lr = 3e-4`.
- **Precision** — mixed precision (`torch.amp.autocast` + `GradScaler`) on CUDA.
- **Schedule** — `ReduceLROnPlateau` (factor 0.5, patience 2) on the monitored metric:
  validation **phase macro-F1** (or pressure macro-F1 for `--task pressure`).
- **Early stopping** — patience 6 epochs on the same metric, max 25 epochs; the best-metric
  checkpoint is saved and reloaded before evaluation.
- **Batch size** — 8 (r2plus1d) / 12 (lstm) by default, tuned for a free-Colab T4.

### 6.2 Cross-validation protocol

`Paddy Pimblett vs Michael Chandler` is reserved as a one-fight final test set. It is never used
for early stopping, threshold or smoothing selection, model/architecture comparison, ablation,
or hyperparameter tuning. The other ten fights are deterministically paired into five folds:
each fold trains on eight complete fights and validates on two. The pairing balances clip count,
non-fight clips, phase labels, and pressure labels, and the exact same pairs are reused by every
candidate. Thus **no clip from a validation fight ever appears in its training fold**. `--lofo`
switches to leave-one-development-fight-out. Per fold the script saves: best checkpoint
(`<tag>_fold<i>.pt`), validation predictions with per-clip fight names
(`<tag>_fold<i>_preds.npz`), and the training history JSON. `<tag>` encodes the variant
(`r2plus1d`, `lstm_pressureonly`, `r2plus1d_nomask`, …) so ablations never overwrite the main run.

When all folds exist, development out-of-fold predictions are concatenated and reported (§7).
The untouched fight makes final evaluation independent even though early stopping happens inside
each development fold.

`--final` trains one model on all **ten development fights**. The notebook uses the median best
epoch from the five folds, freezes the model, and evaluates it exactly once on the untouched
fight. That checkpoint is also used by the inference demo.

### 6.3 Gate (`scripts/train_gate.py`)

The gate uses the same five development fight pairs and untouched test fight as phase/pressure.
`BCEWithLogitsLoss(pos_weight=N_neg/N_pos)` counters the 156-vs-1159 imbalance. AdamW lr 1e-4,
8 epochs, best-validation-AUC checkpoint per fold. Four frames are loaded with one NPZ decode,
flattened through ResNet-18, and averaged back to clip probability. Development OOF predictions
select a cost-aware threshold: maximize non-fight rejection subject to retaining at least 98% of
real fight clips. A final gate trains on all ten development fights for the median best epoch;
that frozen model and threshold are evaluated once on the holdout and stored in `gate.pt`.

---

## 7. Evaluation & artifacts

Out-of-fold aggregation (automatic once all folds finish) produces, per model tag:

- `classification_report` per task (precision/recall/F1 per class),
- confusion matrices `confusion_{phase,pressure}_<tag>.png`,
- `<tag>_metrics.json` — macro-F1 + accuracy per task, full report dict, **and a per-fight
  table** (n clips, phase acc, phase macro-F1, pressure acc) for failure analysis,
- `scripts/evaluate.py` renders the cross-model comparison table + bar chart
  (`model_comparison.png`).

Primary metric: **macro-F1** (each class counts equally — accuracy would reward predicting
Striking everywhere).

---

## 8. Inference pipeline (`scripts/infer.py` → `src/mma/pipeline.py`)

`python scripts/infer.py --video fight.mp4` processes the source in one pass. The Streamlit
interface calls the same `FightAnalyzer` class and adds upload, progress, preview, and download
controls.

Per 5-second window (`win_len = round(fps·5)` frames, last partial window included):

1. **Sample:** choose 16 evenly spaced frames.
2. **Gate:** four sampled frames are resized to 112² and passed through ResNet-18. Their mean
   sigmoid value is P(non-fight). A window above the checkpoint's frozen threshold is labeled
   non-fight and skips the downstream classifiers.
3. **Detect and link:** YOLOv8 detects at most two person boxes per sampled frame. Greedy IoU
   linking forms short tracks, which are ranked by cumulative box area.
4. **Assign identity:**
   - A one-time A/B prompt states which detected track is Fighter 1. Alternatively, both shorts
     colors can be supplied for unattended inference.
   - When two colors are known, the same comparative, non-skin color assignment used during
     preprocessing runs on every window. It scores the straight and swapped pairings across
     both complete tracks and requires a minimum evidence margin.
   - Trusted two-track assignments update appearance and final-position anchors. Temporal
     appearance plus boundary IoU is only used when two sufficiently long, separated tracks
     exist and the straight-vs-swapped margin is clear.
   - A single visible fighter may be identified by color but does not update both anchors.
     Merged boxes, high overlap, weak color evidence, and low temporal margins abstain. This
     prevents uncertain grappling frames from changing fighter identity.
5. **Classify:** the assigned boxes become the ±1 identity mask. The deployed multi-task
   R(2+1)D checkpoint supplies phase, and a pressure-only R(2+1)D checkpoint supplies pressure.
6. **Check pressure reliability:** pressure is named only when identity is available for both
   fighters in at least 25% of sampled frames. Otherwise the overlay says that pressure is
   uncertain, while phase remains visible.
7. **Overlay and write:** boxes are linearly interpolated between sampled frames. Every frame
   receives the fight status, phase, pressure (when reliable), and confidence values.

After the last window, FFmpeg remuxes the original audio. The Streamlit path also converts the
video to browser-compatible H.264 and enables fast start. A JSON timeline records gate,
classification, identity method, identity margin, identity coverage, and reliability for every
window.

---

## 9. Complete artifact reference

| path | producer | contents |
|---|---|---|
| `data/raw/<fight>/…` | `download_data.py` | clips + labels CSVs |
| `data/cache/<fight>/<clip>.npz` | `preprocess.py` | frames `(16,128,228,3)` u8, mask `(16,128,228)` i8 |
| `outputs/gate/gate.pt` | `train_gate.py --final` | dev-trained gate + OOF-selected threshold |
| `outputs/gate/gate_metrics.json` | gate aggregation | five-fold development OOF metrics |
| `outputs/gate/gate_holdout_metrics.json` | `train_gate.py --final` | untouched-fight gate test |
| `outputs/phase/<tag>_fold<i>.pt` | `train_phase.py` | best fold checkpoint (+meta) |
| `outputs/phase/<tag>_fold<i>_preds.npz` | `train_phase.py` | val y_true/y_pred per task + per-clip fight names |
| `outputs/phase/<tag>_metrics.json` | aggregation | OOF metrics, per-class report, per-fight table |
| `outputs/phase/confusion_*_<tag>.png` | aggregation | confusion matrices |
| `outputs/phase/model_comparison.png` | `evaluate.py` | cross-model bar chart |
| `outputs/phase/<tag>_final.pt` | `train_phase.py --final` | deployment model (10 dev fights) |
| `outputs/phase/<tag>_holdout_metrics.json` | `train_phase.py --final` | untouched-fight test |
| `outputs/report/pipeline_holdout_metrics.json` | holdout evaluator | frozen end-to-end test |
| `<video>_labeled.mp4` + `.json` | `infer.py` | annotated video + per-window log |
| `outputs/identity_prompt.png` | `infer.py` | the A/B frame shown at the identity prompt |

## 10. CLI quick reference

```bash
python scripts/download_models.py                       # frozen demo checkpoints
python scripts/download_data.py                         # Drive → data/raw
python scripts/preprocess.py [--yolo-conf 0.35]         # → data/cache  (once)
python scripts/train_gate.py --folds all                # 5-fold development OOF
python scripts/train_gate.py --final                    # dev train + one-shot holdout
python scripts/train_phase.py --model {r2plus1d,lstm}   # 5-fold development CV
        [--task both|phase|pressure] [--no-mask] [--lofo] [--folds 0,2]
        [--k 5] [--epochs 25] [--batch-size N] [--pressure-weight 0.5]
python scripts/train_phase.py --model r2plus1d --run-name deployment_phase --final --final-epochs 8
python scripts/train_phase.py --model r2plus1d --task pressure --run-name deployment_pressure --final --final-epochs 10
python scripts/evaluate.py [--models r2plus1d,lstm]     # comparison chart
python scripts/infer.py --video f.mp4 [--f1-name X --f2-name Y]
        [--f1-color red --f2-color black]               # non-interactive identity
```
