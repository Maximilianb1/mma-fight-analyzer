# Design decisions, rationale & experiment backlog

Working log for the report. Each entry: what we chose, why, and what to write in the report.
"Backlog" items are things to try only if time remains (report+code deadline: 16/07/2026).

---

## 1. Decided

### D1 — Fight-level cross-validation (no clip-level splits)
Folds are grouped by fight (`StratifiedGroupKFold`, default K=4); a validation fight never
contributes clips to training. Clip-level splits would leak fighters/arena/lighting and inflate
scores. `--lofo` runs leave-one-fight-out (11 folds) as a stricter variant.
**Report:** state this explicitly — it is the main methodological claim of the evaluation.

### D2 — No separate held-out test set; early stopping uses the val fold
With 11 fights, a frozen test set of 2–3 fights would be tiny and high-variance. Out-of-fold
CV predictions already come from models that never saw those fights. The one impurity: the
early-stopping *epoch choice* is tuned on the same fold we report, giving a small optimistic
bias; the clean fix (nested CV) costs ~3× compute we don't have.
**Report sentence:** "Model selection (early stopping) uses the validation fold, so results may
be slightly optimistic; we mitigate this by reporting aggregated results across 4 disjoint
fight-level folds and noting it as a limitation."

### D3 — Identity as a 4th input channel (+1 = F1, −1 = F2, 0 = background)
Pressure labels refer to *broadcast metadata* (name left of the timer), invisible in pixels;
without identity the label is not a function of the input and the task is unlearnable.
Masks come from YOLOv8 boxes + per-fight shorts-color anchoring + temporal IoU propagation
(ambiguous frames borrow identity from the nearest confident frame, boxes matched by IoU > 0.2,
forward then backward pass).
**Report:** explain the unlearnability argument — it justifies the whole identity module.

### D4 — Inflated first conv = pure fine-tuning setup
Both backbones are pretrained (Kinetics-400 / ImageNet). The first conv is rebuilt with 4 input
channels: RGB weights copied, mask-channel weights zero-initialized → at init the network is
bit-for-bit the pretrained RGB model; gradient descent grows mask weights only where useful.
Training is standard fine-tuning: ALL weights update, backbone at 10× lower LR
(`backbone_lr_factor=0.1`) than the freshly initialized heads/LSTM.
**Report:** describe as "transfer learning / fine-tuning with a zero-initialized extra input
channel"; note that at initialization the model is exactly the pretrained one.

### D5 — 16 frames per clip @ 112×112
5 s × 30 fps = 150 frames is ~9× redundant compute; phases live at the seconds scale.
16 frames matches the Kinetics pretraining clip length of R(2+1)D-18; 112×112 matches its
pretraining resolution. Frames are cached at short-side 128 so training never decodes video.
Resizing (not cropping) is used at val time, so corner action is *shrunk, not discarded*;
YOLO runs at full 1280×720 before any resize, so masks are computed at full resolution.
**Backlog knob:** if phase confusions look resolution-bound, raise `CACHE_SHORT_SIDE`/`CROP_SIZE`
(e.g. 160/144, ~2× compute) — requires re-running preprocess.

### D6 — Multi-task loss L = CE_phase + 0.5·CE_pressure, class-weighted, label-smoothed (ε=0.05)
Shared backbone + two heads: one training run, shared motion features, mutual regularization.
Pressure down-weighted (0.5) because its labels are subjective/noisier. Class weights
w_c = N/(K·n_c) counter the 56%-Striking / 5%-Clinch imbalance; label smoothing tempers
overconfidence on noisy 5 s labels.
**Comparison built in:** `--task phase` / `--task pressure` train single-task variants →
"one multi-task model vs two single-task models" experiment (outputs tagged `*_phaseonly` /
`*_pressureonly`, never overwriting the main run).

### D7 — Gate = frame-level ResNet-18, 4 frames per clip
Non-fight evidence (replay graphics, walkout, commentary desk) is visible in any single frame —
no temporal model needed. 4 frames per clip: more positives (153 clips → ~600 frames) while
extra frames from the same clip are near-duplicates (highly correlated), so >4 adds compute,
not information. Inference averages 4 sigmoid probabilities; threshold chosen by max-F1 on the
val precision-recall curve (stored in the checkpoint).
**Backlog knob:** if gate AUC is borderline, average more frames at inference (free) before
touching the model.

### D8 — Box interpolation in the overlay (cosmetic only)
Detection runs on 16 of ~150 frames; drawing only those would freeze/jump every 9 frames.
Box coords are linearly interpolated between neighboring sampled frames:
box(t) = (1−α)·box(t_i) + α·box(t_{i+1}). Fighters move ~continuously over 0.3 s so linear looks
smooth. The classifier never sees these boxes — display only.

### D9 — Confidence display (implemented)
Overlay shows top-class softmax probability ("Phase: Striking (81%)"); banner dims/grays when
phase confidence < 0.5 (`overlay.LOW_CONF`). Honest UX: low-confidence windows are usually
genuine transitions. Per-window confidences are also written to the inference JSON log.

### D10 — Augmentation params drawn once per clip
Crop/flip/color-jitter sampled once and applied to all 16 frames AND the identity mask —
preserves temporal coherence and mask↔RGB alignment. Horizontal flip is label-safe for pressure
*because* identity travels in the mask, not in screen position.

---

## 2. Hyperparameters to tune on validation if time remains

| Hyperparameter | Default | Where | Plausible range / note |
|---|---|---|---|
| Pressure loss weight | 0.5 | `--pressure-weight` | 0.3–1.0; raise if pressure lags & phase is fine |
| Learning rate (heads) | 3e-4 | `--lr` | 1e-4–1e-3 |
| Backbone LR factor | 0.1 | `train_utils.train_model` | 0.05–0.3; or freeze backbone first epochs |
| Label smoothing | 0.05 | `train_utils` | 0–0.1 |
| Weight decay | 1e-4 | `train_utils` | 1e-5–1e-3 |
| Batch size | 8 (r2+1d) / 12 (lstm) | `--batch-size` | max that fits the GPU |
| Frames per clip | 16 | `config.NUM_FRAMES` | 8/24/32 — needs preprocess re-run |
| Input resolution | 112 (cache 128) | `config` | 144/160 — needs preprocess re-run |
| Crop scale range | (0.6, 1.0) | `data.augment_clip` | tighter = milder augmentation |
| Gate frames/clip | 4 | `config.GATE_FRAMES` | 2–8 |
| Gate threshold | max-F1 on val | `train_gate.py` | pick by desired precision/recall tradeoff |
| LSTM hidden/layers | 256 / 2 | `models.ResNetLSTMDual` | 128–512 / 1–2 |
| Track-link IoU | 0.3 | `pipeline.TRACK_LINK_IOU` | 0.2–0.5 |
| Anchor EMA | 0.85 | `pipeline.ANCHOR_EMA` | 0.7–0.95 |
| Mask propagation IoU | 0.2 | `identity.MIN_PROPAGATION_IOU` | 0.1–0.4 |
| YOLO confidence | 0.35 | `--yolo-conf` (preprocess) | 0.25–0.5 |

---

## 3. Backlog (try only if time; each is a report paragraph if it works)

- **B1 — Other pretrained video backbones** instead of R(2+1)D-18: MC3-18 / R3D-18
  (same torchvision API, drop-in), X3D or a small video transformer (TimeSformer, VideoMAE)
  via other libs. Also worth trying: freeze the backbone entirely and train heads only
  (linear-probe baseline) — quantifies how much fine-tuning buys.
- **B2 — Temporal smoothing at inference** (UNDECIDED — test before adopting): candidates:
  (a) majority filter over 3 windows; (b) averaging softmax probabilities of neighboring
  windows before argmax; (c) transition prior / simple HMM over phase sequence.
  Evaluate on out-of-fold predictions (no retraining needed) before using in the demo.
- **B3 — Leave-one-fight-out CV** (`--lofo`, implemented): cleaner story, 11 folds, ~3× compute.
  Run if GPU time allows; report alongside the 4-fold numbers.
- **B4 — Per-fight results table** (implemented, automatic in `aggregate`): use as the
  failure-analysis section; check whether hard fights correlate with low identity-mask coverage.
- **B5 — Inter-annotator agreement**: both annotators relabel ~50 clips independently,
  report Cohen's κ — turns "labels are subjective" into a measured statement.
- **B6 — Pressure-weight sweep** (0.3/0.5/1.0) as the multi-task ablation if single-task
  comparison shows interesting interference.
- **B7 — Higher input resolution** (see D5 knob) if confusion matrix suggests distance-related
  errors (e.g. Neutral vs Striking at range).
- **B8 — Combined checkpoint monitor** (observed in r2plus1d fold 0, 2026-07-11): pressure acc
  was still climbing (0.30→0.46 by ep.10) when phase F1 peaked (ep.4-5); checkpointing on phase
  F1 alone froze pressure at its weak early state. Candidate fix: monitor
  `phase_F1 + 0.5·pressure_F1` in `train_utils.train_model`. Decide after both models' 4-fold
  runs finish (changing mid-experiment would break comparability).
