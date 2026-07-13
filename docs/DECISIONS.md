# Design decisions, rationale & experiment backlog

Working log for the report. Each entry: what we chose, why, and what to write in the report.
"Backlog" items are things to try only if time remains (report+code deadline: 16/07/2026).

---

## 1. Decided

### D1 — Fight-level development CV (no clip-level splits)
The ten development fights are partitioned into five deterministic validation pairs: each fold
trains on eight fights and validates on two. A validation fight never contributes clips to its
training fold. The gate, classifiers, architectures, ablations, and tuning candidates reuse the
exact same pairs. Clip-level splits would leak fighters/arena/lighting and inflate scores.
**Report:** state this explicitly — it is the main methodological claim of the evaluation.

### D2 — One untouched final test fight (revised 2026-07-13)
`Paddy Pimblett vs Michael Chandler` is excluded from all development folds and every decision:
early stopping, threshold selection, smoothing selection, architectures, ablations, and tuning.
It was chosen before model results because its 156 fight clips cover all five phase and all three
pressure classes, while its 19 excluded clips also test the gate. Final models train on the ten
development fights for the median best CV epoch and are evaluated exactly once on Paddy–Chandler.
The single-fight test is still high-variance, so development OOF/per-fight results remain part of
the report; the holdout prevents the final headline result from reusing selection data.

### D3 — Identity as a 4th input channel (+1 = F1, −1 = F2, 0 = background)
Pressure labels refer to *broadcast metadata* (name left of the timer), invisible in pixels;
without identity the label is not a function of the input and the task is unlearnable.
Masks come from YOLOv8 boxes + per-fight shorts-color anchoring + temporal IoU propagation.
**Report:** explain the unlearnability argument — it justifies the whole identity module.

**Revised 2026-07-11 after the first training round** (first-round pressure results exposed
mask inversions, diagnosed via a straight-vs-swapped prediction test + visual frame audits):
- **Clip-level decision** replaces per-frame color naming: detections are linked into tracks
  (greedy IoU) and the straight-vs-swapped pairing is decided ONCE per clip from evidence
  aggregated over whole tracks — single badly-lit frames get outvoted.
- **Comparative scoring**: choose the pairing that best explains BOTH tracks' colors, instead of
  naming each box's color absolutely (white shorts with black trim used to out-'black' dark
  maroon shorts and invert Jones–Cormier systematically).
- **Skin exclusion**: shorts regions on shirtless fighters are heavily skin-contaminated and
  skin hues overlap red/orange/gold cloth; ratios are computed over non-skin pixels only.
  Also: red requires S≥140 (skin pollution), white allows V≥150 (arena shadows).
- **Abstention over guessing**: small evidence margins, junk second tracks (area < 20% of the
  main track), and merged fighter-pair boxes yield ZERO masks — "identity unknown" is harmless
  to training, inverted identity is poison.
- **Jones–Cormier 2 is intentionally maskless** (blank colors in fights_meta.csv): both
  fighters dark-skinned, maroon vs shadowed-white shorts, heavy clinch — color anchoring
  stayed unreliable after all fixes (verified visually), so its 152 clips train with
  identity-unknown masks rather than corrupted ones.
- **Known limitation (report!):** in grapple-heavy clips the two fighters merge into one YOLO
  box; identity is then unresolvable by this module and masks are zero or partially wrong
  (occasionally a referee/crowd box slips in as the second track). Rejected fixes: referee
  filtering by torso-skin (fails across skin tones), canvas-below-feet test (fails at the
  fence). A learned detector/tracker (e.g. fine-tuned fighter detector) is the real fix —
  future work.

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
not information. Inference averages 4 sigmoid probabilities. Five-fold development OOF selects
a cost-sensitive threshold that maximizes non-fight rejection while retaining at least 98% of
real-fight clips; the fixed threshold is tested once on the untouched fight and stored in the
checkpoint. The optimized loader decodes each NPZ once for all four frames.
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
| Gate threshold | ≥98% fight retention on dev OOF | `train_gate.py` | vary constraint, never holdout |
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
- **B3 — Leave-one-development-fight-out CV** (`--lofo`, implemented): optional 10-fold
  sensitivity analysis; the final Paddy–Chandler holdout remains untouched.
- **B4 — Per-fight results table** (implemented, automatic in `aggregate`): use as the
  failure-analysis section; check whether hard fights correlate with low identity-mask coverage.
- **B5 — Inter-annotator agreement**: both annotators relabel ~50 clips independently,
  report Cohen's κ — turns "labels are subjective" into a measured statement.
- **B6 — Pressure-weight sweep** (0.3/0.5/1.0) as the multi-task ablation if single-task
  comparison shows interesting interference.
- **B7 — Higher input resolution** (see D5 knob) if confusion matrix suggests distance-related
  errors (e.g. Neutral vs Striking at range).
- **B8 — Combined checkpoint monitor** — IMPLEMENTED 2026-07-11: `train_utils.train_model` now
  monitors `phase_F1 + 0.5·pressure_F1` (was: phase F1 only, which froze pressure at its weak
  early state — observed in r2plus1d fold 0 where pressure was still climbing 0.30→0.46 when
  phase peaked). Bundled into the identity-fix retrain so both architectures rerun under
  identical rules; first-round numbers are not comparable to second-round numbers.
