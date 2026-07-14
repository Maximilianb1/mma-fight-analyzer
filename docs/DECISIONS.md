# Design decisions

This document records the main methodological choices behind the final system. Quantitative results are in [EXPERIMENT_RESULTS_2026-07-14.md](EXPERIMENT_RESULTS_2026-07-14.md), and implementation details are in [TECHNICAL.md](TECHNICAL.md).

## Fight-level evaluation

Clips from the same fight share fighters, arena, lighting, graphics, and camera style. A random clip split would leak this information and overestimate generalization. We therefore use five development folds made from complete fights: each fold trains on eight fights and validates on two.

`Paddy Pimblett vs Michael Chandler` is reserved as an untouched final test. It is not used for architecture comparison, early stopping, gate-threshold selection, ablations, smoothing, or hyperparameter choices. Final models train on the ten development fights for epoch counts selected from development cross-validation, then run once on the held-out fight.

## Five-second windows and 16 frames

Fight phases develop over seconds, while adjacent frames at broadcast frame rate are highly redundant. Every five-second clip is represented by 16 evenly sampled frames. This matches the temporal length used by the pretrained R(2+1)D backbone and keeps training practical on a single GPU. Frames are cached once at short-side 128 and models receive 112 x 112 crops.

## Separate gate, phase, and pressure decisions

Non-fight footage has strong single-frame cues such as replay graphics, walkouts, and studio shots, so the gate uses a lightweight ImageNet-pretrained ResNet-18. It averages four frame probabilities per clip. The decision threshold is chosen from development out-of-fold predictions to retain at least 98% of live-fight clips while rejecting as much non-fight footage as possible.

R(2+1)D-18 performed best for phase. Pressure improved when trained as a separate pressure-only task, so deployment uses the multi-task checkpoint for phase and a pressure-only checkpoint for pressure. Three-window probability averaging and majority voting both reduced development macro-F1, so deployment uses raw five-second predictions.

## Fighter identity as a fourth channel

Pressure labels refer to broadcast identity, not screen position. Fighter 1 is the name left of the timer, and this fact is not inferable from motion alone. The model therefore receives an additional identity mask: `+1` inside Fighter 1's box, `-1` inside Fighter 2's box, and `0` where identity is unknown.

Training masks use YOLOv8 detections, comparative shorts-color assignment across the full clip, skin-excluded color evidence, and temporal IoU propagation. Ambiguous clips produce zero masks. This is intentional: missing identity is uninformative, while inverted identity creates adversarial supervision.

The first convolution of each pretrained video model is expanded from three to four channels. RGB weights are copied and the new mask channel is initialized to zero. The network therefore starts with the original pretrained RGB behavior and learns to use identity only when it improves the loss.

## Runtime identity tracking

Inference uses the same comparative shorts-color logic as preprocessing, fused with temporal appearance and position. A one-time A/B prompt establishes the fighter names. Trusted two-fighter assignments update temporal anchors; single detections, low color margins, and merged grappling boxes do not. When identity cannot be established safely, the video displays pressure as uncertain instead of guessing or swapping names.

## Optimization and imbalance

Both phase and pressure use class-weighted cross-entropy with label smoothing. Class weights are computed only from the training fights in each fold. All backbone layers are fine-tuned with a learning rate ten times smaller than the newly initialized heads. This preserves useful pretrained motion features while adapting them to the MMA domain.

## Known limitations and future work

- Pressure labels are more subjective and less transferable than broad phase labels.
- A generic person detector can merge fighters or select a referee during grappling.
- Shorts colors become unreliable under occlusion, unusual lighting, or similar clothing.
- The held-out estimate comes from one complete fight and therefore has high variance.

The clearest next steps are a fighter-specific detector and re-identification model, more independently labeled fights, and an inter-annotator agreement study for pressure.
