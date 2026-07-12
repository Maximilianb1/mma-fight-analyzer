# Case study: diagnosing and fixing the pressure task (2026-07-11)

Raw material for the report's failure-analysis / method section. Everything here is
reproducible from the repo; round-1 artifacts are in the saved `mma_outputs` (Drive) and
round-2 artifacts in `outputs_round2`. Companion docs: [DECISIONS.md](DECISIONS.md) (D3, B8),
[TECHNICAL.md](TECHNICAL.md) §3-4.

## 1. Symptom

Round-1 training (4-fold fight-level CV, free-Colab T4):

| task | r2plus1d | lstm |
|---|---|---|
| phase acc / macro-F1 (OOF) | **0.744 / 0.673** | 0.627 / 0.539 |
| pressure acc / macro-F1 (OOF) | 0.381 / 0.338 | 0.407 / 0.340 |

Phase was healthy. Pressure was effectively unlearned: accuracy *below* the all-Mutual
baseline (0.49), Fighter-1/Fighter-2 classes at ~0.22 F1 vs a 0.33 chance floor.

## 2. Diagnosis — four steps

### 2.1 The prediction-inversion ("swap") test — `scripts/diagnose_pressure.py`
Re-score each fight's out-of-fold predictions with Fighter 1 <-> Fighter 2 swapped.
Noise doesn't benefit from swapping; **systematic inversion does**. Round-1 results
(r2plus1d): Amanda Nunes 0.100 straight vs 0.700 swapped; Jones-Cormier 2 0.316 vs 0.520
(n=152); overall 0.381 vs 0.439. A 3-class accuracy of 0.10 is only achievable by being
consistently backwards — something in the supervision was inverted.

### 2.2 Ruling out annotation error
Broadcast screenshots confirmed `fights_meta.csv` colors match the Fighter-1 = left-of-timer
convention for every checked fight. The labels themselves passed sanity checks (e.g. the
one-sided Nunes-Rousey fight is labeled Fighter-1-pressuring, correct). The corruption was
NOT human error in labels or metadata.

Subtlety worth reporting: fights *flagged* by the swap test are not necessarily the corrupted
ones. A model trained on partly-poisoned data learns a partly-inverted decision rule, so
CONSISTENT fights can look inverted when scored against it. The test localizes a problem;
attribution needs step 2.3/2.4.

### 2.3 Cross-model agreement
The CNN (r2plus1d) and LSTM are independently trained; data corruption should replicate across
both, model noise should not. Only **Jones-Cormier 2** inverted robustly in both models.
Five other flagged fights disagreed between models -> noise from a barely-learned task.

### 2.4 Visual audit of the identity masks
Running the preprocessing pipeline locally and rendering the assignments frame-by-frame showed
the actual bug: **Jones's white shorts have black side panels and a black waistband that scored
"black 34%", beating Cormier's dark-MAROON (not black) shorts at "black 20%"** — so the
identity mask routinely marked Jones as Fighter 1 while the (correct) labels said Cormier.
152 clips (13% of the dataset, the largest fight) of anti-supervision, whose gradients oppose
the clean fights' gradients and drag the whole task toward chance.

The audit surfaced two more defects:
- the **"red" HSV range matched dark skin** (a clinch box scored "red 87%" with no red cloth
  in frame) — a latent threat to every red-anchored fight;
- in grapple scrambles the two fighters merge into one YOLO box and a **crowd/commentator/
  referee box gets promoted to "second fighter"**.

## 3. Fixes applied (commit `9753d46`)

| # | failure | fix |
|---|---|---|
| 1 | trim/waistband fools absolute color naming | **comparative pairing**: score straight-vs-swapped over BOTH boxes; contamination cancels differentially |
| 2 | skin pollutes color evidence | ratios over **non-skin pixels only**; red needs S>=140; white tolerates V>=150 (shadow) |
| 3 | single bad frames flip identity | **clip-level decision**: link detections into tracks, aggregate evidence over whole tracks, decide the pairing once per clip |
| 4 | wrong guesses poison training | **abstention**: small margins / junk tracks (<20% main-track area) / merged pair-boxes -> zero masks ("identity unknown" is harmless; inversion is poison) |
| 5 | Jones-Cormier 2 unreliable after all fixes (~1/3 of clips still wrong on visual audit: two dark-skinned fighters, maroon vs shadowed white, constant clinch) | fight is **deliberately maskless** (blank colors in meta) — its clips train identity-unknown |
| 6 | checkpoint froze pressure early (B8): pressure acc still climbing 0.30->0.46 when phase F1 peaked at epoch 4-6, and we saved the phase-best epoch | monitor **phase_F1 + 0.5·pressure_F1** for checkpoint/early-stop |

Also corrected: Topuria-Oliveira anchors (black/orange — Oliveira's shorts are orange-gold,
not black).

## 4. Approaches tried and REJECTED (negative results)

- **Referee filtering by torso-skin ratio** (shirtless fighter = high skin fraction, clothed
  referee = low): no single HSV skin band covers all complexions — dark-skinned (Edwards,
  Blaydes) and pale (Pimblett) fighters were misclassified as "clothed" and dropped. Reverted.
- **Canvas-below-the-feet test** (fighters stand on the bright low-saturation canvas, crowd
  sits against dark backgrounds): fails for fighters pressed against the fence, and the
  referee stands on the canvas too — coverage collapsed on clean fights. Reverted.
- **Dummy-anchor trick** (identify only the reliable fighter, complete the pair): measurably
  worse than two-sided comparative evidence (60% vs 75% on a height-proxy metric).

Remaining known limitation: in grapple-heavy clips, merged fighter boxes make identity
unresolvable for this module (masks zero or partially wrong; occasionally a spectator box
slips in). The principled fix is a learned fighter detector/tracker fine-tuned on octagon
footage — future work.

## 5. Verification & expected round-2 readout

- Identity coverage after fixes (4-clip samples/fight): 0.75-1.00 "both" on 9 fights,
  Topuria 0.27 (abstains on its ground-heavy clips — intended), JJDC maskless by design.
- Correctness spot-checks: annotated frames verified visually per fight (Romero, Pimblett,
  Van, Prates, TJ, Amanda, Lopes all correct in standing/separated frames).
- **Round-2 receipt:** `diagnose_pressure.py` should show straight > swapped overall with no
  flagged fights. Round-1 vs round-2 numbers are NOT directly comparable (identity data AND
  checkpoint rule both changed — deliberately bundled into one retrain).
- If pressure improves well above the 0.49 Mutual baseline: report as "supervision corruption
  identified and removed" with the before/after. If it improves only modestly: remaining gap
  is dominated by label subjectivity ("who is pushing" over an arbitrary 5-second window) —
  quantifiable via an inter-annotator agreement study (backlog B5).

## 6. Report-ready sentences

> During error analysis we found that per-fight pressure accuracy on two fights was
> significantly *below* chance. A prediction-inversion test (re-scoring with Fighter 1/2
> swapped) combined with cross-architecture agreement localized the cause to one fight whose
> automatically generated identity masks were systematically inverted: the HSV shorts-color
> classifier scored the white-shorted fighter's black trim above his opponent's dark-maroon
> shorts. We reworked identity assignment to make a single clip-level decision by comparing
> which fighter pairing better explains the color evidence of both tracked fighters over
> skin-excluded pixels, and configured the pipeline to abstain (zero mask) rather than guess
> under ambiguity, since an absent identity channel is uninformative while an inverted one is
> adversarial supervision. One fight remained unreliable after these fixes and was deliberately
> trained without identity masks. Separately, we changed model selection to monitor a combined
> phase+pressure metric after observing that phase-only checkpointing systematically saved
> models whose pressure head had not yet converged.
