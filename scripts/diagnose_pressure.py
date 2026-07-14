"""Detect swapped fighter-identity anchors per fight.

If a fight's f1_color/f2_color in data/fights_meta.csv are swapped relative to how
Fighter 1/2 were defined during labeling (name left of the broadcast timer), the
identity mask contradicts that fight's pressure labels: the model predicts
"Fighter 1" where the label says "Fighter 2" and vice versa. Symptom: per-fight
pressure accuracy BELOW chance, fixed by swapping F1<->F2 in the predictions.

Run after a full K-fold run:
  python scripts/diagnose_pressure.py [--out outputs/phase] [--tag r2plus1d]
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="outputs/phase")
    p.add_argument("--tag", default="r2plus1d")
    args = p.parse_args()

    files = sorted(Path(args.out).glob(f"{args.tag}_fold*_preds.npz"))
    if not files:
        raise SystemExit(f"no {args.tag}_fold*_preds.npz under {args.out}")

    fights, yt, yp = [], [], []
    for f in files:
        d = np.load(f)
        if len(d["pressure_true"]) == 0:
            continue
        fights.append(d["fight"])
        yt.append(d["pressure_true"])
        yp.append(d["pressure_pred"])
    fights, yt, yp = map(np.concatenate, (fights, yt, yp))
    yp_swapped = np.select([yp == 0, yp == 1], [1, 0], default=2)

    print(f"{'fight':<42} {'n':>4} {'straight':>9} {'swapped':>8}  verdict")
    suspects = []
    for fight in sorted(np.unique(fights)):
        m = fights == fight
        straight = float(np.mean(yt[m] == yp[m]))
        swapped = float(np.mean(yt[m] == yp_swapped[m]))
        verdict = ""
        if swapped - straight > 0.10:
            if m.sum() >= 20:
                verdict = "<-- LIKELY SWAPPED COLORS"
                suspects.append(fight)
            else:
                verdict = "(swap scores higher, but n too small to be sure)"
        print(f"{fight:<42} {m.sum():>4} {straight:>9.3f} {swapped:>8.3f}  {verdict}")

    print(
        f"\noverall: straight={np.mean(yt == yp):.3f}  swapped={np.mean(yt == yp_swapped):.3f}"
    )
    if suspects:
        print(
            f"\n{len(suspects)} fight(s) likely have f1_color/f2_color swapped in data/fights_meta.csv:"
        )
        for s in suspects:
            print(f"  - {s}")
        print(
            "\nFix: swap the two colors for those fights in fights_meta.csv, delete their"
        )
        print(
            "folders under data/cache/, rerun scripts/preprocess.py (only they re-process),"
        )
        print(
            "then retrain. Phase is barely affected; pressure supervision becomes consistent."
        )
    else:
        print("\nNo swap signature found - low pressure accuracy is not explained by")
        print(
            "swapped color anchors (see DECISIONS.md B8 and the label-noise discussion)."
        )


if __name__ == "__main__":
    main()
