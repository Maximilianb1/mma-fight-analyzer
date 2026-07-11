"""Compare the trained models: prints a summary table and saves a comparison chart.

  python scripts/evaluate.py            # expects both r2plus1d and lstm metrics
  python scripts/evaluate.py --models r2plus1d
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="outputs/phase")
    p.add_argument("--models", default="r2plus1d,lstm")
    args = p.parse_args()

    out_dir = Path(args.out)
    models = args.models.split(",")
    results = {}
    for m in models:
        path = out_dir / f"{m}_metrics.json"
        if not path.exists():
            print(f"missing {path} — train that model first (all folds)")
            continue
        results[m] = json.loads(path.read_text())

    if not results:
        return

    print(f"\n{'model':<10} {'phase F1':>9} {'phase acc':>10} {'press F1':>9} {'press acc':>10}")
    for m, r in results.items():
        ph, pr = r.get("phase", {}), r.get("pressure", {})
        print(f"{m:<10} {ph.get('macro_f1', float('nan')):>9.3f} "
              f"{ph.get('accuracy', float('nan')):>10.3f} "
              f"{pr.get('macro_f1', float('nan')):>9.3f} "
              f"{pr.get('accuracy', float('nan')):>10.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, task in zip(axes, ("phase", "pressure")):
        names = [m for m in results if task in results[m]]
        x = np.arange(len(names))
        f1s = [results[m][task]["macro_f1"] for m in names]
        accs = [results[m][task]["accuracy"] for m in names]
        ax.bar(x - 0.18, accs, 0.36, label="Accuracy", color="#4C72B0")
        ax.bar(x + 0.18, f1s, 0.36, label="Macro F1", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1)
        ax.set_title(f"{task.title()} (out-of-fold, fight-level splits)")
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150)
    print(f"\nsaved {out_dir / 'model_comparison.png'}")


if __name__ == "__main__":
    main()
