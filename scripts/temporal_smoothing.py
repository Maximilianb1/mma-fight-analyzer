"""Evaluate simple within-fight temporal smoothing on out-of-fold predictions.

Requires prediction artifacts produced by the current train_phase.py, which save
softmax probabilities and filenames. No retraining is performed.

  python scripts/temporal_smoothing.py --tag r2plus1d --task phase --source oof --method all
  python scripts/temporal_smoothing.py --tag deployment_phase --task phase \
      --source holdout --method probability_mean
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mma import config as C  # noqa: E402


def clip_index(filename):
    match = re.search(r"clip_(\d+)", str(filename))
    return int(match.group(1)) if match else 0


def smooth_probabilities(probs, fights, filenames, window):
    """Centered moving average, independently within each fight."""
    out = probs.copy()
    radius = window // 2
    for fight in np.unique(fights):
        ids = np.flatnonzero(fights == fight)
        ids = ids[np.argsort([clip_index(filenames[i]) for i in ids])]
        for pos, idx in enumerate(ids):
            neighbors = ids[max(0, pos - radius):min(len(ids), pos + radius + 1)]
            out[idx] = probs[neighbors].mean(axis=0)
    return out


def majority_smoothing(pred, fights, filenames, window, n_classes):
    out = pred.copy()
    radius = window // 2
    for fight in np.unique(fights):
        ids = np.flatnonzero(fights == fight)
        ids = ids[np.argsort([clip_index(filenames[i]) for i in ids])]
        for pos, idx in enumerate(ids):
            neighbors = ids[max(0, pos - radius):min(len(ids), pos + radius + 1)]
            votes = np.bincount(pred[neighbors], minlength=n_classes)
            winners = np.flatnonzero(votes == votes.max())
            out[idx] = pred[idx] if len(winners) > 1 else int(winners[0])
    return out


def metrics(y, pred, labels):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "report": classification_report(y, pred, labels=range(len(labels)),
                                        target_names=labels, zero_division=0,
                                        output_dict=True),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(len(labels))).tolist(),
    }


def plot_comparison(path, task, labels, results, source):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    axes = np.atleast_1d(axes)
    for ax, (name, result) in zip(axes, results.items()):
        sns.heatmap(np.array(result["confusion_matrix"]), annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"{name}\nmacro-F1={result['macro_f1']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"{task.title()} temporal smoothing ({source})")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="outputs/phase")
    p.add_argument("--tag", required=True)
    p.add_argument("--task", choices=["phase", "pressure", "both"], default="phase")
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--source", choices=["oof", "holdout"], default="oof",
                   help="development OOF folds or the one-shot holdout prediction file")
    p.add_argument("--method", choices=["all", "probability_mean", "majority_vote"],
                   default="all",
                   help="use 'all' only for development selection; fix one method on holdout")
    args = p.parse_args()
    if args.window < 1 or args.window % 2 == 0:
        raise SystemExit("--window must be a positive odd number")

    out_dir = Path(args.out)
    files = (sorted(out_dir.glob(f"{args.tag}_fold*_preds.npz")) if args.source == "oof"
             else [out_dir / f"{args.tag}_holdout_preds.npz"])
    files = [path for path in files if path.exists()]
    if not files:
        raise SystemExit(f"no prediction files for tag {args.tag!r} under {out_dir}")
    parts = [np.load(f) for f in files]
    fights = np.concatenate([x["fight"] for x in parts]).astype(str)
    filenames = np.concatenate([x["filename"] for x in parts]).astype(str)

    tasks = ["phase", "pressure"] if args.task == "both" else [args.task]
    all_results = {"tag": args.tag, "window": args.window, "source": args.source,
                   "method": args.method, "tasks": {}}
    for task in tasks:
        labels = C.PHASE_LABELS if task == "phase" else C.PRESSURE_LABELS
        y = np.concatenate([x[f"{task}_true"] for x in parts])
        if len(y) == 0:
            continue
        if f"{task}_prob" not in parts[0]:
            raise SystemExit("prediction files predate probability saving; rerun evaluation/training")
        probs = np.concatenate([x[f"{task}_prob"] for x in parts])
        raw = probs.argmax(axis=1)
        mean_pred = smooth_probabilities(probs, fights, filenames, args.window).argmax(axis=1)
        majority_pred = majority_smoothing(raw, fights, filenames, args.window, len(labels))
        candidates = {
            "probability_mean": mean_pred,
            "majority_vote": majority_pred,
        }
        results = {"raw": metrics(y, raw, labels)}
        selected = candidates if args.method == "all" else {args.method: candidates[args.method]}
        results.update({name: metrics(y, pred, labels) for name, pred in selected.items()})
        all_results["tasks"][task] = results
        print(f"\n{task.upper()} / {args.tag}")
        for name, result in results.items():
            print(f"  {name:<18} acc={result['accuracy']:.3f} macro-F1={result['macro_f1']:.3f}")
        plot_comparison(out_dir / f"smoothing_{args.source}_{task}_{args.tag}.png",
                        task, labels, results, args.source)

    path = out_dir / f"smoothing_{args.source}_{args.tag}_{args.task}.json"
    path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
