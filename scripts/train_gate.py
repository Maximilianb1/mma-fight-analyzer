"""Train the fight/no-fight gate (excluded-clip detector) on single frames.

Positives are the clips marked "excluded" during labeling (replays, walkouts,
breaks); negatives are live-fight clips. The split is grouped by fight.

  python scripts/train_gate.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mma.data import GateFrameDataset, discover_clips  # noqa: E402
from mma.models import GateNet                          # noqa: E402
from mma.train_utils import set_seed                    # noqa: E402


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.extend(torch.sigmoid(model(x.to(device))).cpu().tolist())
        ys.extend(y.tolist())
    return np.array(probs), np.array(ys)


def clip_level(frame_probs, records, frames_per_clip):
    """Match inference: average the sampled-frame probabilities for each clip."""
    probs = frame_probs.reshape(len(records), frames_per_clip).mean(axis=1)
    labels = records.excluded.astype(int).to_numpy()
    return probs, labels


def save_plots(out_dir, history, ys, probs, threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    prec, rec, _ = precision_recall_curve(ys, probs)
    fpr, tpr, _ = roc_curve(ys, probs)
    pred = probs >= threshold
    cm = confusion_matrix(ys, pred, labels=[0, 1])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o")
    axes[0].set(title="Gate training loss", xlabel="Epoch", ylabel="BCE loss")
    axes[1].plot(fpr, tpr, label=f"AUC={roc_auc_score(ys, probs):.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set(title="Clip-level ROC", xlabel="False-positive rate",
                ylabel="True-positive rate")
    axes[1].legend()
    axes[2].plot(rec, prec, label=f"AP={average_precision_score(ys, probs):.3f}")
    axes[2].axvline(recall_score(ys, pred), linestyle="--", color="gray",
                    label=f"threshold={threshold:.3f}")
    axes[2].set(title="Clip-level precision-recall", xlabel="Recall", ylabel="Precision")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gate_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fight", "Non-fight"], yticklabels=["Fight", "Non-fight"], ax=ax)
    ax.set(xlabel="Predicted", ylabel="True", title="Gate confusion matrix (clip level)")
    fig.tight_layout()
    fig.savefig(out_dir / "gate_confusion.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out", default="outputs/gate")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = discover_clips(args.raw_dir, include_excluded=True)
    n_pos = int(records.excluded.sum())
    print(f"{len(records)} clips ({n_pos} excluded) / {records.fight.nunique()} fights on {device}")

    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=42)
    tr_idx, va_idx = next(gss.split(records, groups=records.fight))
    tr, va = records.iloc[tr_idx], records.iloc[va_idx]
    print(f"train fights: {sorted(tr.fight.unique())}")
    print(f"val fights:   {sorted(va.fight.unique())}")

    train_loader = DataLoader(GateFrameDataset(tr, args.cache_dir, train=True),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(GateFrameDataset(va, args.cache_dir, train=False),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = GateNet().to(device)
    pos_weight = torch.tensor((len(tr) - tr.excluded.sum()) / max(tr.excluded.sum(), 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = {"epoch": [], "train_loss": [], "frame_auc": [], "clip_auc": [],
               "clip_ap": []}
    best_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        probs, ys = collect(model, val_loader, device)
        frame_auc = roc_auc_score(ys, probs)
        clip_probs, clip_ys = clip_level(probs, va, len(val_loader.dataset.slots))
        auc = roc_auc_score(clip_ys, clip_probs)
        ap = average_precision_score(clip_ys, clip_probs)
        train_loss = running / n
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["frame_auc"].append(float(frame_auc))
        history["clip_auc"].append(float(auc))
        history["clip_ap"].append(float(ap))
        print(f"epoch {epoch}/{args.epochs} loss={train_loss:.4f} "
              f"frame AUC={frame_auc:.3f} clip AUC={auc:.3f} AP={ap:.3f}")
        if auc > best_auc:
            best_auc = auc
            prec, rec, thr = precision_recall_curve(clip_ys, clip_probs)
            f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
            threshold = float(thr[int(np.argmax(f1[:-1]))]) if len(thr) else 0.5
            torch.save({"state_dict": model.state_dict(),
                        "meta": {"threshold": threshold, "val_auc": float(auc),
                                 "val_ap": float(ap), "frame_val_auc": float(frame_auc)}},
                       out_dir / "gate.pt")

    ckpt = torch.load(out_dir / "gate.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    probs, _ = collect(model, val_loader, device)
    clip_probs, clip_ys = clip_level(probs, va, len(val_loader.dataset.slots))
    threshold = ckpt["meta"]["threshold"]
    pred = clip_probs >= threshold
    ckpt["meta"].update({
        "accuracy": float(accuracy_score(clip_ys, pred)),
        "precision": float(precision_score(clip_ys, pred, zero_division=0)),
        "recall": float(recall_score(clip_ys, pred, zero_division=0)),
        "f1": float(f1_score(clip_ys, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(clip_ys, pred, labels=[0, 1]).tolist(),
        "train_fights": sorted(tr.fight.unique().tolist()),
        "val_fights": sorted(va.fight.unique().tolist()),
        "n_train_clips": int(len(tr)), "n_val_clips": int(len(va)),
    })
    torch.save(ckpt, out_dir / "gate.pt")
    with open(out_dir / "gate_metrics.json", "w") as f:
        json.dump(ckpt["meta"], f, indent=2)
    with open(out_dir / "gate_history.json", "w") as f:
        json.dump(history, f, indent=2)
    np.savez(out_dir / "gate_val_predictions.npz", probability=clip_probs,
             target=clip_ys, prediction=pred.astype(int), fight=va.fight.values.astype(str),
             filename=va.filename.values.astype(str))
    save_plots(out_dir, history, clip_ys, clip_probs, threshold)
    print(f"best val AUC={ckpt['meta']['val_auc']:.3f}, "
          f"AP={ckpt['meta']['val_ap']:.3f}, F1={ckpt['meta']['f1']:.3f}, "
          f"operating threshold={threshold:.3f} -> {out_dir / 'gate.pt'}")


if __name__ == "__main__":
    main()
