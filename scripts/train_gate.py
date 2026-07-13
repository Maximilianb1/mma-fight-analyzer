"""Train and evaluate the fight/non-fight gate without fight leakage.

Protocol used by the submission notebook:

* reserve one complete fight as an untouched final test set;
* split the other ten fights into five fixed validation pairs;
* select epochs and the operating threshold from development OOF predictions;
* train one deployment model on the ten development fights;
* evaluate that frozen model and threshold once on the held-out fight.

Examples:
  python scripts/train_gate.py --folds all
  python scripts/train_gate.py --folds 0,2,4
  python scripts/train_gate.py --aggregate-only
  python scripts/train_gate.py --final
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mma import config as C                                      # noqa: E402
from mma.data import GateClipDataset, discover_clips             # noqa: E402
from mma.models import GateNet                                   # noqa: E402
from mma.train_utils import make_holdout_folds, set_seed         # noqa: E402


def make_loader(records, cache_dir, train, batch_size, workers):
    return DataLoader(GateClipDataset(records, cache_dir, train),
                      batch_size=batch_size, shuffle=train,
                      num_workers=workers, pin_memory=True, drop_last=False)


@torch.no_grad()
def collect(model, loader, device):
    """Return frame probabilities, clip probabilities, and clip labels."""
    model.eval()
    frame_probs, clip_probs, labels = [], [], []
    for frames, y in loader:
        b, slots, channels, height, width = frames.shape
        logits = model(frames.reshape(b * slots, channels, height, width).to(device))
        probs = torch.sigmoid(logits).reshape(b, slots).cpu().numpy()
        frame_probs.append(probs.reshape(-1))
        clip_probs.append(probs.mean(axis=1))
        labels.append(y.numpy().astype(int))
    return (np.concatenate(frame_probs), np.concatenate(clip_probs),
            np.concatenate(labels))


def threshold_metrics(labels, probs, threshold):
    pred = probs >= threshold
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "auc": float(roc_auc_score(labels, probs)),
        "ap": float(average_precision_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "fight_retention": float(tn / max(tn + fp, 1)),
        "nonfight_rejection": float(tp / max(tp + fn, 1)),
        "confusion_matrix": cm.tolist(),
    }


def choose_thresholds(labels, probs, min_fight_retention):
    """Choose thresholds using development OOF predictions only."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    max_f1_threshold = (float(thresholds[int(np.argmax(f1[:-1]))])
                        if len(thresholds) else 0.5)

    candidates = np.unique(np.r_[0.0, probs, 1.0 + 1e-7])
    valid = []
    for threshold in candidates:
        result = threshold_metrics(labels, probs, float(threshold))
        if result["fight_retention"] >= min_fight_retention:
            valid.append(result)
    # Maximize removal of non-fight clips subject to preserving real fight clips.
    # Precision and F1 break ties; the lower threshold is the final tie-breaker.
    safe = max(valid, key=lambda r: (r["nonfight_rejection"], r["precision"],
                                     r["f1"], -r["threshold"]))
    return safe["threshold"], max_f1_threshold


def save_plots(out_dir, stem, histories, labels, probs, threshold, title_suffix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    precision, recall, _ = precision_recall_curve(labels, probs)
    fpr, tpr, _ = roc_curve(labels, probs)
    metrics = threshold_metrics(labels, probs, threshold)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for fold, history in enumerate(histories):
        axes[0].plot(history["epoch"], history["train_loss"], marker="o",
                     label=f"fold {fold}")
    axes[0].set(title="Gate training loss", xlabel="Epoch", ylabel="BCE loss")
    if histories:
        axes[0].legend(fontsize=8)
    axes[1].plot(fpr, tpr, label=f"AUC={metrics['auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set(title=f"Clip-level ROC ({title_suffix})", xlabel="False-positive rate",
                ylabel="True-positive rate")
    axes[1].legend()
    axes[2].plot(recall, precision, label=f"AP={metrics['ap']:.3f}")
    axes[2].axvline(metrics["recall"], linestyle="--", color="gray",
                    label=f"fixed threshold={threshold:.3f}")
    axes[2].set(title=f"Clip-level precision-recall ({title_suffix})", xlabel="Recall",
                ylabel="Precision")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.asarray(metrics["confusion_matrix"]), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fight", "Non-fight"], yticklabels=["Fight", "Non-fight"], ax=ax)
    ax.set(xlabel="Predicted", ylabel="True",
           title=f"Gate confusion matrix ({title_suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_confusion.png", dpi=180)
    plt.close(fig)


def train_fold(fi, train_records, val_records, pair, args, device, out_dir):
    set_seed(C.RANDOM_SEED + fi)
    train_loader = make_loader(train_records, args.cache_dir, True,
                               args.batch_size, args.workers)
    val_loader = make_loader(val_records, args.cache_dir, False,
                             args.batch_size, args.workers)
    model = GateNet().to(device)
    positives = int(train_records.excluded.sum())
    pos_weight = torch.tensor((len(train_records) - positives) / max(positives, 1),
                              device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    history = {"epoch": [], "train_loss": [], "frame_auc": [],
               "clip_auc": [], "clip_ap": []}
    best_auc = -1.0
    checkpoint = out_dir / f"gate_fold{fi}.pt"

    print(f"fold {fi + 1}/{args.k}: train {len(train_records)} clips; "
          f"validate {len(val_records)} clips from {list(pair)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_frames = 0
        for frames, y in train_loader:
            b, slots, channels, height, width = frames.shape
            frames = frames.reshape(b * slots, channels, height, width).to(device)
            y = y.to(device).repeat_interleave(slots)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                loss = criterion(model(frames), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(y)
            n_frames += len(y)

        frame_probs, clip_probs, labels = collect(model, val_loader, device)
        frame_labels = np.repeat(labels, C.GATE_FRAMES)
        frame_auc = roc_auc_score(frame_labels, frame_probs)
        clip_auc = roc_auc_score(labels, clip_probs)
        clip_ap = average_precision_score(labels, clip_probs)
        train_loss = running / max(n_frames, 1)
        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["frame_auc"].append(float(frame_auc))
        history["clip_auc"].append(float(clip_auc))
        history["clip_ap"].append(float(clip_ap))
        print(f"[gate f{fi}] epoch {epoch}/{args.epochs} loss={train_loss:.4f} "
              f"frame AUC={frame_auc:.3f} clip AUC={clip_auc:.3f} AP={clip_ap:.3f}")
        if clip_auc > best_auc:
            best_auc = clip_auc
            torch.save({"state_dict": model.state_dict(),
                        "meta": {"fold": fi, "best_epoch": epoch,
                                 "val_auc": float(clip_auc), "val_ap": float(clip_ap),
                                 "frame_val_auc": float(frame_auc),
                                 "val_fights": list(pair)}}, checkpoint)

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    frame_probs, clip_probs, labels = collect(model, val_loader, device)
    np.savez(out_dir / f"gate_fold{fi}_preds.npz",
             frame_probability=frame_probs, probability=clip_probs, target=labels,
             fight=val_records.fight.values.astype(str),
             filename=val_records.filename.values.astype(str),
             val_fights=np.asarray(pair))
    (out_dir / f"gate_fold{fi}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")


def aggregate_folds(out_dir, pairs, args):
    # Remove filenames from the superseded single-holdout implementation so a
    # packaged output cannot accidentally mix the two protocols.
    for legacy_name in ("gate_history.json", "gate_val_predictions.npz"):
        (out_dir / legacy_name).unlink(missing_ok=True)
    prediction_paths = [out_dir / f"gate_fold{i}_preds.npz" for i in range(args.k)]
    missing = [str(path) for path in prediction_paths if not path.exists()]
    if missing:
        raise SystemExit("missing gate fold predictions:\n  " + "\n  ".join(missing))
    parts = [np.load(path) for path in prediction_paths]
    probs = np.concatenate([part["probability"] for part in parts])
    labels = np.concatenate([part["target"] for part in parts])
    fights = np.concatenate([part["fight"] for part in parts]).astype(str)
    filenames = np.concatenate([part["filename"] for part in parts]).astype(str)
    threshold, max_f1_threshold = choose_thresholds(
        labels, probs, args.min_fight_retention)
    oof = threshold_metrics(labels, probs, threshold)
    oof["max_f1_threshold"] = float(max_f1_threshold)
    oof["min_fight_retention_constraint"] = args.min_fight_retention

    fold_metrics = []
    histories = []
    best_epochs = []
    for fi, (part, pair) in enumerate(zip(parts, pairs)):
        fold_result = threshold_metrics(part["target"], part["probability"], threshold)
        fold_result.update({"fold": fi, "val_fights": list(pair),
                            "n_clips": int(len(part["target"]))})
        fold_metrics.append(fold_result)
        histories.append(json.loads((out_dir / f"gate_fold{fi}_history.json").read_text()))
        state = torch.load(out_dir / f"gate_fold{fi}.pt", map_location="cpu",
                           weights_only=False)
        best_epochs.append(int(state["meta"]["best_epoch"]))

    per_fight = {}
    for fight in sorted(np.unique(fights)):
        mask = fights == fight
        per_fight[fight] = threshold_metrics(labels[mask], probs[mask], threshold)
        per_fight[fight]["n_clips"] = int(mask.sum())

    result = {
        "protocol": "5-fold paired fight CV on 10 development fights; 1 untouched holdout",
        "positive_class": "Non-fight",
        "holdout_fight": args.holdout_fight,
        "fold_pairs": [list(pair) for pair in pairs],
        "oof": oof,
        "fold_metrics": fold_metrics,
        "per_fight": per_fight,
        "recommended_final_epochs": int(np.median(best_epochs)),
        "n_development_clips": int(len(labels)),
    }
    (out_dir / "gate_metrics.json").write_text(json.dumps(result, indent=2),
                                                encoding="utf-8")
    np.savez(out_dir / "gate_oof_predictions.npz", probability=probs, target=labels,
             prediction=(probs >= threshold).astype(int), fight=fights, filename=filenames)
    save_plots(out_dir, "gate", histories, labels, probs, threshold, "development OOF")
    print(f"OOF AUC={oof['auc']:.3f} AP={oof['ap']:.3f} F1={oof['f1']:.3f} "
          f"fight retention={oof['fight_retention']:.3f} at threshold={threshold:.3f}")
    print(f"recommended final epochs: {result['recommended_final_epochs']}")
    return result


def train_final(dev, test, args, device, out_dir):
    metrics_path = out_dir / "gate_metrics.json"
    if not metrics_path.exists():
        raise SystemExit("run all development folds and --aggregate-only before --final")
    cv = json.loads(metrics_path.read_text(encoding="utf-8"))
    if cv.get("holdout_fight") != args.holdout_fight or "oof" not in cv:
        raise SystemExit("gate_metrics.json is not from the requested holdout-CV protocol")
    epochs = args.final_epochs or int(cv["recommended_final_epochs"])
    threshold = float(cv["oof"]["threshold"])

    set_seed()
    loader = make_loader(dev, args.cache_dir, True, args.batch_size, args.workers)
    model = GateNet().to(device)
    positives = int(dev.excluded.sum())
    pos_weight = torch.tensor((len(dev) - positives) / max(positives, 1), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    history = {"epoch": [], "train_loss": []}
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_frames = 0
        for frames, y in loader:
            b, slots, channels, height, width = frames.shape
            frames = frames.reshape(b * slots, channels, height, width).to(device)
            y = y.to(device).repeat_interleave(slots)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                loss = criterion(model(frames), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(y)
            n_frames += len(y)
        train_loss = running / max(n_frames, 1)
        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        print(f"[gate final] epoch {epoch}/{epochs} loss={train_loss:.4f}")

    test_loader = make_loader(test, args.cache_dir, False, args.batch_size, args.workers)
    frame_probs, probs, labels = collect(model, test_loader, device)
    holdout = threshold_metrics(labels, probs, threshold)
    holdout.update({"protocol": "one-shot untouched fight test",
                    "holdout_fight": args.holdout_fight,
                    "n_clips": int(len(test)), "threshold_source": "development OOF"})
    checkpoint_meta = {
        "threshold": threshold,
        "model_name": "gate",
        "holdout_fight": args.holdout_fight,
        "trained_fights": sorted(dev.fight.unique().tolist()),
        "final_epochs": epochs,
        "development_oof": cv["oof"],
        "holdout_test": holdout,
    }
    torch.save({"state_dict": model.state_dict(), "meta": checkpoint_meta},
               out_dir / "gate.pt")
    (out_dir / "gate_final_history.json").write_text(json.dumps(history, indent=2),
                                                      encoding="utf-8")
    (out_dir / "gate_holdout_metrics.json").write_text(json.dumps(holdout, indent=2),
                                                        encoding="utf-8")
    np.savez(out_dir / "gate_holdout_predictions.npz",
             frame_probability=frame_probs, probability=probs, target=labels,
             prediction=(probs >= threshold).astype(int),
             fight=test.fight.values.astype(str), filename=test.filename.values.astype(str))
    save_plots(out_dir, "gate_holdout", [], labels, probs, threshold, "untouched holdout")
    print(f"HOLDOUT {args.holdout_fight}: AUC={holdout['auc']:.3f} "
          f"AP={holdout['ap']:.3f} F1={holdout['f1']:.3f} "
          f"fight retention={holdout['fight_retention']:.3f}")
    print(f"deployment checkpoint -> {out_dir / 'gate.pt'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--out", default="outputs/gate")
    parser.add_argument("--holdout-fight", default=C.DEFAULT_HOLDOUT_FIGHT)
    parser.add_argument("--k", type=int, default=C.DEV_FOLDS)
    parser.add_argument("--folds", default="all", help="'all' or comma list, e.g. 0,2,4")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--final", action="store_true",
                        help="train on development fights and test once on the holdout")
    parser.add_argument("--final-epochs", type=int, default=None,
                        help="default: median best epoch from development folds")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="clips per batch (each clip contributes four frames)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-fight-retention", type=float, default=0.98,
                        help="OOF threshold must retain at least this fraction of fight clips")
    args = parser.parse_args()
    if not 0 < args.min_fight_retention <= 1:
        raise SystemExit("--min-fight-retention must be in (0, 1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = discover_clips(args.raw_dir, include_excluded=True)
    dev, test, folds, pairs = make_holdout_folds(
        records, holdout_fight=args.holdout_fight, n_splits=args.k)
    print(f"{len(records)} clips / {records.fight.nunique()} fights on {device}")
    print(f"untouched holdout: {args.holdout_fight} ({len(test)} clips)")
    for fi, pair in enumerate(pairs):
        print(f"development fold {fi}: {list(pair)}")

    if args.aggregate_only:
        aggregate_folds(out_dir, pairs, args)
        return
    if args.final:
        train_final(dev, test, args, device, out_dir)
        return

    wanted = range(args.k) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    for fi in wanted:
        if not 0 <= fi < args.k:
            raise SystemExit(f"fold index {fi} outside 0..{args.k - 1}")
        train_idx, val_idx = folds[fi]
        train_fold(fi, dev.iloc[train_idx], dev.iloc[val_idx], pairs[fi],
                   args, device, out_dir)
    if args.folds == "all":
        aggregate_folds(out_dir, pairs, args)
    else:
        print("selected folds complete; run --aggregate-only after all five are present")


if __name__ == "__main__":
    main()
