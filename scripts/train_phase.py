"""Train the phase+pressure classifier with fight-level K-fold cross-validation.

Examples (from the repo root):
  python scripts/train_phase.py --model r2plus1d
  python scripts/train_phase.py --model lstm --folds 0,1         # resume fold-by-fold
  python scripts/train_phase.py --model r2plus1d --task phase    # single-task: phase only
  python scripts/train_phase.py --model r2plus1d --task pressure # single-task: pressure only
  python scripts/train_phase.py --model r2plus1d --no-mask       # RGB-only ablation
  python scripts/train_phase.py --model r2plus1d --lofo          # leave-one-fight-out CV
  python scripts/train_phase.py --model r2plus1d --final         # train on ALL fights for inference

Each fold saves a checkpoint + validation predictions, so a Colab disconnect
only loses the current fold. When all folds are present, out-of-fold predictions
are aggregated into a confusion matrix and a metrics JSON.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mma import config as C                                       # noqa: E402
from mma.data import PhaseClipDataset, discover_clips             # noqa: E402
from mma.models import MODEL_INPUT_STATS, build_phase_model       # noqa: E402
from mma.train_utils import (class_weights, evaluate, make_folds,  # noqa: E402
                             make_lofo_folds, set_seed, train_model)

DEFAULT_BATCH = {"r2plus1d": 8, "r3d": 8, "mc3": 12, "lstm": 12}


def make_loader(records, cache_dir, train, mean, std, use_mask, batch_size, workers):
    ds = PhaseClipDataset(records, cache_dir, train, mean, std, use_mask)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=workers, pin_memory=True, drop_last=False)


def plot_history(out_dir, tag, fold, history):
    """Save a compact report-ready training curve for each fold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(history["train_loss"])
    epochs = np.arange(1, n + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="train")
    if history["val_loss"]:
        axes[0].plot(epochs[:len(history["val_loss"])], history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Weighted loss")
    axes[0].legend()
    if history["val_phase_f1"] and any(x is not None for x in history["val_phase_f1"]):
        axes[1].plot(epochs[:len(history["val_phase_f1"])],
                     [np.nan if x is None else x for x in history["val_phase_f1"]],
                     label="phase macro-F1")
    if history["val_pressure_f1"] and any(x is not None for x in history["val_pressure_f1"]):
        axes[1].plot(epochs[:len(history["val_pressure_f1"])],
                     [np.nan if x is None else x for x in history["val_pressure_f1"]],
                     label="pressure macro-F1")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1))
    axes[1].legend()
    fig.suptitle(f"{tag} / fold {fold}")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"{tag}_fold{fold}_training.png", dpi=180)
    plt.close(fig)


def aggregate(out_dir, model_name, k):
    """Merge per-fold predictions into out-of-fold metrics + confusion matrices."""
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    parts = [np.load(out_dir / f"{model_name}_fold{i}_preds.npz") for i in range(k)]
    result = {"model": model_name, "k_folds": k}
    for task, labels in (("phase", C.PHASE_LABELS), ("pressure", C.PRESSURE_LABELS)):
        y_true = np.concatenate([p[f"{task}_true"] for p in parts])
        y_pred = np.concatenate([p[f"{task}_pred"] for p in parts])
        if len(y_true) == 0:
            continue
        result[task] = {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "accuracy": float(np.mean(y_true == y_pred)),
            "report": classification_report(y_true, y_pred, target_names=labels,
                                            zero_division=0, output_dict=True),
        }
        print(f"\n== {model_name} / {task} (out-of-fold) ==")
        print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{task.title()} — {model_name} (OOF, fight-level folds)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(out_dir / f"confusion_{task}_{model_name}.png", dpi=150)
        plt.close(fig)
    # per-fight breakdown — failure-analysis table for the report
    fights = np.concatenate([p["fight"] for p in parts])
    ph_true = np.concatenate([p["phase_true"] for p in parts])
    ph_pred = np.concatenate([p["phase_pred"] for p in parts])
    pr_true = np.concatenate([p["pressure_true"] for p in parts])
    pr_pred = np.concatenate([p["pressure_pred"] for p in parts])
    has_phase = len(ph_true) == len(fights) > 0
    has_pressure = len(pr_true) == len(fights) > 0
    per_fight = {}
    print(f"\n{'fight':<42} {'n':>4} {'ph_acc':>7} {'ph_F1':>6} {'pr_acc':>7}")
    for fight in sorted(np.unique(fights)):
        m = fights == fight
        row = {"n_clips": int(m.sum())}
        if has_phase:
            row["phase_acc"] = float(np.mean(ph_true[m] == ph_pred[m]))
            row["phase_macro_f1"] = float(f1_score(ph_true[m], ph_pred[m],
                                                   average="macro", zero_division=0))
        if has_pressure:
            row["pressure_acc"] = float(np.mean(pr_true[m] == pr_pred[m]))
        per_fight[fight] = row
        ph_a = f"{row['phase_acc']:.3f}" if has_phase else "-"
        ph_f = f"{row['phase_macro_f1']:.3f}" if has_phase else "-"
        pr_a = f"{row['pressure_acc']:.3f}" if has_pressure else "-"
        print(f"{fight:<42} {row['n_clips']:>4} {ph_a:>7} {ph_f:>6} {pr_a:>7}")
    result["per_fight"] = per_fight

    with open(out_dir / f"{model_name}_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved metrics + confusion matrices to {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["r2plus1d", "r3d", "mc3", "lstm"], required=True)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out", default="outputs/phase")
    p.add_argument("--k", type=int, default=4, help="number of fight-level folds")
    p.add_argument("--lofo", action="store_true",
                   help="leave-one-fight-out CV (one fold per fight) instead of --k folds; "
                        "writes to outputs/phase_lofo unless --out is given")
    p.add_argument("--folds", default="all", help="'all' or comma list, e.g. 0,2")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--backbone-lr-factor", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--pressure-weight", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--task", choices=["both", "phase", "pressure"], default="both",
                   help="'both' = multi-task (default); 'phase'/'pressure' = single-task "
                        "variants for the one-model-vs-two comparison")
    p.add_argument("--no-mask", action="store_true", help="RGB-only ablation (3 channels)")
    p.add_argument("--pressure-head", choices=["flat", "hierarchical"], default="flat",
                   help="flat 3-way head or Mutual-vs-Directional + F1-vs-F2 factorization")
    p.add_argument("--run-name", default=None,
                   help="explicit artifact tag; useful for sweeps and prevents overwrites")
    p.add_argument("--final", action="store_true",
                   help="train one model on ALL fights (for the inference demo)")
    p.add_argument("--final-epochs", type=int, default=12)
    args = p.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.lofo and args.out == "outputs/phase":
        args.out = "outputs/phase_lofo"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = args.batch_size or DEFAULT_BATCH[args.model]
    mean, std = MODEL_INPUT_STATS[args.model]
    in_channels = 3 if args.no_mask else 4
    with_phase = args.task in ("both", "phase")
    with_pressure = args.task in ("both", "pressure")
    # tag keeps single-task/no-mask runs from overwriting the main results
    default_tag = args.model + ("" if args.task == "both" else f"_{args.task}only") \
        + ("_nomask" if args.no_mask else "") \
        + ("_hierarchical" if args.pressure_head == "hierarchical" else "")
    tag = args.run_name or default_tag
    meta = {"model_name": args.model, "in_channels": in_channels,
            "with_phase": with_phase, "with_pressure": with_pressure,
            "pressure_head": args.pressure_head,
            "phase_labels": C.PHASE_LABELS, "pressure_labels": C.PRESSURE_LABELS,
            "training": {"lr": args.lr, "backbone_lr_factor": args.backbone_lr_factor,
                         "weight_decay": args.weight_decay,
                         "label_smoothing": args.label_smoothing,
                         "pressure_weight": args.pressure_weight}}

    records = discover_clips(args.raw_dir)
    print(f"{len(records)} labeled clips / {records.fight.nunique()} fights on {device}")

    if args.final:
        loader = make_loader(records, args.cache_dir, True, mean, std,
                             not args.no_mask, batch, args.workers)
        y_ph = records["phase_label"].map(C.PHASE2IDX).values
        y_pr = records["pressure_label"].map(C.PRESSURE2IDX).values
        model = build_phase_model(args.model, in_channels, with_phase, with_pressure,
                                  pressure_head=args.pressure_head)
        ckpt = out_dir / f"{tag}_final.pt"
        train_model(model, loader, None, device, ckpt, meta,
                    epochs=args.final_epochs, lr=args.lr,
                    backbone_lr_factor=args.backbone_lr_factor,
                    weight_decay=args.weight_decay,
                    pressure_weight=args.pressure_weight,
                    phase_weights=class_weights(y_ph, C.NUM_PHASE_CLASSES),
                    pressure_weights=class_weights(y_pr, C.NUM_PRESSURE_CLASSES),
                    label_smoothing=args.label_smoothing,
                    log_prefix=f"[{tag} final] ")
        print(f"Final model saved to {ckpt}")
        return

    folds = make_lofo_folds(records) if args.lofo else make_folds(records, args.k)
    k = len(folds)
    wanted = range(k) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    for fi in wanted:
        tr_idx, va_idx = folds[fi]
        tr, va = records.iloc[tr_idx], records.iloc[va_idx]
        print(f"\n### {tag} fold {fi + 1}/{k} — "
              f"train {len(tr)} clips / val {len(va)} clips "
              f"(val fights: {sorted(va.fight.unique())})")
        y_ph = tr["phase_label"].map(C.PHASE2IDX).values
        y_pr = tr["pressure_label"].map(C.PRESSURE2IDX).values
        model = build_phase_model(args.model, in_channels, with_phase, with_pressure,
                                  pressure_head=args.pressure_head)
        ckpt = out_dir / f"{tag}_fold{fi}.pt"
        history = train_model(
            model, make_loader(tr, args.cache_dir, True, mean, std, not args.no_mask, batch, args.workers),
            make_loader(va, args.cache_dir, False, mean, std, not args.no_mask, batch, args.workers),
            device, ckpt, meta, epochs=args.epochs, lr=args.lr,
            backbone_lr_factor=args.backbone_lr_factor, weight_decay=args.weight_decay,
            patience=args.patience,
            pressure_weight=args.pressure_weight,
            phase_weights=class_weights(y_ph, C.NUM_PHASE_CLASSES),
            pressure_weights=class_weights(y_pr, C.NUM_PRESSURE_CLASSES),
            label_smoothing=args.label_smoothing,
            log_prefix=f"[{tag} f{fi}] ")

        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        val = evaluate(model, make_loader(va, args.cache_dir, False, mean, std,
                                          not args.no_mask, batch, args.workers),
                       device, torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss(),
                       args.pressure_weight)
        np.savez(out_dir / f"{tag}_fold{fi}_preds.npz",
                 phase_true=val["phase_true"], phase_pred=val["phase_pred"],
                 phase_prob=val["phase_prob"],
                 pressure_true=val["pressure_true"], pressure_pred=val["pressure_pred"],
                 pressure_prob=val["pressure_prob"],
                 fight=va.fight.values.astype(str),
                 filename=va.filename.values.astype(str),
                 val_fights=np.array(sorted(va.fight.unique())))
        with open(out_dir / f"{tag}_fold{fi}_history.json", "w") as f:
            json.dump(history, f, indent=2)
        plot_history(out_dir, tag, fi, history)
        print(f"fold {fi}: phase F1={val['phase_f1']} acc={val['phase_acc']} "
              f"| pressure acc={val['pressure_acc']}")

    if all((out_dir / f"{tag}_fold{i}_preds.npz").exists() for i in range(k)):
        aggregate(out_dir, tag, k)


if __name__ == "__main__":
    main()
