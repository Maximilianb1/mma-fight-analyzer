"""Training loop, grouped folds, and metrics helpers."""

import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from . import config as C
from .models import backbone_and_head_params, save_checkpoint


def set_seed(seed=C.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_folds(records, n_splits=4, seed=C.RANDOM_SEED):
    """Fight-level folds: no clip from a validation fight ever appears in training.
    Stratified on phase so minority classes are spread across folds where possible."""
    y = records["phase_label"].map(C.PHASE2IDX).values
    groups = records["fight"].values
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(records, y, groups))


def make_lofo_folds(records):
    """Leave-one-fight-out: one fold per fight. Cleaner story than K folds, ~3x the compute."""
    idx = np.arange(len(records))
    return [(idx[(records["fight"] != f).values], idx[(records["fight"] == f).values])
            for f in sorted(records["fight"].unique())]


def class_weights(label_indices, n_classes):
    counts = Counter(label_indices)
    n = len(label_indices)
    return torch.tensor([n / (n_classes * counts.get(c, 1)) for c in range(n_classes)],
                        dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, crit_ph, crit_pr, pressure_weight):
    model.eval()
    total_loss, n = 0.0, 0
    ph_true, ph_pred, pr_true, pr_pred = [], [], [], []
    for video, ph, pr in loader:
        video, ph, pr = video.to(device), ph.to(device), pr.to(device)
        logits_ph, logits_pr = model(video)
        loss = video.new_zeros(())
        if logits_ph is not None:
            loss = loss + crit_ph(logits_ph, ph)
            ph_pred.extend(logits_ph.argmax(1).tolist())
            ph_true.extend(ph.tolist())
        if logits_pr is not None:
            loss = loss + pressure_weight * crit_pr(logits_pr, pr)
            pr_pred.extend(logits_pr.argmax(1).tolist())
            pr_true.extend(pr.tolist())
        total_loss += loss.item() * video.size(0)
        n += video.size(0)
    return {
        "loss": total_loss / max(n, 1),
        "phase_true": np.array(ph_true), "phase_pred": np.array(ph_pred),
        "pressure_true": np.array(pr_true), "pressure_pred": np.array(pr_pred),
        "phase_f1": f1_score(ph_true, ph_pred, average="macro", zero_division=0) if ph_true else None,
        "phase_acc": float(np.mean(np.array(ph_true) == np.array(ph_pred))) if ph_true else None,
        "pressure_f1": f1_score(pr_true, pr_pred, average="macro", zero_division=0) if pr_true else None,
        "pressure_acc": float(np.mean(np.array(pr_true) == np.array(pr_pred))) if pr_true else None,
    }


def train_model(model, train_loader, val_loader, device, ckpt_path, ckpt_meta,
                epochs=25, lr=3e-4, backbone_lr_factor=0.1, weight_decay=1e-4,
                patience=6, pressure_weight=0.5,
                phase_weights=None, pressure_weights=None, log_prefix=""):
    """Fine-tune with AMP; early stopping and checkpointing on validation phase macro-F1.
    When val_loader is None (final full-data training) runs all epochs and saves the last."""
    model.to(device)
    crit_ph = nn.CrossEntropyLoss(
        weight=phase_weights.to(device) if phase_weights is not None else None,
        label_smoothing=0.05)
    crit_pr = nn.CrossEntropyLoss(
        weight=pressure_weights.to(device) if pressure_weights is not None else None,
        label_smoothing=0.05)
    bb, head = backbone_and_head_params(model)
    opt = torch.optim.AdamW([
        {"params": bb, "lr": lr * backbone_lr_factor},
        {"params": head, "lr": lr},
    ], weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    history = {"train_loss": [], "val_loss": [], "val_phase_f1": [],
               "val_phase_acc": [], "val_pressure_acc": []}
    best_f1, bad_epochs = -1.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for video, ph, pr in train_loader:
            video, ph, pr = video.to(device), ph.to(device), pr.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                logits_ph, logits_pr = model(video)
                loss = video.new_zeros(())
                if logits_ph is not None:
                    loss = loss + crit_ph(logits_ph, ph)
                if logits_pr is not None:
                    loss = loss + pressure_weight * crit_pr(logits_pr, pr)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * video.size(0)
            n += video.size(0)
        train_loss = running / max(n, 1)
        history["train_loss"].append(train_loss)

        if val_loader is None:
            print(f"{log_prefix}epoch {epoch:>2}/{epochs} loss={train_loss:.4f} "
                  f"({time.time() - t0:.0f}s)")
            continue

        val = evaluate(model, val_loader, device, crit_ph, crit_pr, pressure_weight)
        # combined monitor (phase F1 + 0.5*pressure F1 over available heads):
        # checkpointing on phase alone froze pressure at its weak early state (B8)
        monitor = 0.0
        if val["phase_f1"] is not None:
            monitor += val["phase_f1"]
        if val["pressure_f1"] is not None:
            monitor += 0.5 * val["pressure_f1"]
        sched.step(monitor)
        history["val_loss"].append(val["loss"])
        history["val_phase_f1"].append(val["phase_f1"])
        history["val_phase_acc"].append(val["phase_acc"])
        history["val_pressure_acc"].append(val["pressure_acc"])
        print(f"{log_prefix}epoch {epoch:>2}/{epochs} loss={train_loss:.4f} "
              f"val_loss={val['loss']:.4f} phF1={val['phase_f1'] or 0:.3f} "
              f"phAcc={val['phase_acc'] or 0:.3f} prAcc={val['pressure_acc'] or 0:.3f} "
              f"({time.time() - t0:.0f}s)")

        if monitor > best_f1:
            best_f1, bad_epochs = monitor, 0
            save_checkpoint(ckpt_path, model, ckpt_meta)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"{log_prefix}early stop at epoch {epoch} (best monitored F1 {best_f1:.3f})")
                break

    if val_loader is None:
        save_checkpoint(ckpt_path, model, ckpt_meta)
    return history
