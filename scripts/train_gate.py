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
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
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
        auc = roc_auc_score(ys, probs)
        ap = average_precision_score(ys, probs)
        print(f"epoch {epoch}/{args.epochs} loss={running / n:.4f} val AUC={auc:.3f} AP={ap:.3f}")
        if auc > best_auc:
            best_auc = auc
            prec, rec, thr = precision_recall_curve(ys, probs)
            f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
            threshold = float(thr[int(np.argmax(f1[:-1]))]) if len(thr) else 0.5
            torch.save({"state_dict": model.state_dict(),
                        "meta": {"threshold": threshold, "val_auc": float(auc),
                                 "val_ap": float(ap)}},
                       out_dir / "gate.pt")

    ckpt = torch.load(out_dir / "gate.pt", map_location="cpu", weights_only=False)
    with open(out_dir / "gate_metrics.json", "w") as f:
        json.dump(ckpt["meta"], f, indent=2)
    print(f"best val AUC={ckpt['meta']['val_auc']:.3f}, "
          f"operating threshold={ckpt['meta']['threshold']:.3f} -> {out_dir / 'gate.pt'}")


if __name__ == "__main__":
    main()
