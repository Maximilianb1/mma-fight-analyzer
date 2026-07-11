"""Build the clip cache: 16 sampled frames + identity mask per clip, saved as .npz.

Run once after downloading the data. Training then never touches the videos,
which makes epochs ~10-50x faster on Colab. Resumable: existing cache files
are skipped.

Requires data/fights_meta.csv with shorts colors per fight (f1_color/f2_color);
fights with missing colors get all-zero masks and a warning.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mma import config as C                                   # noqa: E402
from mma.data import cache_path, discover_clips               # noqa: E402
from mma.identity import (COLOR_RANGES, assign_identities,    # noqa: E402
                          build_masks, coverage, detect_fighters, load_yolo)


def read_sampled_frames(video_path, n_frames):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    wanted = set(np.linspace(0, total - 1, n_frames).round().astype(int).tolist())
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames.append(frame)
        i += 1
    cap.release()
    while frames and len(frames) < n_frames:  # decoder returned fewer frames than reported
        frames.append(frames[-1])
    return frames or None


def load_meta(meta_path):
    if not Path(meta_path).exists():
        print(f"ERROR: {meta_path} not found. Create it with columns "
              f"fight,f1_color,f2_color (colors from: {', '.join(COLOR_RANGES)})")
        sys.exit(1)
    meta = pd.read_csv(meta_path)
    colors = {}
    for _, r in meta.iterrows():
        f1, f2 = str(r.get("f1_color", "")).strip().lower(), str(r.get("f2_color", "")).strip().lower()
        if f1 in COLOR_RANGES and f2 in COLOR_RANGES:
            colors[r["fight"]] = (f1, f2)
    return colors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--meta", default="data/fights_meta.csv")
    p.add_argument("--yolo-conf", type=float, default=0.35)
    args = p.parse_args()

    colors = load_meta(args.meta)
    records = discover_clips(args.raw_dir, include_excluded=True)
    print(f"{len(records)} clips ({int(records.excluded.sum())} excluded) "
          f"across {records.fight.nunique()} fights")
    missing_colors = sorted(set(records.fight) - set(colors))
    if missing_colors:
        print(f"WARNING: no shorts colors for {missing_colors} -> zero identity masks")

    yolo = load_yolo()
    stats = {}
    for fight, group in records.groupby("fight"):
        both_sum, any_sum, n_masked = 0.0, 0.0, 0
        for _, row in tqdm(list(group.iterrows()), desc=fight[:40]):
            out = cache_path(args.cache_dir, row.fight, row.filename)
            if out.exists():
                continue
            frames = read_sampled_frames(row.clip_path, C.NUM_FRAMES)
            if frames is None:
                print(f"  unreadable: {row.clip_path}")
                continue
            h, w = frames[0].shape[:2]
            ch = C.CACHE_SHORT_SIDE
            cw = round(w * ch / h)

            if not row.excluded and fight in colors:
                dets = detect_fighters(yolo, frames, conf=args.yolo_conf)
                assigns = assign_identities(frames, dets, *colors[fight])
                mask = build_masks((h, w), assigns, (ch, cw))
                both, any_ = coverage(assigns)
                both_sum += both
                any_sum += any_
                n_masked += 1
            else:
                mask = np.zeros((C.NUM_FRAMES, ch, cw), np.int8)

            small = np.stack([cv2.cvtColor(cv2.resize(f, (cw, ch)), cv2.COLOR_BGR2RGB)
                              for f in frames]).astype(np.uint8)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, frames=small, mask=mask)
        if n_masked:
            stats[fight] = (both_sum / n_masked, any_sum / n_masked)

    if stats:
        print("\nIdentity coverage (fraction of frames with both / >=1 fighter assigned):")
        for fight, (both, any_) in stats.items():
            flag = "  <-- CHECK COLORS" if both < 0.5 else ""
            print(f"  {fight:<42} both={both:.2f}  any={any_:.2f}{flag}")
    print("\nCache complete.")


if __name__ == "__main__":
    main()
