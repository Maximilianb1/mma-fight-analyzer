"""Dataset discovery and PyTorch datasets over the preprocessed clip cache."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from . import config as C


def _to_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() == "true"


def discover_clips(raw_dir, include_excluded=False):
    """Scan data/raw/<Fight>/ folders into one DataFrame.

    Handles Excel-mangled CSVs (TRUE/FALSE strings). Excluded clips carry no
    phase/pressure labels and are only returned when include_excluded=True
    (they are the positives for the gate classifier).
    """
    rows = []
    raw_dir = Path(raw_dir)
    for fight_dir in sorted(raw_dir.iterdir()):
        if not fight_dir.is_dir():
            continue
        # CSV either inside the fight folder or next to it (<Fight>_labels.csv)
        csvs = list(fight_dir.glob("*_labels.csv"))
        sibling = raw_dir / f"{fight_dir.name}_labels.csv"
        if not csvs and sibling.exists():
            csvs = [sibling]
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        df["excluded"] = df["excluded"].map(_to_bool)
        for _, r in df.iterrows():
            excluded = bool(r["excluded"])
            if excluded and not include_excluded:
                continue
            if not excluded and (
                pd.isna(r["phase_label"]) or pd.isna(r["pressure_label"])
            ):
                continue
            rows.append(
                {
                    "fight": fight_dir.name,
                    "filename": r["saved_filename"],
                    "clip_path": str(fight_dir / r["saved_filename"]),
                    "phase_label": None if excluded else r["phase_label"],
                    "pressure_label": None if excluded else r["pressure_label"],
                    "excluded": excluded,
                }
            )
    if not rows:
        raise FileNotFoundError(f"no labeled fights found under {raw_dir}")
    return pd.DataFrame(rows)


def cache_path(cache_dir, fight, filename):
    return Path(cache_dir) / fight / (Path(filename).stem + ".npz")


def load_cached_clip(path):
    with np.load(path) as npz:
        return npz["frames"], npz["mask"]  # uint8 (T,H,W,3), int8 (T,H,W)


def augment_clip(frames, mask, train, crop=C.CROP_SIZE):
    """Geometric + photometric augmentation with ONE parameter draw per clip,
    applied identically to every frame and to the identity mask.

    frames: float tensor (T,3,H,W) in [0,1]; mask: float tensor (T,1,H,W) or None.
    Both train and val map the wide frame to a square, so the distortion the
    model sees is consistent between the two.
    """
    if train:
        i, j, h, w = T.RandomResizedCrop.get_params(
            frames, scale=(0.6, 1.0), ratio=(4 / 3, 16 / 9)
        )
        frames = TF.resized_crop(frames, i, j, h, w, [crop, crop], antialias=True)
        if mask is not None:
            mask = TF.resized_crop(
                mask,
                i,
                j,
                h,
                w,
                [crop, crop],
                interpolation=TF.InterpolationMode.NEAREST,
            )
        if torch.rand(()) < 0.5:
            frames = TF.hflip(frames)
            if mask is not None:
                mask = TF.hflip(mask)
        order, b, c, s, hue = T.ColorJitter.get_params(
            (0.7, 1.3), (0.7, 1.3), (0.7, 1.3), (-0.03, 0.03)
        )
        for fn_id in order:
            if fn_id == 0 and b is not None:
                frames = TF.adjust_brightness(frames, b)
            elif fn_id == 1 and c is not None:
                frames = TF.adjust_contrast(frames, c)
            elif fn_id == 2 and s is not None:
                frames = TF.adjust_saturation(frames, s)
            elif fn_id == 3 and hue is not None:
                frames = TF.adjust_hue(frames, hue)
    else:
        frames = TF.resize(frames, [crop, crop], antialias=True)
        if mask is not None:
            mask = TF.resize(
                mask, [crop, crop], interpolation=TF.InterpolationMode.NEAREST
            )
    return frames, mask


class PhaseClipDataset(Dataset):
    """Yields (video, phase_idx, pressure_idx); video is (C,T,H,W) with C=4 (RGB+mask) or 3."""

    def __init__(self, records, cache_dir, train, mean, std, use_mask=True):
        self.records = records.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.train = train
        self.mean, self.std = mean, std
        self.use_mask = use_mask

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        frames_np, mask_np = load_cached_clip(
            cache_path(self.cache_dir, row.fight, row.filename)
        )
        frames = (
            torch.from_numpy(frames_np).float().div_(255).permute(0, 3, 1, 2)
        )  # (T,3,H,W)
        mask = torch.from_numpy(mask_np).float().unsqueeze(1) if self.use_mask else None
        frames, mask = augment_clip(frames, mask, self.train)
        frames = TF.normalize(frames, self.mean, self.std)
        video = frames.permute(1, 0, 2, 3)  # (3,T,H,W)
        if mask is not None:
            video = torch.cat([video, mask.permute(1, 0, 2, 3)], dim=0)  # (4,T,H,W)
        return video, C.PHASE2IDX[row.phase_label], C.PRESSURE2IDX[row.pressure_label]


class GateFrameDataset(Dataset):
    """Single frames labeled 1=excluded (replay/walkout/break) / 0=live fight."""

    def __init__(
        self,
        records,
        cache_dir,
        train,
        mean=None,
        std=None,
        frames_per_clip=C.GATE_FRAMES,
    ):
        self.records = records.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.train = train
        self.mean = mean or C.IMAGENET_MEAN
        self.std = std or C.IMAGENET_STD
        self.slots = (
            np.linspace(0, C.NUM_FRAMES - 1, frames_per_clip).round().astype(int)
        )
        self.items = [(i, s) for i in range(len(self.records)) for s in self.slots]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec_i, slot = self.items[idx]
        row = self.records.iloc[rec_i]
        frames_np, _ = load_cached_clip(
            cache_path(self.cache_dir, row.fight, row.filename)
        )
        frame = (
            torch.from_numpy(frames_np[slot])
            .float()
            .div_(255)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        frame, _ = augment_clip(frame, None, self.train)
        frame = TF.normalize(frame, self.mean, self.std).squeeze(0)
        return frame, np.float32(row.excluded)


class GateClipDataset(Dataset):
    """Four sampled gate frames loaded with one NPZ decompression per clip.

    Returning ``(S, 3, H, W)`` lets the training script flatten the sampled
    frames for ResNet while averaging their probabilities at clip level.  The
    older frame dataset decompressed the same NPZ four times and left the GPU
    starved on Kaggle.
    """

    def __init__(
        self,
        records,
        cache_dir,
        train,
        mean=None,
        std=None,
        frames_per_clip=C.GATE_FRAMES,
    ):
        self.records = records.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.train = train
        self.mean = mean or C.IMAGENET_MEAN
        self.std = std or C.IMAGENET_STD
        self.slots = (
            np.linspace(0, C.NUM_FRAMES - 1, frames_per_clip).round().astype(int)
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        frames_np, _ = load_cached_clip(
            cache_path(self.cache_dir, row.fight, row.filename)
        )
        frames = (
            torch.from_numpy(frames_np[self.slots])
            .float()
            .div_(255)
            .permute(0, 3, 1, 2)
        )
        frames, _ = augment_clip(frames, None, self.train)
        frames = TF.normalize(frames, self.mean, self.std)
        return frames, np.float32(row.excluded)
