"""Generate qualitative EigenGradCAM figures for the phase classifier.

The script selects the most confident correctly classified examples from an
explicit data split, creates a class-wise panel and an eight-frame temporal
panel, and saves a JSON manifest identifying every selected clip.

Run from the repository root after downloading the dataset, preprocessing its
cache, and downloading the deployment checkpoints::

    python report/scripts/generate_explainability.py

These figures are diagnostics, not a quantitative explanation benchmark.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="outputs/phase/deployment_phase_final.pt",
        help="phase-model checkpoint",
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--out-dir", default="report/figures/generated")
    parser.add_argument(
        "--candidate-split",
        choices=("development", "holdout", "all"),
        default="development",
        help="where examples may be selected; development is training data for the final model",
    )
    parser.add_argument("--samples", type=int, default=40, help="candidates per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or an explicit torch device"
    )
    return parser.parse_args()


def choose_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main():
    args = parse_args()
    try:
        from pytorch_grad_cam import EigenGradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as exc:
        raise SystemExit("Install requirements.txt (grad-cam is required).") from exc

    from mma import config as C
    from mma.data import cache_path, discover_clips, load_cached_clip
    from mma.models import load_phase_model

    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model, meta = load_phase_model(checkpoint, device)
    if not meta.get("with_phase", True):
        raise SystemExit(f"checkpoint has no phase head: {checkpoint}")

    class PhaseOnly(torch.nn.Module):
        def __init__(self, source):
            super().__init__()
            self.source = source

        def forward(self, video):
            phase_logits, _ = self.source(video)
            return phase_logits

    wrapped = PhaseOnly(model).to(device).eval()
    target_layers = [model.backbone.layer3[-1], model.backbone.layer4[-1]]

    records = discover_clips(args.raw_dir)
    holdout = meta.get("holdout_fight", C.DEFAULT_HOLDOUT_FIGHT)
    if args.candidate_split == "development":
        records = records[records.fight != holdout]
    elif args.candidate_split == "holdout":
        records = records[records.fight == holdout]
    if records.empty:
        raise SystemExit(f"no clips found for split: {args.candidate_split}")

    sampled = []
    for class_index, (_, group) in enumerate(records.groupby("phase_label", sort=False)):
        n = min(args.samples, len(group))
        sampled.append(group.sample(n=n, random_state=args.seed + class_index))
    candidates = np.random.default_rng(args.seed).permutation(
        np.concatenate([part.index.to_numpy() for part in sampled])
    )
    candidates = records.loc[candidates]

    def prepare_video(frames_np, mask_np):
        frames = torch.from_numpy(frames_np).float().div_(255).permute(0, 3, 1, 2)
        frames = TF.resize(frames, [C.CROP_SIZE, C.CROP_SIZE], antialias=True)
        frames = TF.normalize(frames, C.KINETICS_MEAN, C.KINETICS_STD)
        video = frames.permute(1, 0, 2, 3)
        if meta["in_channels"] == 4:
            mask = torch.from_numpy(mask_np).float().unsqueeze(1)
            mask = TF.resize(
                mask,
                [C.CROP_SIZE, C.CROP_SIZE],
                interpolation=TF.InterpolationMode.NEAREST,
            )
            video = torch.cat([video, mask.permute(1, 0, 2, 3)], dim=0)
        return video.unsqueeze(0).to(device)

    def display_frame(frames_np, index, size=224):
        return cv2.resize(frames_np[index], (size, size)).astype(np.float32) / 255.0

    def normalize_cam(cam_map, size=224):
        cam_map = cv2.resize(cam_map, (size, size))
        cam_map = cv2.GaussianBlur(cam_map, (0, 0), sigmaX=1.5)
        low, high = float(cam_map.min()), float(cam_map.max())
        return (cam_map - low) / (high - low + 1e-8)

    best = {}
    for _, row in candidates.iterrows():
        class_index = C.PHASE2IDX[row.phase_label]
        cached = cache_path(args.cache_dir, row.fight, row.filename)
        if not cached.exists():
            continue
        frames_np, mask_np = load_cached_clip(cached)
        video = prepare_video(frames_np, mask_np)
        with torch.inference_mode():
            probabilities = torch.softmax(wrapped(video), dim=1)[0]
        predicted = int(probabilities.argmax())
        confidence = float(probabilities[class_index])
        if predicted == class_index and (
            class_index not in best or confidence > best[class_index][0]
        ):
            best[class_index] = (confidence, row, frames_np, mask_np)

    missing = [C.PHASE_LABELS[index] for index in range(C.NUM_PHASE_CLASSES) if index not in best]
    if missing:
        print("No correct candidate for: " + ", ".join(missing))

    cams = {}
    with EigenGradCAM(model=wrapped, target_layers=target_layers) as cam:
        for class_index, (_, _, frames_np, mask_np) in best.items():
            video = prepare_video(frames_np, mask_np)
            cams[class_index] = cam(
                input_tensor=video,
                targets=[ClassifierOutputTarget(class_index)],
            )[0]

    fig, axes = plt.subplots(2, C.NUM_PHASE_CLASSES, figsize=(22, 9))
    for class_index in range(C.NUM_PHASE_CLASSES):
        if class_index not in best:
            axes[:, class_index].flat[0].axis("off")
            axes[:, class_index].flat[1].axis("off")
            continue
        confidence, _, frames_np, _ = best[class_index]
        middle = C.NUM_FRAMES // 2
        rgb = display_frame(frames_np, middle)
        heatmap = normalize_cam(cams[class_index][middle])
        overlay = show_cam_on_image(rgb, heatmap.astype(np.float32), use_rgb=True)
        axes[0, class_index].imshow(rgb)
        axes[0, class_index].set_title(
            f"{C.PHASE_LABELS[class_index]}\n({confidence:.0%} confidence)",
            fontsize=11,
            fontweight="bold",
        )
        axes[1, class_index].imshow(overlay)
        axes[0, class_index].axis("off")
        axes[1, class_index].axis("off")
    fig.suptitle("EigenGradCAM: spatial evidence by phase", fontweight="bold")
    fig.tight_layout()
    phase_path = out_dir / "eigen_gradcam_by_phase.png"
    fig.savefig(phase_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    frame_indices = np.linspace(0, C.NUM_FRAMES - 1, 8).astype(int)
    fig, axes = plt.subplots(C.NUM_PHASE_CLASSES, len(frame_indices), figsize=(26, 16))
    for class_index in range(C.NUM_PHASE_CLASSES):
        if class_index not in best:
            for axis in axes[class_index]:
                axis.axis("off")
            continue
        confidence, _, frames_np, _ = best[class_index]
        for column, frame_index in enumerate(frame_indices):
            rgb = display_frame(frames_np, frame_index)
            heatmap = normalize_cam(cams[class_index][frame_index])
            overlay = show_cam_on_image(rgb, heatmap.astype(np.float32), use_rgb=True)
            axis = axes[class_index, column]
            axis.imshow(overlay)
            axis.axis("off")
            if column == 0:
                axis.text(
                    -0.08,
                    0.5,
                    f"{C.PHASE_LABELS[class_index]}\n({confidence:.0%})",
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    fontweight="bold",
                )
            if class_index == 0:
                axis.set_title(f"t={frame_index * C.CLIP_SECONDS / C.NUM_FRAMES:.1f}s")
    fig.suptitle("Temporal EigenGradCAM across each five-second clip", fontweight="bold")
    fig.tight_layout()
    temporal_path = out_dir / "eigen_gradcam_temporal.png"
    fig.savefig(temporal_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "method": "EigenGradCAM",
        "target_layers": ["backbone.layer3[-1]", "backbone.layer4[-1]"],
        "checkpoint": str(checkpoint),
        "candidate_split": args.candidate_split,
        "holdout_fight": holdout,
        "samples_per_class": args.samples,
        "seed": args.seed,
        "selected": {
            C.PHASE_LABELS[index]: {
                "fight": str(row.fight),
                "filename": str(row.filename),
                "confidence": confidence,
            }
            for index, (confidence, row, _, _) in best.items()
        },
    }
    manifest_path = out_dir / "eigen_gradcam_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {phase_path}, {temporal_path}, and {manifest_path}")


if __name__ == "__main__":
    main()
