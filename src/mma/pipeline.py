"""End-to-end inference: fight video in -> annotated, labeled video out.

Per 5-second window:
  1. gate model decides fight vs non-fight (replay/walkout/break)
  2. YOLO detects fighters on 16 sampled frames; a light IoU tracker links them
  3. fighter identity is anchored once (interactive prompt or shorts colors),
     then propagated across windows via HSV appearance histograms
  4. the phase+pressure model classifies the window from RGB + identity mask
  5. boxes and labels are drawn on every frame and written to the output video
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from . import config as C
from . import identity as ident
from . import overlay
from .models import MODEL_INPUT_STATS, load_gate_model, load_phase_model

TRACK_LINK_IOU = 0.3
ANCHOR_EMA = 0.85


def _appearance_hist(frame_bgr, box):
    """HSV H/S histogram over the torso+shorts region of a box."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    crop = frame_bgr[max(0, y1 + int(h * 0.2)):min(frame_bgr.shape[0], y1 + int(h * 0.65)),
                     max(0, x1):min(frame_bgr.shape[1], x2)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _hist_similarity(a, b):
    if a is None or b is None:
        return 0.0
    return 1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)


def link_tracks(dets):
    """Greedy IoU linking of per-frame detections into tracks.
    Returns list of dicts {sampled_slot: box}."""
    tracks, last_box = [], {}
    for t, boxes in enumerate(dets):
        used = set()
        for box in boxes:
            best_id, best_iou = None, TRACK_LINK_IOU
            for tid, pb in last_box.items():
                if tid in used:
                    continue
                v = ident.iou(box, pb)
                if v > best_iou:
                    best_id, best_iou = tid, v
            if best_id is None:
                best_id = len(tracks)
                tracks.append({})
            tracks[best_id][t] = box
            last_box[best_id] = box
            used.add(best_id)
    return tracks


def top_two_tracks(tracks):
    def score(tr):
        return sum((b[2] - b[0]) * (b[3] - b[1]) for b in tr.values())
    ranked = sorted(tracks, key=score, reverse=True)
    return ranked[:2]


class FightAnalyzer:
    def __init__(self, gate_ckpt, phase_ckpt, device=None, interactive=True,
                 f1_color=None, f2_color=None,
                 f1_name="Fighter 1", f2_name="Fighter 2",
                 out_dir="outputs", yolo_weights="yolov8n.pt"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gate, gate_meta = load_gate_model(gate_ckpt, self.device)
        self.gate_threshold = gate_meta.get("threshold", 0.5)
        self.phase_model, self.phase_meta = load_phase_model(phase_ckpt, self.device)
        self.mean, self.std = MODEL_INPUT_STATS[self.phase_meta["model_name"]]
        self.use_mask = self.phase_meta["in_channels"] == 4
        self.yolo = ident.load_yolo(yolo_weights)
        self.interactive = interactive
        self.f1_color, self.f2_color = f1_color, f2_color
        self.f1_name, self.f2_name = f1_name, f2_name
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.anchor_f1 = None  # appearance histograms, set once then EMA-updated
        self.anchor_f2 = None
        self.prev_f1_box = None
        self.prev_f2_box = None

    # ── identity ──
    def _prompt_user(self, frame, box_a, box_b):
        img_path = self.out_dir / "identity_prompt.png"
        img = overlay.save_identity_prompt_image(img_path, frame, box_a, box_b)
        try:
            cv2.imshow("Who is Fighter 1? (press any key, then answer in terminal)", img)
            cv2.waitKey(1500)
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print(f"\nIdentity check — see {img_path}")
        print("Fighter 1 = the name shown LEFT of the timer in the broadcast overlay.")
        while True:
            ans = input(f"Which box is {self.f1_name} (Fighter 1)? [A/B]: ").strip().upper()
            if ans in ("A", "B"):
                return ans == "A"

    def _track_hist(self, window, sampled_idx, track):
        hists = [_appearance_hist(window[sampled_idx[t]], b) for t, b in track.items()]
        hists = [h for h in hists if h is not None]
        return np.mean(hists, axis=0) if hists else None

    def _track_color(self, window, sampled_idx, track):
        votes = {}
        for t, b in track.items():
            hsv = cv2.cvtColor(window[sampled_idx[t]], cv2.COLOR_BGR2HSV)
            c, _ = ident.classify_shorts_color(ident.get_shorts_region(hsv, b))
            votes[c] = votes.get(c, 0) + 1
        return max(votes, key=votes.get) if votes else "unknown"

    def _assign_identity(self, window, sampled_idx, tracks):
        """Map (up to) two tracks to (f1_track, f2_track)."""
        if not tracks:
            return None, None
        if len(tracks) == 1:
            a, b = tracks[0], None
        else:
            a, b = tracks[0], tracks[1]

        if self.anchor_f1 is None and self.anchor_f2 is None:
            if self.f1_color and self.f2_color:  # non-interactive bootstrap
                ca = self._track_color(window, sampled_idx, a)
                f1, f2 = (a, b) if ca == self.f1_color else (b, a)
            elif self.interactive and b is not None:
                slot = sorted(set(a) & set(b))[len(set(a) & set(b)) // 2]
                a_is_f1 = self._prompt_user(window[sampled_idx[slot]], a[slot], b[slot])
                f1, f2 = (a, b) if a_is_f1 else (b, a)
            else:
                return None, None  # wait for a window where both fighters are visible
        else:
            ha = self._track_hist(window, sampled_idx, a)
            hb = self._track_hist(window, sampled_idx, b) if b is not None else None
            straight = (_hist_similarity(ha, self.anchor_f1)
                        + _hist_similarity(hb, self.anchor_f2))
            swapped = (_hist_similarity(ha, self.anchor_f2)
                       + _hist_similarity(hb, self.anchor_f1))
            f1, f2 = (a, b) if straight >= swapped else (b, a)

        for track, attr in ((f1, "anchor_f1"), (f2, "anchor_f2")):
            if track is None:
                continue
            h = self._track_hist(window, sampled_idx, track)
            if h is None:
                continue
            old = getattr(self, attr)
            setattr(self, attr, h if old is None else ANCHOR_EMA * old + (1 - ANCHOR_EMA) * h)
        return f1, f2

    # ── models ──
    @torch.no_grad()
    def _gate_prob(self, sampled_frames):
        idx = np.linspace(0, len(sampled_frames) - 1, C.GATE_FRAMES).round().astype(int)
        batch = []
        for i in idx:
            f = cv2.cvtColor(sampled_frames[i], cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(f).float().div_(255).permute(2, 0, 1)
            t = TF.resize(t, [C.CROP_SIZE, C.CROP_SIZE], antialias=True)
            batch.append(TF.normalize(t, C.IMAGENET_MEAN, C.IMAGENET_STD))
        logits = self.gate(torch.stack(batch).to(self.device))
        return torch.sigmoid(logits).mean().item()

    @torch.no_grad()
    def _classify(self, sampled_frames, mask):
        frames = torch.stack([
            torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).float().div_(255).permute(2, 0, 1)
            for f in sampled_frames])
        frames = TF.resize(frames, [C.CROP_SIZE, C.CROP_SIZE], antialias=True)
        frames = TF.normalize(frames, self.mean, self.std)
        video = frames.permute(1, 0, 2, 3)
        if self.use_mask:
            m = torch.from_numpy(mask).float().unsqueeze(1)
            m = TF.resize(m, [C.CROP_SIZE, C.CROP_SIZE],
                          interpolation=TF.InterpolationMode.NEAREST)
            video = torch.cat([video, m.permute(1, 0, 2, 3)], dim=0)
        logits_ph, logits_pr = self.phase_model(video.unsqueeze(0).to(self.device))
        phase = pressure = phase_conf = pressure_conf = None
        if logits_ph is not None:
            probs = torch.softmax(logits_ph, dim=1)[0]
            phase, phase_conf = C.IDX2PHASE[int(probs.argmax())], float(probs.max())
        if logits_pr is not None:
            probs = torch.softmax(logits_pr, dim=1)[0]
            pressure, pressure_conf = C.IDX2PRESSURE[int(probs.argmax())], float(probs.max())
        return phase, pressure, phase_conf, pressure_conf

    # ── main loop ──
    def process(self, video_path, out_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        win_len = int(round(fps * C.CLIP_SECONDS))
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        window, w_idx, results = [], 0, []
        while True:
            ok, frame = cap.read()
            if ok:
                window.append(frame)
            if (not ok and window) or len(window) == win_len:
                results.append(self._process_window(window, w_idx, writer))
                window, w_idx = [], w_idx + 1
            if not ok:
                break
        cap.release()
        writer.release()
        self._remux_audio(video_path, out_path)
        return results

    def _process_window(self, window, w_idx, writer):
        sampled_idx = np.linspace(0, len(window) - 1, C.NUM_FRAMES).round().astype(int)
        sampled = [window[i] for i in sampled_idx]
        info = {"window": w_idx, "start_s": w_idx * C.CLIP_SECONDS}

        p_excluded = self._gate_prob(sampled)
        info["gate_prob_excluded"] = round(p_excluded, 3)
        if p_excluded > self.gate_threshold:
            info.update(excluded=True, phase=None, pressure=None)
            overlay.annotate_window(window, sampled_idx, [], [], None, None, excluded=True)
        else:
            dets = ident.detect_fighters(self.yolo, sampled)
            f1_track, f2_track = self._assign_identity(
                window, sampled_idx, top_two_tracks(link_tracks(dets)))
            f1_boxes = [f1_track.get(t) if f1_track else None for t in range(C.NUM_FRAMES)]
            f2_boxes = [f2_track.get(t) if f2_track else None for t in range(C.NUM_FRAMES)]
            assigns = list(zip(f1_boxes, f2_boxes))
            mask = ident.build_masks(window[0].shape[:2], assigns,
                                     (C.CACHE_SHORT_SIDE,
                                      round(C.CACHE_SHORT_SIDE * window[0].shape[1] / window[0].shape[0])))
            phase, pressure, phase_conf, pressure_conf = self._classify(sampled, mask)
            info.update(excluded=False, phase=phase, pressure=pressure,
                        phase_conf=round(phase_conf, 3) if phase_conf else None,
                        pressure_conf=round(pressure_conf, 3) if pressure_conf else None)
            overlay.annotate_window(window, sampled_idx, f1_boxes, f2_boxes, phase, pressure,
                                    self.f1_name, self.f2_name,
                                    phase_conf=phase_conf, pressure_conf=pressure_conf)
        for frame in window:
            writer.write(frame)
        tag = "EXCLUDED" if info["excluded"] else (
            f"{info['phase']} ({info.get('phase_conf')}) | {info['pressure']}")
        print(f"  window {w_idx:>3} [{info['start_s']:>4}s] {tag}")
        return info

    def _remux_audio(self, src, dst):
        """Best-effort: copy the original audio track onto the annotated video."""
        tmp = str(dst) + ".audio.mp4"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(dst), "-i", str(src),
                 "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0?", "-shortest", tmp],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            Path(tmp).replace(dst)
        except (subprocess.CalledProcessError, FileNotFoundError):
            Path(tmp).unlink(missing_ok=True)
