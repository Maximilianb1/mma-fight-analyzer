"""End-to-end inference: fight video in -> annotated, labeled video out.

Per 5-second window:
  1. gate model decides fight vs non-fight (replay/walkout/break)
  2. YOLO detects fighters on 16 sampled frames; a light IoU tracker links them
  3. fighter identity combines a one-time prompt, comparative non-skin shorts
     color evidence, and conservative temporal continuity across windows
  4. the phase+pressure model classifies the window from RGB + identity mask
  5. boxes and labels are drawn on every frame and written to the output video
"""

import math
import shutil
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
TEMPORAL_IDENTITY_MARGIN = 0.12
CONTINUITY_WEIGHT = 0.35
MIN_TEMPORAL_TRACK_SLOTS = 4
MAX_PAIR_OVERLAP = 0.35


def _appearance_hist(frame_bgr, box):
    """HSV H/S histogram over the torso+shorts region of a box."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    crop = frame_bgr[
        max(0, y1 + int(h * 0.2)) : min(frame_bgr.shape[0], y1 + int(h * 0.65)),
        max(0, x1) : min(frame_bgr.shape[1], x2),
    ]
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
    def __init__(
        self,
        gate_ckpt,
        phase_ckpt,
        pressure_ckpt=None,
        device=None,
        interactive=True,
        f1_color=None,
        f2_color=None,
        f1_name="Fighter 1",
        f2_name="Fighter 2",
        out_dir="outputs",
        yolo_weights="yolov8n.pt",
        identity_choice=None,
        web_compatible=False,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.gate, gate_meta = load_gate_model(gate_ckpt, self.device)
        self.gate_threshold = gate_meta.get("threshold", 0.5)
        self.phase_model, self.phase_meta = load_phase_model(phase_ckpt, self.device)
        self.pressure_model = self.pressure_meta = None
        if pressure_ckpt is not None:
            self.pressure_model, self.pressure_meta = load_phase_model(
                pressure_ckpt, self.device
            )
            if not self.pressure_meta.get("with_pressure", True):
                raise ValueError("--pressure-ckpt does not contain a pressure head")
        self.yolo = ident.load_yolo(yolo_weights)
        self.interactive = interactive
        self.identity_choice = identity_choice
        self.f1_color = f1_color.lower() if isinstance(f1_color, str) else f1_color
        self.f2_color = f2_color.lower() if isinstance(f2_color, str) else f2_color
        self.f1_name, self.f2_name = f1_name, f2_name
        self.web_compatible = web_compatible
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.reset_identity(identity_choice)

    def reset_identity(self, identity_choice=None):
        """Clear cross-window tracking and optionally set the web UI's A/B answer."""
        self.identity_choice = identity_choice
        self.anchor_f1 = None  # appearance histograms, set once then EMA-updated
        self.anchor_f2 = None
        self.prev_f1_box = None
        self.prev_f2_box = None

    # ── identity ──
    def _prompt_user(self, frame, box_a, box_b):
        img_path = self.out_dir / "identity_prompt.png"
        img = overlay.save_identity_prompt_image(img_path, frame, box_a, box_b)
        try:
            cv2.imshow(
                "Who is Fighter 1? (press any key, then answer in terminal)", img
            )
            cv2.waitKey(1500)
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print(f"\nIdentity check — see {img_path}")
        print("Fighter 1 = the name shown LEFT of the timer in the broadcast overlay.")
        while True:
            ans = (
                input(f"Which box is {self.f1_name} (Fighter 1)? [A/B]: ")
                .strip()
                .upper()
            )
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

    def _track_color_evidence(self, window, sampled_idx, track):
        """Best non-skin shorts color plus its score and lead over second place."""
        scores = {color: [] for color in ident.COLOR_RANGES}
        for t, box in track.items():
            hsv = cv2.cvtColor(window[sampled_idx[t]], cv2.COLOR_BGR2HSV)
            for color in scores:
                scores[color].append(ident.color_ratio(hsv, box, color))
        ranked = sorted(
            (
                (float(np.mean(values)) if values else 0.0, color)
                for color, values in scores.items()
            ),
            reverse=True,
        )
        best_score, best_color = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        return best_color, best_score, best_score - second_score

    def _learn_prompt_colors(self, window, sampled_idx, f1_track, f2_track):
        """Learn symbolic shorts colors from the user-confirmed prompt when unambiguous."""
        if self.f1_color and self.f2_color:
            return
        c1, s1, g1 = self._track_color_evidence(window, sampled_idx, f1_track)
        c2, s2, g2 = self._track_color_evidence(window, sampled_idx, f2_track)
        if c1 != c2 and min(s1, s2) >= 0.02 and min(g1, g2) >= 0.01:
            self.f1_color, self.f2_color = c1, c2

    @staticmethod
    def _tracks_to_assigns(f1_track, f2_track, n):
        return [
            (
                f1_track.get(t) if f1_track else None,
                f2_track.get(t) if f2_track else None,
            )
            for t in range(n)
        ]

    @staticmethod
    def _assignment_overlap(assigns):
        values = [
            ident.iou(a, b) for a, b in assigns if a is not None and b is not None
        ]
        return float(np.median(values)) if values else 0.0

    def _update_identity_state(self, window, sampled_idx, f1_track, f2_track):
        """Update appearance/position anchors only after a trusted complete assignment."""
        if not f1_track or not f2_track:
            return
        for track, attr in ((f1_track, "anchor_f1"), (f2_track, "anchor_f2")):
            hist = self._track_hist(window, sampled_idx, track)
            if hist is None:
                continue
            old = getattr(self, attr)
            setattr(
                self,
                attr,
                hist if old is None else ANCHOR_EMA * old + (1 - ANCHOR_EMA) * hist,
            )
        self.prev_f1_box = f1_track[max(f1_track)]
        self.prev_f2_box = f2_track[max(f2_track)]

    def _temporal_pair(self, window, sampled_idx, track_a, track_b):
        """Conservative appearance + boundary-IoU pairing; abstain below a fixed margin."""
        if (
            not track_a
            or not track_b
            or self.anchor_f1 is None
            or self.anchor_f2 is None
            or len(track_a) < MIN_TEMPORAL_TRACK_SLOTS
            or len(track_b) < MIN_TEMPORAL_TRACK_SLOTS
        ):
            return None, None, 0.0
        ha = self._track_hist(window, sampled_idx, track_a)
        hb = self._track_hist(window, sampled_idx, track_b)
        straight = _hist_similarity(ha, self.anchor_f1) + _hist_similarity(
            hb, self.anchor_f2
        )
        swapped = _hist_similarity(ha, self.anchor_f2) + _hist_similarity(
            hb, self.anchor_f1
        )
        if self.prev_f1_box is not None and self.prev_f2_box is not None:
            a0, b0 = track_a[min(track_a)], track_b[min(track_b)]
            straight += CONTINUITY_WEIGHT * (
                ident.iou(a0, self.prev_f1_box) + ident.iou(b0, self.prev_f2_box)
            )
            swapped += CONTINUITY_WEIGHT * (
                ident.iou(a0, self.prev_f2_box) + ident.iou(b0, self.prev_f1_box)
            )
        margin = abs(straight - swapped)
        if margin < TEMPORAL_IDENTITY_MARGIN:
            return None, None, margin
        return ((track_a, track_b) if straight > swapped else (track_b, track_a)) + (
            margin,
        )

    def _assign_identity(self, window, sampled_idx, tracks):
        """Prompt bootstrap or conservative temporal fallback for videos without colors."""
        if not tracks:
            return None, None
        if len(tracks) == 1:
            a, b = tracks[0], None
        else:
            a, b = tracks[0], tracks[1]

        if self.anchor_f1 is None and self.anchor_f2 is None:
            if self.identity_choice in ("A", "B") and b is not None:
                f1, f2 = (a, b) if self.identity_choice == "A" else (b, a)
            elif self.f1_color and self.f2_color:  # non-interactive bootstrap
                ca = self._track_color(window, sampled_idx, a)
                f1, f2 = (a, b) if ca == self.f1_color else (b, a)
            elif self.interactive and b is not None:
                slot = sorted(set(a) & set(b))[len(set(a) & set(b)) // 2]
                a_is_f1 = self._prompt_user(window[sampled_idx[slot]], a[slot], b[slot])
                f1, f2 = (a, b) if a_is_f1 else (b, a)
            else:
                return None, None  # wait for a window where both fighters are visible
            if f1 is None or f2 is None:
                return None, None
            self._learn_prompt_colors(window, sampled_idx, f1, f2)
        else:
            f1, f2, _ = self._temporal_pair(window, sampled_idx, a, b)
            if f1 is None:
                return None, None

        self._update_identity_state(window, sampled_idx, f1, f2)
        return f1, f2

    def _assign_runtime(self, window, sampled_idx, sampled, dets):
        """Assign runtime boxes using prompt, robust colors, and conservative continuity."""
        n = len(sampled)
        tracks = top_two_tracks(link_tracks(dets))

        # The first live window honors the user's A/B answer, then learns/keeps
        # shorts colors and seeds clean temporal anchors.
        if (
            self.anchor_f1 is None
            and self.anchor_f2 is None
            and self.identity_choice in ("A", "B")
        ):
            f1_track, f2_track = self._assign_identity(window, sampled_idx, tracks)
            assigns = self._tracks_to_assigns(f1_track, f2_track, n)
            return assigns, {"method": "prompt", "margin": 1.0}

        if self.f1_color and self.f2_color:
            assigns, color_info = ident.assign_identities(
                sampled, dets, self.f1_color, self.f2_color, return_info=True
            )
            overlap = self._assignment_overlap(assigns)
            if color_info["complete"] and overlap <= MAX_PAIR_OVERLAP:
                f1_track = {
                    t: pair[0] for t, pair in enumerate(assigns) if pair[0] is not None
                }
                f2_track = {
                    t: pair[1] for t, pair in enumerate(assigns) if pair[1] is not None
                }
                self._update_identity_state(sampled, np.arange(n), f1_track, f2_track)
                return assigns, {
                    "method": "comparative_color",
                    "margin": color_info["margin"],
                }
            if color_info["confident"] and not color_info["complete"]:
                # One fighter may remain visible, but this evidence is never
                # allowed to rewrite the two-fighter identity anchors.
                return assigns, {
                    "method": "single_track_color",
                    "margin": color_info["margin"],
                }
            if overlap > MAX_PAIR_OVERLAP:
                return [(None, None)] * n, {"method": "abstain_merged", "margin": 0.0}

            # Only clean two-track, low-color-evidence cases may use temporal
            # continuity. One/merged/junk-track cases abstain completely.
            if color_info["reason"] == "low_color_margin":
                f1_track, f2_track, margin = self._temporal_pair(
                    sampled, np.arange(n), color_info["track_a"], color_info["track_b"]
                )
                if f1_track is not None:
                    temporal = self._tracks_to_assigns(f1_track, f2_track, n)
                    if self._assignment_overlap(temporal) <= MAX_PAIR_OVERLAP:
                        return temporal, {
                            "method": "temporal_fallback",
                            "margin": margin,
                        }
            return [(None, None)] * n, {
                "method": f"abstain_{color_info['reason']}",
                "margin": color_info["margin"],
            }

        f1_track, f2_track = self._assign_identity(window, sampled_idx, tracks)
        assigns = self._tracks_to_assigns(f1_track, f2_track, n)
        return assigns, {
            "method": "appearance_only"
            if self.anchor_f1 is not None
            else "abstain_no_identity",
            "margin": 0.0,
        }

    def _identity_prompt_from_window(self, window, w_idx):
        """Return an A/B prompt image for a live window with two usable tracks."""
        if not window:
            return None
        sampled_idx = np.linspace(0, len(window) - 1, C.NUM_FRAMES).round().astype(int)
        sampled = [window[i] for i in sampled_idx]
        p_excluded = self._gate_prob(sampled)
        if p_excluded > self.gate_threshold:
            return None
        tracks = top_two_tracks(link_tracks(ident.detect_fighters(self.yolo, sampled)))
        if len(tracks) < 2:
            return None
        common = sorted(set(tracks[0]) & set(tracks[1]))
        if not common:
            return None
        slot = common[len(common) // 2]
        image_path = self.out_dir / "identity_prompt.png"
        image = overlay.save_identity_prompt_image(
            image_path, window[sampled_idx[slot]], tracks[0][slot], tracks[1][slot]
        )
        return {
            "image_bgr": image,
            "image_path": str(image_path),
            "window": w_idx,
            "start_s": w_idx * C.CLIP_SECONDS,
            "gate_prob_excluded": round(p_excluded, 3),
        }

    def find_identity_prompt(self, video_path, max_windows=None):
        """Scan a video for the first live window where two fighters can be labeled A/B."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        win_len = int(round(fps * C.CLIP_SECONDS))
        window, w_idx = [], 0
        try:
            while max_windows is None or w_idx < max_windows:
                ok, frame = cap.read()
                if ok:
                    window.append(frame)
                if (not ok and window) or len(window) == win_len:
                    prompt = self._identity_prompt_from_window(window, w_idx)
                    if prompt is not None:
                        return prompt
                    window, w_idx = [], w_idx + 1
                if not ok:
                    break
        finally:
            cap.release()
        return None

    def find_identity_prompt_in_clips(self, clip_paths, max_clips=None):
        """Find an A/B identity frame in an ordered collection of 5-second clips."""
        for w_idx, clip_path in enumerate(clip_paths):
            if max_clips is not None and w_idx >= max_clips:
                break
            window = self._read_all_frames(clip_path)
            prompt = self._identity_prompt_from_window(window, w_idx)
            if prompt is not None:
                return prompt
        return None

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
        def prepare(meta):
            mean, std = MODEL_INPUT_STATS[meta["model_name"]]
            frames = torch.stack(
                [
                    torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                    .float()
                    .div_(255)
                    .permute(2, 0, 1)
                    for f in sampled_frames
                ]
            )
            frames = TF.resize(frames, [C.CROP_SIZE, C.CROP_SIZE], antialias=True)
            frames = TF.normalize(frames, mean, std)
            video = frames.permute(1, 0, 2, 3)
            if meta["in_channels"] == 4:
                m = torch.from_numpy(mask).float().unsqueeze(1)
                m = TF.resize(
                    m,
                    [C.CROP_SIZE, C.CROP_SIZE],
                    interpolation=TF.InterpolationMode.NEAREST,
                )
                video = torch.cat([video, m.permute(1, 0, 2, 3)], dim=0)
            return video.unsqueeze(0).to(self.device)

        logits_ph, logits_pr = self.phase_model(prepare(self.phase_meta))
        if self.pressure_model is not None:
            _, logits_pr = self.pressure_model(prepare(self.pressure_meta))
        phase = pressure = phase_conf = pressure_conf = None
        if logits_ph is not None:
            probs = torch.softmax(logits_ph, dim=1)[0]
            phase, phase_conf = C.IDX2PHASE[int(probs.argmax())], float(probs.max())
        if logits_pr is not None:
            probs = torch.softmax(logits_pr, dim=1)[0]
            pressure, pressure_conf = (
                C.IDX2PRESSURE[int(probs.argmax())],
                float(probs.max()),
            )
        return phase, pressure, phase_conf, pressure_conf

    # ── main loop ──
    def process(
        self, video_path, out_path, progress_callback=None, finalize_callback=None
    ):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        win_len = int(round(fps * C.CLIP_SECONDS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_windows = math.ceil(frame_count / win_len) if frame_count > 0 else None
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create output video: {out_path}")
        window, w_idx, results = [], 0, []
        while True:
            ok, frame = cap.read()
            if ok:
                window.append(frame)
            if (not ok and window) or len(window) == win_len:
                info = self._process_window(window, w_idx, writer)
                results.append(info)
                if progress_callback is not None:
                    progress_callback(len(results), total_windows, info)
                window, w_idx = [], w_idx + 1
            if not ok:
                break
        cap.release()
        writer.release()
        if finalize_callback is not None:
            finalize_callback()
        self._remux_audio(video_path, out_path)
        return results

    @staticmethod
    def _read_all_frames(video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        return frames

    def process_clips(
        self, clip_paths, out_path, progress_callback=None, finalize_callback=None
    ):
        """Run ordered 5-second clips as one continuous fight while preserving tracker state."""
        clip_paths = [Path(p) for p in clip_paths]
        if not clip_paths:
            raise ValueError("No clips were provided")
        first = cv2.VideoCapture(str(clip_paths[0]))
        if not first.isOpened():
            raise FileNotFoundError(clip_paths[0])
        fps = first.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
        first.release()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {out_path}")
        results = []
        try:
            for w_idx, clip_path in enumerate(clip_paths):
                window = self._read_all_frames(clip_path)
                if not window:
                    continue
                if window[0].shape[1] != width or window[0].shape[0] != height:
                    window = [cv2.resize(frame, (width, height)) for frame in window]
                info = self._process_window(window, w_idx, writer)
                info["source_clip"] = clip_path.name
                results.append(info)
                if progress_callback is not None:
                    progress_callback(w_idx + 1, len(clip_paths), info)
        finally:
            writer.release()
        # FFmpeg's concat demuxer joins the original per-clip audio streams while
        # the annotated frames are converted to a browser-compatible video.
        if finalize_callback is not None:
            finalize_callback()
        self._remux_audio(clip_paths, out_path)
        return results

    def _process_window(self, window, w_idx, writer):
        sampled_idx = np.linspace(0, len(window) - 1, C.NUM_FRAMES).round().astype(int)
        sampled = [window[i] for i in sampled_idx]
        info = {"window": w_idx, "start_s": w_idx * C.CLIP_SECONDS}

        p_excluded = self._gate_prob(sampled)
        info["gate_prob_excluded"] = round(p_excluded, 3)
        if p_excluded > self.gate_threshold:
            info.update(excluded=True, phase=None, pressure=None)
            overlay.annotate_window(
                window,
                sampled_idx,
                [],
                [],
                None,
                None,
                excluded=True,
                gate_prob_excluded=p_excluded,
            )
        else:
            dets = ident.detect_fighters(self.yolo, sampled)
            assigns, identity_info = self._assign_runtime(
                window, sampled_idx, sampled, dets
            )
            f1_boxes = [pair[0] for pair in assigns]
            f2_boxes = [pair[1] for pair in assigns]
            both_coverage, any_coverage = ident.coverage(assigns)
            mask = ident.build_masks(
                window[0].shape[:2],
                assigns,
                (
                    C.CACHE_SHORT_SIDE,
                    round(C.CACHE_SHORT_SIDE * window[0].shape[1] / window[0].shape[0]),
                ),
            )
            phase, pressure, phase_conf, pressure_conf = self._classify(sampled, mask)
            pressure_reliable = (
                not identity_info["method"].startswith("abstain")
                and both_coverage >= 0.25
            )
            info.update(
                excluded=False,
                phase=phase,
                pressure=pressure,
                phase_conf=round(phase_conf, 3) if phase_conf else None,
                pressure_conf=round(pressure_conf, 3) if pressure_conf else None,
                pressure_reliable=pressure_reliable,
                identity_method=identity_info["method"],
                identity_margin=round(float(identity_info["margin"]), 3),
                identity_both_coverage=round(both_coverage, 3),
                identity_any_coverage=round(any_coverage, 3),
            )
            overlay.annotate_window(
                window,
                sampled_idx,
                f1_boxes,
                f2_boxes,
                phase,
                pressure if pressure_reliable else None,
                self.f1_name,
                self.f2_name,
                phase_conf=phase_conf,
                pressure_conf=pressure_conf,
                gate_prob_excluded=p_excluded,
                identity_uncertain=not pressure_reliable,
            )
        for frame in window:
            writer.write(frame)
        tag = (
            "EXCLUDED"
            if info["excluded"]
            else (f"{info['phase']} ({info.get('phase_conf')}) | {info['pressure']}")
        )
        print(f"  window {w_idx:>3} [{info['start_s']:>4}s] {tag}")
        return info

    def _remux_audio(self, src, dst):
        """Best-effort audio remux and optional H.264 conversion for browser playback.

        ``src`` may be one full video or an ordered list of clips. For a clip
        list, FFmpeg's concat demuxer supplies one continuous audio input.
        """
        if src is None and not self.web_compatible:
            return
        tmp = str(dst) + ".audio.mp4"
        concat_file = None
        try:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                try:
                    from imageio_ffmpeg import get_ffmpeg_exe

                    ffmpeg = get_ffmpeg_exe()
                except ImportError:
                    return
            command = [ffmpeg, "-y", "-i", str(dst)]
            if isinstance(src, (list, tuple)):
                concat_file = Path(str(dst) + ".audio.concat.txt")
                lines = []
                for clip_path in src:
                    value = Path(clip_path).resolve().as_posix().replace("'", "'\\''")
                    lines.append(f"file '{value}'")
                concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
                command.extend(["-f", "concat", "-safe", "0", "-i", str(concat_file)])
            elif src is not None:
                command.extend(["-i", str(src)])
            if self.web_compatible:
                command.extend(
                    [
                        "-c:v",
                        "libx264",
                        "-preset",
                        "ultrafast",
                        "-crf",
                        "25",
                        "-movflags",
                        "+faststart",
                    ]
                )
            else:
                command.extend(["-c:v", "copy"])
            command.extend(["-map", "0:v:0"])
            if src is not None:
                command.extend(["-map", "1:a:0?", "-c:a", "aac", "-shortest"])
            command.append(tmp)
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            Path(tmp).replace(dst)
        except (subprocess.CalledProcessError, FileNotFoundError):
            Path(tmp).unlink(missing_ok=True)
        finally:
            if concat_file is not None:
                concat_file.unlink(missing_ok=True)
