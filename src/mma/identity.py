"""Fighter identity from person detection + shorts-color anchoring.

Identity convention (matches the dataset labels): Fighter 1 is the fighter whose
name appears LEFT of the timer in the broadcast overlay, Fighter 2 on the right.
At training time the mapping is provided per fight via data/fights_meta.csv
(f1_color / f2_color = shorts colors); at inference time it comes from a one-time
user prompt or from --f1-color/--f2-color.
"""

import cv2
import numpy as np

# HSV ranges for shorts colors (H in [0,180], S/V in [0,255])
COLOR_RANGES = {
    # red needs S>=140: dark-skin pixels sit at H<12 with S up to ~130 and were
    # polluting the red evidence on shirtless fighters
    "red": [((0, 140, 60), (12, 255, 255)), ((168, 140, 60), (180, 255, 255))],
    "blue": [((100, 60, 50), (135, 255, 255))],
    "black": [((0, 0, 0), (180, 100, 70))],
    # white allows V>=150: arena shadows push white shorts well below V=180
    "white": [((0, 0, 150), (180, 60, 255))],
    "green": [((35, 60, 50), (85, 255, 255))],
    "gold": [((15, 60, 50), (35, 255, 255))],
    "gray": [((0, 0, 50), (180, 50, 180))],
    "orange": [((10, 60, 50), (20, 255, 255))],
    "purple": [((130, 60, 50), (165, 255, 255))],
}

MIN_PROPAGATION_IOU = 0.2


def load_yolo(weights="yolov8n.pt"):
    from ultralytics import YOLO  # heavy import, keep lazy

    return YOLO(weights)


def detect_fighters(yolo, frames_bgr, conf=0.35, max_dets=2):
    """Person detection on a list of BGR frames -> per-frame list of (x1,y1,x2,y2)."""
    results = yolo(frames_bgr, classes=[0], conf=conf, verbose=False)
    per_frame = []
    for r in results:
        boxes = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
        boxes.sort(key=lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1]), reverse=True)
        per_frame.append(boxes[:max_dets])
    return per_frame


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def get_shorts_region(hsv, box):
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    return hsv[
        max(0, y1 + int(h * 0.35)) : min(hsv.shape[0], y1 + int(h * 0.60)),
        max(0, x1 + int(w * 0.15)) : min(hsv.shape[1], x2 - int(w * 0.15)),
    ]


def classify_shorts_color(hsv_crop):
    if hsv_crop.size == 0:
        return "unknown", 0.0
    total = hsv_crop.shape[0] * hsv_crop.shape[1]
    best, best_ratio = "unknown", 0.0
    for name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv_crop.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv_crop, np.array(lo), np.array(hi))
        ratio = float(mask.sum()) / 255.0 / total
        if ratio > best_ratio:
            best_ratio, best = ratio, name
    return best, best_ratio


# HSV skin band: fighters are shirtless, so shorts regions are heavily
# skin-contaminated, and skin hues overlap the red/orange/gold cloth ranges.
SKIN_LO, SKIN_HI = (3, 40, 70), (25, 180, 255)


def _skin_mask(region_hsv):
    return cv2.inRange(region_hsv, np.array(SKIN_LO), np.array(SKIN_HI))


def color_ratio(hsv, box, color):
    """Fraction of NON-SKIN shorts-region pixels matching a named color."""
    region = get_shorts_region(hsv, box)
    if region.size == 0:
        return 0.0
    total = region.shape[0] * region.shape[1]
    keep = cv2.bitwise_not(_skin_mask(region))
    n_keep = int(keep.sum()) // 255
    if n_keep < 0.15 * total:  # region is nearly all skin -> no cloth evidence
        return 0.0
    mask = np.zeros(region.shape[:2], np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask |= cv2.inRange(region, np.array(lo), np.array(hi))
    mask &= keep
    return float(mask.sum()) / 255.0 / n_keep


def _complete_pair(boxes, f1b, f2b):
    """If exactly one identity is known and another detection exists, it is the other fighter."""
    if f1b is not None and f2b is None:
        others = [b for b in boxes if b != f1b]
        if others:
            f2b = others[0]
    elif f2b is not None and f1b is None:
        others = [b for b in boxes if b != f2b]
        if others:
            f1b = others[0]
    return f1b, f2b


TRACK_LINK_IOU = 0.25
CLIP_DECISION_MARGIN = 0.03  # min clip-aggregated evidence gap between pairings
SINGLE_TRACK_MARGIN = 0.08  # stricter: no differential cancellation with one track
JUNK_TRACK_AREA_FRACTION = 0.2  # 2nd track smaller than this vs 1st = crowd/referee


def _link_tracks(dets):
    """Greedy IoU linking of per-frame detections into tracks: {frame_idx: box}."""
    tracks, last = [], {}
    for t, boxes in enumerate(dets):
        used = set()
        for box in boxes:
            best, best_iou = None, TRACK_LINK_IOU
            for tid, pb in last.items():
                if tid in used:
                    continue
                v = iou(box, pb)
                if v > best_iou:
                    best, best_iou = tid, v
            if best is None:
                best = len(tracks)
                tracks.append({})
            tracks[best][t] = box
            last[best] = box
            used.add(best)
    return tracks


def assign_identities(frames_bgr, dets, f1_color, f2_color, return_info=False):
    """Per-frame (f1_box, f2_box) assignment with a CLIP-LEVEL identity decision.

    Detections are linked into tracks across the clip; color evidence is
    aggregated over each whole track and the straight-vs-swapped pairing is
    decided once — single badly-lit frames get outvoted instead of getting a
    vote. Frames not covered by the two main tracks are then filled by the
    IoU propagation passes.
    """
    n = len(frames_bgr)
    dets = [boxes[:2] for boxes in dets]
    hsvs = {}

    def _hsv(t):
        if t not in hsvs:
            hsvs[t] = cv2.cvtColor(frames_bgr[t], cv2.COLOR_BGR2HSV)
        return hsvs[t]

    tracks = sorted(
        _link_tracks(dets),
        key=lambda tr: sum((b[2] - b[0]) * (b[3] - b[1]) for b in tr.values()),
        reverse=True,
    )
    track_a = tracks[0] if tracks else {}
    track_b = tracks[1] if len(tracks) > 1 else {}

    def mean_area(track):
        return (
            sum((b[2] - b[0]) * (b[3] - b[1]) for b in track.values()) / len(track)
            if track
            else 0.0
        )

    second_track_rejected = False
    if track_b and mean_area(track_b) < JUNK_TRACK_AREA_FRACTION * mean_area(track_a):
        track_b = {}  # crowd/referee fragment, never a pairing partner
        second_track_rejected = True

    def mean_ratio(track, color):
        total = sum(color_ratio(_hsv(t), box, color) for t, box in track.items())
        return total / len(track) if track else 0.0

    f1_track, f2_track = {}, {}
    decision, margin = "abstain", 0.0
    if track_b:
        straight = mean_ratio(track_a, f1_color) + mean_ratio(track_b, f2_color)
        swapped = mean_ratio(track_a, f2_color) + mean_ratio(track_b, f1_color)
        margin = abs(straight - swapped)
        if margin > CLIP_DECISION_MARGIN:
            f1_track, f2_track = (
                (track_a, track_b) if straight > swapped else (track_b, track_a)
            )
            decision = "comparative_color"
    elif track_a:
        # single usable track: identify it alone; merged fighter-pair boxes carry
        # BOTH colors and abstain here on purpose
        ra, rb = mean_ratio(track_a, f1_color), mean_ratio(track_a, f2_color)
        margin = abs(ra - rb)
        if ra - rb > SINGLE_TRACK_MARGIN:
            f1_track = track_a
            decision = "single_track_color"
        elif rb - ra > SINGLE_TRACK_MARGIN:
            f2_track = track_a
            decision = "single_track_color"

    if f1_track or f2_track:
        complete = bool(f1_track) and bool(f2_track)
        assigns = [
            _complete_pair(dets[t], f1_track.get(t), f2_track.get(t))
            if complete
            else (f1_track.get(t), f2_track.get(t))
            for t in range(n)
        ]
    else:
        assigns = [(None, None)] * n  # no reliable evidence -> zero masks

    for order in (range(n), range(n - 1, -1, -1)):
        prev = None
        for t in order:
            f1b, f2b = assigns[t]
            if f1b is None and f2b is None and prev is not None and dets[t]:
                pf1, pf2 = prev
                if pf1 is not None:
                    cand = max(dets[t], key=lambda b: iou(b, pf1))
                    if iou(cand, pf1) > MIN_PROPAGATION_IOU:
                        f1b = cand
                if pf2 is not None:
                    rest = [b for b in dets[t] if b != f1b]
                    if rest:
                        cand = max(rest, key=lambda b: iou(b, pf2))
                        if iou(cand, pf2) > MIN_PROPAGATION_IOU:
                            f2b = cand
                if f1b is not None or f2b is not None:
                    assigns[t] = _complete_pair(dets[t], f1b, f2b)
            if assigns[t][0] is not None or assigns[t][1] is not None:
                prev = assigns[t]
    if return_info:
        reason = None
        if decision == "abstain":
            if second_track_rejected:
                reason = "junk_second_track"
            elif track_b:
                reason = "low_color_margin"
            elif track_a:
                reason = "single_or_merged_track"
            else:
                reason = "no_tracks"
        return assigns, {
            "decision": decision,
            "reason": reason,
            "margin": float(margin),
            "complete": bool(f1_track) and bool(f2_track),
            "confident": bool(f1_track) or bool(f2_track),
            "track_a": track_a,
            "track_b": track_b,
        }
    return assigns


def build_masks(frame_hw, assigns, out_hw):
    """Identity masks: +1 = Fighter 1, -1 = Fighter 2, 0 = background/overlap."""
    h, w = frame_hw
    oh, ow = out_hw
    sy, sx = oh / h, ow / w
    masks = np.zeros((len(assigns), oh, ow), np.int8)
    for t, (f1b, f2b) in enumerate(assigns):
        m = masks[t]
        if f1b is not None:
            x1, y1, x2, y2 = (
                int(f1b[0] * sx),
                int(f1b[1] * sy),
                int(f1b[2] * sx),
                int(f1b[3] * sy),
            )
            m[y1:y2, x1:x2] = 1
        if f2b is not None:
            x1, y1, x2, y2 = (
                int(f2b[0] * sx),
                int(f2b[1] * sy),
                int(f2b[2] * sx),
                int(f2b[3] * sy),
            )
            region = m[y1:y2, x1:x2]
            m[y1:y2, x1:x2] = np.where(region == 1, 0, -1).astype(np.int8)
    return masks


def coverage(assigns):
    """Fraction of frames with both / at least one fighter identified."""
    n = max(len(assigns), 1)
    both = sum(1 for a, b in assigns if a is not None and b is not None)
    any_ = sum(1 for a, b in assigns if a is not None or b is not None)
    return both / n, any_ / n
