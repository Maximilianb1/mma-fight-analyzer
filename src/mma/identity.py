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
    "red":    [((0, 60, 60), (12, 255, 255)), ((168, 60, 60), (180, 255, 255))],
    "blue":   [((100, 60, 50), (135, 255, 255))],
    "black":  [((0, 0, 0), (180, 100, 70))],
    "white":  [((0, 0, 180), (180, 50, 255))],
    "green":  [((35, 60, 50), (85, 255, 255))],
    "gold":   [((15, 60, 50), (35, 255, 255))],
    "gray":   [((0, 0, 50), (180, 50, 180))],
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
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def get_shorts_region(hsv, box):
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    return hsv[max(0, y1 + int(h * 0.35)):min(hsv.shape[0], y1 + int(h * 0.60)),
               max(0, x1 + int(w * 0.15)):min(hsv.shape[1], x2 - int(w * 0.15))]


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


def _assign_frame_by_color(hsv, boxes, f1c, f2c):
    colors = [classify_shorts_color(get_shorts_region(hsv, b))[0] for b in boxes]
    if len(boxes) == 2:
        a, b = colors
        if (a == f1c and b == f2c) or (a == f1c and b != f1c) or (b == f2c and a != f2c):
            return boxes[0], boxes[1]
        if (a == f2c and b == f1c) or (a == f2c and b != f2c) or (b == f1c and a != f1c):
            return boxes[1], boxes[0]
        return None, None
    if len(boxes) == 1:
        if colors[0] == f1c:
            return boxes[0], None
        if colors[0] == f2c:
            return None, boxes[0]
    return None, None


def assign_identities(frames_bgr, dets, f1_color, f2_color):
    """Per-frame (f1_box, f2_box) assignment.

    Pass 1 assigns by shorts color where the colors are decisive; passes 2-3
    propagate identity to ambiguous frames from temporal neighbours via IoU.
    """
    n = len(frames_bgr)
    assigns = []
    for frame, boxes in zip(frames_bgr, dets):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        f1b, f2b = _assign_frame_by_color(hsv, boxes, f1_color, f2_color)
        assigns.append(_complete_pair(boxes, f1b, f2b))

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
            x1, y1, x2, y2 = (int(f1b[0] * sx), int(f1b[1] * sy),
                              int(f1b[2] * sx), int(f1b[3] * sy))
            m[y1:y2, x1:x2] = 1
        if f2b is not None:
            x1, y1, x2, y2 = (int(f2b[0] * sx), int(f2b[1] * sy),
                              int(f2b[2] * sx), int(f2b[3] * sy))
            region = m[y1:y2, x1:x2]
            m[y1:y2, x1:x2] = np.where(region == 1, 0, -1).astype(np.int8)
    return masks


def coverage(assigns):
    """Fraction of frames with both / at least one fighter identified."""
    n = max(len(assigns), 1)
    both = sum(1 for a, b in assigns if a is not None and b is not None)
    any_ = sum(1 for a, b in assigns if a is not None or b is not None)
    return both / n, any_ / n
