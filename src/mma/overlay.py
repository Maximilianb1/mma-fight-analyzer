"""Drawing utilities for the annotated output video."""

import cv2

F1_COLOR = (255, 120, 0)  # BGR: blue-ish for Fighter 1
F2_COLOR = (0, 0, 255)  # red for Fighter 2
BANNER_BG = (20, 20, 20)
LOW_CONF = 0.5  # below this the banner is dimmed ("the system isn't sure")


def interpolate_boxes(sampled_idx, boxes, n_frames):
    """Linearly interpolate per-sampled-frame boxes (some may be None) to every frame."""
    known = [(t, b) for t, b in zip(sampled_idx, boxes) if b is not None]
    out = [None] * n_frames
    if not known:
        return out
    for f in range(n_frames):
        prev = max((k for k in known if k[0] <= f), key=lambda k: k[0], default=None)
        nxt = min((k for k in known if k[0] >= f), key=lambda k: k[0], default=None)
        if prev is None:
            out[f] = nxt[1]
        elif nxt is None or prev[0] == nxt[0]:
            out[f] = prev[1]
        else:
            a = (f - prev[0]) / (nxt[0] - prev[0])
            out[f] = tuple(int((1 - a) * p + a * q) for p, q in zip(prev[1], nxt[1]))
    return out


def draw_fighter_box(frame, box, name, color):
    if box is None:
        return
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
    cv2.putText(
        frame,
        name,
        (x1 + 4, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_banner(frame, lines, alpha=0.65, text_color=(255, 255, 255)):
    """Semi-transparent info banner in the top-left corner."""
    pad, lh = 12, 30
    width = max(
        cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] for t in lines
    )
    h = pad * 2 + lh * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width + pad * 3, h), BANNER_BG, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (pad, pad + lh * i + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
            cv2.LINE_AA,
        )


def annotate_window(
    frames,
    sampled_idx,
    f1_boxes,
    f2_boxes,
    phase,
    pressure,
    f1_name="Fighter 1",
    f2_name="Fighter 2",
    excluded=False,
    phase_conf=None,
    pressure_conf=None,
    gate_prob_excluded=None,
    identity_uncertain=False,
):
    """Draw boxes + labels on every frame of a 5s window, in place.
    Low-confidence windows get a dimmed gray banner."""
    segment_conf = None
    if gate_prob_excluded is not None:
        segment_conf = gate_prob_excluded if excluded else 1.0 - gate_prob_excluded
    if excluded:
        text = "Segment: NON-FIGHT"
        if segment_conf is not None:
            text += f" ({segment_conf:.0%})"
        for frame in frames:
            draw_banner(frame, [text, "Replay / break / broadcast footage"])
        return
    f1_all = interpolate_boxes(sampled_idx, f1_boxes, len(frames))
    f2_all = interpolate_boxes(sampled_idx, f2_boxes, len(frames))
    segment = "Segment: FIGHT"
    if segment_conf is not None:
        segment += f" ({segment_conf:.0%})"
    lines = [segment]
    if phase is not None:
        lines.append(
            f"Phase: {phase}"
            + (f" ({phase_conf:.0%})" if phase_conf is not None else "")
        )
    if pressure is not None:
        who = {"Fighter 1": f1_name, "Fighter 2": f2_name}.get(pressure, "Mutual")
        lines.append(
            f"Pressure: {who}"
            + (f" ({pressure_conf:.0%})" if pressure_conf is not None else "")
        )
    elif identity_uncertain:
        lines.append("Pressure: uncertain (fighter identity unavailable)")
    dim = phase_conf is not None and phase_conf < LOW_CONF
    for f, frame in enumerate(frames):
        draw_fighter_box(frame, f1_all[f], f1_name, F1_COLOR)
        draw_fighter_box(frame, f2_all[f], f2_name, F2_COLOR)
        if lines:
            draw_banner(
                frame,
                lines,
                alpha=0.45 if dim else 0.65,
                text_color=(170, 170, 170) if dim else (255, 255, 255),
            )


def save_identity_prompt_image(path, frame, box_a, box_b):
    img = frame.copy()
    for box, tag, color in ((box_a, "A", (0, 255, 0)), (box_b, "B", (0, 255, 255))):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        cv2.putText(
            img, tag, (x1 + 8, y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4
        )
    cv2.imwrite(str(path), img)
    return img
