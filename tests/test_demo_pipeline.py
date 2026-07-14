from types import MethodType

import cv2
import numpy as np

from mma.overlay import annotate_window
from mma.pipeline import FightAnalyzer
from mma.identity import assign_identities


def test_overlay_marks_live_and_nonfight_windows():
    sampled_idx = np.array([0, 1])
    live = [np.zeros((120, 320, 3), dtype=np.uint8) for _ in range(2)]
    annotate_window(
        live,
        sampled_idx,
        [None, None],
        [None, None],
        "Striking",
        "Mutual",
        gate_prob_excluded=0.1,
        phase_conf=0.8,
        pressure_conf=0.7,
    )
    assert np.count_nonzero(live[0][:100]) > 0

    nonfight = [np.zeros((120, 320, 3), dtype=np.uint8) for _ in range(2)]
    annotate_window(
        nonfight,
        sampled_idx,
        [],
        [],
        None,
        None,
        excluded=True,
        gate_prob_excluded=0.9,
    )
    assert np.count_nonzero(nonfight[0][:100]) > 0


def test_web_identity_choice_maps_box_b_to_fighter_1():
    analyzer = FightAnalyzer.__new__(FightAnalyzer)
    analyzer.identity_choice = "B"
    analyzer.f1_color = analyzer.f2_color = None
    analyzer.interactive = False
    analyzer.anchor_f1 = analyzer.anchor_f2 = None

    def fake_hist(self, window, sampled_idx, track):
        return np.ones((2, 2), dtype=np.float32)

    analyzer._track_hist = MethodType(fake_hist, analyzer)
    track_a = {0: (0, 0, 10, 10)}
    track_b = {0: (20, 0, 30, 10)}
    f1, f2 = analyzer._assign_identity(
        [np.zeros((40, 40, 3), dtype=np.uint8)], np.array([0]), [track_a, track_b]
    )
    assert f1 is track_b
    assert f2 is track_a


def test_comparative_shorts_color_reacquires_identity_and_reports_margin():
    box_black = (5, 5, 25, 35)
    box_red = (35, 5, 55, 35)
    frames, detections = [], []
    for _ in range(6):
        frame = np.full((45, 65, 3), 200, dtype=np.uint8)
        frame[5:35, 5:25] = (0, 0, 0)
        frame[5:35, 35:55] = (0, 0, 255)
        frames.append(frame)
        detections.append([box_black, box_red])

    assignments, info = assign_identities(
        frames, detections, "black", "red", return_info=True
    )

    assert info["decision"] == "comparative_color"
    assert info["complete"] is True
    assert info["margin"] > 0.03
    assert all(f1 == box_black and f2 == box_red for f1, f2 in assignments)


def test_comparative_shorts_color_abstains_when_evidence_is_ambiguous():
    box_a = (5, 5, 25, 35)
    box_b = (35, 5, 55, 35)
    frames = [np.full((45, 65, 3), 120, dtype=np.uint8) for _ in range(6)]
    detections = [[box_a, box_b] for _ in frames]

    assignments, info = assign_identities(
        frames, detections, "black", "red", return_info=True
    )

    assert info["decision"] == "abstain"
    assert info["reason"] == "low_color_margin"
    assert all(f1 is None and f2 is None for f1, f2 in assignments)


def test_video_process_reports_window_progress(tmp_path):
    source = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (64, 48)
    )
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    analyzer = FightAnalyzer.__new__(FightAnalyzer)
    analyzer.web_compatible = False

    def fake_window(self, window, index, output_writer):
        for frame in window:
            output_writer.write(frame)
        return {"window": index, "start_s": index * 5, "excluded": True}

    analyzer._process_window = MethodType(fake_window, analyzer)
    analyzer._remux_audio = MethodType(lambda self, src, dst: None, analyzer)
    updates = []
    results = analyzer.process(
        source,
        output,
        lambda done, total, info: updates.append((done, total, info["window"])),
        lambda: updates.append(("finalizing", None, None)),
    )

    assert len(results) == 2
    assert updates == [(1, 2, 0), (2, 2, 1), ("finalizing", None, None)]
    assert output.is_file()
