"""Run the full pipeline on a fight video and write an annotated copy.

  python scripts/infer.py --video myfight.mp4 --out outputs/myfight_labeled.mp4

By default you are shown one frame with two boxes (A/B) and asked which is
Fighter 1 (the name LEFT of the broadcast timer). For non-interactive runs pass
--f1-color/--f2-color (shorts colors) instead.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mma.pipeline import FightAnalyzer  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--out", default=None, help="output video path")
    p.add_argument("--gate-ckpt", default="outputs/gate/gate.pt")
    p.add_argument("--phase-ckpt", default="outputs/phase/deployment_phase_final.pt")
    p.add_argument(
        "--pressure-ckpt",
        default="outputs/phase/deployment_pressure_final.pt",
        help="separate pressure-only checkpoint; overrides the pressure head in --phase-ckpt",
    )
    p.add_argument("--f1-name", default="Fighter 1")
    p.add_argument("--f2-name", default="Fighter 2")
    p.add_argument(
        "--f1-color", default=None, help="shorts color (skips the interactive prompt)"
    )
    p.add_argument("--f2-color", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    video = Path(args.video)
    out = Path(args.out) if args.out else video.with_name(video.stem + "_labeled.mp4")
    analyzer = FightAnalyzer(
        args.gate_ckpt,
        args.phase_ckpt,
        pressure_ckpt=args.pressure_ckpt,
        device=args.device,
        interactive=not (args.f1_color and args.f2_color),
        f1_color=args.f1_color,
        f2_color=args.f2_color,
        f1_name=args.f1_name,
        f2_name=args.f2_name,
    )
    print(f"Processing {video} -> {out}")
    results = analyzer.process(video, out)
    log = out.with_suffix(".json")
    log.write_text(json.dumps(results, indent=2))
    print(f"\nDone. Video: {out}\nPer-window log: {log}")


if __name__ == "__main__":
    main()
