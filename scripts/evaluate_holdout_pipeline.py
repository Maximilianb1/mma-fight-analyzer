"""Combine frozen gate and classifier predictions on the untouched holdout.

This reports both conditional classifier performance and end-to-end performance,
where a real fight clip rejected by the gate counts as an incorrect phase and
pressure prediction.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def keys(data):
    return np.char.add(
        np.char.add(data["fight"].astype(str), "\0"), data["filename"].astype(str)
    )


def routed_task(gate, downstream, task, n_classes):
    gate_lookup = {key: int(pred) for key, pred in zip(keys(gate), gate["prediction"])}
    y_true = downstream[f"{task}_true"].astype(int)
    y_pred = downstream[f"{task}_pred"].astype(int)
    passed = np.asarray([gate_lookup[key] == 0 for key in keys(downstream)])
    routed = y_pred.copy()
    routed[~passed] = -1  # rejected real-fight clips are pipeline errors
    return {
        "n_fight_clips": int(len(y_true)),
        "gate_pass_rate": float(passed.mean()),
        "conditional_accuracy": float(accuracy_score(y_true, y_pred)),
        "conditional_macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=range(n_classes),
                average="macro",
                zero_division=0,
            )
        ),
        "end_to_end_accuracy": float(accuracy_score(y_true, routed)),
        "end_to_end_macro_f1": float(
            f1_score(
                y_true,
                routed,
                labels=range(n_classes),
                average="macro",
                zero_division=0,
            )
        ),
        "rejected_fight_clips": int((~passed).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", default="outputs/gate/gate_holdout_predictions.npz")
    parser.add_argument("--phase", required=True)
    parser.add_argument(
        "--pressure",
        default=None,
        help="defaults to --phase when one checkpoint predicts both tasks",
    )
    parser.add_argument("--out", default="outputs/report/pipeline_holdout_metrics.json")
    args = parser.parse_args()

    gate = np.load(args.gate)
    phase = np.load(args.phase)
    pressure = np.load(args.pressure or args.phase)
    gate_target = gate["target"].astype(int)
    gate_pred = gate["prediction"].astype(int)
    fight_mask = gate_target == 0
    nonfight_mask = ~fight_mask
    result = {
        "protocol": "one-shot untouched-fight end-to-end evaluation",
        "holdout_fight": str(gate["fight"][0]),
        "gate": {
            "accuracy": float(accuracy_score(gate_target, gate_pred)),
            "fight_retention": float(np.mean(gate_pred[fight_mask] == 0)),
            "nonfight_rejection": float(np.mean(gate_pred[nonfight_mask] == 1)),
            "n_clips": int(len(gate_target)),
        },
        "phase": routed_task(gate, phase, "phase", 5),
        "pressure": routed_task(gate, pressure, "pressure", 3),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
