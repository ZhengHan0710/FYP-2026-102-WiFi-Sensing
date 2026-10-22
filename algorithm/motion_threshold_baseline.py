#!/usr/bin/env python3
"""Train and test an interpretable motion threshold on CSI window features."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


FEATURE = "relative_temporal_diff"


def recording_number(filename: str) -> int:
    match = re.search(r"_(\d+)\.csv$", filename)
    if not match:
        raise ValueError(f"Cannot read recording number from: {filename}")
    return int(match.group(1))


def metrics(rows: list[dict[str, object]], threshold: float) -> dict[str, float | int]:
    tp = tn = fp = fn = 0
    for row in rows:
        actual = int(row["target"])
        predicted = int(float(row[FEATURE]) > threshold)
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    moving_recall = tp / (tp + fn) if tp + fn else 0.0
    nonmoving_recall = tn / (tn + fp) if tn + fp else 0.0
    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "moving_recall": moving_recall,
        "nonmoving_recall": nonmoving_recall,
        "balanced_accuracy": (moving_recall + nonmoving_recall) / 2,
    }


def choose_threshold(rows: list[dict[str, object]]) -> tuple[float, dict[str, float | int]]:
    values = sorted({float(row[FEATURE]) for row in rows})
    if not values:
        raise ValueError("No training windows found")
    candidates = [values[0] - 1e-9]
    candidates.extend((left + right) / 2 for left, right in zip(values, values[1:]))
    candidates.append(values[-1] + 1e-9)

    scored = [(metrics(rows, threshold), threshold) for threshold in candidates]
    best_metrics, best_threshold = max(
        scored,
        key=lambda item: (item[0]["balanced_accuracy"], item[0]["accuracy"]),
    )
    return best_threshold, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path, help="window_features.csv path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with args.features.open(encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            row: dict[str, object] = dict(source)
            row[FEATURE] = float(source[FEATURE])
            row["recording"] = recording_number(source["source_file"])
            row["target"] = int(source["scenario"] == "person_moving")
            rows.append(row)

    train_rows = [row for row in rows if int(row["recording"]) in (1, 2)]
    test_rows = [row for row in rows if int(row["recording"]) == 3]

    threshold, train_metrics = choose_threshold(train_rows)
    test_metrics = metrics(test_rows, threshold)

    prediction_path = args.output / "test_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "scenario", "source_file", "start_row", "end_row", FEATURE,
            "actual_motion", "predicted_motion",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in test_rows:
            writer.writerow({
                "scenario": row["scenario"],
                "source_file": row["source_file"],
                "start_row": row["start_row"],
                "end_row": row["end_row"],
                FEATURE: row[FEATURE],
                "actual_motion": row["target"],
                "predicted_motion": int(float(row[FEATURE]) > threshold),
            })

    report = {
        "task": "person_moving versus nonmoving",
        "feature": FEATURE,
        "rule": f"feature > {threshold:.8f} means moving",
        "split": "recordings 1 and 2 train; recording 3 test",
        "threshold": threshold,
        "train": train_metrics,
        "test": test_metrics,
    }
    report_path = args.output / "motion_threshold_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Simple motion-detection baseline")
    print("Training data: recordings 1 and 2")
    print("Test data: recording 3")
    print(f"Decision rule: {FEATURE} > {threshold:.6f} => moving")
    print("-" * 56)
    print(
        "Train balanced accuracy:",
        f"{100 * float(train_metrics['balanced_accuracy']):.2f}%",
    )
    print("Test windows:", test_metrics["total"])
    print("Test accuracy:", f"{100 * float(test_metrics['accuracy']):.2f}%")
    print(
        "Test balanced accuracy:",
        f"{100 * float(test_metrics['balanced_accuracy']):.2f}%",
    )
    print("Moving recall:", f"{100 * float(test_metrics['moving_recall']):.2f}%")
    print(
        "Nonmoving recall:",
        f"{100 * float(test_metrics['nonmoving_recall']):.2f}%",
    )
    print(
        "Confusion counts: TP={tp}, TN={tn}, FP={fp}, FN={fn}".format(
            **test_metrics
        )
    )
    print("Report:", report_path)
    print("Test predictions:", prediction_path)


if __name__ == "__main__":
    main()
