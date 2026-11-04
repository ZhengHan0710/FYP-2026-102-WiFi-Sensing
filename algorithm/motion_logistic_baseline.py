#!/usr/bin/env python3
"""Two-feature logistic-regression baseline for CSI motion detection."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


FEATURES = ["temporal_std", "relative_temporal_diff"]


def recording_number(filename: str) -> int:
    match = re.search(r"_(\d+)\.csv$", filename)
    if not match:
        raise ValueError(f"Cannot read recording number from: {filename}")
    return int(match.group(1))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    total = len(actual)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path, help="window_features.csv path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with args.features.open(encoding="utf-8", newline="") as handle:
        rows.extend(csv.DictReader(handle))

    train_rows = [row for row in rows if recording_number(row["source_file"]) in (1, 2)]
    test_rows = [row for row in rows if recording_number(row["source_file"]) == 3]

    x_train = np.asarray(
        [[float(row[name]) for name in FEATURES] for row in train_rows],
        dtype=np.float64,
    )
    y_train = np.asarray(
        [row["scenario"] == "person_moving" for row in train_rows],
        dtype=np.float64,
    )
    x_test = np.asarray(
        [[float(row[name]) for name in FEATURES] for row in test_rows],
        dtype=np.float64,
    )
    y_test = np.asarray(
        [row["scenario"] == "person_moving" for row in test_rows],
        dtype=np.int64,
    )

    feature_mean = x_train.mean(axis=0)
    feature_std = x_train.std(axis=0)
    feature_std[feature_std < 1e-12] = 1.0
    x_train_scaled = (x_train - feature_mean) / feature_std
    x_test_scaled = (x_test - feature_mean) / feature_std

    class_counts = np.bincount(y_train.astype(np.int64), minlength=2)
    class_weights = len(y_train) / (2.0 * class_counts)
    sample_weights = class_weights[y_train.astype(np.int64)]

    weights = np.zeros(len(FEATURES), dtype=np.float64)
    bias = 0.0
    for _ in range(args.epochs):
        probabilities = sigmoid(x_train_scaled @ weights + bias)
        errors = (probabilities - y_train) * sample_weights
        weight_gradient = x_train_scaled.T @ errors / len(y_train)
        bias_gradient = float(errors.mean())
        weights -= args.learning_rate * weight_gradient
        bias -= args.learning_rate * bias_gradient

    train_probabilities = sigmoid(x_train_scaled @ weights + bias)
    test_probabilities = sigmoid(x_test_scaled @ weights + bias)
    train_predictions = (train_probabilities >= 0.5).astype(np.int64)
    test_predictions = (test_probabilities >= 0.5).astype(np.int64)

    train_metrics = evaluate(y_train.astype(np.int64), train_predictions)
    test_metrics = evaluate(y_test, test_predictions)

    by_scenario: dict[str, dict[str, float | int]] = {}
    scenario_indexes: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(test_rows):
        scenario_indexes[row["scenario"]].append(index)
    for scenario, indexes in scenario_indexes.items():
        predicted_motion = int(test_predictions[indexes].sum())
        by_scenario[scenario] = {
            "windows": len(indexes),
            "predicted_moving": predicted_motion,
            "predicted_moving_rate": predicted_motion / len(indexes),
        }

    prediction_path = args.output / "test_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "scenario", "source_file", "start_row", "end_row",
            *FEATURES, "moving_probability", "actual_motion", "predicted_motion",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, probability, actual, predicted in zip(
            test_rows, test_probabilities, y_test, test_predictions
        ):
            writer.writerow({
                "scenario": row["scenario"],
                "source_file": row["source_file"],
                "start_row": row["start_row"],
                "end_row": row["end_row"],
                FEATURES[0]: row[FEATURES[0]],
                FEATURES[1]: row[FEATURES[1]],
                "moving_probability": float(probability),
                "actual_motion": int(actual),
                "predicted_motion": int(predicted),
            })

    report = {
        "task": "person_moving versus nonmoving",
        "model": "two-feature logistic regression implemented with NumPy",
        "features": FEATURES,
        "split": "recordings 1 and 2 train; recording 3 test",
        "feature_mean_train": feature_mean.tolist(),
        "feature_std_train": feature_std.tolist(),
        "weights_scaled": weights.tolist(),
        "bias_scaled": bias,
        "train": train_metrics,
        "test": test_metrics,
        "test_by_scenario": by_scenario,
    }
    report_path = args.output / "motion_logistic_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Two-feature logistic-regression baseline")
    print("Features:", ", ".join(FEATURES))
    print("Training data: recordings 1 and 2")
    print("Test data: recording 3")
    print("-" * 60)
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
    for scenario, result in by_scenario.items():
        print(
            f"{scenario}: {result['predicted_moving']}/{result['windows']} "
            f"predicted as moving "
            f"({100 * float(result['predicted_moving_rate']):.2f}%)"
        )
    print("Report:", report_path)
    print("Test predictions:", prediction_path)


if __name__ == "__main__":
    main()
