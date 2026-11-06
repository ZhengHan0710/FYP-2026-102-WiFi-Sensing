#!/usr/bin/env python3
"""Evaluate a frozen motion CNN on labelled CSVs without fitting or training."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from motion_cnn_baseline import (
    SCENARIOS, build_windows, evaluate, load_motion_checkpoint, predict,
    preprocess_from_checkpoint, source_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS.values()),
                        default=list(SCENARIOS.values()),
                        help="Scenario names to evaluate; every selected folder must contain CSVs.")
    parser.add_argument("--test-recording", type=int,
                        help="Optional recording-number subset, e.g. 3 for the old held-out fold.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device(args.device)
    model, checkpoint = load_motion_checkpoint(args.checkpoint, device)
    selected_scenarios = [scenario for scenario in SCENARIOS.values() if scenario in args.scenarios]
    windows, labels, metadata = build_windows(
        args.data_root, checkpoint["window"], checkpoint["step"], scenarios=selected_scenarios
    )
    if args.test_recording is not None:
        indexes = [i for i, row in enumerate(metadata) if row["recording"] == args.test_recording]
        if not indexes:
            raise SystemExit("Selected recording has no valid windows")
        windows, labels = windows[indexes], labels[indexes]
        metadata = [metadata[i] for i in indexes]
    sources = source_manifest(args.data_root, metadata)
    training_hashes = {row["sha256"] for row in checkpoint["training"]["source_files"]}
    reused = [row["path"] for row in sources if row["sha256"] in training_hashes]
    if reused:
        raise SystemExit(f"Evaluation CSV duplicates checkpoint training data: {reused}")
    values = preprocess_from_checkpoint(windows, checkpoint)
    loader = DataLoader(TensorDataset(torch.from_numpy(values), torch.from_numpy(labels)),
                        batch_size=args.batch_size, shuffle=False)
    probabilities, predictions = predict(model, loader, device)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        names = ["scenario", "source_file", "source_relative_path", "recording",
                 "start_row", "end_row", "moving_probability", "actual_motion", "predicted_motion"]
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row, probability, actual, predicted in zip(metadata, probabilities, labels, predictions):
            writer.writerow({**row, "moving_probability": float(probability),
                             "actual_motion": int(actual), "predicted_motion": int(predicted)})
    by_scenario = {}
    for scenario in sorted({row["scenario"] for row in metadata}):
        indexes = [i for i, row in enumerate(metadata) if row["scenario"] == scenario]
        by_scenario[scenario] = {"windows": len(indexes),
                                 "predicted_moving": int(predictions[indexes].sum()),
                                 "predicted_moving_rate": float(predictions[indexes].mean())}
    classes_present = sorted(int(value) for value in np.unique(labels))
    if classes_present == [1]:
        evaluation_scope = (
            "motion recall on positive examples only; nonmoving recall and false-positive "
            "performance are not estimable"
        )
    elif classes_present == [0]:
        evaluation_scope = "nonmoving recall on negative examples only; moving recall is not estimable"
    else:
        evaluation_scope = "binary motion evaluation with both classes present"
    report = {
        "task": "person_moving versus nonmoving",
        "mode": "frozen checkpoint evaluation; no training or normalization fitting",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "training": checkpoint["training"],
        "test_data_root": str(args.data_root.resolve()), "test_source_files": sources,
        "selected_scenarios": selected_scenarios, "classes_present": classes_present,
        "evaluation_scope": evaluation_scope,
        "test_recording": args.test_recording, "input_mode": checkpoint["input_mode"],
        "input_shape": checkpoint["input_shape"], "window": checkpoint["window"],
        "step": checkpoint["step"], "device": str(device),
        "test": evaluate(labels, predictions), "test_by_scenario": by_scenario,
    }
    report_path = args.output / "frozen_evaluation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["test"], ensure_ascii=False, indent=2))
    print("Report:", report_path)


if __name__ == "__main__":
    main()
