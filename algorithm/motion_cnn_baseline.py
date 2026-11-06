#!/usr/bin/env python3
"""Small CNN baseline for motion detection on project CSI recordings.

By default, recordings 1 and 2 train the model and recording 3 is held out.
With --test-data-root, all recordings in --data-root train the model and a
separate acquisition batch is used only for testing. Each example is the full
CSI-amplitude matrix of one time window instead of two summary features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from analyze_scenarios import SCENARIOS, continuous_ranges, load_csv


def recording_number(filename: str) -> int:
    match = re.search(r"_(\d+)\.csv$", filename)
    if not match:
        raise ValueError(f"Cannot read recording number from: {filename}")
    return int(match.group(1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_separate_batches(train_root: Path, test_root: Path) -> None:
    """Reject reused directories and byte-identical CSVs across the split."""
    if train_root.resolve() == test_root.resolve():
        raise ValueError("Training and external test directories must be different")

    train_hashes: dict[str, Path] = {}
    for folder in SCENARIOS:
        for path in sorted((train_root / folder).rglob("*.csv")):
            train_hashes[hashlib.sha256(path.read_bytes()).hexdigest()] = path
    for folder in SCENARIOS:
        for path in sorted((test_root / folder).rglob("*.csv")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in train_hashes:
                raise ValueError(
                    "External test CSV duplicates training data: "
                    f"{path} == {train_hashes[digest]}"
                )


def build_windows(
    data_root: Path,
    window_length: int,
    step: int,
    scenarios: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Build windows for selected scenario names; require every selected folder."""
    selected_scenarios = set(SCENARIOS.values()) if scenarios is None else set(scenarios)
    if not selected_scenarios:
        raise ValueError("At least one scenario must be selected")
    unknown = selected_scenarios - set(SCENARIOS.values())
    if unknown:
        raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
    windows: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, object]] = []

    for folder_name, scenario in SCENARIOS.items():
        if scenario not in selected_scenarios:
            continue
        paths = sorted((data_root / folder_name).rglob("*.csv"))
        if not paths:
            raise FileNotFoundError(f"No CSV files found in {data_root / folder_name}")

        for path in paths:
            sequences, _, amplitudes = load_csv(path)
            recording = recording_number(path.name)
            for run_start, run_end in continuous_ranges(sequences):
                if run_end - run_start < window_length:
                    continue
                for start in range(
                    run_start, run_end - window_length + 1, step
                ):
                    end = start + window_length
                    windows.append(amplitudes[start:end])
                    labels.append(int(scenario == "person_moving"))
                    metadata.append({
                        "scenario": scenario,
                        "source_file": path.name,
                        "source_relative_path": path.relative_to(data_root).as_posix(),
                        "recording": recording,
                        "start_row": start,
                        "end_row": end - 1,
                    })

    if not windows:
        raise ValueError(f"No continuous windows of {window_length} frames in {data_root}")
    return (
        np.stack(windows).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        metadata,
    )


class SmallMotionCNN(nn.Module):
    """A compact 2-D CNN for [time, CSI-position] amplitude maps."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 3), padding=(2, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(values))


def transform_windows(windows: np.ndarray, input_mode: str) -> np.ndarray:
    """Apply the per-window transform without fitting any dataset statistics."""
    if input_mode == "centered":
        return windows - windows.mean(axis=1, keepdims=True)
    if input_mode == "difference":
        return np.diff(windows, axis=1, prepend=windows[:, :1, :])
    if input_mode == "amplitude":
        return windows
    raise ValueError(f"Unknown input mode: {input_mode}")


def source_manifest(data_root: Path, metadata: list[dict[str, object]]) -> list[dict[str, str]]:
    return [
        {"path": str(relative),
         "sha256": hashlib.sha256((data_root / str(relative)).read_bytes()).hexdigest()}
        for relative in sorted({row["source_relative_path"] for row in metadata})
    ]


def save_motion_checkpoint(
    path: Path, model: SmallMotionCNN, *, feature_mean: np.ndarray,
    feature_std: np.ndarray, input_mode: str, window: int, step: int,
    training: dict[str, object],
) -> None:
    """Store inference weights, training-only scaling and a reproducible input contract."""
    checkpoint = {
        "format_version": 1,
        "model_class": "SmallMotionCNN-v1",
        "model_state_dict": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
        "window": window,
        "step": step,
        "input_shape": [1, window, int(feature_mean.shape[-1])],
        "input_mode": input_mode,
        "labels": {"0": "nonmoving", "1": "person_moving"},
        "scenarios": dict(SCENARIOS),
        "preprocessing": {
            "mean": torch.from_numpy(feature_mean.copy()),
            "std": torch.from_numpy(feature_std.copy()),
            "fitted_on": "training recordings only",
        },
        "training": training,
        "torch_version": str(torch.__version__),
    }
    torch.save(checkpoint, path)


def load_motion_checkpoint(
    path: Path, device: torch.device,
) -> tuple[SmallMotionCNN, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("format_version") != 1
            or checkpoint.get("model_class") != "SmallMotionCNN-v1"):
        raise ValueError("Unsupported motion-CNN checkpoint format")
    if checkpoint.get("labels") != {"0": "nonmoving", "1": "person_moving"}:
        raise ValueError("Checkpoint class labels do not match this task")
    if checkpoint.get("scenarios") != SCENARIOS:
        raise ValueError("Checkpoint scenario mapping does not match this task")
    shape = checkpoint["input_shape"]
    if (len(shape) != 3 or shape[0] != 1 or shape[1] != checkpoint["window"]
            or shape[1] < 4 or shape[2] < 4 or checkpoint["step"] <= 0):
        raise ValueError("Invalid checkpoint window, step or input shape")
    for name in ("mean", "std"):
        values = checkpoint["preprocessing"][name]
        if (not isinstance(values, torch.Tensor)
                or tuple(values.shape) != (1, 1, shape[2])
                or not torch.isfinite(values).all()):
            raise ValueError(f"Invalid checkpoint preprocessing {name}")
    if not (checkpoint["preprocessing"]["std"] > 0).all():
        raise ValueError("Checkpoint standard deviations must be positive")
    if checkpoint["input_mode"] not in ("amplitude", "centered", "difference"):
        raise ValueError("Invalid checkpoint input mode")
    model = SmallMotionCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def preprocess_from_checkpoint(windows: np.ndarray, checkpoint: dict[str, object]) -> np.ndarray:
    if windows.ndim != 3 or list(windows.shape[1:]) != checkpoint["input_shape"][1:]:
        raise ValueError("CSI window shape does not match the saved model")
    values = transform_windows(windows, checkpoint["input_mode"])
    mean = checkpoint["preprocessing"]["mean"].numpy()
    std = checkpoint["preprocessing"]["std"].numpy()
    return ((values - mean) / std)[:, None, :, :].astype(np.float32)


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    total = len(actual)
    moving_count = tp + fn
    nonmoving_count = tn + fp
    moving_recall = tp / moving_count if moving_count else None
    nonmoving_recall = tn / nonmoving_count if nonmoving_count else None
    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "moving_count": moving_count,
        "nonmoving_count": nonmoving_count,
        "accuracy": (tp + tn) / total if total else 0.0,
        "moving_recall": moving_recall,
        "nonmoving_recall": nonmoving_recall,
        "balanced_accuracy": (
            (moving_recall + nonmoving_recall) / 2
            if moving_recall is not None and nonmoving_recall is not None else None
        ),
    }


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for values, _ in loader:
            logits = model(values.to(device))
            batch_probabilities = torch.softmax(logits, dim=1)[:, 1]
            probabilities.append(batch_probabilities.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--test-data-root",
        type=Path,
        help=(
            "Independent acquisition batch with Stop/Person/person_stop folders; "
            "train on all --data-root recordings and test only on this batch."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--step", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--test-recording",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="Held-out recording for the original split; ignored with --test-data-root.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("amplitude", "centered", "difference"),
        default="amplitude",
        help=(
            "amplitude keeps the full amplitude map; centered removes each "
            "window's per-position time mean; difference uses adjacent-frame "
            "amplitude changes"
        ),
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weights in the training loss.",
    )
    args = parser.parse_args()

    if args.window <= 0 or args.step <= 0:
        raise SystemExit("--window and --step must be positive")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.test_data_root is not None:
        try:
            check_separate_batches(args.data_root, args.test_data_root)
        except ValueError as error:
            raise SystemExit(str(error)) from error

    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    all_x, all_y, metadata = build_windows(args.data_root, args.window, args.step)
    if args.test_data_root is not None:
        x_train, y_train = all_x, all_y
        train_metadata = metadata
        x_test, y_test, test_metadata = build_windows(
            args.test_data_root, args.window, args.step
        )
        if x_train.shape[1:] != x_test.shape[1:]:
            raise SystemExit(
                f"Train/test input shapes differ: {x_train.shape[1:]} vs {x_test.shape[1:]}"
            )
        split_description = "all old-batch recordings train; independent new batch tests"
    else:
        train_indexes = np.asarray(
            [index for index, row in enumerate(metadata)
             if row["recording"] != args.test_recording],
            dtype=np.int64,
        )
        test_indexes = np.asarray(
            [index for index, row in enumerate(metadata)
             if row["recording"] == args.test_recording],
            dtype=np.int64,
        )
        if not len(train_indexes) or not len(test_indexes):
            raise SystemExit("The recording-level train/test split is empty")
        x_train, y_train = all_x[train_indexes], all_y[train_indexes]
        x_test, y_test = all_x[test_indexes], all_y[test_indexes]
        train_metadata = [metadata[index] for index in train_indexes]
        test_metadata = [metadata[index] for index in test_indexes]
        split_description = (
            f"recording {args.test_recording} test; the other recordings train"
        )

    raw_x_test = x_test
    x_train = transform_windows(x_train, args.input_mode)
    x_test = transform_windows(x_test, args.input_mode)

    # Fit normalization on training recordings only to avoid test-data leakage.
    feature_mean = x_train.mean(axis=(0, 1), keepdims=True)
    feature_std = x_train.std(axis=(0, 1), keepdims=True)
    feature_std[feature_std < 1e-6] = 1.0
    x_train = ((x_train - feature_mean) / feature_std)[:, None, :, :]
    x_test = ((x_test - feature_mean) / feature_std)[:, None, :, :]

    train_dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train)
    )
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    train_eval_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = SmallMotionCNN().to(device)
    train_counts = np.bincount(y_train, minlength=2)
    if np.any(train_counts == 0):
        raise SystemExit(f"Training split is missing a class: {train_counts.tolist()}")
    class_weights = len(y_train) / (2.0 * train_counts)
    loss_weights = None
    if not args.no_class_weights:
        loss_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_function = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        for values, targets in train_loader:
            values = values.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(values)
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            seen += len(targets)

        epoch_result = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "train_accuracy": correct / seen,
        }
        history.append(epoch_result)
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print(
                f"epoch {epoch:03d}: loss={epoch_result['train_loss']:.4f}, "
                f"train_acc={100 * epoch_result['train_accuracy']:.2f}%"
            )

    elapsed_seconds = time.perf_counter() - started
    train_probabilities, train_predictions = predict(
        model, train_eval_loader, device
    )
    test_probabilities, test_predictions = predict(model, test_loader, device)
    train_metrics = evaluate(y_train, train_predictions)
    test_metrics = evaluate(y_test, test_predictions)

    checkpoint_path = args.output / "motion_cnn.pt"
    save_motion_checkpoint(
        checkpoint_path, model, feature_mean=feature_mean, feature_std=feature_std,
        input_mode=args.input_mode, window=args.window, step=args.step,
        training={
            "data_root": str(args.data_root.resolve()),
            "source_files": source_manifest(args.data_root, train_metadata),
            "split": split_description,
            "seed": args.seed, "epochs": args.epochs,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "class_weighted_loss": not args.no_class_weights,
            "class_weights": class_weights.tolist(),
        },
    )
    reloaded_model, checkpoint = load_motion_checkpoint(checkpoint_path, device)
    reloaded_x_test = preprocess_from_checkpoint(raw_x_test, checkpoint)
    reloaded_loader = DataLoader(
        TensorDataset(torch.from_numpy(reloaded_x_test), torch.from_numpy(y_test)),
        batch_size=args.batch_size, shuffle=False,
    )
    reloaded_probabilities, reloaded_predictions = predict(reloaded_model, reloaded_loader, device)
    reload_verification = {
        "labels_identical": bool(np.array_equal(test_predictions, reloaded_predictions)),
        "probabilities_close": bool(np.allclose(
            test_probabilities, reloaded_probabilities, rtol=1e-6, atol=1e-7
        )),
        "max_probability_difference": float(np.max(np.abs(
            test_probabilities - reloaded_probabilities
        ))),
        "test_windows": len(y_test),
    }
    if not (reload_verification["labels_identical"] and reload_verification["probabilities_close"]):
        raise RuntimeError(f"Checkpoint reload verification failed: {reload_verification}")

    # Diagnostic ablations: these are not additional training runs.  They test
    # whether the trained CNN depends on temporal order/dynamics or can classify
    # recordings from a mostly static CSI fingerprint.
    shuffle_rng = np.random.default_rng(args.seed + 1000)
    shuffled_x_test = x_test.copy()
    for sample in shuffled_x_test:
        sample[0] = sample[0, shuffle_rng.permutation(sample.shape[1]), :]
    shuffled_loader = DataLoader(
        TensorDataset(torch.from_numpy(shuffled_x_test), torch.from_numpy(y_test)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    _, shuffled_predictions = predict(model, shuffled_loader, device)
    shuffled_metrics = evaluate(y_test, shuffled_predictions)

    time_mean_x_test = np.repeat(
        x_test.mean(axis=2, keepdims=True), args.window, axis=2
    )
    time_mean_loader = DataLoader(
        TensorDataset(torch.from_numpy(time_mean_x_test), torch.from_numpy(y_test)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    _, time_mean_predictions = predict(model, time_mean_loader, device)
    time_mean_metrics = evaluate(y_test, time_mean_predictions)

    by_scenario: dict[str, dict[str, float | int]] = {}
    scenario_indexes: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(test_metadata):
        scenario_indexes[str(row["scenario"])].append(index)
    for scenario, indexes in scenario_indexes.items():
        predicted_motion = int(test_predictions[indexes].sum())
        actual_motion = int(y_test[indexes].sum())
        by_scenario[scenario] = {
            "windows": len(indexes),
            "actual_moving": actual_motion,
            "predicted_moving": predicted_motion,
            "predicted_moving_rate": predicted_motion / len(indexes),
        }

    prediction_path = args.output / "test_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "scenario", "source_file", "source_relative_path", "recording", "start_row", "end_row",
            "moving_probability", "actual_motion", "predicted_motion",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, probability, actual, predicted in zip(
            test_metadata, test_probabilities, y_test, test_predictions
        ):
            writer.writerow({
                **row,
                "moving_probability": float(probability),
                "actual_motion": int(actual),
                "predicted_motion": int(predicted),
            })

    report = {
        "task": "person_moving versus nonmoving",
        "model": "small 2-D CNN on CSI amplitude windows",
        "input_mode": args.input_mode,
        "input_shape": [1, args.window, int(all_x.shape[2])],
        "split": split_description,
        "test_recording": args.test_recording if args.test_data_root is None else None,
        "train_data_root": str(args.data_root.resolve()),
        "test_data_root": str((args.test_data_root or args.data_root).resolve()),
        "train_source_files": sorted({row["source_relative_path"] for row in train_metadata}),
        "test_source_files": sorted({row["source_relative_path"] for row in test_metadata}),
        "normalization": "per-CSI-position mean and std fitted on training recordings only",
        "class_weighted_loss": not args.no_class_weights,
        "class_weights": class_weights.tolist(),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": elapsed_seconds,
        "window_counts": {
            "all": len(x_train) + len(x_test),
            "train": len(x_train),
            "test": len(x_test),
        },
        "train_class_counts": train_counts.tolist(),
        "train": train_metrics,
        "test": test_metrics,
        "diagnostics": {
            "time_shuffled_test": shuffled_metrics,
            "time_mean_repeated_test": time_mean_metrics,
        },
        "test_by_scenario": by_scenario,
        "history": history,
        "checkpoint": {"file": checkpoint_path.name,
                       "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                       "reload_verification": reload_verification},
    }
    report_path = args.output / "motion_cnn_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("-" * 60)
    print("Small CNN motion-detection baseline")
    print("Input shape:", report["input_shape"])
    print("Input mode:", args.input_mode)
    print("Split:", split_description)
    print("Training directory:", report["train_data_root"])
    print("Test directory:", report["test_data_root"])
    print("Train/test windows:", len(x_train), "/", len(x_test))
    print("Class-weighted loss:", not args.no_class_weights)
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
    print(
        "Time-shuffled test accuracy:",
        f"{100 * float(shuffled_metrics['accuracy']):.2f}%",
    )
    print(
        "Time-mean-only test accuracy:",
        f"{100 * float(time_mean_metrics['accuracy']):.2f}%",
    )
    for scenario, values in by_scenario.items():
        print(
            f"{scenario}: {values['predicted_moving']}/{values['windows']} "
            "predicted as moving"
        )
    print("Report:", report_path)
    print("Predictions:", prediction_path)
    print("Saved model:", checkpoint_path)
    print("Reload verification:", reload_verification)


if __name__ == "__main__":
    main()
