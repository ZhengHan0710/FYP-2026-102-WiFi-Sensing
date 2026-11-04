#!/usr/bin/env python3
"""A documented reconstruction of the paper-described bidirectional LSTM baseline."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "datasets" / "wifi_csi_har_dataset"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

CLASS_NAMES = [
    "standing",
    "walking",
    "get_down",
    "sitting",
    "get_up",
    "lying",
    "no_person",
]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

SPLIT_PRESETS = {
    "cross_room": {
        "train": [
            "room_1/1", "room_1/2", "room_1/3",
            "room_3/1", "room_3/2", "room_3/3", "room_3/4",
        ],
        "val": ["room_1/4", "room_3/5"],
        "test": ["room_2/1"],
    },
    "same_room": {
        "train": ["room_1/1", "room_1/2"],
        "val": ["room_1/3"],
        "test": ["room_1/4"],
    },
}


@dataclass(frozen=True)
class Profile:
    split_name: str
    window: int
    step: int
    hidden: int
    batch_size: int
    epochs: int
    row_limit: int | None
    max_train_batches: int | None
    max_eval_batches: int | None
    all_sessions: bool
    min_purity: float
    balanced_sampling: bool
    window_standardize: bool = False
    mean_pool: bool = False
    feature_mode: str = "amplitude_phase"
    model_name: str = "bilstm"
    weighted_loss: bool = False


PROFILES = {
    # Environment/shape test only. Its accuracy is not a research result.
    "smoke": Profile("cross_room", 64, 64, 16, 2, 1, 256, 1, 1, False, 0.0, False),
    # A small end-to-end learning run for understanding the output files.
    "quick": Profile("cross_room", 256, 32, 64, 8, 3, 2200, None, None, True, 0.0, False),
    # Shorter, mostly single-activity windows with balanced training sampling.
    "balanced_quick": Profile("cross_room", 64, 16, 32, 16, 5, None, None, None, True, 0.8, True),
    # Controlled comparison: same model and preprocessing, only the room split changes.
    "same_room": Profile("same_room", 64, 16, 32, 16, 10, None, None, None, True, 0.8, True),
    "cross_room": Profile("cross_room", 64, 16, 32, 16, 10, None, None, None, True, 0.8, True),
    # Reduce room-specific static offsets before the LSTM and aggregate the whole sequence.
    "cross_room_normalized": Profile(
        "cross_room", 64, 16, 32, 16, 10,
        None, None, None, True, 0.8, True, True, True,
    ),
    # Ablation: remove phase and keep the other settings unchanged.
    "cross_room_amplitude": Profile(
        "cross_room", 64, 16, 32, 16, 10,
        None, None, None, True, 0.8, True,
        feature_mode="amplitude",
    ),
    # Model comparison: use the same windows and features with a temporal 1D CNN.
    "cross_room_cnn": Profile(
        "cross_room", 64, 16, 32, 16, 10,
        None, None, None, True, 0.8, True,
        model_name="cnn",
    ),
    # Imbalance comparison: keep the original training distribution and put
    # inverse-frequency weights in the loss instead of resampling windows.
    "cross_room_weighted": Profile(
        "cross_room", 64, 16, 32, 16, 10,
        None, None, None, True, 0.8, False,
        weighted_loss=True,
    ),
    # One optimization step using the paper's full input/model shape.
    "paper_check": Profile("cross_room", 1024, 8, 256, 1, 1, 1050, 1, 1, False, 0.0, False),
    # Main parameters explicitly reported in the paper.
    "paper": Profile("cross_room", 1024, 8, 256, 2, 100, None, None, None, True, 0.0, False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="公开HAR数据集根目录（其中包含room_1、room_2和room_3）",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="训练结果输出目录",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sanitize_phase(phases: np.ndarray) -> np.ndarray:
    """Unwrap and remove the common linear phase slope for each antenna pair."""
    shaped = phases.reshape(-1, 4, 114)
    unwrapped = np.unwrap(shaped, axis=2)
    index = np.arange(114, dtype=np.float32)[None, None, :]
    slope = (unwrapped[:, :, -1:] - unwrapped[:, :, :1]) / 113.0
    centered = unwrapped - slope * index
    centered -= centered.mean(axis=2, keepdims=True)
    return centered.reshape(-1, 456).astype(np.float32, copy=False)


def load_session(
    data_root: Path,
    relative: str,
    row_limit: int | None,
) -> tuple[np.ndarray, list[str]]:
    folder = data_root / relative
    # data.csv columns: 114 subcarrier numbers, 456 amplitudes, 456 phases.
    values = pd.read_csv(
        folder / "data.csv",
        header=None,
        usecols=range(114, 1026),
        nrows=row_limit,
        dtype=np.float32,
    ).to_numpy(copy=True)
    labels = pd.read_csv(
        folder / "label.csv",
        header=None,
        usecols=[1],
        nrows=row_limit,
        dtype=str,
    ).iloc[:, 0].tolist()
    length = min(len(values), len(labels))
    return values[:length], labels[:length]


def load_raw_splits(
    profile: Profile,
    data_root: Path,
) -> dict[str, list[tuple[np.ndarray, list[str]]]]:
    result: dict[str, list[tuple[np.ndarray, list[str]]]] = {}
    for split, sessions in SPLIT_PRESETS[profile.split_name].items():
        # Smoke/quick need only one recording per split; paper uses all recordings.
        chosen = sessions if profile.all_sessions else sessions[:1]
        result[split] = [
            load_session(data_root, session, profile.row_limit)
            for session in chosen
        ]
    return result


def preprocess(
    raw: dict[str, list[tuple[np.ndarray, list[str]]]],
    feature_mode: str,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    train_amplitudes = [values[:, :456] for values, _ in raw["train"]]
    amp_min = min(float(values.min()) for values in train_amplitudes)
    amp_max = max(float(values.max()) for values in train_amplitudes)
    scale = max(amp_max - amp_min, np.finfo(np.float32).eps)

    processed: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for split, sessions in raw.items():
        processed[split] = []
        for values, labels in sessions:
            amplitudes = (values[:, :456] - amp_min) / scale
            if feature_mode == "amplitude":
                features = amplitudes.astype(np.float32, copy=False)
            elif feature_mode == "amplitude_phase":
                phases = sanitize_phase(values[:, 456:])
                features = np.concatenate([amplitudes, phases], axis=1).astype(np.float32)
            else:
                raise ValueError(f"未知特征模式: {feature_mode}")
            try:
                targets = np.asarray([CLASS_TO_INDEX[label] for label in labels], dtype=np.int64)
            except KeyError as exc:
                raise ValueError(f"未知标签: {exc.args[0]}") from exc
            processed[split].append((features, targets))
    return processed


def import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        raise SystemExit(
            "未安装 PyTorch。请先在 wifi_csi_har_repro 目录运行: "
            "./setup.command"
        )
    return torch, nn, DataLoader, Dataset


def choose_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def window_count(length: int, window: int, step: int) -> int:
    return max(0, (length - window) // step + 1)


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.profile]
    set_seed(args.seed)
    torch, nn, DataLoader, Dataset = import_torch()
    torch.manual_seed(args.seed)

    class WindowDataset(Dataset):
        def __init__(
            self,
            sessions,
            window: int,
            step: int,
            min_purity: float,
            window_standardize: bool,
        ):
            self.sessions = sessions
            self.window = window
            self.window_standardize = window_standardize
            self.index = []
            for session_index, (features, targets) in enumerate(sessions):
                for start in range(0, len(features) - window + 1, step):
                    end = start + window
                    window_targets = targets[start:end]
                    counts = np.bincount(window_targets, minlength=len(CLASS_NAMES))
                    label = int(counts.argmax())
                    purity = float(counts[label]) / window
                    if purity >= min_purity:
                        self.index.append((session_index, start, label))

        def __len__(self):
            return len(self.index)

        def __getitem__(self, item):
            session_index, start, label = self.index[item]
            features, _ = self.sessions[session_index]
            end = start + self.window
            window_features = features[start:end]
            if self.window_standardize:
                mean = window_features.mean(axis=0, keepdims=True)
                std = window_features.std(axis=0, keepdims=True)
                window_features = (window_features - mean) / np.maximum(std, 1e-5)
            return torch.from_numpy(window_features.astype(np.float32, copy=False)), label

        def class_counts(self):
            return np.bincount(
                [label for _, _, label in self.index], minlength=len(CLASS_NAMES)
            )

    class PaperBiLSTM(nn.Module):
        def __init__(self, input_features: int, hidden: int, mean_pool: bool):
            super().__init__()
            self.mean_pool = mean_pool
            self.lstm = nn.LSTM(
                input_size=input_features,
                hidden_size=hidden,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden * 2, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, len(CLASS_NAMES)),
            )

        def forward(self, values):
            sequence, _ = self.lstm(values)
            representation = sequence.mean(dim=1) if self.mean_pool else sequence[:, -1, :]
            return self.classifier(representation)

    class TemporalCNN(nn.Module):
        def __init__(self, input_features: int):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(input_features, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, len(CLASS_NAMES)),
            )

        def forward(self, values):
            # Input [batch, time, feature] -> Conv1d [batch, feature, time].
            return self.classifier(self.feature_extractor(values.transpose(1, 2)))

    print(f"正在加载数据，profile={args.profile} ...", flush=True)
    raw = load_raw_splits(profile, args.data_root)
    data = preprocess(raw, profile.feature_mode)
    input_features = 456 if profile.feature_mode == "amplitude" else 912
    datasets = {
        split: WindowDataset(
            sessions,
            profile.window,
            profile.step,
            profile.min_purity,
            profile.window_standardize,
        )
        for split, sessions in data.items()
    }
    for split, dataset in datasets.items():
        if not len(dataset):
            raise SystemExit(f"{split} 没有足够数据生成长度 {profile.window} 的窗口")

    train_sampler = None
    if profile.balanced_sampling:
        train_counts = datasets["train"].class_counts().astype(np.float64)
        sample_weights = [
            1.0 / max(train_counts[label], 1.0)
            for _, _, label in datasets["train"].index
        ]
        train_sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=profile.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
        ),
        "val": DataLoader(datasets["val"], batch_size=profile.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=profile.batch_size, shuffle=False),
    }
    device = choose_device(torch)
    if profile.model_name == "bilstm":
        model = PaperBiLSTM(input_features, profile.hidden, profile.mean_pool).to(device)
    elif profile.model_name == "cnn":
        model = TemporalCNN(input_features).to(device)
    else:
        raise ValueError(f"未知模型: {profile.model_name}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if profile.weighted_loss:
        counts = datasets["train"].class_counts().astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights /= weights.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    else:
        weights = None
        criterion = nn.CrossEntropyLoss()

    run_name = time.strftime(f"{args.profile}_seed{args.seed}_%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "profile": args.profile,
        "profile_parameters": asdict(profile),
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "device": str(device),
        "input_features": input_features,
        "loss_class_weights": weights.tolist() if weights is not None else None,
        "class_names": CLASS_NAMES,
        "split_sessions": SPLIT_PRESETS[profile.split_name],
        "window_counts": {key: len(value) for key, value in datasets.items()},
        "window_class_counts": {
            split: {
                CLASS_NAMES[index]: int(count)
                for index, count in enumerate(dataset.class_counts())
            }
            for split, dataset in datasets.items()
        },
        "note": "smoke/quick/paper_check are engineering checks, not paper reproduction scores",
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"设备: {device}; 窗口数: {config['window_counts']}")
    print(f"窗口类别数: {config['window_class_counts']}")
    print(f"模型: {profile.model_name}; 特征: {profile.feature_mode}")
    print(f"单个输入 shape: [{profile.window}, {input_features}]")

    def evaluate(loader, max_batches=None):
        model.eval()
        total_loss = total = correct = 0
        confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
        with torch.no_grad():
            for batch_index, (features, targets) in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                features, targets = features.to(device), targets.to(device)
                logits = model(features)
                loss = criterion(logits, targets)
                predictions = logits.argmax(dim=1)
                total_loss += float(loss) * len(targets)
                total += len(targets)
                correct += int((predictions == targets).sum())
                for true, pred in zip(targets.cpu().tolist(), predictions.cpu().tolist()):
                    confusion[true, pred] += 1
        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
            "samples": total,
            "confusion": confusion,
        }

    history = []
    best_val = -1.0
    for epoch in range(1, profile.epochs + 1):
        model.train()
        running_loss = seen = 0
        for batch_index, (features, targets) in enumerate(loaders["train"]):
            if profile.max_train_batches is not None and batch_index >= profile.max_train_batches:
                break
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(targets)
            seen += len(targets)

        validation = evaluate(loaders["val"], profile.max_eval_batches)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "val_samples": validation["samples"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}: train_loss={row['train_loss']:.4f}, "
            f"val_loss={row['val_loss']:.4f}, val_acc={row['val_accuracy']:.2%}"
        )
        if validation["accuracy"] > best_val:
            best_val = validation["accuracy"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    test = evaluate(loaders["test"], profile.max_eval_batches)
    row_totals = test["confusion"].sum(axis=1)
    per_class_recall = np.divide(
        np.diag(test["confusion"]),
        row_totals,
        out=np.zeros(len(CLASS_NAMES), dtype=np.float64),
        where=row_totals > 0,
    )
    prediction_counts = test["confusion"].sum(axis=0)
    result = {
        "best_validation_accuracy": best_val,
        "test_accuracy": test["accuracy"],
        "macro_recall": float(per_class_recall.mean()),
        "per_class_recall": {
            name: float(per_class_recall[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "prediction_counts": {
            name: int(prediction_counts[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "test_samples": test["samples"],
        "warning": "Do not cite smoke/quick accuracy as a paper reproduction result.",
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (run_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    np.savetxt(run_dir / "confusion_matrix.csv", test["confusion"], fmt="%d", delimiter=",")
    with (run_dir / "per_class_recall.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "recall"])
        writer.writerows(zip(CLASS_NAMES, per_class_recall.tolist()))
    if args.profile in {"smoke", "paper_check"}:
        print(
            f"流程检查: PASS（诊断准确率 {test['accuracy']:.2%}, "
            f"仅 {test['samples']} windows，不是实验结果）"
        )
    else:
        print(f"测试准确率: {test['accuracy']:.2%} ({test['samples']} windows)")
        print(f"宏平均召回率: {per_class_recall.mean():.2%}")
        print(
            "预测类别数: "
            + str({
                name: int(prediction_counts[index])
                for index, name in enumerate(CLASS_NAMES)
            })
        )
    print(f"输出目录: {run_dir}")


if __name__ == "__main__":
    main()
