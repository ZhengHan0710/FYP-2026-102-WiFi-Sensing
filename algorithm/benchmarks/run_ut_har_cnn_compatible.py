#!/usr/bin/env python3
"""Run the SenseFi UT-HAR CNN-5 baseline on a current PyTorch install.

This keeps the convolutional architecture, optimizer, learning rate, batch
sizes, and official validation+test evaluation protocol from SenseFi.  The
data loader uses memory mapping so the 885 MB dataset does not need to be
duplicated in RAM.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset


CLASS_NAMES = [
    "lie_down",
    "fall",
    "walk",
    "pick_up",
    "run",
    "sit_down",
    "stand_up",
]


class UTHARDataset(Dataset):
    def __init__(self, data_path: Path, label_path: Path) -> None:
        self.data = np.load(data_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")
        if self.data.shape[0] != self.labels.shape[0]:
            raise ValueError("The number of samples and labels does not match")
        if self.data.shape[1:] != (250, 90):
            raise ValueError(f"Unexpected UT-HAR sample shape: {self.data.shape}")
        self.minimum = float(np.min(self.data))
        self.maximum = float(np.max(self.data))
        self.scale = self.maximum - self.minimum
        if self.scale <= 0:
            raise ValueError("UT-HAR data has no numeric range")

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = np.array(self.data[index], dtype=np.float32, copy=True)
        sample = (sample - self.minimum) / self.scale
        x = torch.from_numpy(sample).unsqueeze(0)
        y = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return x, y


class SenseFiCNN5(nn.Module):
    """SenseFi's UT_HAR_LeNet (CNN-5) architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=(3, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 4), stride=(2, 2), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, (3, 3), stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(CLASS_NAMES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.classifier(x.flatten(1))


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, object]:
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            predictions = model(inputs.to(device)).argmax(dim=1).cpu().numpy()
            targets = labels.numpy()
            for target, prediction in zip(targets, predictions):
                confusion[int(target), int(prediction)] += 1

    recalls = np.divide(
        np.diag(confusion),
        confusion.sum(axis=1),
        out=np.zeros(len(CLASS_NAMES), dtype=np.float64),
        where=confusion.sum(axis=1) != 0,
    )
    accuracy = float(np.trace(confusion) / confusion.sum())
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(recalls.mean()),
        "class_recall": {
            name: float(recall) for name, recall in zip(CLASS_NAMES, recalls)
        },
        "confusion_matrix": confusion.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("Data/UT_HAR"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/ut_har_cnn"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    train_set = UTHARDataset(
        args.data_root / "data/X_train.csv",
        args.data_root / "label/y_train.csv",
    )
    val_set = UTHARDataset(
        args.data_root / "data/X_val.csv",
        args.data_root / "label/y_val.csv",
    )
    test_set = UTHARDataset(
        args.data_root / "data/X_test.csv",
        args.data_root / "label/y_test.csv",
    )
    official_test_set = ConcatDataset([val_set, test_set])

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=64,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    test_loader = DataLoader(
        official_test_set,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    model = SenseFiCNN5().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    started = time.time()

    print(f"device={device}; train={len(train_set)}; official_test={len(official_test_set)}")
    print("input_shape=[1, 250, 90]; classes=7; model=SenseFi CNN-5")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += batch_size
        print(
            f"epoch {epoch:03d}: train_loss={total_loss / seen:.4f}, "
            f"train_acc={correct / seen:.2%}"
        )

    metrics = evaluate(model, test_loader, device)
    report = {
        "source": "SenseFi UT-HAR CNN-5 compatibility run",
        "data_root": str(args.data_root),
        "seed": args.seed,
        "epochs": args.epochs,
        "device": str(device),
        "train_samples": len(train_set),
        "official_test_samples": len(official_test_set),
        "elapsed_seconds": time.time() - started,
        **metrics,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / f"seed_{args.seed}_epochs_{args.epochs}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"test_acc={metrics['accuracy']:.2%}")
    print(f"test_balanced_acc={metrics['balanced_accuracy']:.2%}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
