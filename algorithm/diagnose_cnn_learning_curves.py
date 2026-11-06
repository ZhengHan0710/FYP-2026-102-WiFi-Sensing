#!/usr/bin/env python3
"""Plot original loss and rerun fixed CNN settings with per-epoch diagnostics.

Held-out recordings are monitored for diagnosis, not an untouched final test.
No epoch selection, early stopping, tuning, or overwriting of prior runs occurs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from motion_cnn_baseline import (
    SmallMotionCNN, build_windows, check_separate_batches, evaluate, save_motion_checkpoint, set_seed,
    source_manifest, transform_windows,
)


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "algorithm/results"
REFERENCES = {
    "old": RESULTS / "week10_checkpoint/fold3",
    "new": RESULTS / "week10_cross_batch_2026-11-04/new_batch_internal_diagnostic",
}


def read_batch(root: Path) -> dict:
    x, y, metadata = build_windows(root, 64, 16)
    train = np.asarray([i for i, row in enumerate(metadata) if row["recording"] != 3])
    heldout = np.asarray([i for i, row in enumerate(metadata) if row["recording"] == 3])
    if not len(train) or not len(heldout):
        raise ValueError(f"Empty recording-level split: {root}")
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite inputs: {root}")
    if set(np.unique(y[train])) != {0, 1} or set(np.unique(y[heldout])) != {0, 1}:
        raise ValueError(f"Both classes must be present: {root}")
    return {"root": root, "x": x, "y": y, "metadata": metadata,
            "train": train, "heldout": heldout}


def prepare(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    centered = transform_windows(x, "centered")
    return ((centered - mean) / std)[:, None].astype(np.float32)


def score(model: nn.Module, x: np.ndarray, y: np.ndarray,
          weights: torch.Tensor, batch_size: int) -> dict:
    model.eval()
    predictions = []
    numerator = 0.0
    denominator = 0.0
    with torch.inference_mode():
        for start in range(0, len(y), batch_size):
            targets = torch.from_numpy(y[start:start + batch_size])
            logits = model(torch.from_numpy(x[start:start + batch_size]))
            numerator += float(nn.functional.cross_entropy(
                logits, targets, weight=weights, reduction="sum").item())
            denominator += float(weights[targets].sum().item())
            predictions.append(logits.argmax(1).numpy())
    predicted = np.concatenate(predictions)
    return {"loss": numerator / denominator, **evaluate(y, predicted)}


def run(name: str, batch: dict, external: dict | None, args: argparse.Namespace,
        reference_dir: Path) -> dict:
    set_seed(args.seed)
    train, heldout = batch["train"], batch["heldout"]
    raw_train = batch["x"][train]
    transformed = transform_windows(raw_train, "centered")
    mean = transformed.mean(axis=(0, 1), keepdims=True)
    std = transformed.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = prepare(raw_train, mean, std)
    y_train = batch["y"][train]
    sets = {
        "train": (x_train, y_train),
        "heldout": (prepare(batch["x"][heldout], mean, std), batch["y"][heldout]),
    }
    if external is not None:
        sets["cross_batch"] = (prepare(external["x"], mean, std), external["y"])
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    model = SmallMotionCNN()
    counts = np.bincount(y_train, minlength=2)
    weights = torch.tensor(len(y_train) / (2.0 * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(targets)
            seen += len(targets)
        row = {"epoch": epoch, "optimization_loss": loss_sum / seen}
        # Evaluation uses slicing, not a DataLoader iterator, so it consumes no
        # training RNG state. Dropout is disabled for every displayed loss.
        for split, (x, y) in sets.items():
            row[split] = score(model, x, y, weights, args.batch_size)
        history.append(row)
        message = (f"{name} epoch {epoch:02d}/{args.epochs}: "
                   f"train_loss={row['train']['loss']:.6f}, "
                   f"heldout_loss={row['heldout']['loss']:.6f}, "
                   f"heldout_acc={row['heldout']['accuracy']:.2%}")
        if "cross_batch" in row:
            message += (f", cross_loss={row['cross_batch']['loss']:.6f}, "
                        f"cross_motion_recall={row['cross_batch']['moving_recall']:.2%}")
        print(message, flush=True)
    output = args.output / name
    output.mkdir()
    train_meta = [batch["metadata"][i] for i in train]
    save_motion_checkpoint(output / "motion_cnn.pt", model,
        feature_mean=mean, feature_std=std, input_mode="centered", window=64, step=16,
        training={"data_root": str(batch["root"]),
                  "source_files": source_manifest(batch["root"], train_meta),
                  "split": "recordings 1/2 train; recording 3 monitored for diagnosis",
                  "seed": args.seed, "epochs": args.epochs,
                  "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                  "class_weighted_loss": True, "class_weights": weights.tolist()})
    reference_path = reference_dir / "motion_cnn.pt"
    same_weights = None
    if reference_path.is_file():
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
        same_weights = all(key in reference["model_state_dict"] and
                           torch.equal(value, reference["model_state_dict"][key])
                           for key, value in model.state_dict().items())
    result = {"run": name, "input_shape": [1, 64, 52],
              "train_windows": len(train), "heldout_windows": len(heldout),
              "external_windows": len(external["y"]) if external is not None else 0,
              "class_weights": weights.tolist(), "history": history,
              "same_final_weights_as_previous_fixed_run": same_weights,
              "heldout_min_loss_epoch": min(history, key=lambda row: row["heldout"]["loss"])["epoch"],
              "claim_boundary": "Recording labels; monitored sets are diagnostic, not untouched final tests. Final epoch is fixed; no best-epoch selection."}
    (output / "learning_history.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def plots(runs: dict, output: Path, references: dict[str, Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    original_count = 0
    for name, directory in references.items():
        report_path = directory / "motion_cnn_report.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text())
        history = report["history"]
        ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history],
                label=f"{name.title()} batch: original training loss", linewidth=2)
        original_count += 1
    if original_count:
        ax.set(xlabel="Epoch", ylabel="Training loss", title="Original recorded training loss (no validation logged)", yscale="log")
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
        fig.savefig(output / "original_training_loss.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for col, name in enumerate(["old", "new"]):
        result = runs[name]; h = result["history"]; epochs = [r["epoch"] for r in h]
        for split, color, label in [("train", "#2563eb", "Training (eval mode)"),
                                     ("heldout", "#d97706", "Held-out recording (diagnostic)")]:
            axes[0, col].plot(epochs, [r[split]["loss"] for r in h], color=color, label=label, linewidth=2)
            axes[1, col].plot(epochs, [100*r[split]["accuracy"] for r in h], color=color, label=label, linewidth=2)
        axes[0, col].set(title=f"{name.title()} batch: {result['train_windows']} train / {result['heldout_windows']} held-out",
                         ylabel="Weighted cross-entropy loss (log scale)", yscale="log")
        axes[1, col].set(xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 102))
        for ax in axes[:, col]: ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle("Same CNN and fixed 30-epoch protocol; no early stopping or epoch selection" if len(runs['old']['history']) == 30 else "Fixed-epoch diagnostic runs; no epoch selection")
    fig.tight_layout(); fig.savefig(output / "within_batch_learning_curves.png");
    fig.savefig(output / "within_batch_learning_curves.pdf"); plt.close(fig)
    h = runs["old"]["history"]; epochs = [r["epoch"] for r in h]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for split, label in [("train", "Old training"), ("heldout", "Old held-out"), ("cross_batch", "New batch (diagnostic)")]:
        axes[0].plot(epochs, [r[split]["loss"] for r in h], label=label, linewidth=2)
    axes[0].set(title="Old-trained CNN: loss", xlabel="Epoch", ylabel="Weighted cross-entropy loss (log scale)", yscale="log")
    for split, label in [("heldout", "Old held-out motion recall"), ("cross_batch", "New-batch motion recall")]:
        axes[1].plot(epochs, [100*r[split]["moving_recall"] for r in h], label=label, linewidth=2)
    axes[1].set(title="Old-trained CNN: motion recall", xlabel="Epoch", ylabel="Motion recall (%)", ylim=(0, 102))
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(output / "cross_batch_learning_curves.png");
    fig.savefig(output / "cross_batch_learning_curves.pdf"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-data-root", type=Path, default=REPO / "data")
    parser.add_argument("--new-data-root", type=Path, default=REPO / "data_batches/2026-11-04")
    parser.add_argument("--reference-results-root", type=Path, default=RESULTS,
                        help="Optional prior reports/checkpoints; missing references do not block training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--check-only", action="store_true", help="Check data/config only; do not train or write results")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch size and learning rate must be positive")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wifi-csi-matplotlib"))
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    try:
        check_separate_batches(args.old_data_root, args.new_data_root)
        batches = {"old": read_batch(args.old_data_root),
                   "new": read_batch(args.new_data_root)}
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if batches["old"]["x"].shape[1:] != batches["new"]["x"].shape[1:]:
        parser.error("Old and new batches have different CSI window shapes")
    references = {
        name: args.reference_results_root / directory.relative_to(RESULTS)
        for name, directory in REFERENCES.items()
    }
    for name, batch in batches.items():
        print(f"{name}: train={len(batch['train'])}, heldout={len(batch['heldout'])}, shape={list(batch['x'].shape[1:])}", flush=True)
    if args.check_only:
        print("Checks passed. No training started; no result files written.")
        return
    if args.output.exists():
        parser.error("Output already exists. Choose a new --output directory to preserve previous results.")
    protected = [REPO / "algorithm/motion_cnn_baseline.py"]
    for directory in references.values():
        protected.extend(p for p in [directory / "motion_cnn.pt", directory / "motion_cnn_report.json"]
                         if p.is_file())
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    args.output.mkdir(parents=True)
    runs = {"old": run("old", batches["old"], batches["new"], args, references["old"])}
    runs["new"] = run("new", batches["new"], None, args, references["new"])
    plots(runs, args.output, references)
    with (args.output / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "epoch", "optimization_loss", "train_eval_loss", "heldout_loss", "train_accuracy", "heldout_accuracy", "cross_batch_loss", "cross_batch_motion_recall"])
        for name, result in runs.items():
            for row in result["history"]:
                cross = row.get("cross_batch", {})
                writer.writerow([name, row["epoch"], row["optimization_loss"], row["train"]["loss"], row["heldout"]["loss"], row["train"]["accuracy"], row["heldout"]["accuracy"], cross.get("loss", ""), cross.get("moving_recall", "")])
    unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    summary = {"original_artifacts_unchanged": unchanged,
               "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "loss_definition": "Displayed losses use eval mode (Dropout off) and the same training-class weights, globally normalized by sum of target weights. Optimization loss is logged separately.",
               "scope": "Diagnostic replays of already-used recordings. These curves must not be presented as untouched final-test evidence.",
               "runs": {name: {k:v for k,v in r.items() if k != "history"} for name,r in runs.items()}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not unchanged: raise RuntimeError("An original artifact changed unexpectedly")
    print(f"Done. Figures and per-epoch metrics: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
