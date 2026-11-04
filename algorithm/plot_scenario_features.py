#!/usr/bin/env python3
"""Plot interpretable feature distributions produced by analyze_scenarios.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


SCENARIOS = ["empty", "person_moving", "person_stationary"]
DISPLAY_NAMES = ["Empty", "Person moving", "Person stationary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path, help="window_features.csv path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    values = {
        label: {"temporal_std": [], "relative_temporal_diff": []}
        for label in SCENARIOS
    }
    with args.features.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = row["scenario"]
            if label not in values:
                continue
            values[label]["temporal_std"].append(float(row["temporal_std"]))
            values[label]["relative_temporal_diff"].append(
                float(row["relative_temporal_diff"])
            )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plots = [
        ("temporal_std", "Temporal standard deviation"),
        ("relative_temporal_diff", "Normalized frame-to-frame change"),
    ]
    for axis, (feature, title) in zip(axes, plots):
        data = [values[label][feature] for label in SCENARIOS]
        axis.boxplot(data, tick_labels=DISPLAY_NAMES, showfliers=False)
        axis.set_title(title)
        axis.set_ylabel("Feature value")
        axis.grid(axis="y", color="#dddddd", linewidth=0.8)
        axis.tick_params(axis="x", rotation=12)

    figure.suptitle("CSI temporal feature comparison across three scenarios")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print("Feature comparison figure:", args.output)


if __name__ == "__main__":
    main()
