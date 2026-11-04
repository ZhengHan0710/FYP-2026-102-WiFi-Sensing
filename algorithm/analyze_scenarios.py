#!/usr/bin/env python3
"""Extract simple, interpretable CSI amplitude features for three scenarios."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SCENARIOS = {
    "Stop": "empty",
    "Person": "person_moving",
    "person_stop": "person_stationary",
}


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences: list[int] = []
    rssis: list[int] = []
    amplitudes: list[np.ndarray] = []

    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            raw = json.loads(row["data"])
            declared_length = int(row["len"])
            if declared_length != len(raw) or declared_length % 2:
                raise ValueError(f"{path.name}第{line_number}行长度无效")
            if int(row["first_word_invalid"]) != 0:
                continue

            values = np.asarray(raw, dtype=np.float32)
            imaginary = values[0::2]
            real = values[1::2]
            amplitudes.append(np.hypot(real, imaginary))
            sequences.append(int(row["seq"]))
            rssis.append(int(row["rssi"]))

    return (
        np.asarray(sequences, dtype=np.int64),
        np.asarray(rssis, dtype=np.float32),
        np.stack(amplitudes).astype(np.float32),
    )


def continuous_ranges(sequences: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) ranges that do not cross a sequence-number gap."""
    if len(sequences) == 0:
        return []
    starts = [0]
    for index in range(1, len(sequences)):
        if sequences[index] != sequences[index - 1] + 1:
            starts.append(index)
    ends = starts[1:] + [len(sequences)]
    return list(zip(starts, ends))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--step", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    frame_counts: dict[str, int] = defaultdict(int)
    all_rssis: dict[str, list[float]] = defaultdict(list)

    for folder_name, label in SCENARIOS.items():
        files = sorted((args.data_root / folder_name).glob("*.csv"))
        if not files:
            raise SystemExit(f"没有找到CSV: {args.data_root / folder_name}")

        for path in files:
            sequences, rssis, amplitudes = load_csv(path)
            frame_counts[label] += len(sequences)
            all_rssis[label].extend(rssis.tolist())

            for run_start, run_end in continuous_ranges(sequences):
                run_length = run_end - run_start
                if run_length < args.window:
                    continue
                for start in range(run_start, run_end - args.window + 1, args.step):
                    end = start + args.window
                    window = amplitudes[start:end]
                    amplitude_mean = float(window.mean())
                    temporal_std = float(window.std(axis=0).mean())
                    temporal_diff = float(np.abs(np.diff(window, axis=0)).mean())
                    relative_diff = temporal_diff / max(amplitude_mean, 1e-6)
                    rows.append({
                        "scenario": label,
                        "source_file": path.name,
                        "start_row": start,
                        "end_row": end - 1,
                        "rssi_median": float(np.median(rssis[start:end])),
                        "amplitude_mean": amplitude_mean,
                        "temporal_std": temporal_std,
                        "temporal_diff": temporal_diff,
                        "relative_temporal_diff": relative_diff,
                    })

    feature_path = args.output / "window_features.csv"
    with feature_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict[str, float | int]] = {}
    for label in SCENARIOS.values():
        selected = [row for row in rows if row["scenario"] == label]
        summary[label] = {
            "frames": frame_counts[label],
            "windows": len(selected),
            "rssi_median": float(np.median(all_rssis[label])),
            "amplitude_mean_median": float(np.median([
                row["amplitude_mean"] for row in selected
            ])),
            "temporal_std_median": float(np.median([
                row["temporal_std"] for row in selected
            ])),
            "temporal_diff_median": float(np.median([
                row["temporal_diff"] for row in selected
            ])),
            "relative_temporal_diff_median": float(np.median([
                row["relative_temporal_diff"] for row in selected
            ])),
        }

    summary_path = args.output / "scenario_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("第一版三场景特征统计")
    print("窗口长度:", args.window, "帧; 步长:", args.step, "帧")
    for label, values in summary.items():
        print("-" * 60)
        print("场景:", label)
        print("帧数:", values["frames"], "窗口数:", values["windows"])
        print("RSSI中位数:", f"{values['rssi_median']:.1f} dBm")
        print("幅度均值中位数:", f"{values['amplitude_mean_median']:.4f}")
        print("时间标准差中位数:", f"{values['temporal_std_median']:.4f}")
        print("相邻帧变化中位数:", f"{values['temporal_diff_median']:.4f}")
        print(
            "归一化变化中位数:",
            f"{values['relative_temporal_diff_median']:.6f}",
        )
    print("输出:", feature_path)
    print("汇总:", summary_path)


if __name__ == "__main__":
    main()
