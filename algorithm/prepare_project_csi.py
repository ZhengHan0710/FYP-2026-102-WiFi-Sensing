#!/usr/bin/env python3
"""Validate the team's 28-column ESP32-S3 CSV and build model-ready windows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = {
    "type", "seq", "timestamp", "taget_seq", "taget", "mac", "rssi",
    "channel_primary", "local_timestamp", "len", "first_word_invalid", "data",
}


def subcarrier_name(index: int) -> str:
    return f"m{abs(index)}" if index < 0 else f"p{index}"


def load_subcarriers(folder: Path, complex_count: int) -> list[int]:
    session_file = folder / "session.json"
    if session_file.exists():
        session = json.loads(session_file.read_text(encoding="utf-8"))
        values = session.get("output_subcarriers")
        if isinstance(values, list) and len(values) == complex_count:
            return [int(value) for value in values]
    return list(range(complex_count))


def parse_file(path: Path) -> dict:
    metadata: list[dict] = []
    amplitudes: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    rejected = 0

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"没有表头: {path}")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"缺少字段 {sorted(missing)}: {path}")

        previous_seq: int | None = None
        previous_device_time: int | None = None
        expected_length: int | None = None
        for line_number, row in enumerate(reader, start=2):
            try:
                if row["type"] not in {"CSI_DATA", "CSI"}:
                    rejected += 1
                    continue
                seq = int(row["seq"])
                device_time = int(row["local_timestamp"])
                declared_length = int(row["len"])
                invalid = int(row["first_word_invalid"])
                raw = json.loads(row["data"])
                if not isinstance(raw, list) or len(raw) != declared_length:
                    raise ValueError("data数量与len不一致")
                if declared_length <= 0 or declared_length % 2:
                    raise ValueError("len必须为正偶数")
                if any(type(value) is not int or not -128 <= value <= 127 for value in raw):
                    raise ValueError("I/Q必须是-128到127的整数")
                if invalid != 0:
                    rejected += 1
                    continue
                if expected_length is None:
                    expected_length = declared_length
                elif declared_length != expected_length:
                    raise ValueError(
                        f"同一文件出现不同CSI长度: {expected_length}和{declared_length}"
                    )
                if previous_seq is not None and seq <= previous_seq:
                    raise ValueError("seq没有严格递增")
                if previous_device_time is not None and device_time <= previous_device_time:
                    raise ValueError("local_timestamp没有严格递增")

                values = np.asarray(raw, dtype=np.float32)
                imaginary = values[0::2]
                real = values[1::2]
                amplitudes.append(np.hypot(real, imaginary))
                phases.append(np.arctan2(imaginary, real))
                metadata.append({
                    "source_file": path.name,
                    "seq": seq,
                    "timestamp": row["timestamp"],
                    "segment": row["taget_seq"],
                    "label": row["taget"],
                    "mac": row["mac"],
                    "rssi": int(row["rssi"]),
                    "channel": int(row["channel_primary"]),
                    "local_timestamp": device_time,
                })
                previous_seq = seq
                previous_device_time = device_time
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path.name}第{line_number}行无效: {exc}") from exc

    if not metadata:
        raise ValueError(f"没有可用CSI帧: {path}")
    amplitude_array = np.stack(amplitudes).astype(np.float32)
    phase_array = np.stack(phases).astype(np.float32)
    return {
        "metadata": metadata,
        "amplitude": amplitude_array,
        "phase": phase_array,
        "features": np.concatenate([amplitude_array, phase_array], axis=1),
        "rejected": rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="包含一个或多个CSI CSV的会话目录")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--step", type=int, default=16)
    parser.add_argument("--min-purity", type=float, default=0.8)
    args = parser.parse_args()
    if args.window <= 0 or args.step <= 0:
        parser.error("window和step必须大于0")
    if not 0 < args.min_purity <= 1:
        parser.error("min-purity必须在0到1之间")

    files = sorted(args.input.glob("*.csv"))
    if not files:
        raise SystemExit(f"目录中没有CSV: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    parsed = [parse_file(path) for path in files]
    complex_count = parsed[0]["amplitude"].shape[1]
    if any(item["amplitude"].shape[1] != complex_count for item in parsed):
        raise SystemExit("不同文件的复数CSI位置数量不一致，不能直接合并")
    subcarriers = load_subcarriers(args.input, complex_count)

    all_metadata = [row for item in parsed for row in item["metadata"]]
    all_features = np.concatenate([item["features"] for item in parsed], axis=0)
    all_amplitudes = np.concatenate([item["amplitude"] for item in parsed], axis=0)
    all_phases = np.concatenate([item["phase"] for item in parsed], axis=0)

    feature_header = (
        [f"amplitude_sc_{subcarrier_name(value)}" for value in subcarriers]
        + [f"phase_sc_{subcarrier_name(value)}" for value in subcarriers]
    )
    metadata_header = [
        "source_file", "seq", "timestamp", "segment", "label", "mac",
        "rssi", "channel", "local_timestamp",
    ]
    with (args.output / "frames_features.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(metadata_header + feature_header)
        for row, features in zip(all_metadata, all_features):
            writer.writerow([row[name] for name in metadata_header] + features.tolist())

    windows: list[np.ndarray] = []
    window_labels: list[str] = []
    window_sources: list[str] = []
    window_starts: list[int] = []
    file_reports = []
    offset = 0
    for path, item in zip(files, parsed):
        labels = [row["label"] for row in item["metadata"]]
        count_before = len(windows)
        for start in range(0, len(labels) - args.window + 1, args.step):
            label_counts = Counter(labels[start:start + args.window])
            label, count = label_counts.most_common(1)[0]
            if count / args.window < args.min_purity:
                continue
            windows.append(item["features"][start:start + args.window])
            window_labels.append(label)
            window_sources.append(path.name)
            window_starts.append(start)

        device_times = np.asarray(
            [row["local_timestamp"] for row in item["metadata"]], dtype=np.float64
        )
        intervals = np.diff(device_times) / 1_000_000.0
        file_reports.append({
            "file": path.name,
            "frames": len(labels),
            "labels": dict(Counter(labels)),
            "rejected_frames": item["rejected"],
            "complex_positions": complex_count,
            "feature_count": item["features"].shape[1],
            "median_interval_seconds": float(np.median(intervals)),
            "observed_fps_from_median": float(1.0 / np.median(intervals)),
            "windows": len(windows) - count_before,
            "global_frame_offset": offset,
        })
        offset += len(labels)

    if not windows:
        raise SystemExit("当前设置无法生成任何时间窗口")
    window_array = np.stack(windows).astype(np.float32)
    np.savez_compressed(
        args.output / "model_windows.npz",
        X=window_array,
        y=np.asarray(window_labels),
        source_file=np.asarray(window_sources),
        start_index=np.asarray(window_starts, dtype=np.int64),
        feature_names=np.asarray(feature_header),
    )
    np.save(args.output / "amplitudes.npy", all_amplitudes)
    np.save(args.output / "phases_raw.npy", all_phases)

    unique_labels = sorted(set(window_labels))
    report = {
        "input_directory": str(args.input),
        "source_files": file_reports,
        "component_order": "imaginary, real",
        "frames": len(all_metadata),
        "complex_positions_per_frame": complex_count,
        "features_per_frame": all_features.shape[1],
        "feature_order": "all amplitudes, then all raw phases",
        "window": args.window,
        "step": args.step,
        "window_shape": list(window_array.shape),
        "window_labels": dict(Counter(window_labels)),
        "can_train_classifier": len(unique_labels) >= 2,
        "limitation": (
            "Only one label is present; these windows validate the data pipeline "
            "but cannot train a classifier."
            if len(unique_labels) < 2 else None
        ),
    }
    (args.output / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"读取CSV: {len(files)}个")
    print(f"有效CSI帧: {len(all_metadata)}")
    print(f"每帧: {complex_count}个复数位置 -> {all_features.shape[1]}个幅相特征")
    print(f"模型窗口shape: {list(window_array.shape)}")
    print(f"窗口标签: {dict(Counter(window_labels))}")
    if len(unique_labels) < 2:
        print("分类训练: 暂不可用（只有一个标签）")
    print(f"输出目录: {args.output.resolve()}")


if __name__ == "__main__":
    main()
