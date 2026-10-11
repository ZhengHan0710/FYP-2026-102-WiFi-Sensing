"""Record ESP32-S3 CSI using the exact 28-column schema of csi_data.zip."""
import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

HEADER = (
    "type,seq,timestamp,taget_seq,taget,mac,rssi,rate,sig_mode,mcs,cwb,"
    "smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,"
    "ampdu_cnt,channel_primary,channel_secondary,local_timestamp,ant,"
    "sig_len,rx_state,len,first_word_invalid,data"
).split(",")
SHANGHAI = timezone(timedelta(hours=8))
SUBCARRIERS = list(range(-26, 0)) + list(range(1, 27))


def parse_frame(text):
    if not text.startswith("CSI_DATA,"):
        return None
    try:
        row = next(csv.reader([text]))
        if len(row) != len(HEADER):
            return None
        for i in [1] + list(range(6, 27)):
            int(row[i])
        data = json.loads(row[27])
        if not isinstance(data, list) or len(data) != int(row[25]):
            return None
        if not data or len(data) % 2:
            return None
        if any(type(x) is not int or not -128 <= x <= 127 for x in data):
            return None
        if int(row[26]) not in (0, 1):
            return None
        return row, data
    except (ValueError, csv.Error):
        return None


def format_frame(row, data, *, mode, segment, label, timestamp):
    result = list(row)
    if mode == "lltf52":
        # LLTF-only, HT20, no secondary channel: indices 0..31, -32..-1.
        # Reject invalid data rather than replacing it with fabricated values.
        if (len(data) != 128 or int(row[10]) != 0
                or int(row[20]) != 0 or int(row[8]) not in (0, 1)
                or int(row[26]) != 0):
            return None
        indices = [k if k >= 0 else k + 64 for k in SUBCARRIERS]
        data = [data[2 * i + j] for i in indices for j in (0, 1)]
        result[25] = "104"
        result[26] = "0"  # all selected pairs are valid; raw flag retained in serial.log
    result[0] = "CSI_DATA"
    result[2] = timestamp
    result[3] = str(segment)
    result[4] = label
    result[27] = json.dumps(data)
    return result


def check_health(text):
    if "DISCONNECTED" in text or "Experiment stopped" in text:
        raise RuntimeError("热点断开：本轮采集不完整，请重启开发板后重新采集。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="例如 COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--mode", choices=("raw", "lltf52"), default="raw")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--gap", type=float, default=10)
    parser.add_argument("--warmup", type=float, default=15)
    parser.add_argument("--label", default="empty")
    parser.add_argument("--out", default="data")
    args = parser.parse_args()
    if args.seconds <= 0 or args.segments < 1 or args.gap < 0 or args.warmup < 0:
        parser.error("seconds/segments 必须大于零；gap/warmup 不能为负")
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.label):
        parser.error("label 只能包含英文字母、数字、下划线和连字符")

    try:
        import serial
    except ImportError:
        parser.exit(1, "请先运行：python -m pip install -r requirements.txt\n")

    folder = Path(args.out) / datetime.now(SHANGHAI).strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running", "chip": "ESP32-S3", "idf_expected": "5.4.x",
        "schema": HEADER, "settings": vars(args),
        "timestamp_meaning": "PC serial receive time, UTC+08:00; not RF reception wall time",
        "taget_seq_meaning": "segment number assigned by this recorder",
        "taget_meaning": "experiment label supplied by --label",
        "ltf": "LLTF only; no merge; no software filtering; automatic scaling",
        "component_order": "imaginary, real",
        "output_subcarriers": SUBCARRIERS if args.mode == "lltf52" else "driver order, unchanged",
        "original_dataset_subcarrier_mapping": "unknown; equal length does not establish equivalence",
        "invalid_policy": "raw: preserve flag; lltf52: reject frame if first_word_invalid=1",
        "length_meaning": "raw: driver bytes; lltf52: selected output bytes (104)",
        "segments": [],
    }
    meta = folder / "session.json"

    def save_report():
        meta.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    save_report()
    port = serial.Serial(port=None, baudrate=args.baud, timeout=0.2)
    port.dtr = False
    port.rts = False
    port.port = args.port
    last_seq = None
    print(f"输出目录：{folder.resolve()}")

    try:
        with port, (folder / "serial.log").open("w", encoding="utf-8") as log:
            def read():
                nonlocal last_seq
                text = port.readline().decode("utf-8", errors="replace").strip()
                stamp = datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                if text:
                    log.write(text + "\n")
                check_health(text)
                parsed = parse_frame(text)
                if parsed:
                    seq = int(parsed[0][1])
                    if last_seq is not None and seq <= last_seq:
                        raise RuntimeError("CSI 序号回退：可能发生重启，请重新采集。")
                    last_seq = seq
                return text, stamp, parsed

            print("等待有效 CSI，最多 45 秒...")
            deadline = time.monotonic() + 45
            while True:
                text, stamp, parsed = read()
                if parsed:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("未收到 CSI：检查热点、串口、CSI 开关和 Ping 回复。")

            print(f"预热 {args.warmup:g} 秒，请离开实验区域。")
            until = time.monotonic() + args.warmup
            while time.monotonic() < until:
                read()

            for segment in range(1, args.segments + 1):
                start_wall = datetime.now(SHANGHAI)
                file_tag = start_wall.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
                suffix = "104" if args.mode == "lltf52" else "raw"
                name = f"{file_tag}_{suffix}_{segment}.csv"
                stats = {
                    "file": name, "segment": segment, "rows": 0,
                    "rejected_by_mode": 0, "malformed_lines": 0,
                    "seq_gaps_within_segment": 0,
                    "start_time": start_wall.isoformat(),
                }
                report["segments"].append(stats)
                lengths, channels, modes = Counter(), Counter(), Counter()
                previous = None
                began = time.monotonic()
                deadline = began + args.seconds
                progress = began
                print(f"开始第 {segment}/{args.segments} 段：{args.seconds:g} 秒")
                with (folder / name).open("w", newline="", encoding="utf-8") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(HEADER)
                    while time.monotonic() < deadline:
                        text, stamp, parsed = read()
                        if parsed:
                            row, data = parsed
                            seq = int(row[1])
                            if previous is not None:
                                stats["seq_gaps_within_segment"] += max(0, seq - previous - 1)
                            previous = seq
                            output = format_frame(row, data, mode=args.mode,
                                segment=segment, label=args.label, timestamp=stamp)
                            if output is None:
                                stats["rejected_by_mode"] += 1
                            else:
                                writer.writerow(output)
                                stats["rows"] += 1
                                lengths[output[25]] += 1
                                channels[output[19]] += 1
                                modes[output[8]] += 1
                        elif text.startswith("CSI_DATA,"):
                            stats["malformed_lines"] += 1
                        now = time.monotonic()
                        if now - progress >= 10:
                            print(f"  {now-began:.1f} 秒，保存 {stats['rows']} 帧")
                            progress = now
                stats.update(
                    elapsed_seconds=time.monotonic() - began,
                    lengths=dict(lengths), channels=dict(channels), sig_modes=dict(modes),
                )
                stats["frames_per_second"] = stats["rows"] / stats["elapsed_seconds"]
                save_report()
                print(f"已保存 {stats['rows']} 帧，{stats['frames_per_second']:.2f} 帧/秒；长度 {dict(lengths)}")
                if not stats["rows"]:
                    raise RuntimeError("本段没有可保存数据。lltf52 模式可先改 raw 排查长度/带宽/无效标记。")
                if segment != args.segments:
                    print(f"间隔 {args.gap:g} 秒，设备保持不动。")
                    until = time.monotonic() + args.gap
                    while time.monotonic() < until:
                        read()
        report["status"] = "complete"
        print("采集完成。")
    except BaseException as exc:
        report["status"] = "incomplete"
        report["error"] = str(exc) or type(exc).__name__
        raise
    finally:
        save_report()


if __name__ == "__main__":
    main()
