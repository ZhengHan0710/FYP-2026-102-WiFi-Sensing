"""Batch separation and command-line regression checks with synthetic CSI."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_cnn_baseline import SCENARIOS, check_separate_batches


def write_fixture(root, recording, offset):
    for scene_index, folder in enumerate(SCENARIOS):
        target = root / folder / "session"
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"fixture_104_{recording}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["seq", "rssi", "len", "first_word_invalid", "data"]
            )
            writer.writeheader()
            for frame in range(72):
                values = [offset + scene_index + frame % 7, 10] * 52
                writer.writerow({
                    "seq": frame + 1, "rssi": -40, "len": 104,
                    "first_word_invalid": 0, "data": json.dumps(values),
                })


class BatchTests(unittest.TestCase):
    def test_same_root_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "must be different"):
                check_separate_batches(root, root)

    def test_renamed_nested_copy_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, test = root / "train", root / "test"
            write_fixture(train, 1, 0)
            copied = test / "person_stop" / "session" / "renamed_104_9.csv"
            copied.parent.mkdir(parents=True)
            copied.write_bytes((train / "Stop/session/fixture_104_1.csv").read_bytes())
            with self.assertRaisesRegex(ValueError, "duplicates training data"):
                check_separate_batches(train, test)

    def test_external_and_original_cli_splits(self):
        script = Path(__file__).resolve().parents[1] / "motion_cnn_baseline.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, test = root / "train", root / "test"
            write_fixture(train, 1, 0)
            write_fixture(train, 3, 10)
            write_fixture(test, 8, 20)
            check_separate_batches(train, test)
            for name, arguments, counts in [
                ("external", ["--test-data-root", str(test)], (6, 3)),
                ("original", ["--test-recording", "3"], (3, 3)),
            ]:
                output = root / name
                result = subprocess.run(
                    [sys.executable, str(script), "--data-root", str(train),
                     "--output", str(output), "--epochs", "1",
                     "--input-mode", "centered", *arguments],
                    capture_output=True, text=True, timeout=90,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads((output / "motion_cnn_report.json").read_text())
                self.assertEqual(report["window_counts"]["train"], counts[0])
                self.assertEqual(report["window_counts"]["test"], counts[1])
                self.assertEqual(report["window_counts"]["all"], sum(counts))
                self.assertEqual(report["test"]["total"], counts[1])
                self.assertEqual(len(report["train_source_files"]), counts[0])
                self.assertEqual(len(report["test_source_files"]), counts[1])
                if name == "external":
                    self.assertIsNone(report["test_recording"])
                    self.assertEqual(report["test_data_root"], str(test.resolve()))
                else:
                    self.assertEqual(report["test_recording"], 3)


if __name__ == "__main__":
    unittest.main()
