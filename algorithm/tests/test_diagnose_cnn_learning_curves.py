"""Shareable learning-curve CLI checks without private reference artifacts."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_cnn_learning_curves import score


def write_batch(root: Path, offset: int) -> None:
    for folder_index, folder in enumerate(["Stop", "Person", "person_stop"]):
        for recording in [1, 2, 3]:
            directory = root / folder
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"fixture_16_{recording}.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["seq", "rssi", "len", "first_word_invalid", "data"])
                writer.writeheader()
                for frame in range(80):
                    values = []
                    for position in range(8):
                        values.extend([offset + folder_index + recording + frame % 7, 10 + position])
                    writer.writerow({"seq": frame + 1, "rssi": -40, "len": 16,
                                     "first_word_invalid": 0, "data": json.dumps(values)})


class LearningCurveTests(unittest.TestCase):
    def test_displayed_weighted_loss_is_independent_of_batch_partition(self):
        x = np.asarray([[[[1, 2]]], [[[3, -1]]], [[[.5, 1]]], [[[2, 2]]], [[[-1, 3]]]], dtype=np.float32)
        labels = np.asarray([0, 1, 1, 0, 1], dtype=np.int64)
        weights = torch.tensor([.75, 1.5])
        model = torch.nn.Flatten()
        expected = float(torch.nn.functional.cross_entropy(
            torch.from_numpy(x[:, 0, 0]), torch.from_numpy(labels),
            weight=weights, reduction="sum") / weights[torch.from_numpy(labels)].sum())
        for batch_size in [1, 2, 5]:
            result = score(model, x, labels, weights, batch_size)
            self.assertAlmostEqual(result["loss"], expected, places=6)
            self.assertEqual(result["total"], 5)

    def test_cli_runs_without_any_reference_checkpoint_or_report(self):
        script = Path(__file__).resolve().parents[1] / "diagnose_cnn_learning_curves.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new, output = root / "old", root / "new", root / "output"
            write_batch(old, 0)
            write_batch(new, 30)
            protected = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for data in [old, new] for p in data.rglob("*.csv")}
            env = dict(os.environ, MPLCONFIGDIR=str(root / "matplotlib"))
            command = [sys.executable, str(script), "--old-data-root", str(old),
                       "--new-data-root", str(new), "--reference-results-root", str(root / "absent_refs"),
                       "--output", str(output), "--epochs", "1", "--batch-size", "4"]
            checked = subprocess.run(command + ["--check-only"], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertFalse(output.exists())
            completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["original_artifacts_unchanged"])
            for name in ["old", "new"]:
                result = summary["runs"][name]
                self.assertIsNone(result["same_final_weights_as_previous_fixed_run"])
                self.assertEqual((result["train_windows"], result["heldout_windows"]), (12, 6))
                self.assertTrue((output / name / "motion_cnn.pt").is_file())
            for name in ["within_batch_learning_curves.png", "cross_batch_learning_curves.png", "epoch_metrics.csv"]:
                self.assertGreater((output / name).stat().st_size, 0)
            self.assertFalse((output / "original_training_loss.png").exists())
            self.assertEqual(protected, {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in protected})
            repeated = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("Output already exists", repeated.stderr)

    def test_cli_rejects_same_batch_as_cross_batch_input(self):
        script = Path(__file__).resolve().parents[1] / "diagnose_cnn_learning_curves.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_batch(root / "data", 0)
            result = subprocess.run([sys.executable, str(script),
                "--old-data-root", str(root / "data"), "--new-data-root", str(root / "data"),
                "--output", str(root / "output"), "--check-only"],
                capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be different", result.stderr)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
