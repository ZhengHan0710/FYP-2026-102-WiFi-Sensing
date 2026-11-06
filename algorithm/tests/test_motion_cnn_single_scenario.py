"""Single-scenario frozen inference and missing-class metric regression checks."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_cnn_baseline import (
    SCENARIOS,
    SmallMotionCNN,
    build_windows,
    evaluate,
    save_motion_checkpoint,
)


def write_scenario(root, folder, offset):
    target = root / folder / "session"
    target.mkdir(parents=True, exist_ok=True)
    path = target / "synthetic_16_1.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["seq", "rssi", "len", "first_word_invalid", "data"]
        )
        writer.writeheader()
        for frame in range(16):
            values = []
            for position in range(8):
                values.extend([offset + frame % 5 + position, 10 + position])
            writer.writerow({"seq": frame + 1, "rssi": -40, "len": len(values),
                             "first_word_invalid": 0, "data": json.dumps(values)})
    return path


class SingleScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_selection_is_explicit_and_default_still_requires_three_scenarios(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_scenario(root, "Person", 20)
            windows, labels, metadata = build_windows(root, 8, 8, scenarios=["person_moving"])
            self.assertEqual(windows.shape, (2, 8, 8))
            np.testing.assert_array_equal(labels, [1, 1])
            self.assertEqual({row["scenario"] for row in metadata}, {"person_moving"})
            with self.assertRaisesRegex(FileNotFoundError, "Stop"):
                build_windows(root, 8, 8)
            with self.assertRaisesRegex(FileNotFoundError, "person_stop"):
                build_windows(root, 8, 8, scenarios=["person_stationary"])
            with self.assertRaises(ValueError):
                build_windows(root, 8, 8, scenarios=["unknown"])
            write_scenario(root, "Stop", 3)
            write_scenario(root, "person_stop", 30)
            default_windows, default_labels, default_metadata = build_windows(root, 8, 8)
            self.assertEqual(default_windows.shape, (6, 8, 8))
            np.testing.assert_array_equal(default_labels, [0, 0, 1, 1, 0, 0])
            self.assertEqual({row["scenario"] for row in default_metadata}, set(SCENARIOS.values()))
            np.testing.assert_array_equal(default_windows[2:4], windows)

    def test_missing_class_recall_and_balanced_accuracy_are_not_estimated(self):
        for actual, predicted, counts, expected_recalls in (
            ([1, 1, 1], [1, 0, 1], (2, 0, 0, 1), (2 / 3, None)),
            ([0, 0, 0], [0, 1, 0], (0, 2, 1, 0), (None, 2 / 3)),
            ([0, 0, 1, 1], [0, 1, 1, 0], (1, 1, 1, 1), (0.5, 0.5)),
        ):
            with self.subTest(actual=actual):
                metrics = evaluate(np.asarray(actual), np.asarray(predicted))
                self.assertEqual(tuple(metrics[name] for name in ("tp", "tn", "fp", "fn")), counts)
                self.assertEqual(metrics["total"], len(actual))
                self.assertEqual(metrics["moving_count"], actual.count(1))
                self.assertEqual(metrics["nonmoving_count"], actual.count(0))
                self.assertEqual(metrics["accuracy"], sum(a == p for a, p in zip(actual, predicted)) / len(actual))
                self.assertEqual(metrics["moving_recall"], expected_recalls[0])
                self.assertEqual(metrics["nonmoving_recall"], expected_recalls[1])
                if len(set(actual)) == 1:
                    self.assertIsNone(metrics["balanced_accuracy"])
                else:
                    self.assertEqual(metrics["balanced_accuracy"], 0.5)
                self.assertEqual(json.loads(json.dumps(metrics)), metrics)

    def test_cli_motion_only_and_default_keep_checkpoint_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            write_scenario(data, "Person", 50)
            torch.manual_seed(43)
            model = SmallMotionCNN().eval()
            mean = np.arange(8, dtype=np.float32)[None, None, :] / 10
            std = np.linspace(2, 5, 8, dtype=np.float32)[None, None, :]
            checkpoint = root / "random_cnn.pt"
            training = {"source_files": [], "test_fixture": "fixed training statistics"}
            save_motion_checkpoint(
                checkpoint, model, feature_mean=mean, feature_std=std,
                input_mode="centered", window=8, step=8, training=training,
            )
            original_checkpoint = checkpoint.read_bytes()
            script = Path(__file__).resolve().parents[1] / "evaluate_saved_motion_cnn.py"

            def run_cli(output, *extra):
                result = subprocess.run(
                    [sys.executable, str(script), "--checkpoint", str(checkpoint),
                     "--data-root", str(data), "--output", str(output), "--device", "cpu",
                     "--batch-size", "2", *extra],
                    capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return json.loads((output / "frozen_evaluation_report.json").read_text(encoding="utf-8"))

            selected_output = root / "motion_only"
            selected_report = run_cli(selected_output, "--scenarios", "person_moving")
            self.assertEqual(selected_report["selected_scenarios"], ["person_moving"])
            self.assertEqual(selected_report["classes_present"], [1])
            self.assertTrue(selected_report["evaluation_scope"])
            self.assertEqual(selected_report["training"], training)
            self.assertEqual(selected_report["test"]["total"], 2)
            self.assertEqual(selected_report["test"]["moving_count"], 2)
            self.assertEqual(selected_report["test"]["nonmoving_count"], 0)
            self.assertIsNone(selected_report["test"]["nonmoving_recall"])
            self.assertIsNone(selected_report["test"]["balanced_accuracy"])
            self.assertEqual(selected_report["test"]["accuracy"], selected_report["test"]["moving_recall"])
            self.assertEqual(len(selected_report["test_source_files"]), 1)
            windows, labels, _ = build_windows(data, 8, 8, scenarios=["person_moving"])
            expected_values = (
                ((windows - windows.mean(axis=1, keepdims=True)) - mean) / std
            )[:, None].astype(np.float32)
            with torch.no_grad():
                logits = model(torch.from_numpy(expected_values))
            with (selected_output / "test_predictions.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            np.testing.assert_array_equal([int(row["actual_motion"]) for row in rows], labels)
            np.testing.assert_array_equal([int(row["predicted_motion"]) for row in rows],
                                          logits.argmax(dim=1).numpy())
            np.testing.assert_allclose([float(row["moving_probability"]) for row in rows],
                                       logits.softmax(dim=1)[:, 1].numpy(), rtol=1e-6, atol=1e-7)

            write_scenario(data, "Stop", 3)
            write_scenario(data, "person_stop", 30)
            default_report = run_cli(root / "all_scenarios")
            self.assertEqual(default_report["selected_scenarios"], list(SCENARIOS.values()))
            self.assertEqual(default_report["classes_present"], [0, 1])
            self.assertEqual(default_report["test"]["total"], 6)
            self.assertEqual(len(default_report["test_source_files"]), 3)
            self.assertIsInstance(default_report["test"]["balanced_accuracy"], float)
            self.assertEqual(checkpoint.read_bytes(), original_checkpoint)


if __name__ == "__main__":
    unittest.main()
