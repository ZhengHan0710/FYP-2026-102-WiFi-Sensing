"""Frozen CNN inference, preprocessing and data-leakage regression checks."""

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
    load_motion_checkpoint,
    preprocess_from_checkpoint,
    save_motion_checkpoint,
    source_manifest,
    transform_windows,
)


def write_fixture(root, recording, offset):
    """Write two short windows per scenario using the real CSI CSV format."""
    for scene_index, folder in enumerate(SCENARIOS):
        target = root / folder / "session"
        target.mkdir(parents=True, exist_ok=True)
        with (target / f"fixture_104_{recording}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["seq", "rssi", "len", "first_word_invalid", "data"]
            )
            writer.writeheader()
            for frame in range(16):
                values = [offset + scene_index + frame % 7, 10] * 52
                writer.writerow({
                    "seq": frame + 1, "rssi": -40, "len": 104,
                    "first_word_invalid": 0, "data": json.dumps(values),
                })


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.device = torch.device("cpu")
        torch.manual_seed(17)
        self.model = SmallMotionCNN()
        rng = np.random.default_rng(29)
        self.training_windows = rng.normal(3, 2, (5, 8, 6)).astype(np.float32)
        self.evaluation_windows = (
            rng.normal(40, 7, (3, 8, 6)).astype(np.float32)
            + np.arange(8, dtype=np.float32)[None, :, None] * 3
        )

    def save_checkpoint(self, mode="amplitude", windows=None, training=None):
        if windows is None:
            windows = self.training_windows
        transformed = transform_windows(windows, mode)
        mean = transformed.mean(axis=(0, 1), keepdims=True)
        std = transformed.std(axis=(0, 1), keepdims=True)
        std[std < 1e-6] = 1.0
        path = self.root / f"{mode}.pt"
        save_motion_checkpoint(
            path, self.model, feature_mean=mean, feature_std=std,
            input_mode=mode, window=windows.shape[1], step=8,
            training=training if training is not None else {"source_files": []},
        )
        return path, mean, std

    def test_three_input_transforms_have_expected_values(self):
        windows = np.asarray([[[1, 9], [3, 5], [8, 1]]], dtype=np.float32)
        original = windows.copy()
        np.testing.assert_array_equal(transform_windows(windows, "amplitude"), original)
        np.testing.assert_allclose(
            transform_windows(windows, "centered"),
            np.asarray([[[-3, 4], [-1, 0], [4, -4]]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            transform_windows(windows, "difference"),
            np.asarray([[[0, 0], [2, -4], [5, -4]]], dtype=np.float32),
        )
        np.testing.assert_array_equal(windows, original)

    def test_round_trip_preserves_probabilities_labels_weights_and_eval_mode(self):
        for mode in ("amplitude", "centered", "difference"):
            with self.subTest(input_mode=mode):
                path, mean, std = self.save_checkpoint(mode)
                values = (
                    (transform_windows(self.evaluation_windows, mode) - mean) / std
                )[:, None].astype(np.float32)
                self.model.eval()
                with torch.no_grad():
                    original_logits = self.model(torch.from_numpy(values))
                    original_probabilities = original_logits.softmax(dim=1)
                    original_labels = original_logits.argmax(dim=1)

                # A saved training-mode model must still load in inference mode.
                self.model.train()
                self.save_checkpoint(mode)
                loaded_model, checkpoint = load_motion_checkpoint(path, self.device)
                self.assertFalse(loaded_model.training)
                self.assertTrue(all(not module.training for module in loaded_model.modules()))
                for name, tensor in self.model.state_dict().items():
                    torch.testing.assert_close(loaded_model.state_dict()[name], tensor, rtol=0, atol=0)
                restored_values = preprocess_from_checkpoint(self.evaluation_windows, checkpoint)
                np.testing.assert_array_equal(restored_values, values)
                with torch.no_grad():
                    loaded_logits = loaded_model(torch.from_numpy(restored_values))
                    repeated_logits = loaded_model(torch.from_numpy(restored_values))
                torch.testing.assert_close(loaded_logits.softmax(dim=1), original_probabilities,
                                           rtol=0, atol=0)
                torch.testing.assert_close(loaded_logits.argmax(dim=1), original_labels,
                                           rtol=0, atol=0)
                torch.testing.assert_close(loaded_logits, repeated_logits, rtol=0, atol=0)

    def test_preprocessing_reuses_training_statistics_for_every_mode(self):
        for mode in ("amplitude", "centered", "difference"):
            with self.subTest(input_mode=mode):
                path, mean, std = self.save_checkpoint(mode)
                _, checkpoint = load_motion_checkpoint(path, self.device)
                original_windows = self.evaluation_windows.copy()
                values = preprocess_from_checkpoint(self.evaluation_windows, checkpoint)
                transformed = transform_windows(self.evaluation_windows, mode)
                expected = ((transformed - mean) / std)[:, None].astype(np.float32)
                np.testing.assert_array_equal(values, expected)
                self.assertEqual(values.dtype, np.float32)
                self.assertEqual(values.shape, (3, 1, 8, 6))
                refitted = (
                    (transformed - transformed.mean(axis=(0, 1), keepdims=True))
                    / transformed.std(axis=(0, 1), keepdims=True)
                )[:, None]
                self.assertFalse(np.allclose(values, refitted))
                # The first example must not depend on other evaluation examples.
                alone = preprocess_from_checkpoint(self.evaluation_windows[:1], checkpoint)
                np.testing.assert_array_equal(alone[0], values[0])
                np.testing.assert_array_equal(checkpoint["preprocessing"]["mean"].numpy(), mean)
                np.testing.assert_array_equal(checkpoint["preprocessing"]["std"].numpy(), std)
                np.testing.assert_array_equal(self.evaluation_windows, original_windows)

    def test_preprocessing_rejects_mismatched_window_shapes(self):
        path, _, _ = self.save_checkpoint()
        _, checkpoint = load_motion_checkpoint(path, self.device)
        for shape in ((8, 6), (3, 7, 6), (3, 8, 5), (3, 1, 8, 6)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "window shape"):
                    preprocess_from_checkpoint(np.zeros(shape, dtype=np.float32), checkpoint)

    def test_loading_rejects_invalid_input_contract(self):
        path, _, _ = self.save_checkpoint()
        original = torch.load(path, weights_only=True)
        for shape in ([1, 8], [2, 8, 6], [1, 9, 6], [1, 8, 3]):
            with self.subTest(input_shape=shape):
                broken = dict(original, input_shape=shape)
                damaged_path = self.root / "bad_shape.pt"
                torch.save(broken, damaged_path)
                with self.assertRaisesRegex(ValueError, "input shape"):
                    load_motion_checkpoint(damaged_path, self.device)

    def test_loading_rejects_nonpositive_nonfinite_and_wrong_shape_std(self):
        path, _, _ = self.save_checkpoint()
        original = torch.load(path, weights_only=True)
        for name, std in (
            ("zero", torch.zeros(1, 1, 6)),
            ("negative", -torch.ones(1, 1, 6)),
            ("nan", torch.full((1, 1, 6), float("nan"))),
            ("infinity", torch.full((1, 1, 6), float("inf"))),
            ("wrong_shape", torch.ones(6)),
            ("wrong_type", [1.0] * 6),
        ):
            with self.subTest(std=name):
                broken = dict(original)
                broken["preprocessing"] = dict(original["preprocessing"], std=std)
                damaged_path = self.root / "bad_std.pt"
                torch.save(broken, damaged_path)
                with self.assertRaisesRegex(ValueError, "standard deviations|preprocessing std"):
                    load_motion_checkpoint(damaged_path, self.device)

    def test_cli_evaluates_frozen_model_and_rejects_renamed_training_csv(self):
        train, evaluation = self.root / "train", self.root / "evaluation"
        write_fixture(train, 1, 0)
        write_fixture(evaluation, 9, 20)
        train_windows, _, train_metadata = build_windows(train, 8, 8)
        training = {"source_files": source_manifest(train, train_metadata), "seed": 17}
        path, _, _ = self.save_checkpoint("centered", train_windows, training)
        script = Path(__file__).resolve().parents[1] / "evaluate_saved_motion_cnn.py"
        output = self.root / "valid_output"
        command = [sys.executable, str(script), "--checkpoint", str(path),
                   "--data-root", str(evaluation), "--output", str(output),
                   "--batch-size", "2", "--device", "cpu"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((output / "frozen_evaluation_report.json").read_text())
        self.assertEqual(report["training"], training)
        self.assertEqual(report["test"]["total"], 6)
        self.assertEqual(len(report["test_source_files"]), 3)
        with (output / "test_predictions.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        windows, labels, _ = build_windows(evaluation, 8, 8)
        model, checkpoint = load_motion_checkpoint(path, self.device)
        with torch.no_grad():
            logits = model(torch.from_numpy(preprocess_from_checkpoint(windows, checkpoint)))
        np.testing.assert_allclose(
            [float(row["moving_probability"]) for row in rows],
            logits.softmax(dim=1)[:, 1].numpy(), rtol=1e-6, atol=1e-7,
        )
        np.testing.assert_array_equal([int(row["actual_motion"]) for row in rows], labels)
        np.testing.assert_array_equal([int(row["predicted_motion"]) for row in rows],
                                      logits.argmax(dim=1).numpy())

        copied = evaluation / "person_stop" / "different_session" / "renamed_104_9.csv"
        copied.parent.mkdir(parents=True)
        copied.write_bytes((train / "Stop/session/fixture_104_1.csv").read_bytes())
        rejected_output = self.root / "rejected_output"
        command[command.index(str(output))] = str(rejected_output)
        rejected = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("duplicates checkpoint training data", rejected.stdout + rejected.stderr)
        self.assertIn("renamed_104_9.csv", rejected.stdout + rejected.stderr)
        self.assertFalse(rejected_output.exists())


if __name__ == "__main__":
    unittest.main()
