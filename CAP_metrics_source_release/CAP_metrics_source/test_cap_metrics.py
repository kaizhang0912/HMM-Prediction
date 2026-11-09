"""Run: python -m unittest test_cap_metrics.py"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from cap_metrics import (compute_subject_ocr, compute_dwell_times, compute_transition_matrices,
                         stationary_distribution, trajectory_entropy, export_metrics)


class MetricChecks(unittest.TestCase):
    def test_run_boundaries(self):
        data = {"subject_example": {"run_a": [0, 0], "run_b": [0, 1, 1]}}
        ocr = compute_subject_ocr(data, 3)
        np.testing.assert_allclose(ocr.iloc[0, 1:].to_numpy(dtype=float), [.6, .4, 0])
        dwell = compute_dwell_times(data, 3)["subject_example"]
        self.assertEqual(dwell, {0: [2, 1], 1: [2], 2: []})
        P = compute_transition_matrices(data, 3)["subject_example"]
        np.testing.assert_allclose(P, [[.5, .5, 0], [0, 1, 0], [0, 0, 0]])
        with self.assertRaises(ValueError):
            trajectory_entropy(P)

    def test_stationarity_and_entropy(self):
        P = np.array([[.9, .1], [.2, .8]])
        np.testing.assert_allclose(stationary_distribution(P), [2/3, 1/3])
        np.testing.assert_allclose(trajectory_entropy(np.full((2, 2), .5)), np.full((2, 2), 2.))
        np.testing.assert_allclose(trajectory_entropy(np.array([[0., 1.], [1., 0.]])), 0.)
        with self.assertRaises(ValueError):
            trajectory_entropy(np.eye(2))

    def test_exports_and_missing_states(self):
        data = {"subject_example": {"run": [0, 0, 1, 1, 0]}, "empty_example": {"run": []}}
        with tempfile.TemporaryDirectory() as folder:
            export_metrics(data, folder, 2)
            frame = pd.read_csv(Path(folder) / "OCR.csv")
            self.assertEqual(frame.Subject.tolist(), list(data))
            self.assertTrue(frame.iloc[1, 1:].isna().all())
            with np.load(Path(folder) / "metrics.npz") as output:
                self.assertEqual(output["EMT"].shape, (2, 2, 2))
                self.assertTrue(np.isnan(output["EMT"][1]).all())
                np.testing.assert_allclose(output["EMT_mean"][0], output["EMT"][0].mean())


if __name__ == "__main__":
    unittest.main()
