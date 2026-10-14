import math
import unittest

import numpy as np

from scripts.phase2_training_metrics import binary_metrics, macro_metrics


class MacroMetricTests(unittest.TestCase):
    def test_macro_metrics_does_not_equal_pooled_when_instance_sizes_differ(self):
        first_labels = np.array([1, 0, 0, 0], dtype=np.int8)
        first_scores = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float64)
        second_labels = np.array([0, 1], dtype=np.int8)
        second_scores = np.array([0.9, 0.8], dtype=np.float64)
        first = binary_metrics(first_labels, first_scores)
        second = binary_metrics(second_labels, second_scores)
        pooled = binary_metrics(np.concatenate([first_labels, second_labels]),
                                np.concatenate([first_scores, second_scores]))
        result = macro_metrics([first, second], pooled)
        self.assertTrue(math.isclose(result["average_precision"],
                                     (first["average_precision"] + second["average_precision"]) / 2))
        self.assertEqual(result["instances"], 2)
        self.assertFalse(math.isclose(result["average_precision"], pooled["average_precision"]))

    def test_undefined_auroc_instances_are_reported_and_excluded(self):
        positive_only = binary_metrics(np.array([1, 1]), np.array([0.1, 0.9]))
        mixed = binary_metrics(np.array([1, 0]), np.array([0.9, 0.1]))
        pooled = binary_metrics(np.array([1, 1, 1, 0]), np.array([0.1, 0.9, 0.9, 0.1]))
        result = macro_metrics([positive_only, mixed], pooled)
        self.assertEqual(result["ap_instances"], 2)
        self.assertEqual(result["auroc_instances"], 1)
        self.assertEqual(result["auroc_undefined_instances"], 1)
        self.assertEqual(positive_only["average_precision"], 1.0)
        self.assertTrue(math.isnan(positive_only["auroc"]))


if __name__ == "__main__":
    unittest.main()
