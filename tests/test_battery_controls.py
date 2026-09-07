import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "battery"))

from battery_controls import _record, decide


class DecideTests(unittest.TestCase):
    def test_sensitivity_at_or_below_point_fifteen_is_no_fault(self):
        for sensitivity in (-0.01, 0.0, 0.15):
            with self.subTest(sensitivity=sensitivity):
                self.assertEqual(decide(False, sensitivity), "NO-FAULT")
                self.assertEqual(decide(True, sensitivity), "NO-FAULT")

    def test_uncontrolled_sensitivity_above_point_fifteen_is_indeterminate(self):
        for sensitivity in (0.150001, 0.1875, 0.50, 1.0):
            with self.subTest(sensitivity=sensitivity):
                self.assertEqual(decide(False, sensitivity), "INDETERMINATE")

    def test_grid_controlled_sensitivity_above_point_fifteen_is_detect(self):
        for sensitivity in (0.150001, 0.1875, 0.50, 1.0):
            with self.subTest(sensitivity=sensitivity):
                self.assertEqual(decide(True, sensitivity), "DETECT")

    def test_record_averages_variant_populations_without_pooling_cases(self):
        calls = []

        def detector(nominal, changed, target_fpr):
            calls.append((nominal.shape, changed.shape, target_fpr))
            return float(changed[0, 0])

        results = []
        result = _record(
            results, np.zeros((3, 5)), [np.full((3, 5), 0.1), np.full((3, 5), 0.3)],
            1, "operating", "variants", "DETECT", detector=detector,
        )
        self.assertEqual(calls, [((3, 5), (3, 5), 0.05), ((3, 5), (3, 5), 0.05)])
        self.assertAlmostEqual(result["sensitivity"], 0.2)


if __name__ == "__main__":
    unittest.main()
