import math
import unittest

from vreparo_artifact.metrics import format_number, mean, paired_metrics


class MetricsTests(unittest.TestCase):
    def test_cancellation_does_not_hide_deviation(self):
        result = paired_metrics([0, 1, 0, 1], [1, 0, 0, 1])
        self.assertEqual(result["delta"], 0)
        self.assertEqual(result["sd"], 0.25)
        self.assertEqual(result["si"], 0.25)
        self.assertEqual(result["sdv"], 0.5)

    def test_continuous_scores(self):
        result = paired_metrics([0.1, 0.8, 0.5], [0.4, 0.6, 0.5])
        self.assertAlmostEqual(result["delta"], result["sd"] - result["si"])
        self.assertAlmostEqual(result["sdv"], result["sd"] + result["si"])
        self.assertAlmostEqual(result["sdv"], 0.5 / 3)

    def test_signed_points_not_relative_percent(self):
        result = paired_metrics([0, 0, 0, 1], [0, 0, 1, 1])
        self.assertEqual(format_number(100 * result["delta"], 2, signed=True), "+25.00")
        self.assertEqual(format_number(-100 * result["delta"], 2, signed=True), "-25.00")

    def test_zero_formatting(self):
        self.assertEqual(format_number(-0.00001, 2), "0.00")
        self.assertEqual(format_number(0, 2, signed=True), "0.00")

    def test_bad_scores_fail(self):
        for values in ([], [math.nan], [math.inf], [True], ["0.5"], [-0.1], [1.1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                paired_metrics(values, values)
        with self.assertRaises(ValueError):
            paired_metrics([0], [0, 1])

    def test_mean_accepts_generator(self):
        self.assertEqual(mean(value for value in [2, 4]), 3)

    def test_invalid_format(self):
        for digits in (-1, 13, True, 1.5):
            with self.subTest(digits=digits), self.assertRaises(ValueError):
                format_number(0.1, digits)


if __name__ == "__main__":
    unittest.main()
