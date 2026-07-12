import unittest

from data.generate_report_protocol_seven_heatmap import (
    holdout_improvement_statistics,
    summarize_six_model_holdout_improvement,
)


def _row(series, model, validation_rmse, usable=True):
    return {
        "series": series,
        "model": model,
        "validation_rmse": validation_rmse,
        "usable_for_comparison": usable,
    }


class SixModelHoldoutImprovementTests(unittest.TestCase):
    def test_selects_best_comparable_fear_model_and_excludes_other_rows(self):
        rows = [
            _row("improved", "baseline", 10.0),
            _row("improved", "fear_memory", 8.0),
            _row("improved", "fear_instant", 7.0, usable=False),
            _row("improved", "fear_foraging", 9.0),
            _row("improved", "bda_fear", 1.0),
        ]

        summary = summarize_six_model_holdout_improvement(rows)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["best_fear_model"], "fear_memory")
        self.assertAlmostEqual(summary[0]["absolute_improvement"], 2.0)
        self.assertAlmostEqual(summary[0]["relative_improvement_percent"], 20.0)

    def test_handles_improvement_equality_and_worsening(self):
        rows = [
            _row("improved", "baseline", 10.0),
            _row("improved", "fear_memory", 9.0),
            _row("equal", "baseline", 10.0),
            _row("equal", "fear_instant", 10.0),
            _row("worse", "baseline", 10.0),
            _row("worse", "fear_handling", 12.0),
        ]

        summary = summarize_six_model_holdout_improvement(rows)

        by_series = {row["series"]: row for row in summary}
        self.assertAlmostEqual(by_series["improved"]["relative_improvement_percent"], 10.0)
        self.assertAlmostEqual(by_series["equal"]["relative_improvement_percent"], 0.0)
        self.assertAlmostEqual(by_series["worse"]["relative_improvement_percent"], -20.0)

    def test_statistics_count_thresholds(self):
        rows = [
            {"relative_improvement_percent": -2.0},
            {"relative_improvement_percent": 0.0},
            {"relative_improvement_percent": 0.5},
            {"relative_improvement_percent": 1.0},
            {"relative_improvement_percent": 5.0},
            {"relative_improvement_percent": 10.0},
        ]

        stats = holdout_improvement_statistics(rows)

        self.assertEqual(stats["n_series"], 6)
        self.assertEqual(stats["n_positive"], 4)
        self.assertEqual(stats["n_equal"], 1)
        self.assertEqual(stats["n_positive_below_1_percent"], 1)
        self.assertEqual(stats["n_at_least_1_percent"], 3)
        self.assertEqual(stats["n_at_least_5_percent"], 2)
        self.assertEqual(stats["n_at_least_10_percent"], 1)
        self.assertAlmostEqual(stats["median_all_percent"], 0.75)
        self.assertAlmostEqual(stats["median_positive_percent"], 3.0)


if __name__ == "__main__":
    unittest.main()
