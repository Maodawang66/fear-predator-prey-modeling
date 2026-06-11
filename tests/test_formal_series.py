import unittest

import numpy as np

from data.formal_series import FORMAL_SERIES_IDS, load_formal_series


class FormalSeriesTests(unittest.TestCase):
    def test_formal_set_contains_exactly_fifteen_series_without_lynxhare(self):
        series = load_formal_series()
        self.assertEqual([item.name for item in series], list(FORMAL_SERIES_IDS))
        self.assertEqual(len(series), 15)
        self.assertNotIn("lynxhare", {item.name for item in series})

    def test_new_series_metadata_and_values(self):
        by_name = {item.name: item for item in load_formal_series()}
        expected = {
            "isle_royale_wolf_moose_pre_2018": (38, 1980, 2017),
            "windermere_north_pike_perch": (23, 1944, 1966),
            "windermere_south_pike_perch": (23, 1944, 1966),
            "komi_lynx_hare": (21, 1922, 1942),
        }
        for name, (n_points, year_start, year_end) in expected.items():
            item = by_name[name]
            self.assertEqual(item.n_points, n_points)
            self.assertEqual(item.meta["year_start"], year_start)
            self.assertEqual(item.meta["year_end"], year_end)
            self.assertTrue(np.all(np.diff(item.t) > 0))
            self.assertTrue(np.all(np.isfinite(item.prey)))
            self.assertTrue(np.all(np.isfinite(item.predator)))

    def test_formal_series_start_with_joint_positive_observations(self):
        by_name = {item.name: item for item in load_formal_series()}
        for item in by_name.values():
            self.assertGreater(item.prey[0], 0.0)
            self.assertGreater(item.predator[0], 0.0)

        glerl = by_name["glerl_m110_zoop_1994-201"]
        self.assertEqual(glerl.n_points, 172)
        self.assertEqual(glerl.meta["leading_observations_dropped"], 3)
        self.assertEqual(glerl.meta["original_n_points"], 175)
        self.assertEqual(glerl.meta["n_points"], 172)
        self.assertEqual(glerl.meta["formal_start_rule"], "first_joint_positive_observation")
        self.assertEqual(glerl.t[0], 0.0)
        self.assertAlmostEqual(glerl.prey[0], 3.753)
        self.assertAlmostEqual(glerl.predator[0], 0.064)


if __name__ == "__main__":
    unittest.main()
