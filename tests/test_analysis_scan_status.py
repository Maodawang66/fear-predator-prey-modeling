import unittest

import numpy as np

from src.analysis import scan_bda_fear_k, scan_delta, scan_phi


class AnalysisScanStatusTests(unittest.TestCase):
    def test_scans_include_convergence_fields(self):
        cases = (
            (scan_phi(np.array([0.0]), t_end=10.0), "phi"),
            (scan_delta(np.array([1.0]), t_end=10.0), "delta"),
            (scan_bda_fear_k(np.array([0.08]), t_end=10.0), "k"),
        )
        for result, parameter_key in cases:
            with self.subTest(scan=parameter_key):
                self.assertIn(parameter_key, result)
                self.assertIn("convergence_status", result)
                self.assertIn("t_end_used", result)
                self.assertIn("extensions", result)
                self.assertEqual(result["convergence_status"].shape, (1,))
                self.assertEqual(result["t_end_used"].shape, (1,))
                self.assertEqual(result["extensions"].shape, (1,))


if __name__ == "__main__":
    unittest.main()
