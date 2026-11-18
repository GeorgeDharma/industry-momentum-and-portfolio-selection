from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from equity.allocation import block_interval


class AllocationsTests(unittest.TestCase):
    def test_paired_intervals_keep_zero_active_return_at_zero(self):
        np.testing.assert_allclose(block_interval(np.zeros(36), 400, 12, 1), [0, 0])
        np.testing.assert_allclose(
            block_interval(np.full(36, 0.001), 400, 12, 1), [0.012, 0.012]
        )

    def test_sleeve_endpoints_reproduce_standalone_net_returns(self):
        root = Path(__file__).resolve().parents[1] / "outputs"
        if not (root / "allocation_monthly.csv").is_file():
            self.skipTest("Generate the monthly results with run_analysis.py first.")
        race = pd.read_csv(root / "monthly.csv")
        import json

        selected = json.loads(
            (root / "selection_lock.json").read_text(encoding="utf-8")
        )["selected_policy"]
        mixes = pd.read_csv(root / "allocation_monthly.csv")
        for weight, label in [(0, "Equal industries"), (1, selected)]:
            actual = (
                mixes[mixes.sleeve_weight == weight].sort_values("month").net.to_numpy()
            )
            expected = (
                race[race.policy == label].sort_values("month").net_return.to_numpy()
            )
            np.testing.assert_allclose(actual, expected, atol=2e-10)
