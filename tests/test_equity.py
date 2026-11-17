import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from equity.engine import *


class DataAndSignalTests(unittest.TestCase):
    def test_first_block_units_and_missing_values(self):
        text = "Source\n Value weighted\n,A,B\n200001,1.0,-99.99\n200002,2.0,3.0\n\n Equal weighted\n,A,B\n200001,90,90\n"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.csv"
            p.write_text(text)
            r = read_monthly(p)
        self.assertEqual(len(r), 2)
        self.assertEqual(r.iloc[0, 0], 0.01)
        self.assertTrue(np.isnan(r.iloc[0, 1]))

    def test_missing_month_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.csv"
            p.write_text(",A\n200001,1\n200003,2\n")
            with self.assertRaises(ValueError):
                read_monthly(p)

    def test_signal_months_are_exact(self):
        ix = pd.period_range("2000-01", periods=20, freq="M")
        r = pd.DataFrame({"A": np.arange(20) / 100}, index=ix)
        s = signals(r)
        self.assertAlmostEqual(s.iloc[12, 0], np.prod(1 + r.iloc[:11, 0]) - 1)
        self.assertTrue(np.isnan(s.iloc[11, 0]))

    def test_skipped_month_does_not_change_rank_score(self):
        ix = pd.period_range("2000-01", periods=20, freq="M")
        r = pd.DataFrame({"A": np.arange(20) / 100}, index=ix)
        a = signals(r).iloc[12, 0]
        r.iloc[11, 0] = 5
        self.assertEqual(signals(r).iloc[12, 0], a)

    def test_missing_history_is_not_zero_filled(self):
        ix = pd.period_range("2000-01", periods=20, freq="M")
        r = pd.DataFrame({"A": 0.01}, index=ix)
        r.iloc[4, 0] = np.nan
        self.assertTrue(np.isnan(signals(r).iloc[12, 0]))

    def test_deterministic_tie_order(self):
        s = pd.Series([1.0, 1.0, 0.5], index=["B", "A", "C"])
        self.assertEqual(ranked_codes(s), ["A", "B", "C"])

    def test_future_data_cannot_change_past_signal(self):
        ix = pd.period_range("2000-01", periods=30, freq="M")
        r = pd.DataFrame(np.random.default_rng(2).normal(0, 0.05, (30, 5)), index=ix)
        a = signals(r)
        r.iloc[20:] = 0.9
        b = signals(r)
        pd.testing.assert_frame_equal(a.iloc[:21], b.iloc[:21])


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.s = pd.Series(np.arange(20, 0, -1), index=[f"I{i:02}" for i in range(20)])

    def test_constraints(self):
        for policy in ["Equal industries", "Top 10 momentum", "Buffered momentum"]:
            w = target_weights(self.s, policy)
            self.assertAlmostEqual(w.sum(), 1)
            self.assertTrue((w >= 0).all())
            self.assertLessEqual(w.max(), 0.1 + 1e-12)

    def test_buffer_retains_and_fills(self):
        w = target_weights(self.s, "Buffered momentum", previous=["I10", "I14", "I18"])
        self.assertEqual(w["I10"], 0.1)
        self.assertEqual(w["I14"], 0.1)
        self.assertEqual(w["I18"], 0)
        self.assertEqual((w > 0).sum(), 10)

    def test_buffer_boundary_equals_plain(self):
        a = target_weights(
            self.s, "Buffered momentum", previous=["I10", "I14"], buffer_rank=10
        )
        b = target_weights(self.s, "Top 10 momentum")
        pd.testing.assert_series_equal(a, b)

    def test_entry_cost_is_funded(self):
        t = rebalance([0.5, 0.5], [0, 0], 10)
        self.assertAlmostEqual(t["cost"], 0.001 / 1.001)
        self.assertAlmostEqual(t["buys"] + t["cost"], 1)
        self.assertEqual(t["sales"], 0)

    def test_rotation_and_cash_identity(self):
        t = rebalance([1, 0], [0, 1], 10)
        self.assertAlmostEqual(t["cost"], 0.002 / 1.001)
        self.assertAlmostEqual(t["buys"] - t["sales"] + t["cost"], 0)
        self.assertAlmostEqual(t["cost"], 0.001 * t["gross_traded"])

    def test_no_trade_and_zero_cost(self):
        t = rebalance([0.4, 0.6], [0.4, 0.6], 10)
        self.assertEqual(t["cost"], 0)
        t = rebalance([1, 0], [0, 1], 0)
        self.assertEqual(t["cost"], 0)
        self.assertEqual(t["turnover"], 1)

    def test_rebalance_validation(self):
        for args in [
            ([0.4, 0.4], [0, 0], 10),
            ([0.5, 0.5], [0.2, 0.2], 10),
            ([0.5, 0.5], [0, 0], -1),
        ]:
            with self.assertRaises(ValueError):
                rebalance(*args)

    def test_monthly_accounting_and_drift(self):
        ix = pd.period_range("2000-01", periods=3, freq="M")
        r = pd.DataFrame({"A": [0.1, 0, 0], "B": [0, 0, 0]}, index=ix)
        s = pd.DataFrame({"A": 2.0, "B": 1.0}, index=ix)
        b, h = backtest(r, s, "2000-01", "2000-03", "Equal industries", 10, count=1)
        self.assertAlmostEqual(b.net_return.iloc[0], 1.05 / 1.001 - 1)
        second = h[h.month == "2000-02"].set_index("industry")
        self.assertAlmostEqual(second.loc["A", "pretrade_weight"], 0.55 / 1.05)
        np.testing.assert_allclose(
            b.net_return, (1 - b.cost_fraction) * (1 + b.gross_return) - 1
        )

    def test_held_missing_return_is_rejected(self):
        ix = pd.period_range("2000-01", periods=2, freq="M")
        r = pd.DataFrame({"A": [0.1, np.nan], "B": [0.1, 0.1]}, index=ix)
        s = pd.DataFrame({"A": 2.0, "B": 1.0}, index=ix)
        with self.assertRaises(ValueError):
            backtest(r, s, "2000-01", "2000-02", "Top 10 momentum", count=1)

    def test_future_changes_leave_prior_holdings_unchanged(self):
        ix = pd.period_range("2000-01", periods=30, freq="M")
        r = pd.DataFrame(np.random.default_rng(4).normal(0, 0.05, (30, 20)), index=ix)
        a, ha = backtest(r, signals(r), "2001-01", "2002-06", "Buffered momentum")
        r.iloc[22:] = 0.5
        b, hb = backtest(r, signals(r), "2001-01", "2002-06", "Buffered momentum")
        pd.testing.assert_frame_equal(
            ha[ha.month < "2001-11"].reset_index(drop=True),
            hb[hb.month < "2001-11"].reset_index(drop=True),
        )


class StatisticalTests(unittest.TestCase):
    def test_ols_coefficients(self):
        x = np.arange(100, dtype=float)
        r = hac_ols(2 + 3 * x, x)
        np.testing.assert_allclose(r["coefficients"], [2, 3], atol=1e-10)

    def test_hac_mean_matches_manual_lag_zero(self):
        y = np.array([1, 2, 4, 8, 10.0])
        r = mean_hac(y, 0)
        self.assertAlmostEqual(r["coefficients"], y.mean())
        self.assertAlmostEqual(r["se"], y.std(ddof=1) / np.sqrt(len(y)))

    def test_metrics_include_initial_drawdown(self):
        f = pd.DataFrame(
            {
                "net_return": [-0.2, 0, 0.1],
                "gross_return": [-0.2, 0, 0.1],
                "turnover": [1, 0, 0],
                "cost_fraction": [0, 0, 0],
                "max_target_weight": [0.1] * 3,
                "holdings": [10] * 3,
            }
        )
        m = metrics(f, [0, 0, 0])
        self.assertAlmostEqual(m["max_drawdown"], -0.2)
        self.assertAlmostEqual(m["final_wealth"], 0.88)

    def test_information_coefficient_is_rank_based(self):
        ix = pd.period_range("2000-01", periods=2, freq="M")
        r = pd.DataFrame(
            [[0.01, 0.02, 0.03], [-0.01, -0.02, -0.03]],
            index=ix,
            columns=["A", "B", "C"],
        )
        s = pd.DataFrame([[1, 2, 3], [1, 2, 3]], index=ix, columns=r.columns)
        d = predictive_diagnostics(r, s, "2000-01", "2000-02", count=1)
        np.testing.assert_allclose(d.rank_ic, [1, -1])
        np.testing.assert_allclose(d.spread, [0.02, -0.02])


if __name__ == "__main__":
    unittest.main()
