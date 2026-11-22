"""Regression checks for timing, accounting and experiment isolation."""

import unittest

import numpy as np
import pandas as pd

from research_engine import Strategy, metrics, rebalance, simulate, targets
from research_lab import select_strategy
from market_data import parse_prices, timestamp


class ResearchChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        dates = pd.bdate_range("2010-01-01", "2025-12-31")
        self.prices = pd.DataFrame(100 * np.exp(np.cumsum(
            rng.normal(0.0002, 0.01, (len(dates), 3)), axis=0)),
            index=dates, columns=["A", "B", "C"])

    def test_self_financing(self):
        holdings = np.array([0.3, 0.2, 0.1])
        desired = np.array([0.1, 0.6, 0.0])
        updated, cash, fee, _ = rebalance(holdings, 0.4, desired, 0.002)
        self.assertAlmostEqual(updated.sum() + cash + fee, 1)
        np.testing.assert_allclose(updated / (1 - fee), desired)
        self.assertGreaterEqual(cash, 0)

    def test_entry_and_exit_cost_hand_calculation(self):
        prices = pd.DataFrame({"A": [100, 100, 110, 121]}, index=pd.bdate_range("2020-01-01", periods=4))
        weights = prices * 0 + 1
        curve = simulate(prices, weights, str(prices.index[2].date()), "2021-01-01", 10, 999)
        self.assertAlmostEqual(curve.equity.iloc[-1], 1.21 * (1 - 0.001) / (1 + 0.001))
        # Entry notional is reduced by its own cost; turnover uses pretrade NAV.
        self.assertAlmostEqual(curve.turnover.sum(), 1 / 1.001 + 1)

    def test_signal_execution_lag(self):
        prices = pd.DataFrame({"A": [100, 100, 100, 200, 200, 200]}, index=pd.bdate_range("2020-01-01", periods=6))
        weights = prices * 0
        weights.iloc[2:] = 1  # Day 2 signal trades at day 3 close, after the jump.
        curve = simulate(prices, weights, str(prices.index[2].date()), "2021-01-01", 0, 1)
        self.assertAlmostEqual(curve.equity.iloc[-1], 1)

    def test_future_prices_do_not_change_past_targets(self):
        changed = self.prices.copy()
        changed.iloc[1000:] *= 3
        for family in ("trend", "relative_momentum", "reversal"):
            config = Strategy(family, 63)
            pd.testing.assert_frame_equal(targets(self.prices, config).iloc[:1000],
                                          targets(changed, config).iloc[:1000])

    def test_selection_ignores_final_test(self):
        changed = self.prices.copy()
        changed.loc["2024-01-01":] *= 5
        before, after = [], []
        a = select_strategy(self.prices, "2021-01-01", "2024-01-01", 5, before, "test")
        b = select_strategy(changed, "2021-01-01", "2024-01-01", 5, after, "test")
        self.assertEqual(a, b)
        self.assertEqual(before, after)

    def test_cash_and_cost_sensitivity(self):
        cash = self.prices * 0
        curve = simulate(self.prices, cash, "2024-01-01", "2026-01-01")
        self.assertEqual(metrics(curve)["total_return"], 0)
        weights = targets(self.prices, Strategy("trend", 21))
        cheap = simulate(self.prices, weights, "2024-01-01", "2026-01-01", 0)
        expensive = simulate(self.prices, weights, "2024-01-01", "2026-01-01", 20)
        self.assertLessEqual(expensive.equity.iloc[-1], cheap.equity.iloc[-1])

    def test_invalid_weights_rejected(self):
        with self.assertRaises(ValueError):
            simulate(self.prices, self.prices * 0 + 1, "2024-01-01", "2026-01-01")

    def china_response(self):
        return {"chart": {"result": [{
            "meta": {"symbol": "510300.SS", "currency": "CNY"},
            "timestamp": [timestamp("2014-01-02"), timestamp("2024-12-31")],
            "indicators": {"adjclose": [{"adjclose": [2.0, 4.0]}]},
        }]}}

    def test_china_symbol_currency_and_dates(self):
        prices = parse_prices(self.china_response(), "510300.SS")
        self.assertEqual(str(prices.index[0].date()), "2014-01-02")
        with self.assertRaises(ValueError):
            parse_prices(self.china_response(), "159915.SZ")
        response = self.china_response()
        response["chart"]["result"][0]["meta"]["currency"] = "USD"
        with self.assertRaises(ValueError):
            parse_prices(response, "510300.SS")

    def test_missing_china_price_rejected(self):
        response = self.china_response()
        response["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"][0] = None
        with self.assertRaises(ValueError):
            parse_prices(response, "510300.SS")


if __name__ == "__main__":
    unittest.main()
