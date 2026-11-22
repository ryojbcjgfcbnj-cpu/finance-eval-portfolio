"""Check the solver and timing boundaries used by the risk study."""

import unittest

import numpy as np
import pandas as pd

from risk_models import LABEL_LAG, RIDGE, fit_qlike, forecast, inputs, objective
from risk_study import block_interval, portfolio_targets


class RiskChecks(unittest.TestCase):
    def test_intercept_recovers_mean_variance(self):
        values = np.array([0.1, 0.5, 1.7, 3.0])
        beta, _ = fit_qlike(np.ones((len(values), 1)), values)
        self.assertAlmostEqual(np.exp(beta[0]), values.mean())

    def test_solver_fits_ridge_objective(self):
        feature = np.linspace(-2, 2, 101)
        design = np.column_stack((np.ones(len(feature)), feature))
        values = np.exp(0.7 * feature)
        beta, _ = fit_qlike(design, values)
        penalty = np.array([0.0, RIDGE])
        value, weighted = objective(beta, design, values, penalty)
        initial_value, _ = objective(np.zeros(2), design, values, penalty)
        self.assertLess(value, initial_value)
        np.testing.assert_allclose(design.T @ (1 - weighted) / len(values) + penalty * beta,
                                   np.zeros(2), atol=1e-8)

    def test_label_starts_after_execution(self):
        daily = np.arange(15) / 100
        prices = pd.DataFrame({"A": 100 * (1 + daily).cumprod()})
        _, label, _ = inputs(prices)
        self.assertAlmostEqual(label.iloc[3, 0], (daily[5:10] ** 2).mean() * 252)
        self.assertTrue(label.iloc[-LABEL_LAG:].isna().all().all())

    def test_future_prices_leave_fits_predictions_and_exposure_unchanged(self):
        rng = np.random.default_rng(2)
        dates = pd.bdate_range("2015-01-01", periods=1400)
        prices = pd.DataFrame(100 * np.exp(rng.normal(0, 0.01, (len(dates), 3)).cumsum(axis=0)), index=dates)
        changed = prices.copy()
        cut = dates[1200]
        changed.loc[cut:] *= rng.uniform(0.8, 1.2, size=changed.loc[cut:].shape)
        original, _, fits = forecast(prices)
        altered, _, changed_fits = forecast(changed)
        for name in original:
            pd.testing.assert_frame_equal(original[name].loc[dates < cut], altered[name].loc[dates < cut])
        self.assertEqual([row for row in fits if pd.Timestamp(row["fit_date"]) < cut],
                         [row for row in changed_fits if pd.Timestamp(row["fit_date"]) < cut])
        for row in fits:
            end = dates.get_loc(pd.Timestamp(row["latest_training_origin"])) + LABEL_LAG
            self.assertEqual(str(dates[end].date()), row["latest_label_end"])
            self.assertLessEqual(row["latest_label_end"], row["fit_date"])
        w = portfolio_targets(prices, {"ewma": original["ewma"]})["ewma"]
        w2 = portfolio_targets(changed, {"ewma": altered["ewma"]})["ewma"]
        pd.testing.assert_frame_equal(w.loc[dates < cut], w2.loc[dates < cut])
        self.assertTrue((w.sum(axis=1) <= 1 + 1e-12).all())

    def test_bootstrap_constant_paired_difference(self):
        self.assertEqual(block_interval(np.full(90, -0.25), 20), [-0.25, -0.25])


if __name__ == "__main__":
    unittest.main()
