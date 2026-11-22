"""Causal features, mature labels and convex variance forecasts."""

import numpy as np
import pandas as pd


HORIZON = 5
LABEL_LAG = HORIZON + 1
FLOOR = 1e-8  # Annualized variance
MIN_TRAIN = 500
TRAIN_DAYS = 756
RIDGE = 0.03
MODELS = ("historical", "ewma", "har_log", "har_qlike", "extended_qlike")


def inputs(prices):
    returns = prices.pct_change(fill_method=None)
    squared = returns.pow(2)
    variance = {f"rv{window}": squared.rolling(window).mean() * 252
                for window in (5, 21, 63)}
    features = {key: np.log(value.clip(lower=FLOOR)) for key, value in variance.items()}
    downside = returns.clip(upper=0).pow(2).rolling(21).sum()
    features["downside_share"] = downside / squared.rolling(21).sum().replace(0, np.nan)
    features["return21"] = prices.pct_change(21, fill_method=None)
    # Cross-asset state is observable at the same close as the other features.
    cross_asset = features["rv21"].mean(axis=1, skipna=False).to_numpy()
    features["market_risk"] = pd.DataFrame(
        np.broadcast_to(cross_asset[:, None], prices.shape), index=prices.index, columns=prices.columns)
    # At signal t, trade at close t+1. First realized return ends at t+2.
    label = squared.rolling(HORIZON).mean().shift(-LABEL_LAG) * 252
    x = np.stack([feature.to_numpy() for feature in features.values()], axis=-1)
    observed = {"historical": variance["rv21"],
                "ewma": squared.ewm(alpha=0.06, adjust=False, min_periods=63).mean() * 252}
    return x, label, observed


def qlike(actual, predicted):
    """QLIKE up to an additive target-dependent term, including zero targets."""
    return np.log(predicted) + actual / predicted


def objective(beta, design, scaled_y, penalty):
    linear = design @ beta
    with np.errstate(over="ignore", invalid="ignore"):
        weighted = scaled_y * np.exp(-linear)
        value = float(np.mean(linear + weighted) + 0.5 * np.dot(penalty * beta, beta))
    return value, weighted


def fit_qlike(design, scaled_y, ridge=RIDGE):
    """Newton updates with an unpenalized intercept and Armijo backtracking."""
    beta = np.zeros(design.shape[1])
    penalty = np.full(beta.size, ridge)
    penalty[0] = 0
    for iteration in range(61):
        value, weighted = objective(beta, design, scaled_y, penalty)
        gradient = design.T @ (1 - weighted) / len(scaled_y) + penalty * beta
        if np.linalg.norm(gradient, ord=np.inf) < 1e-8:
            return beta, iteration
        hessian = (design.T * weighted) @ design / len(scaled_y) + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        directional = float(gradient @ step)
        for backtrack in range(25):
            rate = 0.5 ** backtrack
            candidate = beta - rate * step
            candidate_value, _ = objective(candidate, design, scaled_y, penalty)
            if np.isfinite(candidate_value) and candidate_value <= value - 1e-4 * rate * directional:
                beta = candidate
                break
        else:
            raise ArithmeticError("QLIKE line search failed")
    raise ArithmeticError("QLIKE fit did not converge")


def forecast(prices, start="2019-01-01"):
    x, labels, observed = inputs(prices)
    y = labels.to_numpy()
    predictions = {name: frame.clip(lower=FLOOR).copy() for name, frame in observed.items()}
    histories = []
    for name, columns in (("har_log", 3), ("har_qlike", 3), ("extended_qlike", 6)):
        values = np.full(y.shape, np.nan)
        fit_month = None
        for t, date in enumerate(prices.index):
            if date < pd.Timestamp(start):
                continue
            month = (date.year, date.month)
            if month != fit_month:
                stop = t - LABEL_LAG + 1  # Last label ends at this close.
                first = max(0, stop - TRAIN_DAYS)
                window = x[first:stop, :, :columns].reshape(-1, columns)
                outcome = y[first:stop].reshape(-1)
                valid = np.isfinite(window).all(axis=1) & np.isfinite(outcome)
                if valid.sum() < MIN_TRAIN * prices.shape[1]:
                    raise ValueError("insufficient mature training observations")
                window, outcome = window[valid], outcome[valid]
                center, scale = window.mean(axis=0), window.std(axis=0)
                scale = np.where(scale > 1e-12, scale, 1.0)
                design = np.column_stack((np.ones(len(window)), (window - center) / scale))
                unit = max(float(outcome.mean()), FLOOR)
                scaled_y = outcome / unit
                if name == "har_log":
                    log_y = np.log(np.maximum(outcome, FLOOR) / unit)
                    penalty = np.eye(design.shape[1]) * RIDGE
                    penalty[0, 0] = 0.0
                    beta = np.linalg.solve(design.T @ design / len(log_y) + penalty,
                                           design.T @ log_y / len(log_y))
                    # Approximate conditional mean after retransformation.
                    correction = float(np.exp(log_y - design @ beta).mean())
                    iterations = 0
                else:
                    beta, iterations = fit_qlike(design, scaled_y)
                    correction = 1.0
                fit_month = month
                histories.append({"model": name, "fit_date": str(date.date()),
                                  "latest_label_end": str(prices.index[stop - 1 + LABEL_LAG].date()),
                                  "latest_training_origin": str(prices.index[stop - 1].date()),
                                  "train_rows": int(valid.sum()), "iterations": iterations})
            current = x[t, :, :columns]
            design = np.column_stack((np.ones(len(current)), (current - center) / scale))
            variance = unit * correction * np.exp(design @ beta)
            if not np.isfinite(variance).all():
                raise ArithmeticError("non-finite forecast")
            values[t] = np.maximum(variance, FLOOR)
        predictions[name] = pd.DataFrame(values, index=prices.index, columns=prices.columns)
    return predictions, labels, histories
