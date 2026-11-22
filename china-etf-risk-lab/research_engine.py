"""Causal portfolio targets and self-financing daily portfolio simulation."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Strategy:
    family: str
    lookback: int
    top_k: int = 2
    volatility_target: float = 0.10

    def to_dict(self):
        return asdict(self)


def targets(prices, strategy):
    """Targets at signal-day close. Execution lag is enforced by simulate()."""
    if strategy.family not in {"trend", "relative_momentum", "reversal"}:
        raise ValueError("unknown strategy family")
    if strategy.lookback < 2 or not 1 <= strategy.top_k <= prices.shape[1]:
        raise ValueError("invalid lookback or top_k")
    if not 0 < strategy.volatility_target <= 1:
        raise ValueError("invalid volatility target")
    returns = prices.pct_change(fill_method=None)
    vol = returns.rolling(63, min_periods=63).std() * np.sqrt(252)
    signal = prices.pct_change(strategy.lookback, fill_method=None)
    if strategy.family == "trend":
        selected = signal > 0
    else:
        ranks = signal.rank(axis=1, ascending=strategy.family == "reversal", method="first")
        selected = ranks <= strategy.top_k
    # Inverse-vol allocation. Sum(weight_i * sigma_i) is an upper bound on
    # portfolio volatility for any valid correlation matrix. No leverage.
    inverse_vol = (1 / vol.replace(0, np.nan)).where(selected, 0).fillna(0)
    normalized = inverse_vol.div(inverse_vol.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    bound = (normalized * vol.fillna(0)).sum(axis=1)
    scale = (strategy.volatility_target / bound.replace(0, np.nan)).clip(upper=1).fillna(0)
    weights = normalized.mul(scale, axis=0)
    # No strategy trades before every indicator has a full history.
    ready = signal.notna().all(axis=1) & vol.notna().all(axis=1)
    return weights.where(ready, 0)


def rebalance(holdings, cash, desired, fee):
    """Solve E_after = E_before - fee * sum(abs(target * E_after - holdings))."""
    equity = float(holdings.sum() + cash)
    after_cost = equity
    for _ in range(100):
        traded = float(np.abs(desired * after_cost - holdings).sum())
        updated = equity - fee * traded
        if abs(updated - after_cost) <= 1e-12 * equity:
            after_cost = updated
            break
        after_cost = updated
    else:
        raise ArithmeticError("transaction-cost solver failed to converge")
    new_holdings = desired * after_cost
    cost = fee * float(np.abs(new_holdings - holdings).sum())
    new_cash = equity - cost - float(new_holdings.sum())
    return new_holdings, new_cash, cost, traded / equity


def simulate(prices, desired, start, end, cost_bps=5, rebalance_days=21):
    """Start/end are date bounds, end exclusive. Each episode starts in cash.

    A close-s signal trades at close s+1 and first earns the s+1 -> s+2 return.
    Existing holdings drift between rebalances. Terminal holdings are liquidated.
    """
    if not 0 <= cost_bps <= 100 or not np.isfinite(cost_bps):
        raise ValueError("cost_bps must lie in [0,100]")
    if rebalance_days < 1:
        raise ValueError("rebalance_days must be positive")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("price dates must be unique and increasing")
    if not np.isfinite(prices.to_numpy()).all() or (prices.to_numpy() <= 0).any():
        raise ValueError("prices must be positive and finite")
    if not desired.index.equals(prices.index) or not desired.columns.equals(prices.columns):
        raise ValueError("target axes must match prices")
    values = desired.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values.sum(axis=1) > 1 + 1e-10).any():
        raise ValueError("targets must be finite, long-only and unlevered")
    indices = np.flatnonzero((prices.index >= start) & (prices.index < end))
    if len(indices) < 2 or indices[0] < 2:
        raise ValueError("episode needs history and at least two observations")
    first, last = int(indices[0]), int(indices[-1])
    fee = cost_bps / 10000
    price_values = prices.to_numpy()
    holdings = np.zeros(prices.shape[1])
    # Initial order executes on the close preceding the first return date.
    holdings, cash, pending_cost, pending_turnover = rebalance(holdings, 1.0, values[first - 2], fee)
    previous_equity = 1.0
    rows = []
    for t in indices:
        holdings *= price_values[t] / price_values[t - 1]
        cost, turnover = pending_cost, pending_turnover
        pending_cost = pending_turnover = 0.0
        if t == last:
            holdings, cash, extra_cost, extra_turnover = rebalance(holdings, cash, np.zeros_like(holdings), fee)
            cost += extra_cost
            turnover += extra_turnover
        elif (t - first + 1) % rebalance_days == 0:
            holdings, cash, extra_cost, extra_turnover = rebalance(holdings, cash, values[t - 1], fee)
            cost += extra_cost
            turnover += extra_turnover
        equity = float(holdings.sum() + cash)
        rows.append({"date": prices.index[t], "equity": equity,
                     "net_return": equity / previous_equity - 1,
                     "cost": cost, "turnover": turnover,
                     "exposure": float(holdings.sum()) / equity})
        previous_equity = equity
    return pd.DataFrame(rows).set_index("date")


def metrics(curve):
    returns = curve.net_return
    n = len(returns)
    volatility = float(returns.std(ddof=1) * np.sqrt(252))
    peak = curve.equity.cummax().clip(lower=1)
    return {
        "observations": n,
        "total_return": float(curve.equity.iloc[-1] - 1),
        "cagr": float(curve.equity.iloc[-1] ** (252 / n) - 1),
        "annualized_volatility": volatility,
        "sharpe_rf_zero": float(returns.mean() * 252 / volatility) if volatility > 0 else 0.0,
        "max_drawdown": float((1 - curve.equity / peak).max()),
        "annualized_turnover": float(curve.turnover.sum() * 252 / n),
        "cost_in_initial_equity_units": float(curve.cost.sum()),
    }
