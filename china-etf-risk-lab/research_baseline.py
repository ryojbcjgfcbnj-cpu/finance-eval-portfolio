"""Lesson 1: reproducible research runner, using synthetic daily closes.

Run with Python 3.10+. Uses only the standard library.
"""

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path


def synthetic_closes(seed, days=1100):
    """AR(1) log returns are a toy process, not a market dataset."""
    rng = random.Random(seed)
    closes = [100.0]
    previous_return = 0.0
    for _ in range(days - 1):
        daily_return = 0.0001 + 0.08 * previous_return + rng.gauss(0, 0.012)
        closes.append(closes[-1] * math.exp(daily_return))
        previous_return = daily_return
    return closes


def momentum_positions(closes, lookback):
    """Long/cash. Signal at close s executes at close s+1.

    positions[t] earns close[t-1] -> close[t], so its signal uses close[t-2].
    """
    if lookback < 1:
        raise ValueError("lookback must be positive")
    positions = [0.0] * len(closes)
    for t in range(lookback + 2, len(closes)):
        signal_day = t - 2
        momentum = closes[signal_day] / closes[signal_day - lookback] - 1
        positions[t] = float(momentum > 0)
    return positions


def backtest(closes, positions, start, stop, cost_bps):
    """Fresh portfolio per split. Charge entry, rebalance and final exit.

    Daily net return uses equity-normalized notional cost. Flat cash earns zero.
    Fractional positions, no borrow, slippage included in per-side cost_bps.
    """
    if len(closes) != len(positions) or not 1 <= start < stop <= len(closes):
        raise ValueError("invalid backtest inputs")
    if cost_bps < 0 or not math.isfinite(cost_bps):
        raise ValueError("cost_bps must be finite and nonnegative")
    equity = peak = 1.0
    previous_position = 0.0
    max_drawdown = total_turnover = 0.0
    daily_returns = []
    for t in range(start, stop):
        position = positions[t]
        turnover = abs(position - previous_position)
        if t == stop - 1:
            turnover += abs(position)
        gross_return = position * (closes[t] / closes[t - 1] - 1)
        net_return = gross_return - turnover * cost_bps / 10000
        equity *= 1 + net_return
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, 1 - equity / peak)
        total_turnover += turnover
        daily_returns.append(net_return)
        previous_position = position
    volatility = statistics.stdev(daily_returns)
    sharpe = 0.0
    if volatility > 0:
        sharpe = statistics.mean(daily_returns) / volatility * math.sqrt(252)
    return {
        "observations": len(daily_returns),
        "total_return": equity - 1,
        "annualized_sharpe_rf_zero": sharpe,
        "max_drawdown": max_drawdown,
        "total_turnover": total_turnover,
    }


def run(output, seed, cost_bps):
    # Fixed before inspecting any results. Reserve test until model selection ends.
    splits = {"train": (62, 600), "validation": (600, 850), "test": (850, 1100)}
    candidates = (5, 20, 60)
    closes = synthetic_closes(seed)
    trajectory = []
    for lookback in candidates:
        positions = momentum_positions(closes, lookback)
        train = backtest(closes, positions, *splits["train"], cost_bps)
        validation = backtest(closes, positions, *splits["validation"], cost_bps)
        trajectory.append({
            "actor": "scripted_grid_baseline",
            "action": {"tool": "evaluate_momentum", "lookback": lookback},
            "observation": {"train": train, "validation": validation},
        })
    winner = max(trajectory, key=lambda row: row["observation"]["validation"]["annualized_sharpe_rf_zero"])
    selected_lookback = winner["action"]["lookback"]
    selected_positions = momentum_positions(closes, selected_lookback)
    report = {
        "data": "synthetic_AR1_log_returns",
        "seed": seed,
        "cost_bps_per_side": cost_bps,
        "splits_half_open": splits,
        "selection_rule": "highest_validation_sharpe_among_fixed_candidates",
        "candidates": candidates,
        "selected_lookback": selected_lookback,
        "test": backtest(closes, selected_positions, *splits["test"], cost_bps),
        "test_cash": backtest(closes, [0.0] * len(closes), *splits["test"], cost_bps),
        "test_buy_hold": backtest(closes, [1.0] * len(closes), *splits["test"], cost_bps),
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "prices.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["day", "synthetic_close"])
        writer.writerows(enumerate(closes))
    with (output / "trajectory.jsonl").open("w", encoding="utf-8") as stream:
        for step in trajectory:
            stream.write(json.dumps(step, ensure_ascii=False) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()
    run(args.output, args.seed, args.cost_bps)
