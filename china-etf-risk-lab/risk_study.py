"""Reproduce the China ETF variance forecasting and risk targeting study."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_data import load_prices
from research_engine import metrics, simulate
from risk_models import HORIZON, LABEL_LAG, MODELS, RIDGE, TRAIN_DAYS, forecast, qlike


ROOT = Path(__file__).resolve().parent
TEST_START = "2023-01-01"
TEST_END = "2025-01-01"
TARGET = 0.10
REBALANCE_DAYS = 5
CORRELATION_SHRINKAGE = 0.25


def block_interval(values, length, seed=24):
    """Bootstrap contiguous dates after averaging the same-date asset panel."""
    sample = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    count = (len(sample) + length - 1) // length
    starts = generator.integers(0, len(sample) - length + 1, size=(4000, count, 1))
    indices = (starts + np.arange(length)).reshape(4000, -1)[:, :len(sample)]
    return [float(v) for v in np.quantile(sample[indices].mean(axis=1), (0.025, 0.975))]


def evaluate(predictions, actual):
    mask = (actual.index >= TEST_START) & (actual.index < TEST_END) & actual.notna().all(axis=1)
    actual = actual.loc[mask]
    losses = {name: qlike(actual, estimates.loc[mask]) for name, estimates in predictions.items()}
    if not all(np.isfinite(loss).all().all() for loss in losses.values()):
        raise ValueError("test forecasts are incomplete")
    scores = {}
    for name, loss in losses.items():
        difference = (loss - losses["ewma"]).mean(axis=1)
        # Blocks reflect dependence from overlapping horizons and market states.
        scores[name] = {"qlike": float(loss.to_numpy().mean()),
                        "delta_vs_ewma": float(difference.mean()),
                        "ci95_by_block_days": {str(length): block_interval(difference, length)
                                               for length in (10, 20, 40)},
                        "volatility_mae_annualized": float(
                            (np.sqrt(predictions[name].loc[mask]) - np.sqrt(actual)).abs().to_numpy().mean()),
                        "qlike_by_asset": loss.mean().to_dict(),
                        "delta_by_year": difference.groupby(difference.index.year).mean().to_dict()}
    ablation = (losses["extended_qlike"] - losses["har_qlike"]).mean(axis=1)
    loss_ablation = (losses["har_qlike"] - losses["har_log"]).mean(axis=1)
    comparisons = {"extended_minus_har_qlike": ablation, "har_qlike_minus_har_log": loss_ablation}
    checks = {name: {"delta": float(diff.mean()), "ci95_block20": block_interval(diff, 20)}
              for name, diff in comparisons.items()}
    return actual, losses, scores, checks


def portfolio_targets(prices, predictions):
    """Hold the same equal-weight sleeve and vary only gross exposure."""
    returns = prices.pct_change(fill_method=None)
    desired = {name: prices * 0 for name in predictions}
    count = prices.shape[1]
    equal = np.full(count, 1 / count)
    for t, date in enumerate(prices.index):
        if t < 63 or date < pd.Timestamp("2019-01-01"):
            continue
        correlation = returns.iloc[t - 62:t + 1].corr().to_numpy()
        correlation = np.nan_to_num(correlation, nan=0.0)
        np.fill_diagonal(correlation, 1.0)
        correlation = (1 - CORRELATION_SHRINKAGE) * correlation + CORRELATION_SHRINKAGE * np.eye(count)
        for name, variance in predictions.items():
            weighted_sigma = np.sqrt(variance.iloc[t].to_numpy()) * equal
            volatility = np.sqrt(max(float(weighted_sigma @ correlation @ weighted_sigma), 1e-8))
            desired[name].iloc[t] = equal * min(1, TARGET / volatility)
    return desired


def write_report(path, report):
    data = report["data"]
    extended = report["forecasts"]["extended_qlike"]
    portfolios = report["portfolio_by_cost_bps"]
    lo, hi = extended["ci95_by_block_days"]["20"]
    lines = ["# 中国 ETF 波动率预测与组合风险控制", "",
             f"真实行情覆盖 {data['first_date']} 至 {data['last_date']}，六只 ETF、{data['trading_days']:,} 个共同交易日。"
             f"2023—2024 年按时间前推评估，预测实验含 {report['forecast_origin_count']} 个完整预测日。", "",
             "## 结果概览", "",
             f"扩展模型相对 EWMA 的 QLIKE 差值为 {extended['delta_vs_ewma']:+.4f}，"
             f"95% 区间为 [{lo:+.4f}, {hi:+.4f}]。2023 和 2024 年的差值分别为 "
             f"{extended['delta_by_year'][2023]:+.4f} 和 {extended['delta_by_year'][2024]:+.4f}。", "",
             "## 预测表现", "",
             "QLIKE 越低越好。差值按同日六资产平均，再按日期做移动区块自助抽样。区间使用 20 日区块，10 和 40 日敏感性结果见 JSON。", "",
             "| 模型 | QLIKE | 相对 EWMA 差值 | 95% 区间 | 年化波动率 MAE |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in report["forecasts"].items():
        lo, hi = row["ci95_by_block_days"]["20"]
        lines.append(f"| {name} | {row['qlike']:.4f} | {row['delta_vs_ewma']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {row['volatility_mae_annualized']:.2%} |")
    lines += ["", "## 消融实验", "", "| 比较 | QLIKE 差值 | 95% 区间 |",
              "| --- | ---: | ---: |"]
    for name, row in report["ablations"].items():
        lo, hi = row["ci95_block20"]
        lines.append(f"| {name} | {row['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ["", "## 组合表现", "", "同一个等权资产组合，所有预测模型使用相同的相关矩阵收缩、调仓时点和成交费率。单边成本 5 bps。", "",
              "| 模型 | 累计收益 | 年化波动率 | 与 10% 目标相差 | 最大回撤 | 年化换手 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in portfolios["5"].items():
        lines.append(f"| {name} | {row['total_return']:.2%} | {row['annualized_volatility']:.2%} | {row['volatility_target_gap']:+.2%} | {row['max_drawdown']:.2%} | {row['annualized_turnover']:.2f}× |")
    lines += ["", "| 扩展模型的单边费用 | 累计收益 |", "| --- | ---: |"]
    for cost, result in portfolios.items():
        lines.append(f"| {cost} bps | {result['extended_qlike']['total_return']:.2%} |")
    lines += ["", "## 适用范围", "", "样本是固定存续 ETF，价格来自同一历史快照。2023—2024 年曾用于仓库中的另一项回测，此处为新模型的历史时间隔离评估。滚动过程中已成熟的标签会进入下一次拟合。",
              "以日收益平方估计短期实现方差。重叠标签和共同市场状态会降低有效样本量。区间用于衡量固定样本下的差值不确定性，多项比较采用探索性解释。",
              "组合使用连续权重、零现金收益和统一费率，成交模拟尚未纳入交易单位、涨跌停、成交容量与最低佣金。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(cache, output):
    prices, sources = load_prices(cache)
    predictions, labels, fits = forecast(prices)
    actual, losses, scores, ablations = evaluate(predictions, labels)
    weights = portfolio_targets(prices, predictions)
    weights["unscaled_equal_weight"] = prices * 0 + 1 / prices.shape[1]
    comparisons, curves = {}, {}
    for cost in (0, 5, 20):
        result = {}
        for name, desired in weights.items():
            curve = simulate(prices, desired, TEST_START, TEST_END, cost, REBALANCE_DAYS)
            stats = metrics(curve)
            stats["volatility_target_gap"] = stats["annualized_volatility"] - TARGET
            stats["average_exposure"] = float(curve.exposure.mean())
            result[name] = stats
            if cost == 5:
                curves[name] = curve.equity
        comparisons[str(cost)] = result
    report = {
        "data": {"first_date": str(prices.index[0].date()), "last_date": str(prices.index[-1].date()),
                 "trading_days": len(prices), "snapshots": sources},
        "protocol": {"models": MODELS, "test_start": TEST_START, "test_end_exclusive": TEST_END,
                     "variance_horizon_days": HORIZON, "last_label_offset": LABEL_LAG,
                     "train_window_days": TRAIN_DAYS, "fit_frequency": "monthly", "ridge": RIDGE,
                     "target_volatility": TARGET, "rebalance_days": REBALANCE_DAYS,
                     "correlation_shrinkage": CORRELATION_SHRINKAGE, "cost_bps_per_side": [0, 5, 20],
                     "bootstrap_draws": 4000, "bootstrap_seed": 24},
        "runtime": {"numpy": np.__version__, "pandas": pd.__version__},
        "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("market_data.py", "risk_models.py", "research_engine.py", "risk_study.py")},
        "forecast_origin_count": len(actual), "forecasts": scores, "ablations": ablations,
        "portfolio_by_cost_bps": comparisons,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "risk_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_report(output / "RISK_REPORT.md", report)
    pd.DataFrame(fits).to_csv(output / "fit_audit.csv", index=False)
    pd.DataFrame({name: loss.mean(axis=1) for name, loss in losses.items()}).to_csv(output / "daily_loss.csv")
    pd.DataFrame(curves).to_csv(output / "equity_cost5.csv")
    print(output / "RISK_REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "risk_study")
    args = parser.parse_args()
    run(args.cache, args.output)
