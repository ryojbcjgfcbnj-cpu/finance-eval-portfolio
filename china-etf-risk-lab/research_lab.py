"""Real-data walk-forward research with auditable tool-call trajectories."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from market_data import ASSETS, BENCHMARK, PRICE_FIELD, SOURCE, load_prices
from research_engine import Strategy, metrics, simulate, targets


ROOT = Path(__file__).resolve().parent
CANDIDATES = [Strategy(family, lookback) for family in
              ("trend", "relative_momentum", "reversal") for lookback in (21, 63, 126)]


def select_strategy(prices, start, end, cost_bps, trajectory, episode):
    # Pass a physically truncated frame to indicator computation and simulation.
    visible = prices.loc[prices.index < end]
    experiments = []
    for candidate in CANDIDATES:
        curve = simulate(visible, targets(visible, candidate), start, end, cost_bps)
        result = metrics(curve)
        score = result["sharpe_rf_zero"] - 2 * result["max_drawdown"]
        record = {"episode": episode, "actor": "scripted_grid_researcher",
                  "action": {"tool": "evaluate_strategy", "arguments": candidate.to_dict()},
                  "observation": {"start": start, "end_exclusive": end, "metrics": result},
                  "selection_score": score}
        trajectory.append(record)
        experiments.append((score, candidate))
    return max(experiments, key=lambda item: item[0])[1]


def run(cache, output, refresh=False, cost_bps=5):
    prices, provenance = load_prices(cache, refresh)
    output.mkdir(parents=True, exist_ok=True)
    trajectory, folds, out_of_sample = [], [], []
    for year in range(2018, 2023):
        start, end = f"{year}-01-01", f"{year + 1}-01-01"
        candidate = select_strategy(prices, f"{year - 3}-01-01", start,
                                    cost_bps, trajectory, f"walk_forward_{year}")
        available = prices.loc[prices.index < end]
        curve = simulate(available, targets(available, candidate), start, end, cost_bps)
        folds.append({"year": year, "strategy": candidate.to_dict(), "metrics": metrics(curve)})
        out_of_sample.append(curve)

    # Freeze final specification using only data ending in 2022.
    selected = select_strategy(prices, "2020-01-01", "2023-01-01", cost_bps,
                               trajectory, "final_selection")
    final_targets = targets(prices, selected)
    curves = {"selected": simulate(prices, final_targets, "2023-01-01", "2025-01-01", cost_bps)}
    equal = pd.DataFrame(1 / prices.shape[1], index=prices.index, columns=prices.columns)
    benchmark = equal * 0
    benchmark[BENCHMARK] = 1.0
    curves["equal_weight"] = simulate(prices, equal, "2023-01-01", "2025-01-01", cost_bps)
    curves["csi300_buy_hold"] = simulate(prices, benchmark, "2023-01-01", "2025-01-01", cost_bps,
                                      rebalance_days=len(prices) + 1)
    curves["cash"] = simulate(prices, equal * 0, "2023-01-01", "2025-01-01", cost_bps)
    sensitivity = {}
    for fee in (0, 5, 20):
        sensitivity[str(fee)] = metrics(simulate(prices, final_targets, "2023-01-01", "2025-01-01", fee))
    combined = pd.concat(out_of_sample)
    combined["equity"] = (1 + combined.net_return).cumprod()
    # Episode costs are denominated in each year's initial equity; report folds
    # separately and avoid adding them as if they shared a denominator.
    oos_metrics = metrics(combined)
    oos_metrics.pop("cost_in_initial_equity_units")
    report = {
        "data": {"source": SOURCE, "field": PRICE_FIELD,
                 "market": "China mainland exchange-listed ETFs", "currency": "CNY",
                 "assets": ASSETS, "benchmark": BENCHMARK, "rows": len(prices),
                 "first_date": str(prices.index[0].date()),
                 "last_date": str(prices.index[-1].date()), "snapshots": provenance},
        "experiment": {"cost_bps_per_side": cost_bps, "rebalance_trading_days": 21,
                       "selection": "3-year Sharpe(rf=0) minus 2 * max_drawdown",
                       "candidate_count": len(CANDIDATES), "volatility_target": 0.10,
                       "final_test": ["2023-01-01", "2025-01-01"],
                       "engine_sha256": hashlib.sha256((ROOT / "research_engine.py").read_bytes()).hexdigest()},
        "walk_forward": folds, "walk_forward_aggregate": oos_metrics,
        "selected_strategy": selected.to_dict(),
        "final_test": {name: metrics(curve) for name, curve in curves.items()},
        "cost_sensitivity_fixed_strategy_bps": sensitivity,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "trajectory.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trajectory), encoding="utf-8")
    combined.to_csv(output / "walk_forward_curve.csv")
    for name, curve in curves.items():
        curve.to_csv(output / f"test_{name}.csv")
    lines = ["# 中国市场 ETF 研究实验", "", "## 数据与协议", "",
             f"沪深交易所上市 ETF，人民币计价。Yahoo Finance 调整收盘价，{len(prices)} 个共同交易日，{report['data']['first_date']} 至 {report['data']['last_date']}。",
             "", f"最终候选 `{selected.to_dict()}`。选择阶段截止 2022 年末，最终测试覆盖 2023–2024 年。",
             "", "## 最终测试结果", "", "| 策略 | 累计收益 | 年化收益 | Sharpe（rf=0） | 最大回撤 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, result in report["final_test"].items():
        lines.append(f"| {name} | {result['total_return']:.2%} | {result['cagr']:.2%} | {result['sharpe_rf_zero']:.3f} | {result['max_drawdown']:.2%} |")
    lines += ["", "## 滚动样本外实验", "", "| 年份 | 选择的策略 | 收益 | Sharpe（rf=0） | 最大回撤 |",
              "| --- | --- | ---: | ---: | ---: |"]
    for fold in folds:
        s, m = fold["strategy"], fold["metrics"]
        lines.append(f"| {fold['year']} | {s['family']} / {s['lookback']} | {m['total_return']:.2%} | {m['sharpe_rf_zero']:.3f} | {m['max_drawdown']:.2%} |")
    lines += ["", "## 适用范围", "",
              "固定六只存续 ETF 的研究样本存在资产选择偏差；数据为下载时的修订与调整口径。",
              "调整价格作为收益指数使用，组合允许碎股；现金利率为零；交易成本使用统一单边费率。",
              "数据源的2025年行情存在缺失，国内备用接口连接不稳定，本轮使用完整的2014—2024年窗口。",
              "策略采用波动率约束，沪深300ETF基准满仓，需结合风险暴露比较收益。",
              "股票ETF按日收盘低频调仓，持仓至少隔一交易日；黄金与债券ETF同样使用该保守频率。",
              "仍使用碎份额与统一费率近似，尚未实现100份交易单位、最低佣金、涨跌停与停牌成交限制。",
              "滚动回测每年独立建仓并平仓，已计边界交易成本。",
              "候选选择使用脚本搜索。LLM 决策与权重训练仍在后续实现范围。",
              "本轮测试结果一旦用于修改模型或策略，应改称开发结果，并另设时间段或资产作为后续独立测试。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(prices), "selected": selected.to_dict(),
                      "final_test": report["final_test"], "output": str(output)}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "china_market")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cost-bps", type=float, default=5)
    args = parser.parse_args()
    run(args.cache, args.output, args.refresh, args.cost_bps)
