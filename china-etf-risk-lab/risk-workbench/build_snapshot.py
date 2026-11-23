"""Export derived risk snapshots from the parent research project."""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
sys.path.insert(0, str(LAB))

from market_data import ASSETS, load_prices
from risk_models import forecast
from risk_study import CORRELATION_SHRINKAGE


def build():
    prices, sources = load_prices(LAB / "data" / "cache")
    forecasts, _, _ = forecast(prices)
    report = json.loads((LAB / "examples" / "risk_report.json").read_text(encoding="utf-8"))
    if [item["sha256"] for item in sources] != [item["sha256"] for item in report["data"]["snapshots"]]:
        raise ValueError("Rerun the risk study and update the example report to match these snapshots")
    daily = prices.pct_change(fill_method=None)
    snapshots = {}
    for date in ("2024-12-31", "2024-10-08", "2023-12-29"):
        corr = daily.loc[:date].tail(63).corr().to_numpy()
        corr = (1 - CORRELATION_SHRINKAGE) * corr + CORRELATION_SHRINKAGE * np.eye(len(ASSETS))
        snapshots[date] = {"correlation": corr.round(9).tolist(), "variance": {
            name: series.loc[date].to_numpy().round(12).tolist() for name, series in forecasts.items()}}
    subset = {key: report[key] for key in ("forecast_origin_count", "forecasts", "ablations", "portfolio_by_cost_bps")}
    data = {"assets": list(ASSETS.items()), "snapshots": snapshots, "study": subset,
            "source": report["data"]["snapshots"], "firstDate": report["data"]["first_date"]}
    output = HERE / "dist" / "snapshot.js"
    output.write_text("window.riskSnapshot = " + json.dumps(data, ensure_ascii=False, allow_nan=False) + ";\n", encoding="utf-8")


if __name__ == "__main__":
    build()
