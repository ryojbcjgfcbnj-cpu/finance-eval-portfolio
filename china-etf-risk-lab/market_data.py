"""China exchange-listed ETF adjusted daily history, with source snapshots."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ASSETS = {
    "510300.SS": "华泰柏瑞沪深300ETF",
    "510500.SS": "南方中证500ETF",
    "159915.SZ": "易方达创业板ETF",
    "510880.SS": "华泰柏瑞上证红利ETF",
    "518880.SS": "华安黄金ETF",
    "511010.SS": "国泰上证5年期国债ETF",
}
SYMBOLS = tuple(ASSETS)
BENCHMARK = "510300.SS"
START = "2014-01-01"
# The source has a missing trading day in 2025 for this universe. Use the
# complete 2014-2024 window, without silently dropping dates or filling prices.
END = "2025-01-01"
SOURCE = "Yahoo Finance chart API"
PRICE_FIELD = "adjclose"


def timestamp(date):
    return int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp())


def parse_prices(document, symbol):
    results = document.get("chart", {}).get("result")
    if not results:
        raise ValueError(f"{symbol}: missing chart data")
    result = results[0]
    if result["meta"]["symbol"] != symbol or result["meta"]["currency"] != "CNY":
        raise ValueError(f"{symbol}: expected matching China ETF in CNY")
    index = pd.to_datetime(result["timestamp"], unit="s", utc=True)
    index = index.tz_convert("Asia/Shanghai").tz_localize(None).normalize()
    price = pd.Series(result["indicators"]["adjclose"][0]["adjclose"], index=index, name=symbol, dtype=float)
    price = price.loc[(price.index >= START) & (price.index < END)]
    if price.index.has_duplicates or not price.index.is_monotonic_increasing:
        raise ValueError(f"{symbol}: invalid date ordering")
    if price.empty or not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError(f"{symbol}: invalid adjusted closes")
    if price.index[0] > pd.Timestamp("2014-01-08") or price.index[-1] < pd.Timestamp("2024-12-24"):
        raise ValueError(f"{symbol}: incomplete requested history")
    return price


def load_prices(cache_dir, refresh=False):
    """Validate and hash snapshots. Fail instead of imputing missing prices."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    series, provenance = [], []
    for symbol in SYMBOLS:
        path = cache_dir / f"{symbol}_{START}_{END}.json"
        meta_path = path.with_suffix(".meta.json")
        params = urlencode({"period1": timestamp(START), "period2": timestamp(END),
                            "interval": "1d", "events": "div,splits"})
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{params}"
        if refresh or not path.exists() or not meta_path.exists():
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            for attempt in range(3):
                try:
                    with urlopen(request, timeout=30) as response:
                        raw = response.read()
                    parse_prices(json.loads(raw), symbol)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            metadata = {"symbol": symbol, "name": ASSETS[symbol], "currency": "CNY",
                        "source": SOURCE, "price_field": PRICE_FIELD, "url": url,
                        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                        "sha256": hashlib.sha256(raw).hexdigest()}
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(raw)
            temporary.replace(path)
            meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        raw = path.read_bytes()
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
            raise ValueError(f"{symbol}: cached snapshot hash mismatch")
        price = parse_prices(json.loads(raw), symbol)
        series.append(price)
        provenance.append(metadata | {"rows": len(price)})
    prices = pd.concat(series, axis=1, join="outer")
    if prices.isna().any().any():
        raise ValueError("Asset calendars differ or data is missing; inspect raw snapshots")
    return prices, provenance
