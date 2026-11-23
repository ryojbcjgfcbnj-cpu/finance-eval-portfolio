"""Retrieve a fixed sample of Chinese A-share adjusted daily closes."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from market_data import timestamp


UNIVERSE = {
    '600519.SS': '贵州茅台', '000858.SZ': '五粮液', '600809.SS': '山西汾酒',
    '000333.SZ': '美的集团', '000651.SZ': '格力电器', '600690.SS': '海尔智家',
    '600036.SS': '招商银行', '601398.SS': '工商银行', '601288.SS': '农业银行',
    '601318.SS': '中国平安', '601601.SS': '中国太保', '601628.SS': '中国人寿',
    '600276.SS': '恒瑞医药', '000538.SZ': '云南白药', '600436.SS': '片仔癀',
    '002594.SZ': '比亚迪', '300750.SZ': '宁德时代', '600104.SS': '上汽集团',
    '600900.SS': '长江电力', '600011.SS': '华能国际', '600886.SS': '国投电力',
    '601899.SS': '紫金矿业', '600547.SS': '山东黄金', '600111.SS': '北方稀土',
}
START, END = '2019-01-01', '2025-01-01'


def fetch(symbol, cache):
    query = urlencode({'period1': timestamp(START), 'period2': timestamp(END), 'interval': '1d'})
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}'
    path = cache / f'{symbol}_{START}_{END}.json'
    if path.exists():
        raw = path.read_bytes()
    else:
        request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(request, timeout=12) as response:
            raw = response.read()
    item = json.loads(raw)['chart']['result'][0]
    meta = item['meta']
    if meta['currency'] != 'CNY' or meta['symbol'] != symbol:
        raise ValueError(f'{symbol}: symbol or currency mismatch')
    dates = pd.to_datetime(item['timestamp'], unit='s', utc=True).tz_convert('Asia/Shanghai').tz_localize(None).normalize()
    close = np.array(item['indicators']['adjclose'][0]['adjclose'], dtype=float)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError(f'{symbol}: duplicate or unordered dates')
    within = (dates >= START) & (dates < END)
    good = np.isfinite(close) & (close > 0)
    if (within & good).sum() < 250:
        raise ValueError(f'{symbol}: insufficient observations')
    if not path.exists():
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(raw)
        temporary.replace(path)
    digest = hashlib.sha256(raw).hexdigest()
    frame = pd.DataFrame({'symbol': symbol, 'trade_date': dates[within & good],
                          'adj_close': close[within & good], 'source_hash': digest})
    return frame, {'symbol': symbol, 'name': UNIVERSE[symbol], 'url': url, 'sha256': digest,
                   'rows': len(frame), 'invalid_rows': int((within & ~good).sum())}


def load_universe(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {symbol: pool.submit(fetch, symbol, cache) for symbol in UNIVERSE}
        for symbol, job in jobs.items():
            try:
                results.append(job.result())
            except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                failures.append({'symbol': symbol, 'reason': str(error)})
    if not results:
        raise RuntimeError(f'No real market data retrieved: {failures}')
    frames, provenance = zip(*results)
    return pd.concat(frames, ignore_index=True), list(provenance), failures
