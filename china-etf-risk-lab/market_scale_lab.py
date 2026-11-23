"""Build an A-share Parquet snapshot, screen risk, and benchmark SQL windows."""

import argparse
import hashlib
import json
import statistics
import tempfile
from pathlib import Path
from time import perf_counter

import duckdb
import numpy as np
import pandas as pd

from market_scale_data import UNIVERSE, load_universe


ROOT = Path(__file__).resolve().parent
SCAN_SQL = """
WITH calendar AS (
    SELECT trade_date, row_number() OVER (ORDER BY trade_date) AS day_number
    FROM (SELECT DISTINCT trade_date FROM panel)
), returns AS (
    SELECT symbol, trade_date,
           CASE WHEN day_number - lag(day_number) OVER asset = 1
                THEN adj_close / lag(adj_close) OVER asset - 1 END AS ret
    FROM panel JOIN calendar USING (trade_date)
    WINDOW asset AS (PARTITION BY symbol ORDER BY trade_date)
), rolling AS (
    SELECT symbol, trade_date,
           CASE WHEN count(ret) OVER recent = 20 THEN stddev_samp(ret) OVER recent END AS vol20,
           CASE WHEN count(ret) OVER prior = 60 THEN stddev_samp(ret) OVER prior END AS baseline60
    FROM returns
    WINDOW recent AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
           prior AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 79 PRECEDING AND 20 PRECEDING)
)
SELECT symbol, vol20 * sqrt(252) AS annual_vol20,
       baseline60 * sqrt(252) AS annual_baseline60, vol20 / nullif(baseline60, 0) AS risk_ratio
FROM rolling WHERE trade_date = ?
ORDER BY risk_ratio DESC NULLS LAST, symbol
"""


def connect(path=':memory:'):
    db = duckdb.connect(str(path))
    db.execute('SET threads = 4')
    return db


def ingest(db, rows):
    """Idempotent primary-key merge. Freeze past observations on revision conflicts."""
    db.execute('''CREATE TABLE IF NOT EXISTS market (
        symbol VARCHAR NOT NULL, trade_date DATE NOT NULL,
        adj_close DOUBLE NOT NULL CHECK (isfinite(adj_close) AND adj_close > 0),
        source_hash VARCHAR NOT NULL, PRIMARY KEY (symbol, trade_date))''')
    db.register('incoming', rows)
    started = perf_counter()
    try:
        db.begin()
        duplicate = db.execute('''SELECT count(*) FROM (
            SELECT symbol, trade_date FROM incoming GROUP BY ALL HAVING count(*) > 1)''').fetchone()[0]
        if duplicate:
            raise ValueError(f'{duplicate} duplicate input keys')
        changed = db.execute('''SELECT count(*) FROM market JOIN incoming USING (symbol, trade_date)
            WHERE market.adj_close IS DISTINCT FROM incoming.adj_close''').fetchone()[0]
        if changed:
            raise ValueError(f'{changed} historical price revisions; ingest into a new snapshot')
        count = db.execute('''INSERT INTO market
            SELECT symbol, trade_date, adj_close, source_hash FROM incoming
            ON CONFLICT (symbol, trade_date) DO NOTHING''').fetchone()[0]
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.unregister('incoming')
    return {'input_rows': len(rows), 'inserted_rows': count, 'seconds': perf_counter() - started}


def scan(db, day):
    return db.execute(SCAN_SQL, [day]).df().set_index('symbol')


def pandas_scan(panel, day):
    frame = panel.sort_values(['symbol', 'trade_date']).copy()
    calendar = pd.Index(frame.trade_date.unique()).sort_values()
    frame['day_number'] = calendar.get_indexer(frame.trade_date)
    by_symbol = frame.groupby('symbol', sort=False)
    ret = by_symbol.adj_close.pct_change(fill_method=None).where(by_symbol.day_number.diff() == 1)
    grouped = ret.groupby(frame.symbol, sort=False)
    vol20 = grouped.rolling(20).std().droplevel(0)
    baseline = grouped.rolling(60).std().droplevel(0).groupby(frame.symbol, sort=False).shift(20)
    frame['annual_vol20'] = vol20 * np.sqrt(252)
    frame['annual_baseline60'] = baseline * np.sqrt(252)
    frame['risk_ratio'] = vol20 / baseline.replace(0, np.nan)
    return frame.loc[frame.trade_date == pd.Timestamp(day),
                     ['symbol', 'annual_vol20', 'annual_baseline60', 'risk_ratio']].set_index('symbol')


def publish_parquet(db, output, sources):
    """Publish an immutable, content-addressed release after all ingestion succeeds."""
    source_ids = sorted((item['symbol'], item['sha256']) for item in sources)
    version = hashlib.sha256(json.dumps(['market-v1', source_ids]).encode()).hexdigest()[:16]
    destination = output / 'snapshots' / version
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
            staging = Path(temp) / 'release'
            db.execute('''COPY (SELECT *, year(trade_date) AS yr FROM market ORDER BY symbol, trade_date)
                TO ? (FORMAT PARQUET, PARTITION_BY (yr), COMPRESSION ZSTD)''', [str(staging)])
            (staging / 'sources.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding='utf-8')
            try:
                staging.rename(destination)
            except FileExistsError:
                if not destination.is_dir():
                    raise
    return version, destination


def benchmark(db, day, factor):
    """Replicate real inputs under synthetic IDs solely for computing load."""
    db.execute('''CREATE OR REPLACE TEMP TABLE benchmark_panel AS
        SELECT symbol || ':' || i::VARCHAR AS symbol, trade_date, adj_close
        FROM market CROSS JOIN range(?) AS series(i)''', [factor])
    rows, symbols = db.execute('SELECT count(*), count(DISTINCT symbol) FROM benchmark_panel').fetchone()
    db.execute('CREATE OR REPLACE TEMP VIEW panel AS SELECT * FROM benchmark_panel')
    sql_times, pandas_times = [], []
    for _ in range(3):
        began = perf_counter()
        sql = scan(db, day)
        sql_times.append(perf_counter() - began)
        began = perf_counter()
        frame = db.execute('SELECT * FROM benchmark_panel').df()
        pandas = pandas_scan(frame, day)
        pandas_times.append(perf_counter() - began)
        pd.testing.assert_index_equal(sql.sort_index().index, pandas.sort_index().index)
        np.testing.assert_allclose(sql.sort_index(), pandas.sort_index(), atol=1e-8, rtol=1e-7, equal_nan=True)
    median = statistics.median(sql_times)
    return {'rows': rows, 'series': symbols, 'replication_factor': factor,
            'sql_seconds': sql_times, 'pandas_seconds_including_read': pandas_times,
            'sql_median_seconds': median, 'pandas_median_seconds': statistics.median(pandas_times),
            'sql_rows_per_second': rows / median, 'results_match': True}


def run(output, day, factor):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, sources, failures = load_universe(ROOT / 'data' / 'cache' / 'scale')
    day = pd.Timestamp(day).normalize()
    if day not in pd.DatetimeIndex(rows.trade_date):
        raise ValueError('scan date has no source observations')
    if factor < 1:
        raise ValueError('replication factor must be positive')
    with connect() as db:
        before = rows[rows.trade_date < '2024-01-01']
        after = rows[rows.trade_date >= '2024-01-01']
        batches = {'initial': ingest(db, before), 'increment': ingest(db, after), 'replay': ingest(db, after)}
        version, snapshot = publish_parquet(db, output, sources)
        partition = str(snapshot / 'yr=*' / '*.parquet')
        began = perf_counter()
        db.execute('CREATE OR REPLACE TEMP TABLE visible AS SELECT symbol, trade_date, adj_close FROM '
                   'read_parquet(?, hive_partitioning=true) WHERE yr <= ? AND trade_date <= ?',
                   [partition, day.year, day.date()])
        read_seconds = perf_counter() - began
        db.execute('CREATE TEMP VIEW panel AS SELECT * FROM visible')
        began = perf_counter()
        ranking = scan(db, day.date())
        scan_seconds = perf_counter() - began
        eligible = ranking.dropna(subset=['risk_ratio'])
        top = eligible.head(10).reset_index()
        top.insert(1, 'name', top.symbol.map(UNIVERSE))
        result = {'real_data': {'rows': len(rows), 'symbols': len(sources), 'requested_symbols': len(UNIVERSE),
                                'dates': [str(rows.trade_date.min().date()), str(rows.trade_date.max().date())],
                                'invalid_close_rows': sum(item['invalid_rows'] for item in sources),
                                'source_failures': failures, 'snapshot_id': version,
                                'partitions': len(list(snapshot.glob('yr=*')))},
                  'ingest': batches,
                  'scan': {'date': str(day.date()), 'observed': len(ranking), 'eligible': len(eligible),
                           'parquet_read_seconds': read_seconds,
                           'above_1_5': int((eligible.risk_ratio >= 1.5).sum()), 'seconds': scan_seconds,
                           'ranking': top.to_dict(orient='records')},
                  'benchmark': benchmark(db, day.date(), factor),
                  'runtime': {'duckdb': duckdb.__version__, 'pandas': pd.__version__, 'duckdb_threads': 4}}
    (output / 'scale_report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(output / 'scale_report.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', default='2024-10-08')
    parser.add_argument('--factor', type=int, default=64, help='replication factor for load test only')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'market_scale')
    args = parser.parse_args()
    run(args.output, args.date, args.factor)
