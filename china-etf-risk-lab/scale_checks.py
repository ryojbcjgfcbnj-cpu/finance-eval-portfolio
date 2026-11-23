import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from market_scale_lab import connect, ingest, pandas_scan, publish_parquet, scan


class MarketScaleChecks(unittest.TestCase):
    def setUp(self):
        calendar = pd.bdate_range('2023-01-01', periods=150)
        rng = np.random.default_rng(10)
        self.panel = pd.concat([
            pd.DataFrame({'symbol': symbol, 'trade_date': calendar, 'source_hash': symbol,
                          'adj_close': 100 * np.exp(rng.normal(0, .02, len(calendar)).cumsum())})
            for symbol in ('A', 'B')], ignore_index=True)
        self.cut = calendar[100]

    def test_replay_and_revision_are_safe(self):
        with connect() as db:
            self.assertEqual(ingest(db, self.panel)['inserted_rows'], 300)
            self.assertEqual(ingest(db, self.panel)['inserted_rows'], 0)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                ingest(db, pd.concat([self.panel.iloc[:1], self.panel.iloc[:1]]))
            conflicting = self.panel.iloc[:2].copy()
            conflicting.loc[0, 'adj_close'] *= 3
            with self.assertRaisesRegex(ValueError, 'revisions'):
                ingest(db, conflicting)
            self.assertEqual(db.execute('SELECT count(*) FROM market').fetchone()[0], 300)
            stored = db.execute("SELECT adj_close FROM market WHERE symbol='A' ORDER BY trade_date LIMIT 1").fetchone()[0]
            self.assertEqual(stored, self.panel.loc[0, 'adj_close'])

    def test_parquet_round_trip_and_gap_detection(self):
        missing = self.panel.drop(index=[99])
        sources = [{'symbol': name, 'sha256': name} for name in ('A', 'B')]
        with tempfile.TemporaryDirectory() as directory, connect() as db:
            ingest(db, missing)
            version, path = publish_parquet(db, Path(directory), sources)
            self.assertEqual((version, path), publish_parquet(db, Path(directory), sources))
            db.execute('CREATE TEMP TABLE panel AS SELECT * FROM read_parquet(?)', [str(path / 'yr=*' / '*.parquet')])
            result = scan(db, self.cut.date()).sort_index()
            self.assertTrue(np.isnan(result.loc['A', 'risk_ratio']))
            self.assertTrue(np.isfinite(result.loc['B', 'risk_ratio']))
            np.testing.assert_allclose(result, pandas_scan(missing, self.cut).sort_index(), atol=1e-9, equal_nan=True)

    def test_subsequent_prices_do_not_change_historical_ranking(self):
        with connect() as db:
            db.register('panel', self.panel)
            original = scan(db, self.cut.date())
            changed = self.panel.copy()
            changed.loc[changed.trade_date > self.cut, 'adj_close'] *= 5
            db.unregister('panel')
            db.register('panel', changed)
            pd.testing.assert_frame_equal(original, scan(db, self.cut.date()))


if __name__ == '__main__':
    unittest.main()
