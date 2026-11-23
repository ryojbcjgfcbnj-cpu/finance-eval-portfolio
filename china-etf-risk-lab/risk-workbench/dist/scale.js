'use strict';

const {real_data: market, ingest, scan, benchmark} = window.marketScale;
const field = id => document.getElementById(id);
const put = (id, value) => field(id).textContent = value;
const whole = value => value.toLocaleString('zh-CN', {maximumFractionDigits: 0});
const pct = value => `${(100 * value).toFixed(2)}%`;
const seconds = value => `${value.toFixed(3)} 秒`;
put('period', market.dates.join(' 至 '));
put('release', `快照 ${market.snapshot_id}`);
put('real-rows', whole(market.rows));
put('real-symbols', `${market.symbols} 只股票 · ${market.partitions} 个年份分区`);
put('inserted', whole(ingest.increment.inserted_rows));
put('replay', `同批重跑新增 ${whole(ingest.replay.inserted_rows)} 条`);
put('bench-rows', whole(benchmark.rows));
put('runtime', seconds(benchmark.sql_median_seconds));
put('throughput', `约 ${whole(benchmark.sql_rows_per_second)} 输入行/秒`);
put('coverage', `${scan.date} · 覆盖 ${scan.observed} 只 · ${scan.eligible} 只有效 · 展示前 ${scan.ranking.length} 名`);

const audit = [
  ['源请求失败', `${market.source_failures.length} / ${market.requested_symbols}`],
  ['无效收盘价记录', whole(market.invalid_close_rows)],
  ['真实行情读取与扫描', seconds(scan.parquet_read_seconds + scan.seconds)],
  ['压测运行次数', `${benchmark.sql_seconds.length} 次`],
  ['pandas 中位耗时（含读取）', seconds(benchmark.pandas_median_seconds)],
  ['DuckDB 与 pandas 结果交叉核对', benchmark.results_match ? '一致' : '有差异'],
];

function appendRows(id, rows) {
  const body = field(id);
  body.replaceChildren();
  for (const row of rows) {
    const tr = body.insertRow();
    const header = document.createElement('th');
    header.scope = 'row';
    header.textContent = row[0];
    tr.append(header);
    for (const value of row.slice(1)) tr.insertCell().textContent = value;
  }
}

function updateRanking() {
  const threshold = Number(field('threshold').value);
  const rows = scan.ranking.filter(item => item.risk_ratio >= threshold).map(item => [
    `${item.name} ${item.symbol.split('.')[0]}`, pct(item.annual_vol20), pct(item.annual_baseline60), `${item.risk_ratio.toFixed(2)}×`,
  ]);
  appendRows('ranks', rows.length ? rows : [['前列样本中无符合条件的证券', '—', '—', '—']]);
}

field('threshold').addEventListener('change', updateRanking);
appendRows('audit', audit);
updateRanking();
