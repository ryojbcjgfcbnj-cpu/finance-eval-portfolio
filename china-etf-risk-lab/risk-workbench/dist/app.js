'use strict';

const data = window.riskSnapshot;
const labels = { historical: '历史方差', ewma: 'EWMA', har_log: '对数 HAR', har_qlike: 'QLIKE HAR', extended_qlike: '扩展 QLIKE', unscaled_equal_weight: '全仓等权' };
const assetLabels = ['沪深300', '中证500', '创业板', '上证红利', '黄金', '国债'];
const portfolios = { balanced: [25, 15, 10, 10, 15, 25], equal: [16.67, 16.67, 16.67, 16.67, 16.66, 16.66], equity: [35, 25, 20, 10, 5, 5] };
const scenarios = { equity: [-7, -9, -11, -5, 3, 1], rates: [-4, -5, -6, -3, -2, -3], gold: [-1, -1.5, -2, -0.5, -10, 0.5] };
const $ = id => document.getElementById(id);
const pct = value => `${(100 * value).toFixed(2)}%`;
const money = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 0 });
const yuan = value => `¥${money.format(value)}`;
const signed = value => `${value > 0 ? '+' : ''}${value.toFixed(4)}`;

function option(value, label = value) { return new Option(label, value); }
for (const date of Object.keys(data.snapshots)) $('date').add(option(date));
for (const key of Object.keys(data.study.forecasts)) $('model').add(option(key, labels[key]));
$('model').value = 'extended_qlike';

for (const [i, [symbol, name]] of data.assets.entries()) {
  const asset = `<span title="${name}">${assetLabels[i]}</span><small>${symbol.split('.')[0]}</small>`;
  $('positions').insertAdjacentHTML('beforeend', `<tr><th scope="row">${asset}</th><td><input id="w${i}" type="number" min="0" max="100" step="0.01" aria-label="${name} 权重百分比"></td><td id="sigma${i}"></td><td><div class="barcell"><span id="share${i}"></span><b aria-hidden="true"><i id="bar${i}"></i></b></div></td></tr>`);
  $('stresses').insertAdjacentHTML('beforeend', `<tr><th scope="row">${asset}</th><td><input id="s${i}" type="number" min="-100" step="0.1" aria-label="${name} 冲击百分比"></td><td id="stress${i}"></td></tr>`);
}

function fill(prefix, values) { values.forEach((value, i) => $(prefix + i).value = value); }
function read() {
  const weights = data.assets.map((_, i) => $('w' + i).valueAsNumber / 100);
  const shocks = data.assets.map((_, i) => $('s' + i).valueAsNumber / 100);
  if (shocks.some(shock => !Number.isFinite(shock) || shock < -1)) throw Error('各项冲击须为数字，最低为 −100%。');
  const date = $('date').value;
  const model = $('model').value;
  const nav = $('nav').valueAsNumber;
  const target = $('target').valueAsNumber / 100;
  const { variance, correlation } = data.snapshots[date];
  return { weights, shocks, date, model, nav, target, variance: variance[model],
    result: analyze(weights, variance[model], correlation, shocks, nav, target) };
}

function update() {
  $('date-label').textContent = `评估日 ${$('date').value}`;
  let state;
  try {
    state = read();
    $('error').hidden = true;
  } catch (error) {
    $('error').textContent = error.message;
    $('error').hidden = false;
    for (const id of ['vol', 'exposure', 'pnl', 'largest', 'limit', 'cash', 'pnl-percent', 'largest-share']) $(id).textContent = '—';
    data.assets.forEach((_, i) => {
      for (const prefix of ['sigma', 'share', 'stress']) $(prefix + i).textContent = '—';
      $('bar' + i).style.width = '0';
    });
    $('apply').disabled = true;
    $('download').disabled = true;
    $('action-text').textContent = '请修正输入以继续评估。';
    return;
  }
  const { result, variance, target, nav } = state;
  const within = result.scale >= 1 - 1e-9;
  const top = result.contribution.indexOf(Math.max(...result.contribution));
  $('vol').textContent = pct(result.volatility);
  $('limit').textContent = `${within ? '限额内' : '超出限额'} · 限额 ${pct(target)}`;
  $('limit').className = within ? 'positive' : 'negative';
  $('exposure').textContent = pct(result.exposure);
  $('cash').textContent = `现金 ${pct(result.cash)}`;
  $('pnl').textContent = yuan(result.pnl);
  $('pnl').className = result.pnl < 0 ? 'negative' : 'positive';
  $('pnl-percent').textContent = `占净值 ${pct(result.pnl / nav)}`;
  $('largest').textContent = result.volatility ? assetLabels[top] : '现金';
  $('largest-share').textContent = `方差贡献 ${pct(result.contribution[top])}`;
  data.assets.forEach((_, i) => {
    $('sigma' + i).textContent = pct(Math.sqrt(variance[i]));
    $('share' + i).textContent = pct(result.contribution[i]);
    $('bar' + i).style.width = pct(Math.min(1, Math.abs(result.contribution[i])));
    $('bar' + i).className = result.contribution[i] < 0 ? 'hedge' : '';
    $('stress' + i).textContent = money.format(result.stress[i]);
    $('stress' + i).className = result.stress[i] < 0 ? 'negative' : 'positive';
  });
  $('apply').disabled = within;
  $('download').disabled = false;
  $('action-text').textContent = within ? '当前组合处于波动率限额内。' :
    `保持资产相对比例，将总仓位从 ${pct(result.exposure)} 降至约 ${pct(result.exposure * result.scale)}。减少的仓位转为现金。`;
}

function renderValidation() {
  const block = $('block').value;
  $('forecast-table').innerHTML = Object.entries(data.study.forecasts).map(([name, row]) => {
    const band = row.ci95_by_block_days[block];
    const interval = name === 'ewma' ? '基准' : `[${signed(band[0])}, ${signed(band[1])}]`;
    return `<tr><th scope="row">${labels[name]}</th><td>${row.qlike.toFixed(4)}</td><td>${signed(row.delta_vs_ewma)}</td><td>${interval}</td><td>${pct(row.volatility_mae_annualized)}</td></tr>`;
  }).join('');
  const row = data.study.forecasts.extended_qlike;
  const [low, high] = row.ci95_by_block_days[block];
  $('finding').textContent = `扩展 QLIKE 相对 EWMA 的差值为 ${signed(row.delta_vs_ewma)}。${low <= 0 && high >= 0 ? '区间跨零，历史评估中的平均改善仍有不确定性。' : '该区块长度下区间未跨零。'}`;
  $('simulation-table').innerHTML = Object.entries(data.study.portfolio_by_cost_bps[$('cost').value]).map(([name, row]) =>
    `<tr><th scope="row">${labels[name]}</th><td>${pct(row.total_return)}</td><td>${pct(row.annualized_volatility)}</td><td>${pct(row.max_drawdown)}</td><td>${row.annualized_turnover.toFixed(2)}×</td></tr>`).join('');
}

for (const [id, sets, prefix] of [['preset', portfolios, 'w'], ['scenario', scenarios, 's']]) {
  $(id).addEventListener('change', () => { fill(prefix, sets[$(id).value]); update(); });
}
for (const id of ['date', 'model', 'nav', 'target']) $(id).addEventListener('input', update);
data.assets.forEach((_, i) => {
  $('w' + i).addEventListener('input', () => { $('preset').value = 'custom'; update(); });
  $('s' + i).addEventListener('input', () => { $('scenario').value = 'custom'; update(); });
});
$('apply').addEventListener('click', () => {
  const { weights, result } = read();
  fill('w', weights.map(weight => Math.floor(weight * result.scale * 10000) / 100));
  $('preset').value = 'custom';
  update();
});
document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('[data-view]').forEach(tab => {
    const active = tab === button;
    tab.setAttribute('aria-pressed', active);
    $(tab.dataset.view).hidden = !active;
  });
  $('download').hidden = button.dataset.view !== 'portfolio';
}));
$('block').addEventListener('change', renderValidation);
$('cost').addEventListener('change', renderValidation);
$('download').addEventListener('click', () => {
  const { date, model, nav, target, weights, shocks, result } = read();
  const lines = ['# 中国 ETF 组合风险简报', '', `评估日 ${date} · ${labels[model]} · 人民币`, '',
    `- 组合净值 ${yuan(nav)}`, `- 预测年化波动率 ${pct(result.volatility)}，限额 ${pct(target)}`,
    `- 资产仓位 ${pct(result.exposure)}，现金 ${pct(result.cash)}`, `- 压力情景损益 ${yuan(result.pnl)}（${pct(result.pnl / nav)}）`, '',
    '| 资产 | 权重 | 方差贡献 | 价格冲击 | 情景损益 |', '| --- | ---: | ---: | ---: | ---: |',
    ...data.assets.map(([symbol, name], i) => `| ${name} ${symbol} | ${pct(weights[i])} | ${pct(result.contribution[i])} | ${pct(shocks[i])} | ${yuan(result.stress[i])} |`),
    '', '历史快照基于 Yahoo Finance 调整收盘价。情景损益为线性重估。', ''];
  const url = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/markdown;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `ETF-risk-${date}.md`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
fill('w', portfolios.balanced);
fill('s', scenarios.equity);
update();
renderValidation();
