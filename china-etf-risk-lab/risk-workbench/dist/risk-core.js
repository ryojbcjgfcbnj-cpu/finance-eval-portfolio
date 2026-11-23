function analyze(weights, variance, correlation, shocks, nav, target) {
  if (weights.some(weight => !Number.isFinite(weight) || weight < 0)) throw Error('权重必须是非负数字。');
  const exposure = weights.reduce((sum, weight) => sum + weight, 0);
  if (exposure > 1 + 1e-10) throw Error('总权重不能超过 100%。');
  if (!Number.isFinite(nav) || nav <= 0) throw Error('组合净值必须大于零。');
  if (!Number.isFinite(target) || target <= 0 || target > 1) throw Error('波动率限额须在 0% 至 100% 之间。');
  const sigma = variance.map(Math.sqrt);
  const covarianceTimesWeights = correlation.map((row, i) => row.reduce(
    (sum, rho, j) => sum + rho * sigma[i] * sigma[j] * weights[j], 0));
  const contributions = weights.map((weight, i) => weight * covarianceTimesWeights[i]);
  const portfolioVariance = contributions.reduce((sum, value) => sum + value, 0);
  const volatility = Math.sqrt(Math.max(0, portfolioVariance));
  const stress = weights.map((weight, i) => weight * shocks[i] * nav);
  return { exposure, cash: Math.max(0, 1 - exposure), volatility,
    contribution: contributions.map(value => portfolioVariance > 0 ? value / portfolioVariance : 0),
    stress, pnl: stress.reduce((sum, value) => sum + value, 0),
    scale: volatility > target ? target / volatility : 1 };
}

if (typeof module === 'object') module.exports = { analyze };
