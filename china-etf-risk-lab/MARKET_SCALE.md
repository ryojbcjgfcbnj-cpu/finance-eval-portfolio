# A 股行情与风险扫描

数据流水线为历史风险筛查提供可核验的输入。示例覆盖沪深交易所固定 24 只 A 股。按 `(symbol, trade_date)` 在 DuckDB 事务内合入数据，再发布按年份分区的 Parquet 快照。日线原始响应缓存在本地，快照保存各证券的来源与 SHA-256。批次重跑保持主键唯一，已保存价格发生变化时整批拒绝，新的来源快照可以独立构建。

```sh
python -m pip install -r requirements-scale.txt
python market_scale_lab.py --date 2024-10-08
python -m unittest -v scale_checks
python risk-workbench/build_scale_demo.py
```

`outputs/market_scale/scale_report.json` 记录数据覆盖、输入质量、首次载入与批次重跑、风险排名及压测。`outputs/market_scale/snapshots` 保存 Parquet 和快照来源。最后一条命令刷新工作台显示的汇总结果。

## 风险筛查

用观察到的全样本日期构造交易日索引。若证券两条价格之间有缺口，该跨间隔收益视作缺失。近期 20 日和此前 60 日使用不重叠的有效收益，标准差以 `sqrt(252)` 年化。任何一个窗口不够完整时，该证券不参与排名。结果是风险变化线索，样本缺口无法识别全部市场停牌或源端缺失。

## 压测口径

将真实的观察序列复制 64 份，以合成证券代码运行同一窗口算法，测量处理约 223 万条输入记录的耗时。复制数据用于单机计算压力测试。它没有独立证券之间的截面差异，压缩与缓存特性也可能和更大的真实样本不同。

三次运行分别记录 DuckDB SQL 查询耗时，以及从同一张 DuckDB 临时表读取到 pandas 后完成相同任务的耗时。后者包含物化成本。每次逐项核对结果，包含缺失值。耗时反映本机和该数据布局，运行时长与吞吐不代表单独的数据库读取性能。

历史示例的股票池固定，供应商调整价格来自当前快照。全市场历史回放还需要可追溯的逐日股票池、交易日历与供应商修订记录。

[DuckDB Parquet 文档](https://duckdb.org/docs/stable/data/parquet/overview) · [DuckDB 窗口函数](https://duckdb.org/docs/stable/sql/functions/window_functions)
