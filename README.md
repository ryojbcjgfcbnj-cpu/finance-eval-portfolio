# Financial Evaluation Pipeline

**A small, runnable example of audit-friendly evaluation for financial AI.**

The demo processes three synthetic checks using supported, conflicting and insufficient-evidence states. A simulated judge encounters temporary failures, follows a bounded exponential backoff schedule, then uses a backup. A separate preflight check flags oversized evidence using a UTF-8 byte budget. The JSON report records the scenario, actual decision source and retry events.

## Run locally

```sh
python -m unittest -v
python demo.py
```

Python 3.10 or newer. No external dependencies, API keys or network calls. The fault events are simulated and the wait schedule is recorded without sleeping.

## What to look at

| File | Demonstrates |
| --- | --- |
| `demo.py` | Three-way evidence states, bounded retries, fallback and context preflight |
| `test_demo.py` | Regression checks for abstention, retry limits, permanent failures and a Unicode boundary |
| `example-report.json` | Structured output produced by running the demo |

The public examples are synthetic. This repository demonstrates engineering behavior and provides no financial accuracy or hosted-provider performance benchmark.

## 中文说明

这是「较真一下」的合成数据评测示例。可本地运行四项回归测试，查看证据不足、限流重试、备用路径和超长输入如何进入结构化报告。示例全部为合成数据，长文本使用 UTF-8 字节预算。
