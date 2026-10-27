# 第三方来源与许可

项目中的事件盘点保留固定版本源码，供课程教学、逐项引用和测试定位。`research/sources` 保存审计来源，每个框架目录附有对应原始 `LICENSE`。这些文件的版权与使用条件以各自原始文件为准。

| 来源 | 固定提交 | 原始许可文件 |
|---|---|---|
| [RQAlpha](https://github.com/ricequant/rqalpha) | `284b78a1975bd2d6c4bee8fde35586e6846bfd96` | [RQAlpha 原始许可](research/sources/rqalpha/LICENSE) |
| [LEAN](https://github.com/QuantConnect/Lean) | `a54e9cc6cee9030e865d4f401c03c0e2373ba199` | [Apache 2.0](research/sources/lean/LICENSE) |
| [Zipline](https://github.com/quantopian/zipline) | `014f1fc339dc8b7671d29be2d85ce57d3daec343` | [Apache 2.0](research/sources/zipline/LICENSE) |
| [Backtrader](https://github.com/mementum/backtrader) | `b853d7c90b6721476eb5a5ea3135224e33db1f14` | [GPL 3](research/sources/backtrader/LICENSE) |
| [Qlib](https://github.com/microsoft/qlib) | `54355232463878d2eebb91fe0ee5fa7fa1f5976c` | [MIT](research/sources/qlib/LICENSE) |

RQAlpha 的原始许可分别列出非商业使用与商业使用条件，使用其源码时请阅读该原文。上游文件保留原有版权标记；固定版本、原始 URL 和逐文件 SHA-256 记录在 [来源清单](research/sources/manifest.json)。

课程资料与数据由 [pku_quantllm](https://github.com/aslan9/pku_quantllm) 发布；行情压缩包使用课程的 [数据 Release](https://github.com/aslan9/pku_quantllm/releases/tag/data-backtest-20261016)。历史 ST 状态由本机程序通过 BaoStock 下载并保存来源与时间口径。

Python 依赖按 `pyproject.toml` 安装，其许可随各自发行包提供。项目运行代码在 `src/ashare_bt`；上游快照用于事件审计与教学引用。财务、股本及分红的公开接口与版本口径详见主报告第 3 节。
