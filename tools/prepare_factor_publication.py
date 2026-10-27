"""为十因子研究生成独立、可校验的 GitHub 发布副本和 Issue 摘要。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/celestial-toymaker/factor-search"
DESTINATION = ROOT / "work/factor-search-publication/repository"
TABLES = ("orders", "trades", "journal", "equity", "positions", "fees", "exits")
RUN_FILES = ("report.md", "equity-curve.png", "summary.json", "config.json", "manifest.json",
             "data_profile.json", "status_coverage.json", "equity.csv", "exit_summary.csv", "input_events.jsonl")
TOOLS = ("run_factor_study.py", "check_factor_study.py", "prepare_status.py",
         "prepare_factor_publication.py", "check_publication.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def public_files(root):
    return sorted(path for path in root.rglob("*") if path.is_file()
                  and not any(part in {"__pycache__", ".pytest_cache", ".git"} or part.endswith(".egg-info")
                              for part in path.relative_to(root).parts)
                  and path.suffix not in {".pyc", ".pyo"} and path.name != ".DS_Store")


def clean_value(value):
    if isinstance(value, str):
        return value.replace(str(ROOT) + "/", "").replace(str(ROOT), ".")
    if isinstance(value, list):
        return [clean_value(item) for item in value]
    if isinstance(value, dict):
        return {clean_value(key): clean_value(item) for key, item in value.items()}
    return value


def factor_index(summary, prefix=""):
    return "\n".join(f"| {item['factor_name']} | [报告]({prefix}runs/{item['run_id']}/report.md) | "
                     f"[配置]({prefix}runs/{item['run_id']}/config.json) | "
                     f"[精确指标]({prefix}runs/{item['run_id']}/summary.json) |"
                     for item in summary["ranking"])


def readme(summary):
    return f'''# A 股十大经典因子研究

统一评估价值、小市值、中期动量、短期反转、低波动、质量、成长、红利、流动性溢价和保守投资。研究期间为 **2025-01-02 至 2026-09-30**，共 424 个交易日；初始资金 10 亿元，最多 100 只股票，95% 目标仓位，每 5 个交易日调仓。

小市值在本样本的扣费后夏普排名首位：区间净收益 **39.48%**，最大回撤 **10.84%**，夏普 **1.670**，平均实际仓位 **42.86%**。本研究采用免费公开数据的保守探索性口径，完整解释见主报告。

## 1. 报告入口

- [完整主报告](reports/classic-factors/report.md)：净收益、Rank IC、理论分组、因子相关性、覆盖、成交容量、分期结果及完整来源。
- [完整指标 CSV](reports/classic-factors/comparison.csv)、[逐日净值](reports/classic-factors/equity.csv)、[财务与请求来源索引](reports/classic-factors/sources.json)。
- [研究核验](reports/classic-factors/validation.json)、[报告核验](reports/classic-factors/artifact_validation.json)、[本次发布测试](validation/classic-factors-tests.xml)。
- [文件校验清单](PUBLICATION-MANIFEST.json)、[发布转换与原始文件哈希](validation/publication-lineage.json)、[第三方来源与许可](THIRD_PARTY_NOTICES.md)。

![净值与回撤](reports/classic-factors/equity_drawdown.png)

<a id="factor-reports"></a>

## 2. 单因子报告目录

| 因子 | 单因子报告 | 运行配置 | 精确指标 |
|---|---|---|---|
{factor_index(summary)}

每个运行目录发布订单、成交、账簿、净值、持仓、费用和退出记录七类 Parquet 表，以及退出摘要、源码快照、设备与输入指纹。完整本地执行还生成逐日选股与事件诊断缓存；重新运行研究可生成这些文件。

## 3. 数据来源与时间口径

| 数据 | 来源与实际字段 |
|---|---|
| 行情、成交、复权 | [课程 Release data-backtest-20261016](https://github.com/aslan9/pku_quantllm/releases/tag/data-backtest-20261016) 的 open、high、low、close、amt、num_trades、cumadj 七个 Feather 矩阵；底层行情供应商以课程发布说明为溯源边界 |
| 股本 | [东方财富数据接口](https://datacenter-web.eastmoney.com/api/data/v1/get)，RPT_VALUEANALYSIS_DET / TOTAL_SHARES；市值由本地原始收盘价乘股本计算 |
| 资产负债表 | RPT_F10_FINANCE_GBALANCE、BBALANCE、IBALANCE、SBALANCE；TOTAL_ASSETS、TOTAL_PARENT_EQUITY |
| 利润表 | RPT_F10_FINANCE_GINCOME；TOTAL_OPERATE_INCOME（营业总收入）、PARENT_NETPROFIT |
| 已实施分红 | RPT_SHAREBONUS_DET；ASSIGN_PROGRESS、PRETAX_BONUS_RMB、TOTAL_SHARES、EX_DIVIDEND_DATE、NOTICE_DATE 等 |
| 历史 ST 状态 | BaoStock 0.9.4 / query_daily_history_k_AStock 的 isST；按当日开盘前可知推定 |

行情压缩包发布 SHA-256：`5766950f2a105244867b842709cf4bad890e4e11df0eebeeaa4fec42dc05803d`。行情实际加载 2023-01-03 至 2026-09-30，研究前数据用于预热。

东方财富抓取时间为 **2026-10-27 11:42:46 至 12:04:27（Asia/Seoul，UTC+09:00）**，536 组查询、1,469 页响应；财报期为 2021-03-31 至 2026-06-30。请求 URL、参数、时间、原始缓存相对路径与哈希见来源索引。行情、财务响应、归一化特征及状态原始缓存按来源说明准备，数据文件的原始哈希保留在已发布记录中。

财务记录自 max(REPORT_DATE, NOTICE_DATE, UPDATE_DATE) 日期结束后的下一交易日进入信号。当前快照与时间推迟构成历史版本近似，严格历史时点研究需要完整修订记录。财务计算采用结构化资产负债表和利润表；现金流量表未参与计算，公司原始财报 PDF 未逐份提取校对。共同股票池筛选、信息滞后、日线开盘成交、容量及费用假设均影响结果。主报告第 3、4、6 节提供全部字段、时间范围、交易规则与因子定义参考。

## 4. 安装与结果核验

使用 Apple Silicon Mac、Python 3.11 和可用的 Apple MPS。批量因子计算使用 MPS，金额与费用使用 Decimal。先在仓库根目录执行：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -c "import torch; assert torch.backends.mps.is_available(); print(torch.__version__, 'MPS available')"
.venv/bin/python tools/check_publication.py
.venv/bin/python -m pytest --junitxml=validation/classic-factors-tests.xml
.venv/bin/python tools/check_factor_study.py
```

文件校验以原始克隆内容为基准；本地测试或报告重建会更新验收时间及部分文件哈希，可在执行这些命令之前单独运行 `tools/check_publication.py`。

研究原始验证已对 10 项策略逐一重放七类表；`validation.json` 保存这次完整重放的证据。本次发布副本另行运行测试和报告数据核验。研究指纹为 `{summary['study_fingerprint']}`。

## 5. 数据准备与完整重算

建议使用独立克隆副本，例如 `factor-search-reproduce`。先按课程 Release 下载并核对压缩包哈希，将七个 Feather 文件放入根目录 `feathers/`。再运行：

```bash
.venv/bin/python tools/prepare_status.py
.venv/bin/python tools/run_factor_study.py --config configs/classic_factors.toml --prepare-only
```

首条命令取得历史 ST 状态，第二条命令取得并校验东方财富财务、股本和分红响应。已有匹配的冻结缓存可按来源索引中的相对路径放置。公开接口的后续更新可能改变覆盖、因子值和运行指纹，精确重现历史数值要求全部原始输入哈希一致。

将已发布的结果目录成对移入基准目录，保留其内部相对链接，然后生成新结果并逐表重放：

```bash
mkdir baseline
mv reports runs baseline/
.venv/bin/python tools/run_factor_study.py --config configs/classic_factors.toml --verify-replay
.venv/bin/python tools/check_factor_study.py
```

配置采用收盘决策、次日开盘复核、T+1、历史数量与涨跌幅规则、先卖后买；单证券共享容量为决策日成交额与此前 20 日均成交额较小值的 1%。末日按 strict_end 尝试退出，剩余持仓按有效行情估值。配置文件列明佣金、最低佣金、滑点、印花税及容量参数。

## 6. 报告重建与发布指纹

仅使用已发布净值、成交和诊断表即可重建报告与图表：

```bash
.venv/bin/python tools/run_factor_study.py --config configs/classic_factors.toml --report-only
.venv/bin/python tools/check_factor_study.py
```

原始运行指纹、输入数据哈希及源码快照哈希描述研究执行时的材料；发布校验清单描述仓库中的文件。发布过程将本机工作区绝对路径转换为相对路径、将测试主机名统一为 `apple-silicon-mps`，并刷新受影响的文件校验清单。原始文件哈希与转换范围见发布转换记录。收益、费用、持仓、配置参数和原始输入指纹沿用本次研究结果。
'''


def comment(summary, sha, test_count):
    base = f"{REPOSITORY}/blob/{sha}"
    raw = f"https://raw.githubusercontent.com/celestial-toymaker/factor-search/{sha}"
    rows = "\n".join(f"| {i} | {item['factor_name']} | {item['return']:.2%} | {item['max_drawdown']:.2%} | "
                     f"{item['sharpe']:.3f} | {item['average_exposure']:.2%} | {item['rank_ic_mean']:.3f} |"
                     for i, item in enumerate(summary["ranking"], 1))
    return f'''## A 股十大经典因子：选股能力与实际交易表现

本次对价值、小市值、中期动量、短期反转、低波动、质量、成长、红利、流动性溢价、保守投资做统一口径的探索性评估，比较因子排序效果与实际账户收益。完整报告及复现材料已上传至 [factor-search]({REPOSITORY})。

### 研究设置

2025-01-02 至 2026-09-30，共 **424 个交易日**；初始资金 **10 亿元**，最多 **100 只**股票，目标股票仓位 **95%**，每 **5 个交易日**调仓。十因子采用共同有效股票池，基础条件含非 ST、历史长度与成交额要求；共同股票池日均 3,721 只。

收盘决策、次日原始开盘价复核，执行 T+1、数量与涨跌幅规则、先卖后买。佣金单边万一且每单最低 5 元、卖出印花税万五、独立滑点万五，另计历史过户费。单证券共享容量为决策日成交额与此前 20 日均成交额较小值的 **1%**。末日按规则尝试退出，剩余持仓按有效行情估值。因子计算使用 Apple MPS，金额与费用采用 Decimal 记账。

### 主报告结果

按扣费后夏普排序；年化按 252 个交易日，夏普的无风险利率设为零。最大回撤列表示回落幅度。

| 排名 | 因子 | 区间净收益 | 最大回撤 | 夏普 | 平均实际仓位 | 平均 Rank IC |
|---:|---|---:|---:|---:|---:|---:|
{rows}

Rank IC 为因子排序与随后 5 日收益排序的相关性，标签采用次日开盘至第六日开盘，逐日标签相互重叠。共同股票池等权理论参照区间收益 **38.82%**，采用零费用、分数份额与充足容量假设；实际账户收益包含费用与成交限制。

![十因子净值与回撤]({raw}/reports/classic-factors/equity_drawdown.png)

![Rank IC]({raw}/reports/classic-factors/rank_ic.png)

### 主要发现

1. **小市值的样本内净夏普最高。** 净收益 39.48%、最大回撤 10.84%、夏普 1.670；平均实际仓位 42.86%，容量限制明显影响资金利用率。
2. **预测排序与实际可交易收益需要联合评价。** 流动性溢价平均 Rank IC 为 0.052，理论 Q5 收益 73.89%；实际净收益 15.92%，平均仓位 20.42%。理论分组与实际组合的证券数量、费用和容量条件存在差异。
3. **正 IC 仍可能对应负净收益。** 短期反转平均 Rank IC 为 0.029，实际净收益 −2.36%，累计交易成本约 3,483.84 万元；换手与成交条件会改变策略的经济表现。

### 数据来源与解释边界

- **行情、成交、复权：** [课程 Release data-backtest-20261016](https://github.com/aslan9/pku_quantllm/releases/tag/data-backtest-20261016) 的七个 Feather 矩阵；加载 2023-01-03 至 2026-09-30，前段用于预热。原包 SHA-256 及逐文件指纹列于主报告和输入清单。
- **财报、股本、分红：** [东方财富数据接口](https://datacenter-web.eastmoney.com/api/data/v1/get)，使用历史总股本、一般企业及金融行业资产负债表、利润表、已实施现金分红。财务字段包括 TOTAL_ASSETS、TOTAL_PARENT_EQUITY、TOTAL_OPERATE_INCOME、PARENT_NETPROFIT。财报期 2021-03-31 至 2026-06-30；于 2026-10-27 抓取 536 组查询、1,469 页响应。
- **历史 ST：** BaoStock 0.9.4 / query_daily_history_k_AStock 的逐日 isST 缓存，按开盘前可知推定。

财务记录从 max(REPORT_DATE, NOTICE_DATE, UPDATE_DATE) 日期结束后的下一交易日进入信号。当前快照加保守时间推迟构成历史版本近似，**尚非严格历史时点数据库**；公司原始财报 PDF 未逐份提取校对，现金流量表未参与计算。共同有效样本的财报年龄中位数再取日均为 379 天。覆盖选择、财报滞后、重叠标签和日线成交假设均限制结论的推广，本次结果属于所选区间与冻结数据上的探索性比较。

### 完整材料与复现

- [完整主报告]({base}/reports/classic-factors/report.md)（含五分组、相关性、覆盖、分期表现及全部数据来源）。
- [10 份单因子报告目录]({base}/README.md#factor-reports)（含退出、费用、持仓与逐笔核对表）。
- [复现代码与运行说明]({base}/README.md#4-安装与结果核验)、[研究配置]({base}/configs/classic_factors.toml)、[原始请求索引]({base}/reports/classic-factors/sources.json)。
- [研究与七类表重放证据]({base}/reports/classic-factors/validation.json)、[发布副本测试记录]({base}/validation/classic-factors-tests.xml)：{test_count} 项测试通过；10 项策略的订单、成交、账簿、净值、持仓、费用和退出记录已逐表核对。

以上材料固定于提交 `{sha}`。
'''


def finalize(destination):
    test_path = destination / "validation/classic-factors-tests.xml"
    test_content = test_path.read_text(encoding="utf-8")
    test_content = re.sub(r'hostname="[^"]*"', 'hostname="apple-silicon-mps"', test_content)
    test_path.write_text(clean_value(test_content), encoding="utf-8")
    lineage_path = destination / "validation/publication-lineage.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["transformed_files"] = [name for name, original in lineage["original_file_sha256"].items()
                                    if digest(destination / name) != original]
    write_json(lineage_path, lineage)
    main = destination / "reports/classic-factors"
    manifest = json.loads((main / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = {path.name: digest(path) for path in sorted(main.iterdir())
                         if path.is_file() and path.name != "manifest.json"}
    write_json(main / "manifest.json", manifest)
    entries = {}
    for path in public_files(destination):
        relative = path.relative_to(destination).as_posix()
        if relative == "PUBLICATION-MANIFEST.json":
            continue
        data = path.read_bytes()
        entries[relative] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                             "git_blob_sha": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()}
    write_json(destination / "PUBLICATION-MANIFEST.json", {
        "format": 1, "repository": REPOSITORY, "study_fingerprint":
        json.loads((main / "summary.json").read_text(encoding="utf-8"))["study_fingerprint"],
        "file_count": len(entries), "total_bytes": sum(item["bytes"] for item in entries.values()), "files": entries})
    print(json.dumps({"directory": str(destination), "files": len(entries) + 1,
                      "bytes": sum(item["bytes"] for item in entries.values())}, ensure_ascii=False))


def prepare(destination):
    if destination.exists():
        raise FileExistsError(f"发布目录已存在，请核验后选择新的目标：{destination}")
    summary = json.loads((ROOT / "reports/classic-factors/summary.json").read_text(encoding="utf-8"))
    files = set()
    for name in ("src/ashare_bt", "tests", "configs", "examples", "research", "reports/classic-factors"):
        files.update(path for path in public_files(ROOT / name))
    files.update(ROOT / "tools" / name for name in TOOLS)
    files.update(ROOT / name for name in ("pyproject.toml", "THIRD_PARTY_NOTICES.md", "validation/classic-factors-tests.xml"))
    for item in summary["ranking"]:
        directory = ROOT / "runs" / item["run_id"]
        files.update(directory / name for name in RUN_FILES)
        files.update(directory / f"{name}.parquet" for name in TABLES)
        files.update(public_files(directory / "source"))
    origins, transformed = {}, []
    for source in sorted(files):
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        origins[relative.as_posix()] = digest(source)
        if target.suffix == ".json":
            value = json.loads(target.read_text(encoding="utf-8"))
            cleaned = clean_value(value)
            if cleaned != value:
                write_json(target, cleaned)
                transformed.append(relative.as_posix())
        elif target.suffix == ".xml":
            original = target.read_text(encoding="utf-8")
            cleaned = re.sub(r'hostname="[^"]*"', 'hostname="apple-silicon-mps"', original)
            cleaned = clean_value(cleaned)
            if original != cleaned:
                target.write_text(cleaned, encoding="utf-8")
                transformed.append(relative.as_posix())
    for item in summary["ranking"]:
        directory = destination / "runs" / item["run_id"]
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["publication"] = {"original_manifest_sha256": origins[manifest_path.relative_to(destination).as_posix()],
                                   "files": {path.relative_to(directory).as_posix(): digest(path)
                                             for path in public_files(directory) if path != manifest_path}}
        write_json(manifest_path, manifest)
    notices = (destination / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    notices = notices.replace("Python 和前端依赖按 `pyproject.toml` 与 `web/package-lock.json` 安装，其许可随各自发行包提供。项目运行代码在 `src/ashare_bt`，浏览器界面在 `web/src`；上游快照用于事件审计与教学引用。",
                              "Python 依赖按 `pyproject.toml` 安装，其许可随各自发行包提供。项目运行代码在 `src/ashare_bt`；上游快照用于事件审计与教学引用。财务、股本及分红的公开接口与版本口径详见主报告第 3 节。")
    (destination / "THIRD_PARTY_NOTICES.md").write_text(notices, encoding="utf-8")
    (destination / "README.md").write_text(readme(summary), encoding="utf-8")
    (destination / ".gitignore").write_text(".DS_Store\n.venv/\n__pycache__/\n*.py[cod]\n*.egg-info/\n.pytest_cache/\nfeathers/\ndata/\nwork/\nbaseline/\n.env\n.env.*\n", encoding="utf-8")
    write_json(destination / "validation/publication-lineage.json", {
        "study_fingerprint": summary["study_fingerprint"], "original_file_sha256": origins,
        "path_normalization": "工作区绝对前缀转换为仓库根目录相对路径；测试主机名统一为 apple-silicon-mps",
        "normalized_files": transformed, "original_evidence": "运行、原始输入及源码指纹描述原始研究；PUBLICATION-MANIFEST.json 描述发布文件",
        "included_run_tables": list(TABLES), "large_inputs": "行情、财务原始响应、特征与历史状态缓存按主报告来源和哈希准备"})
    finalize(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--comment-sha")
    parser.add_argument("--comment-output", type=Path, default=ROOT / "work/factor-search-publication/issue-comment.md")
    args = parser.parse_args()
    if args.comment_sha:
        if not re.fullmatch(r"[0-9a-f]{40}", args.comment_sha):
            raise ValueError("评论链接需要完整的 40 位提交 SHA")
        summary = json.loads((args.destination / "reports/classic-factors/summary.json").read_text(encoding="utf-8"))
        suites = ET.parse(args.destination / "validation/classic-factors-tests.xml").getroot().iter("testsuite")
        test_count = sum(int(suite.get("tests", "0")) for suite in suites)
        args.comment_output.parent.mkdir(parents=True, exist_ok=True)
        args.comment_output.write_text(comment(summary, args.comment_sha, test_count), encoding="utf-8")
        print(str(args.comment_output))
    elif args.finalize:
        finalize(args.destination)
    else:
        prepare(args.destination)


if __name__ == "__main__":
    main()
