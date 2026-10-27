"""核验经典因子报告、净值金额、费用、链接与重放证据。"""
from decimal import Decimal
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image

from ashare_bt.utils import atomic_json, sha256


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / "reports/classic-factors"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    validation_path = directory / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    sources = json.loads((directory / "sources.json").read_text(encoding="utf-8"))
    assert validation["completed"] == validation["total"] == len(summary["ranking"]) == 10
    assert len(sources["manifests"]) == sources["requested"] and sources["errors"] == []
    assert all(len(record["replay_tables_identical"]) == 7 for record in validation["runs"])
    assert validation["device"]["device"] == "mps"
    comparison = pd.read_csv(directory / "comparison.csv").set_index("factor")
    documents = [directory / "report.md"]
    checked = []
    for item in summary["ranking"]:
        run_dir = root / "runs" / item["run_id"]
        documents.append(run_dir / "report.md")
        equity = pd.read_parquet(run_dir / "equity.parquet")
        fees = pd.read_parquet(run_dir / "fees.parquet")
        assert len(equity) == 424
        assert equity.date.iloc[0] == "2025-01-02" and equity.date.iloc[-1] == "2026-09-30"
        nav = equity.unit_nav.to_numpy(float)
        assert abs(nav[-1] - 1 - item["return"]) < 1e-12
        curve = np.r_[1., nav]
        drawdown = float(-(curve / np.maximum.accumulate(curve) - 1).min())
        assert abs(drawdown - item["max_drawdown"]) < 1e-12
        costs = sum((Decimal(value) for value in fees.amount), Decimal(0))
        assert costs == Decimal(item["total_costs"])
        for record in equity.to_dict("records"):
            expected = sum((Decimal(record[key]) for key in ("cash", "market_value", "receivables", "pending_equity")), Decimal(0)) - Decimal(record["liabilities"])
            assert abs(expected - Decimal(record["nav"])) <= Decimal("0.01")
        for field in ("return", "annual_return", "max_drawdown", "sharpe", "average_exposure", "rank_ic_mean"):
            assert abs(float(comparison.loc[item["factor"], field]) - item[field]) < 1e-12
        assert Decimal(item["ending_nav"]) - Decimal(item["initial_cash"]) == Decimal(item["net_profit"])
        actual = item["actual_exit"]
        exits = pd.read_csv(run_dir / "exit_summary.csv", dtype=str, keep_default_na=False)
        terminal = exits.loc[exits.planned_start == item["end"]]
        trades = pd.read_parquet(run_dir / "trades.parquet")
        terminal_sells = trades.loc[(trades.date == item["end"]) & (trades.side == "sell") &
                                   (trades.reason == "按规则计划退出") & trades.security.isin(terminal.security)]
        assert len(terminal) == actual["terminal_planned_securities"]
        assert terminal_sells.security.nunique() == actual["terminal_sold_securities"]
        assert sum((Decimal(value) for value in terminal_sells.notional), Decimal(0)) == Decimal(actual["terminal_gross_proceeds"])
        assert sum(Decimal(value) > 0 for value in exits.remaining_quantity) == actual["remaining_securities"] == item["ending_positions"]
        assert sum((Decimal(value) for value in exits.remaining_market_value), Decimal(0)) == Decimal(actual["remaining_market_value"]) == Decimal(item["ending_market_value"])
        checked.append(item["run_id"])
    links = 0
    for path in documents:
        content = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if target.startswith(("http:", "https:", "#")):
                continue
            assert (path.parent / target).resolve().exists(), (str(path), target)
            links += 1
        if path.parent != directory:
            headings = re.findall(r"^## (\d+)\.", content, re.MULTILINE)
            assert headings == [str(number) for number in range(1, 8)]
            assert "保守探索性口径" in content
            assert "## 2. 实际退出记录" in content and "exit_summary.csv" in content
    images = []
    for path in sorted(directory.glob("*.png")):
        with Image.open(path) as picture:
            dimensions = picture.size
            assert dimensions[0] >= 800 and dimensions[1] >= 400
            picture.verify()
        images.append({"file": path.name, "dimensions": list(dimensions), "sha256": sha256(path)})
    assert len(images) == 5
    test_root = ET.parse(root / "validation/classic-factors-tests.xml").getroot()
    suites = list(test_root.iter("testsuite"))
    assert sum(int(suite.get("failures", "0")) + int(suite.get("errors", "0")) for suite in suites) == 0
    result = {"runs_checked": checked, "documents_read": len(documents), "local_links_checked": links,
              "equity_and_fees_reconciled": True, "comparison_values_reconciled": True,
              "images": images, "python_tests": sum(int(suite.get("tests", "0")) for suite in suites),
              "python_skipped": sum(int(suite.get("skipped", "0")) for suite in suites),
              "source_requests": sources["requested"], "source_errors": sources["errors"]}
    atomic_json(directory / "artifact_validation.json", result)
    validation["artifact_checks"] = result
    atomic_json(validation_path, validation)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = {path.name: sha256(path) for path in sorted(directory.iterdir()) if path.is_file() and path.name != "manifest.json"}
    atomic_json(manifest_path, manifest)
    print(json.dumps({"runs": len(checked), "documents": len(documents), "links": links,
                      "images": len(images), "python_tests": result["python_tests"], "status": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
