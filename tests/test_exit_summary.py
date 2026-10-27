from decimal import Decimal

import pandas as pd

from ashare_bt.factor_analytics import actual_exit_section, actual_exit_summary


def test_exit_summary_counts_plans_once_and_uses_real_fills():
    summary = {"end": "2026-09-30", "ending_nav": "10000", "exit_label": "末日按规则退出",
               "exit_planning": {"disclosure": "按逐日可获知信息执行所选末日规则。"}}
    exits = pd.DataFrame([
        {"date": "2026-09-29", "security": "A", "exit_active": True, "triggered_on": "2026-09-29", "planned_start": "2026-09-30", "sold_quantity": "0"},
        {"date": "2026-09-30", "security": "A", "exit_active": True, "triggered_on": "2026-09-29", "planned_start": "2026-09-30", "sold_quantity": "60"},
        {"date": "2026-09-30", "security": "B", "exit_active": True, "triggered_on": "2026-09-29", "planned_start": "2026-09-30", "sold_quantity": "100"},
        {"date": "2026-09-28", "security": "C", "exit_active": True, "triggered_on": "2026-09-25", "planned_start": "2026-09-28", "sold_quantity": "30"},
        {"date": "2026-09-29", "security": "C", "exit_active": True, "triggered_on": "2026-09-25", "planned_start": "2026-09-28", "sold_quantity": "50"},
    ])
    trades = pd.DataFrame([
        {"date": "2026-09-30", "security": "A", "side": "sell", "reason": "按规则计划退出", "notional": "600", "fees": "6"},
        {"date": "2026-09-30", "security": "B", "side": "sell", "reason": "按规则计划退出", "notional": "1000", "fees": "10"},
        {"date": "2026-09-28", "security": "C", "side": "sell", "reason": "按规则计划退出", "notional": "300", "fees": "3"},
        {"date": "2026-09-29", "security": "C", "side": "sell", "reason": "按规则计划退出", "notional": "200", "fees": "2"},
        {"date": "2026-09-20", "security": "A", "side": "sell", "reason": "排名调仓", "notional": "5000", "fees": "50"},
    ])
    orders = pd.DataFrame([
        {"execution_date": "2026-09-30", "security": "A", "side": "sell", "reason": "按规则计划退出", "remaining": "20", "last_reason": "共享成交容量不足或低于合法数量"},
        {"execution_date": "2026-09-30", "security": "A", "side": "sell", "reason": "按规则计划退出", "remaining": "20", "last_reason": "共享成交容量不足或低于合法数量"},
        {"execution_date": "2026-09-30", "security": "B", "side": "sell", "reason": "按规则计划退出", "remaining": "0", "last_reason": "开盘成交"},
    ])
    positions = pd.DataFrame([
        {"date": "2026-09-29", "security": "B", "quantity": "100", "market_value": "1000"},
        {"date": "2026-09-30", "security": "A", "quantity": "40", "market_value": "400"},
    ])
    actual, records = actual_exit_summary(summary, exits, trades, orders, positions)
    assert len(records) == 3
    assert actual["terminal_planned_securities"] == actual["terminal_sold_securities"] == 2
    assert actual["terminal_cleared_securities"] == 1
    assert actual["remaining_securities"] == 1
    assert Decimal(actual["terminal_gross_proceeds"]) == Decimal("1600")
    assert Decimal(actual["terminal_net_proceeds"]) == Decimal("1584")
    assert Decimal(actual["remaining_market_value"]) == Decimal("400")
    assert actual["earlier_planned_securities"] == actual["earlier_cleared_securities"] == 1
    assert actual["remaining_reasons"] == [{"reason": "共享成交容量不足或低于合法数量", "securities": 1}]
    assert records.set_index("security").loc["C", "gross_proceeds"] == "500"
    section = actual_exit_section(summary, actual)
    assert "2026-09-29" in section and "2026-09-30" in section
    assert "4.00%" in section and "exit_summary.csv" in section


def test_exit_summary_reports_cash_portfolio_with_empty_records():
    summary = {"end": "2026-09-30", "ending_nav": "10000", "exit_label": "末日按规则退出",
               "exit_planning": {"disclosure": "按逐日可获知信息执行所选末日规则。"}}
    actual, records = actual_exit_summary(summary, *(pd.DataFrame() for _ in range(4)))
    assert records.empty
    assert actual["terminal_planned_securities"] == actual["terminal_sold_securities"] == 0
    assert actual["remaining_market_value"] == "0"
    assert "持仓已结清" in actual_exit_section(summary, actual)
