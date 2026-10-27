from decimal import Decimal as D

import pytest

from ashare_bt.events import Event, EventSource
from ashare_bt.ledger import Ledger


def action(ledger, kind, ident, payload, day="2025-01-03", security="000001.SZ", known=None):
    ledger.day = day
    ledger.phase = "pre_open"
    event = Event(ident, kind, day + "T09:00:00", known or day + "T09:00:00", security, payload, "manual-fixture")
    return ledger.apply(event)


def owned(quantity=100, cash=10000):
    ledger = Ledger(str(cash), "explicit", "2025-01-02")
    ledger.buy("000001.SZ", quantity, 10, {}, "2025-01-03")
    ledger.mark({"000001.SZ": 10}, "2025-01-02")
    return ledger


def test_dividend_dates_tax_duplicate_and_double_entry():
    ledger = owned()
    action(ledger, "dividend_record", "record", {"action_id":"D1", "cash_per_share":1}, "2025-01-02")
    assert ledger.cash == 9000 and ledger.nav == 10000
    action(ledger, "dividend_ex", "ex", {"action_id":"D1"})
    assert ledger.nav == 10000
    ledger.mark({"000001.SZ":9}, "2025-01-03")
    assert ledger.receivables["D1"]["amount"] == 100 and ledger.nav == 10000
    action(ledger, "dividend_pay", "pay", {"action_id":"D1"}, "2025-01-06")
    assert ledger.cash == 9100 and ledger.nav == 10000
    action(ledger, "tax", "tax", {"amount":20}, "2025-01-06")
    assert ledger.cash == 9080 and ledger.nav == 9980
    assert ledger.fees[-1]['type']=='tax' and ledger.fees[-1]['amount']=='20.00'
    assert ledger.fees[-1]['event_id']=='tax'
    assert action(ledger,"tax","tax",{"amount":20},"2025-01-06") == {"duplicate":True}
    assert len(ledger.fees)==1
    ledger.assert_invariants()


def test_split_fraction_bonus_and_settlement():
    ledger = owned(101)
    action(ledger, "split", "S1", {"ratio":"1.5"})
    assert ledger.positions["000001.SZ"].quantity == 151
    assert ledger.pending["S1:fraction"]["quantity"] == D("0.5")
    assert ledger.nav == 10000
    action(ledger, "fraction_settlement", "F1", {"action_id":"S1:fraction","price":"6.666666666666666666"})
    assert ledger.cash == D("8993.33")
    action(ledger, "stock_dividend", "B1", {"ratio":"0.13", "tradable_at":"2025-01-08"})
    ledger.assert_invariants()
    ledger.unlock("2025-01-08")
    assert ledger.positions["000001.SZ"].quantity == 170
    assert ledger.pending["B1:fraction"]["quantity"] == D("0.63")
    ledger.assert_invariants()


def test_rights_default_waive_subscribe_delivery():
    ledger = owned()
    action(ledger, "rights_offer", "R1", {"ratio":"0.3"})
    assert ledger.cash == 9000 and not ledger.pending
    action(ledger, "rights_subscribe", "R2", {"action_id":"R1","quantity":30,"subscription_price":5,"tradable_at":"2025-01-07"})
    assert ledger.cash == 8850 and ledger.pending["R2"]["cost"] == 150
    ledger.unlock("2025-01-06")
    assert ledger.positions["000001.SZ"].quantity == 100
    ledger.unlock("2025-01-07")
    assert ledger.positions["000001.SZ"].quantity == 130
    ledger.assert_invariants()


def test_conversion_spinoff_code_and_delist():
    ledger = owned()
    action(ledger,"spinoff","SP",{"target_security":"600001.SH","ratio":"0.5","valuation":2,"cost_allocation":"0.1","tradable_at":"2025-01-07"})
    assert ledger.nav == 10000 and ledger.pending["SP:shares"]["cost"] == 100
    ledger.unlock("2025-01-07")
    action(ledger,"merger","M",{"target_security":"600002.SH","ratio":2,"tradable_at":"2025-01-08","cash_per_share":1,"pay_at":"2025-01-09"},"2025-01-07")
    assert ledger.nav == 10000 and "000001.SZ" not in ledger.positions
    ledger.unlock("2025-01-08")
    action(ledger,"code_change","C",{"target_security":"600003.SH"},"2025-01-08","600002.SH")
    ledger.unlock("2025-01-09")
    action(ledger,"delist_settlement","D",{"price":4,"pay_at":"2025-01-10"},"2025-01-09","600003.SH")
    assert ledger.receivables["D"]["amount"] == 800
    ledger.unlock("2025-01-10")
    assert ledger.cash == 9900
    paid=[row for row in ledger.journal if row['type']=='receivable_paid']
    assert paid[-1]['source_event']=='D:payment:2025-01-10'
    ledger.assert_invariants()


def test_external_flows_unitization_and_proxy_mode():
    ledger = owned()
    action(ledger,"cash_deposit","IN",{"amount":5000})
    assert ledger.nav == 15000 and ledger.unit_nav == 1
    action(ledger,"cash_withdrawal","OUT",{"amount":3000})
    assert ledger.nav == 12000 and ledger.unit_nav == 1 and ledger.net_external_flows == 2000
    proxy = Ledger("10000")
    proxy.buy("000001.SZ",100,10,{},"2025-01-03")
    proxy.apply_factor("000001.SZ",2)
    assert proxy.positions["000001.SZ"].quantity == 200 and proxy.nav == 10000
    with pytest.raises(ValueError):
        action(proxy,"dividend_ex","E",{"cash_per_share":1})


def test_event_order_idempotency_and_knowledge():
    late = Event("b","tax","2025-01-03T09:00:00","2025-01-06T09:00:00",payload={"amount":1})
    earlier = Event("a","cash_deposit","2025-01-03T09:00:00","2025-01-03T09:00:00",payload={"amount":1})
    source = EventSource([late,earlier,earlier])
    assert source.due(earlier.ready_at) == [earlier]
    assert source.due(late.ready_at) == [late]
    assert source.due(late.ready_at) == []


def test_full_redemption_preserves_return_and_resubscription():
    ledger = owned()
    ledger.unlock("2025-01-03")
    ledger.sell("000001.SZ",100,20,{})
    assert ledger.unit_nav == D("1.1")
    action(ledger,"cash_withdrawal","ALL",{"amount":11000})
    assert ledger.nav == 0 and ledger.unit_count == 0 and ledger.unit_nav == D("1.1")
    action(ledger,"cash_deposit","NEW",{"amount":11000})
    assert ledger.unit_count == 10000 and ledger.unit_nav == D("1.1")
    ledger.assert_invariants()


def test_conversion_keeps_fractional_entitlement_until_cash_settlement():
    ledger=owned(101)
    action(ledger,'conversion','C1',{'target_security':'600001.SH','ratio':'.5','valuation':20,'tradable_at':'2025-01-07'})
    assert ledger.pending['C1:shares']['quantity']==50
    assert ledger.pending['C1:fraction']['quantity']==D('.5')
    assert ledger.nav==10000
    ledger.unlock('2025-01-07')
    assert ledger.positions['600001.SH'].quantity==50
    assert 'C1:fraction' in ledger.pending
    action(ledger,'fraction_settlement','CF',{'action_id':'C1:fraction','amount':10},'2025-01-08')
    assert ledger.nav==10000 and ledger.cash==9000
    ledger.assert_invariants()


def test_small_residual_sale_fee_and_terminal_liability():
    ledger=Ledger('10','explicit','2025-01-02')
    ledger.buy('000001.SZ',1,10,{},'2025-01-03')
    ledger.unlock('2025-01-03')
    with pytest.raises(ValueError):ledger.sell('000001.SZ',1,'.01',{'commission':D('5')})
    assert ledger.cash==0 and ledger.positions['000001.SZ'].quantity==1
    ledger.sell('000001.SZ',1,'.01',{'commission':D('5')},allow_fee_liability=True)
    assert ledger.cash==0 and ledger.liabilities['commission']==D('4.99')
    assert ledger.nav==D('-4.99')
    ledger.assert_invariants()


def test_spinoff_with_cash_preserves_total_asset_value():
    ledger=owned()
    action(ledger,'spinoff','SC',{'target_security':'600001.SH','ratio':'.5','valuation':2,
                                 'cost_allocation':'.1','cash_per_share':1,'pay_at':'2025-01-08','tradable_at':'2025-01-07'})
    assert ledger.positions['000001.SZ'].last_price==8
    assert ledger.pending['SC:shares']['quantity']==50
    assert ledger.receivables['SC:cash']['amount']==100
    assert ledger.nav==10000
    ledger.unlock('2025-01-08')
    assert ledger.cash==9100 and ledger.nav==10000
    ledger.assert_invariants()
