from decimal import Decimal as D

from ashare_bt.execution import Broker, ExitPlanner
from ashare_bt.ledger import Ledger
from ashare_bt.rules import FeeModel, candidate_limits, legal_quantity


def setup(market):
    config,data,features,status = market
    row = data.date_index[config.start] - 1
    ledger = Ledger(config.initial_cash, day=data.dates[row])
    return Broker(config,data,features,status,ledger),row


def test_next_open_cash_capacity_expiry_and_cancel(market):
    broker,row = setup(market)
    broker.config.participation = "0.00001"
    orders = broker.submit("000001.SZ","buy",100000,row,row+1)
    assert broker.ledger.frozen > 0
    broker.ledger.day = broker.data.dates[row+1]
    broker.match_open(row+1)
    assert orders[0].status == "expired" and 0 < orders[0].filled < 100000
    assert D(broker.trades[0]["notional"]) <= 10000
    assert broker.ledger.frozen == 0
    new = broker.submit("600001.SH","buy",100,row+1,row+2)[0]
    rebuilt = broker.replace(new.order_id,200,row+1,row+2)
    assert new.status == "cancelled" and rebuilt[0].quantity == 200
    broker.cancel(rebuilt[0].order_id)
    broker.ledger.assert_invariants()


def test_blocked_open_then_intraday_opening_and_zero_amount(market):
    broker,row = setup(market)
    data = broker.data
    previous = data.arrays["close"][row,0]
    upper = D(candidate_limits("000001.SZ",data.dates[row+1],previous,is_st=False)[0]["upper"])
    data.arrays["open"][row+1,0] = float(upper)
    data.arrays["low"][row+1,0] = previous
    order = broker.submit("000001.SZ","buy",100,row,row+1)[0]
    broker.ledger.day = data.dates[row+1]
    broker.match_open(row+1)
    assert order.filled == 0 and "候选涨停" in order.last_reason
    data.arrays["amt"][row+2,0] = 0
    order = broker.submit("000001.SZ","buy",100,row+1,row+2)[0]
    broker.match_open(row+2)
    assert order.filled == 0 and "成交额" in order.last_reason


def test_t1_st_open_recheck_and_quantity_rules(market):
    broker,row = setup(market)
    broker.status.open_values[row+1,0] = 1
    order = broker.submit("000001.SZ","buy",100,row,row+1)[0]
    broker.ledger.day = broker.data.dates[row+1]
    broker.match_open(row+1)
    assert order.filled == 0 and "非 ST" in order.last_reason
    broker.ledger.buy("000001.SZ",100,10,{},broker.data.dates[row+2])
    assert broker.ledger.positions["000001.SZ"].sellable(broker.data.dates[row+1]) == 0
    assert broker.ledger.positions["000001.SZ"].sellable(broker.data.dates[row+2]) == 100
    assert legal_quantity("688001.SH",201,"buy") == 201
    assert legal_quantity("000001.SZ",201,"buy") == 200
    assert legal_quantity("000001.SZ",37,"sell",37) == 37


def test_fees_and_history_bands(market):
    config,*_ = market
    fees = FeeModel(config).quote("600001.SH","2025-01-02","sell",10000)
    assert fees == {"commission":D(5),"stamp_tax":D(5),"transfer":D("0.10"),"slippage":D(5)}
    assert candidate_limits("600001.SH","2026-07-03",10,is_st=True)[0]["upper"] == "10.50"
    assert candidate_limits("600001.SH","2026-07-06",10,is_st=True)[0]["upper"] == "11.00"
    assert candidate_limits("300001.SZ","2025-01-02",10,is_st=True)[0]["upper"] == "12.00"


def test_exit_planner_uses_actual_future_capacity(market):
    broker,row = setup(market)
    planner = ExitPlanner(broker.config,broker.data,broker)
    first = planner.estimate("000001.SZ",100000,row)
    broker.data.arrays["amt"][row+1:,0] *= .0001
    second = planner.estimate("000001.SZ",100000,row)
    assert first["actual_latest_start"] != second["actual_latest_start"]
    assert first["uses_future_market_data"] and second["uses_future_market_data"]


def test_actual_path_starts_before_capacity_falls(market):
    broker,row = setup(market)
    data=broker.data
    last=data.date_index[broker.config.end]
    row=last-3
    for field in ('open','high','low','close'):
        data.arrays[field][-25:,0]=10
    data.arrays['amt'][-25:,0]=1e7
    data.arrays['amt'][last-2,0]=1e6
    data.arrays['amt'][last-1,0]=1e5
    broker.ledger.day=data.dates[row]
    broker.ledger.buy('000001.SZ',500,10,{},data.dates[row])
    broker.ledger.unlock(data.dates[row])
    planner=ExitPlanner(broker.config,data,broker)
    plan=planner.estimate('000001.SZ',500,row)
    assert plan['actual_latest_start']==data.dates[last-1]
    assert plan['path_feasible'] and plan['projected_terminal_quantity']=='0'
    assert plan['required_sessions']==1


def test_actual_path_rejects_buy_with_no_t1_exit(market):
    broker,row=setup(market)
    for field in ('open','high','low','close'):
        broker.data.arrays[field][row+2:,0]=float('nan')
    broker.data.arrays['amt'][row+2:,0]=0
    planner=ExitPlanner(broker.config,broker.data,broker)
    assert not planner.allow_buy('000001.SZ',100,row)
    assert not broker.orders and not broker.ledger.positions


def test_split_order_release_clears_decimal_reservation(market):
    broker,row=setup(market)
    broker.ledger=Ledger('100000000',day=broker.data.dates[row])
    ledger=broker.ledger
    ledger.buy('000001.SZ',1774749,10,{},broker.data.dates[row],factor=D('2.46191'))
    ledger.unlock(broker.data.dates[row])
    orders=broker.submit('000001.SZ','sell',1774749,row,row+1)
    ratio=D('2.4834')/D('2.46191')
    ledger.apply_factor('000001.SZ',D('2.4834'))
    broker.adjust_factor('000001.SZ',ratio)
    for order in orders:broker.cancel(order.order_id)
    assert ledger.positions['000001.SZ'].reserved==0
    ledger.sell('000001.SZ',int(ledger.positions['000001.SZ'].quantity),10,{})
    assert '000001.SZ' not in ledger.positions
    ledger.assert_invariants()


def test_proxy_tail_exits_at_valid_open(market):
    broker,row=setup(market)
    broker.ledger.buy('000001.SZ',D('.5'),10,{},broker.data.dates[row])
    broker.ledger.unlock(broker.data.dates[row])
    broker.submit('000001.SZ','sell',D('.5'),row,row+1)
    broker.ledger.day=broker.data.dates[row+1]
    broker.match_open(row+1)
    assert not broker.ledger.positions
    assert broker.ledger.frozen==0
    broker.ledger.assert_invariants()


def test_exit_path_announcement_access_uses_current_known_time(market):
    from ashare_bt.events import Event,EventSource,at
    broker,row=setup(market)
    event=Event('future-suspension','security_status',broker.config.end+'T09:00:00',
                broker.config.end+'T08:00:00','000001.SZ',{'suspended':True})
    source=EventSource([event])
    planner=ExitPlanner(broker.config,broker.data,broker,lambda:source.known(at(broker.ledger.day,'decision')))
    assert planner.actual_path.events('000001.SZ')==()
    broker.ledger.day=broker.config.end
    assert planner.actual_path.events('000001.SZ')==(event,)


def test_exit_path_stops_at_configured_end(market):
    broker,row=setup(market)
    broker.config.end=broker.data.dates[-4]
    planner=ExitPlanner(broker.config,broker.data,broker)
    first=planner.estimate('000001.SZ',100000,row)
    for field in ('open','high','low','close','amt','cumadj'):
        broker.data.arrays[field][-3:,0]*=.00001
    second=planner.estimate('000001.SZ',100000,row)
    assert first==second


def test_actual_path_replays_known_split_and_order_adjustment(market):
    from ashare_bt.events import Event
    broker,row=setup(market)
    broker.config.accounting_mode='explicit'
    broker.ledger=Ledger('100000','explicit',broker.data.dates[row])
    broker.ledger.buy('000001.SZ',1000,10,{},broker.data.dates[row])
    broker.ledger.unlock(broker.data.dates[row])
    event=Event('known-split','split',broker.data.dates[row+2]+'T09:00:00',
                broker.data.dates[row]+'T12:00:00','000001.SZ',{'ratio':'2'})
    planner=ExitPlanner(broker.config,broker.data,broker,lambda:(event,))
    plan=planner.estimate('000001.SZ',1000,row)
    assert plan['path_feasible'] and plan['projected_terminal_quantity']=='0'
    import json
    assert sum(D(fill['quantity']) for fill in json.loads(plan['capacity_path']))==2000
    assert broker.ledger.positions['000001.SZ'].quantity==1000
    assert not broker.ledger.processed_events and not broker.trades
