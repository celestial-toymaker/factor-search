from decimal import Decimal as D
import copy

import numpy as np
import pytest

from ashare_bt.compute import compute_features
from ashare_bt.config import Config
from ashare_bt.engine import run
from ashare_bt.events import Event, EventSource
from ashare_bt.execution import Broker
from ashare_bt.ledger import Ledger
from ashare_bt.strategies import Strategy, TargetWeights
from ashare_bt.utils import Cancelled


def test_announcements_available_when_known_and_future_edits_are_isolated(market):
    config,data,features,status=market
    class Observe(Strategy):
        def __init__(self):self.seen=[]
        def on_close(self,history,portfolio):
            self.seen.append((history.date,[(e.event_id,e.payload) for e in history.announcements]))
            return TargetWeights({})
    event=Event('announcement','user','2025-01-20T09:00:00','2025-01-10T18:00:00',payload={'value':1})
    first=Observe(); run(config,first,data=data,features=features,status=status,event_source=EventSource([event]),output=False)
    changed=Event('announcement','user',event.effective_at,event.known_at,payload={'value':99})
    second=Observe(); run(config,second,data=data,features=features,status=status,event_source=EventSource([changed]),output=False)
    assert [(d,e) for d,e in first.seen if d<'2025-01-13']==[(d,e) for d,e in second.seen if d<'2025-01-13']
    assert next(e for d,e in first.seen if d=='2025-01-13')==[('announcement',{'value':1})]
    assert next(e for d,e in first.seen if d=='2025-01-20')==[('announcement',{'value':1})]


def test_split_updates_live_orders_and_locked_shares(market):
    config,data,features,status=market
    row=data.date_index[config.start]-1
    ledger=Ledger('100000','explicit',data.dates[row]);broker=Broker(config,data,features,status,ledger)
    ledger.buy('000001.SZ',200,10,{},data.dates[row+1]);ledger.unlock(data.dates[row+1])
    sell=broker.submit('000001.SZ','sell',200,row,row+1)[0]
    event=Event('split','split',data.dates[row+1]+'T09:00:00',data.dates[row]+'T18:00:00','000001.SZ',{'ratio':2})
    ledger.apply(event);broker.on_action(event)
    assert ledger.positions['000001.SZ'].quantity==400 and sell.quantity==400
    assert ledger.positions['000001.SZ'].reserved==400
    assert sell.reference_price==D(str(data.arrays['close'][row,0]))/2
    broker.cancel(sell.order_id)
    assert ledger.positions['000001.SZ'].reserved==0
    ledger.assert_invariants()


def test_st_switch_at_open_and_market_rule_override(market):
    config,data,features,status=market
    row=data.date_index[config.start]-1
    ledger=Ledger(config.initial_cash,day=data.dates[row]);broker=Broker(config,data,features,status,ledger)
    ledger.apply(Event('rule','rule_change',data.dates[row]+'T09:00:00',data.dates[row]+'T09:00:00',
                       payload={'minimum':10,'step':10,'maximum':1000,'limit_rate':'.15'}))
    orders=broker.submit('000001.SZ','buy',210,row,row+1)
    assert orders[0].quantity==210
    band=broker.opening_context(row+1,'000001.SZ')[5][0]
    assert band['rate']=='0.15'
    ledger.apply(Event('st','security_status',data.dates[row+1]+'T09:00:00',data.dates[row+1]+'T09:00:00','000001.SZ',{'is_st':1}))
    ledger.day=data.dates[row+1];broker.match_open(row+1)
    assert not broker.trades and '非 ST' in orders[0].last_reason


@pytest.mark.parametrize('mode',['latest_feasible','force_cash','strict_end'])
def test_exit_modes_under_interrupted_quotes(market,mode):
    config,data,features,status=market
    config.exit_mode=mode
    # 首批正常建仓，随后中断完整行情并沿用最近有效估值。
    start=data.date_index[config.start]
    for field in ('open','high','low','close'):
        data.arrays[field][start+3:]=np.nan
    data.arrays['amt'][start+3:]=0
    features=compute_features(data,config,use_cache=False)
    result=run(config,data=data,features=features,status=status,run_id='interrupted-'+mode)
    if mode=='force_cash':
        text=(result.run_dir/'report.md').read_text()
        assert '终点假设清算明细' in text and '估值价格' in text
    if mode in {'latest_feasible','force_cash'}:
        assert D(result.summary['ending_market_value'])==0
        assert bool(result.summary['terminal_assumptions']) == (mode == 'force_cash')
        if mode == 'latest_feasible':
            assert result.trades.date.max() < data.dates[start+3]
    else:
        assert D(result.summary['ending_market_value'])>0
        assert D(result.summary['stale_value'])>0
        assert result.positions.stale_days.max()>0
    assert result.summary['first_trade_date']==config.start


def test_cancel_before_data_and_handler_idempotency(market,tmp_path):
    config,data,features,status=market
    cancel=tmp_path/'cancel';cancel.touch()
    with pytest.raises(Cancelled):run(config,cancel_file=cancel,output=False)
    ledger=Ledger('10000','explicit')
    calls=[]
    ledger.register_handler('custom_action',lambda account,event:calls.append(event.event_id))
    event=Event('X','custom_action','2025-01-02T09:00:00','2025-01-02T09:00:00')
    ledger.apply(event);ledger.apply(event)
    assert calls==['X']


def test_event_known_between_preopen_and_open_changes_first_fill(market):
    config,data,features,status=market
    event=Event('open-risk','security_status',config.start+'T09:10:00',config.start+'T09:15:00','000001.SZ',{'is_st':1})
    config.strategy='buy_hold';config.parameters={'symbol':'000001.SZ'}
    source=EventSource([event,event])
    first=run(config,data=data,features=features,status=status,event_source=source,output=False)
    repeated=run(config,data=data,features=features,status=status,event_source=source,output=False)
    assert first.trades.empty and first.journal.to_json()==repeated.journal.to_json()
    applied=next(record for record in first.events.to_dict('records') if record['event_id']=='open-risk')
    import json
    assert json.loads(applied['payload'])['applied']['applied_at'].endswith('09:30:00+08:00')


def test_trade_callback_observes_cash_after_each_individual_trade(market):
    config,data,features,status=market
    class Observe(Strategy):
        def __init__(self):self.cash=[]
        def on_close(self,history,portfolio):
            return TargetWeights({'000001.SZ':D('.4'),'600001.SH':D('.4')})
        def on_event(self,event,context):
            if event.type=='trade':self.cash.append((event.payload['trade_id'],str(context.portfolio.cash)))
    strategy=Observe()
    result=run(config,strategy,data=data,features=features,status=status,output=False)
    assert strategy.cash==list(zip(result.trades.trade_id,result.trades.cash_after))
