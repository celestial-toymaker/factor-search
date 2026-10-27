import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

from ashare_bt.config import Config
from ashare_bt.data import load_data, METRICS, FactorProvider
from ashare_bt.status import StatusProvider


def test_input_cutoff_and_warmup_fingerprint(tmp_path):
    directory=tmp_path/'feathers'
    directory.mkdir()
    dates=pd.to_datetime(['2024-12-30','2024-12-31','2025-01-02','2026-09-30','2026-10-16'])
    def save(tail):
        for metric in METRICS:
            frame=pd.DataFrame({'date':dates,'000001.SZ':np.array([10.,11.,12.,13.,tail])})
            feather.write_feather(pa.Table.from_pandas(frame),directory/(metric+'.feather'))
    config=Config(root=tmp_path)
    save(14.)
    first=load_data(config)
    save(99999.)
    second=load_data(config)
    assert first.dates==['2024-12-30','2024-12-31','2025-01-02','2026-09-30']
    assert first.fingerprint==second.fingerprint
    for metric in METRICS:
        assert np.array_equal(first.arrays[metric],second.arrays[metric])


def test_status_effective_and_knowledge_time(tmp_path):
    dates=['2025-01-03','2025-01-06','2025-01-07']
    path=tmp_path/'overrides.jsonl'
    pd.DataFrame([{'date':'2025-01-03','security':'000001.SZ','is_st':1,
                   'effective_at':'2025-01-03T09:00:00+08:00','known_at':'2025-01-06T12:00:00+08:00'}]).to_json(path,orient='records',lines=True)
    status=StatusProvider(dates,['000001.SZ'],tmp_path/'source',path)
    assert status.values[:,0].tolist()==[-1,1,1]
    assert status.open_values[:,0].tolist()==[-1,-1,1]


def test_fundamental_provider_point_in_time():
    frame=pd.DataFrame([{'security':'000001.SZ','factor':'roe','value':.15,
                         'effective_at':'2025-01-03T00:00:00+08:00','known_at':'2025-04-20T18:00:00+08:00'}])
    provider=FactorProvider(frame)
    assert len(provider.as_of('2025-04-20T15:02:00+08:00'))==0
    assert len(provider.as_of('2025-04-21T15:02:00+08:00'))==1


def test_naive_factor_time_uses_shanghai():
    frame=pd.DataFrame([{'security':'000001.SZ','factor':'roe','value':.15,
                         'effective_at':'2025-01-03T00:00:00','known_at':'2025-04-20T18:00:00'}])
    provider=FactorProvider(frame)
    assert len(provider.as_of('2025-04-20T18:00:00+08:00'))==1


def test_status_event_coverage_and_unknown_local_flag(tmp_path):
    from ashare_bt.events import Event
    dates=['2025-01-02','2025-01-03']
    status=StatusProvider(dates,['000001.SZ'],tmp_path/'source')
    event=Event('normal','security_status','2025-01-02T09:00:00','2025-01-02T09:15:00','000001.SZ',{'is_st':0})
    known=status.with_events([event])
    assert known.coverage(np.ones((2,1),bool))['unknown_stock_days']==0
    assert status.values[:,0].tolist()==[-1,-1]
    assert known.open_values[:,0].tolist()==[0,0]
    path=tmp_path/'unknown.jsonl'
    pd.DataFrame([{'date':dates[0],'security':'000001.SZ','is_st':-1}]).to_json(path,orient='records',lines=True)
    assert StatusProvider(dates,['000001.SZ'],tmp_path/'source',path).values[:,0].tolist()==[-1,-1]


def test_factor_cache_preserves_current_period_and_source_provenance(market):
    from ashare_bt.compute import compute_features
    config,data,features,status=market
    compute_features(data,config,use_cache=True)
    config.start='2025-01-10'
    data.source_files=[{'metric':'close','sha256':'current-whole-file-digest'}]
    reused=compute_features(data,config,use_cache=True)
    assert reused.metadata['cache_hit']
    assert reused.profile['dates']['start']==config.start
    assert reused.profile['dates']['trading_days']==len(data.trading_indices(config))
    assert reused.profile['source_files']==data.source_files
