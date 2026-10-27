from pathlib import Path
import ast
import json

from ashare_bt.extensions import EXTERNAL_REQUIREMENTS

ROOT=Path(__file__).resolve().parents[1]


def test_each_pinned_event_has_complete_mapping_and_source():
    records=json.loads((ROOT/'research/event_catalog.json').read_text())
    required={'original_name','source_path','source_line','source_url','meaning','trigger','required_fields',
              'processing_order','ledger_effect','implementation_status','implementation','test_location','commit'}
    assert len({r['id'] for r in records})==len(records)
    for record in records:
        assert all(record.get(key) for key in required),record
        source=ROOT/'research/sources'/record['framework']/record['source_path']
        assert source.exists()
        assert record['commit'] in record['source_url']
        assert 0 < record['source_line'] <= len(source.read_text().splitlines())
        assert (ROOT/record['test_location'].split('::')[0]).exists()
    assert len([r for r in records if r['framework']=='rqalpha' and r['category']=='event'])==45
    assert len([r for r in records if r['framework']=='zipline' and r['category']=='datasource'])==13
    assert len([r for r in records if r['framework']=='zipline' and r['category']=='clock'])==5
    assert len([r for r in records if r['framework']=='backtrader' and r['category']=='feed_status'])==8


def test_external_contracts():
    assert {r.family for r in EXTERNAL_REQUIREMENTS} >= {'tick','orderbook','margin','futures','options','live','fundamentals'}
    for item in EXTERNAL_REQUIREMENTS:
        assert item.trigger and item.fields and item.interface and item.asset_model
