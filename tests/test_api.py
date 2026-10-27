from fastapi.testclient import TestClient

from ashare_bt.engine import run
from ashare_bt.server import create_app


def test_saved_run_ui_cli_tables_and_trade_explanation(market):
    config,data,features,status=market
    result=run(config,data=data,features=features,status=status,run_id="api-example")
    app=create_app(config.root)
    client=TestClient(app)
    jobs=client.get('/api/runs').json()
    assert jobs[0]['summary']['ending_nav']==result.summary['ending_nav']
    equity=client.get('/api/runs/api-example/table/equity?limit=3000').json()
    assert equity['total']==len(result.equity)
    assert equity['rows'][-1]['nav']==result.summary['ending_nav']
    trades=client.get('/api/runs/api-example/table/trades').json()
    trade=trades['rows'][0]
    detail=client.get('/api/runs/api-example/trade/'+trade['trade_id']).json()
    assert detail['trade']['notional']==trade['notional']
    assert detail['order'][0]['order_id']==trade['order_id']
    assert sum(float(r['amount']) for r in detail['journal'])==0
    report=client.get('/api/runs/api-example/file/report.md')
    assert report.status_code==200 and '# ' in report.text and '事后规划' in report.text
    image=client.get('/api/runs/api-example/file/equity-curve.png')
    assert image.status_code==200 and image.headers['content-type']=='image/png'
    assert image.content.startswith(bytes.fromhex('89504e470d0a1a0a'))


def test_job_queue_persistence_cancel_and_local_origin(tmp_path):
    client=TestClient(create_app(tmp_path))
    job=client.post('/api/runs',json={"name":"持久化任务"}).json()
    assert job['status']=='queued'
    assert TestClient(create_app(tmp_path)).get('/api/runs').json()[0]['id']==job['id']
    assert client.post('/api/runs/'+job['id']+'/cancel').status_code==200
    assert client.get('/api/runs').json()[0]['status']=='cancelled'
    assert client.post('/api/runs',json={},headers={'Origin':'https://example.com'}).status_code==403
