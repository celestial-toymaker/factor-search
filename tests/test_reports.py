"""资金曲线的图片交付与原始净值对应关系。"""
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg

from ashare_bt.analytics import chart_png
from ashare_bt.engine import run


def test_png_uses_all_original_daily_values(tmp_path,monkeypatch):
    equity=pd.DataFrame({'date':['2025-01-02','2025-01-03','2025-01-06'],
                         'unit_nav':[1.02,.98,1.05]})
    original=equity.copy(deep=True)
    captured={}
    save=FigureCanvasAgg.print_png
    def inspect(canvas,*args,**kwargs):
        for line in canvas.figure.axes[0].lines:
            if line.get_color() in {'#315ee8'}:
                captured[line.get_color()]=np.asarray(line.get_ydata(),dtype=float)
                assert len(line.get_xdata())==len(equity)
        result=save(canvas,*args,**kwargs)
        axis=canvas.figure.axes[0]
        for label in [*axis.get_xticklabels(),*axis.get_yticklabels()]:
            box=label.get_window_extent(canvas.get_renderer())
            assert 0 <= box.x0 <= box.x1 <= 1880
            assert 0 <= box.y0 <= box.y1 <= 580
        return result
    monkeypatch.setattr(FigureCanvasAgg,'print_png',inspect)
    path=chart_png(tmp_path,equity)
    np.testing.assert_allclose(captured['#315ee8'],[2.,-2.,5.],atol=1e-12)
    assert set(captured)=={'#315ee8'}
    pd.testing.assert_frame_equal(equity,original)
    with Image.open(path) as image:
        assert image.format=='PNG' and image.size==(1880,580)
        assert image.convert('RGB').getpixel((0,0))==(255,255,255)
    with Image.open(path) as image:
        image.verify()


def test_zero_trade_result_generates_flat_png_and_markdown(market):
    config,data,features,status=market
    config.strategy='buy_hold';config.parameters={'symbol':'689009.SH'}
    result=run(config,data=data,features=features,status=status,run_id='empty-report')
    assert result.trades.empty and result.equity.unit_nav.eq(1).all()
    text=(result.run_dir/'report.md').read_text()
    assert '](equity-curve.png)' in text
    with Image.open(result.run_dir/'equity-curve.png') as image:
        assert image.size==(1880,580)
        image.verify()
    import json
    manifest=json.loads((result.run_dir/'manifest.json').read_text())
    assert manifest['report']['chart_format']=='png'
    assert manifest['report']['chart_size']==[1880,580]
