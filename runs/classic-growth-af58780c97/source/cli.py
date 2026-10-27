import argparse
from pathlib import Path
import json
import sys
import webbrowser

from .config import Config, ROOT
from .utils import atomic_json, dumps


def main():
    parser = argparse.ArgumentParser(prog="ashare-bt",description="观澜 · A 股全市场回测工作台")
    sub = parser.add_subparsers(dest="command",required=True)
    audit = sub.add_parser("audit-data",help="校验七矩阵、GPU 因子和历史状态覆盖")
    audit.add_argument("--config",default=str(ROOT/"configs/default.toml"))
    audit.add_argument("--sync-status",action="store_true")
    runner = sub.add_parser("run",help="运行策略并保存中文报告")
    runner.add_argument("--config",default=str(ROOT/"configs/default.toml"))
    runner.add_argument("--strategy")
    runner.add_argument("--device",choices=["auto","mps","cuda"])
    runner.add_argument("--exit-mode",choices=["latest_feasible","force_cash","strict_end"])
    runner.add_argument("--run-id")
    runner.add_argument("--replay", type=Path, help="使用已有运行的配置与 input_events.jsonl 完整重放")
    explain = sub.add_parser("explain-trade",help="逐笔解释决策、成交与资金变化")
    explain.add_argument("run_directory",type=Path)
    explain.add_argument("trade_id")
    events = sub.add_parser("list-events",help="查看事件覆盖表或运行事件流")
    events.add_argument("--run",type=Path)
    events.add_argument("--framework")
    events.add_argument("--limit",type=int,default=2000)
    ui = sub.add_parser("ui",help="启动本机浏览器工作台")
    ui.add_argument("--port",type=int,default=8765)
    ui.add_argument("--no-browser",action="store_true")
    args = parser.parse_args()
    if args.command=="ui":
        import uvicorn
        from .server import create_app
        if not (ROOT/"web/dist/index.html").exists():
            parser.error("请先运行 npm --prefix web install 和 npm --prefix web run build")
        if not args.no_browser:
            from threading import Timer
            Timer(1,lambda:webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
        uvicorn.run(create_app(),host="127.0.0.1",port=args.port,log_level="info")
        return
    if args.command=="explain-trade":
        from .server import explain_trade
        print(dumps(explain_trade(args.run_directory.resolve(),args.trade_id),pretty=True))
        return
    if args.command=="list-events":
        import pandas as pd
        records = (pd.read_parquet(args.run/"events.parquet").to_dict("records") if args.run else
                   json.loads((ROOT/"research/event_catalog.json").read_text()))
        if args.framework:
            records = [r for r in records if r.get("framework")==args.framework]
        print(dumps(records[:args.limit],pretty=True))
        return
    config = Config.load(args.config)
    if getattr(args,"replay",None):
        config = Config.load(args.replay / "config.json")
        config.events_file = str((args.replay / "input_events.jsonl").resolve())
    if getattr(args,"strategy",None) and args.strategy != config.strategy:
        config.parameters = {key:value for key,value in config.parameters.items() if key=="allocation"}
    for key in ("strategy","device","exit_mode"):
        value = getattr(args,key,None)
        if value:
            setattr(config,key,value)
    def progress(**record):
        print(record.get("message",record),flush=True)
    if args.command=="audit-data":
        from .data import load_data
        from .compute import compute_features
        from .status import StatusProvider, ensure_status
        from .engine import status_requirement
        data = load_data(config,progress)
        features = compute_features(data,config,progress)
        status = ensure_status(data,config,progress) if args.sync_status else StatusProvider(data.dates,data.securities,config.path("status_dir"))
        report = {"profile":features.profile,"compute":features.metadata,"status":status.coverage(status_requirement(data,features,config))}
        atomic_json(config.root/"data/profile.json",features.profile)
        atomic_json(config.root/"data/audit.json",report)
        print(dumps(report,pretty=True))
    else:
        from .engine import run
        result = run(config,progress=progress,run_id=args.run_id)
        print(f"报告：{result.run_dir / 'report.md'}")
        print(dumps({key:result.summary[key] for key in ("net_profit","return","max_drawdown","average_exposure","first_trade_date")},pretty=True))


if __name__=="__main__":
    main()
