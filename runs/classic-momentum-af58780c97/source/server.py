from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from threading import Event as ThreadEvent, Thread, Lock
from uuid import uuid4
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Config, ROOT, EXIT_MODES
from .engine import FILTER_REASONS
from .strategies import StrategyRegistry
from .utils import atomic_json, contained, dumps


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def records(frame):
    return json.loads(frame.replace([np.inf,-np.inf],np.nan).to_json(orient="records",date_format="iso",force_ascii=False))


class JobQueue:
    def __init__(self,root):
        self.root = root
        self.directory = root / "work" / "jobs"
        self.directory.mkdir(parents=True,exist_ok=True)
        self.stop = ThreadEvent()
        self.lock = Lock()
        self.process = None
        self.thread = Thread(target=self.loop,daemon=True)

    def jobs(self):
        result = [read_json(path) for path in self.directory.glob("*.json")]
        known = {item["id"] for item in result}
        for path in (self.root / "runs").glob("*/summary.json"):
            summary = read_json(path)
            if summary["run_id"] not in known:
                result.append({"id":summary["run_id"],"status":"completed","progress":1,"message":"回测已完成",
                               "created_at":summary["created_at"],"config":summary["config"],"summary":summary})
        return sorted(result,key=lambda item:item.get("created_at",""),reverse=True)

    def add(self,config):
        config = Config.from_dict(config,self.root)
        if config.strategy not in StrategyRegistry().specs:
            raise ValueError("工作台请选择已登记的策略；Python 插件可通过 CLI 接入")
        ident = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8]
        job = {"id":ident,"root":str(self.root),"config":config.public(),"status":"queued","progress":0,
               "created_at":datetime.now().astimezone().isoformat(),"message":"已排队，等待 GPU"}
        atomic_json(self.directory / (ident + ".json"),job)
        return job

    def loop(self):
        while not self.stop.wait(.5):
            if self.process:
                if self.process.poll() is None:
                    continue
                self.process = None
            running = [j for j in self.jobs() if j["status"] == "running"]
            busy = False
            for job in running:
                try:
                    os.kill(job.get("pid",0),0)
                    busy = True
                except (OSError,ValueError):
                    job.update(status="failed",message="执行进程已结束，可重新运行已保存配置")
                    atomic_json(self.directory / (job["id"] + ".json"),job)
            if busy:
                continue
            pending = sorted((j for j in self.jobs() if j["status"] == "queued"),key=lambda j:j["created_at"])
            if pending:
                job = pending[0]
                log = (self.directory / (job["id"] + ".log")).open("ab")
                try:
                    self.process = subprocess.Popen([sys.executable,"-m","ashare_bt.worker",str(self.directory / (job["id"]+".json"))],
                                                    cwd=self.root,stdout=log,stderr=log,start_new_session=True)
                finally:
                    log.close()

    def cancel(self,ident):
        path = contained(self.directory,ident+".json")
        job = read_json(path)
        if job is None:
            raise HTTPException(404,"任务编号待核对")
        path.with_suffix(".cancel").touch()
        if job["status"] == "queued":
            job.update(status="cancelled",message="排队任务已取消")
            atomic_json(path,job)
        return {"status":"requested"}


def create_app(root=ROOT):
    root = Path(root)
    queue = JobQueue(root)

    @asynccontextmanager
    async def lifespan(app):
        queue.thread.start()
        yield
        queue.stop.set()
        queue.thread.join(timeout=2)

    app = FastAPI(title="观澜 · A 股回测工作台",lifespan=lifespan)
    app.state.queue = queue

    @app.middleware("http")
    async def local_access(request:Request,call_next):
        host = request.headers.get("host","").split(":")[0]
        origin = request.headers.get("origin","")
        if host not in {"127.0.0.1","localhost","testserver"}:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail":"请通过本机地址访问工作台"},status_code=403)
        if request.method in {"POST","PUT","DELETE","PATCH"} and origin and origin.split("//")[-1].split(":")[0] not in {"127.0.0.1","localhost"}:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail":"请在工作台中提交任务"},status_code=403)
        return await call_next(request)

    def run_path(ident):
        try:
            return contained(root / "runs",ident)
        except ValueError:
            raise HTTPException(400,"请使用有效运行编号")

    @app.get("/api/meta")
    def meta():
        import torch
        profile = read_json(root / "data" / "profile.json",{})
        return {"defaults":Config(root=root).public(),"strategies":StrategyRegistry().list(),"exit_modes":EXIT_MODES,
                "filter_reasons":FILTER_REASONS,"profile":profile,
                "devices":{"mps":torch.backends.mps.is_available(),"cuda":torch.cuda.is_available()},
                "status_sync":read_json(root / "data/status/sync_progress.json",{}),
                "status_coverage":read_json(root / "data/status/coverage.json",{})}

    @app.get("/api/runs")
    def jobs():
        return [{k:v for k,v in j.items() if k not in {"root","error"}} for j in queue.jobs()]

    @app.post("/api/runs",status_code=202)
    def start(payload:dict):
        try:
            return queue.add(payload)
        except (ValueError,TypeError) as exc:
            raise HTTPException(422,str(exc))

    @app.post("/api/runs/{ident}/cancel")
    def cancel(ident:str):
        return queue.cancel(ident)

    @app.get("/api/runs/{ident}")
    def detail(ident:str):
        job = next((j for j in queue.jobs() if j["id"]==ident),None)
        if job is None:
            raise HTTPException(404,"请从历史任务选择运行")
        return {k:v for k,v in job.items() if k not in {"root"}}

    @app.get("/api/runs/{ident}/table/{name}")
    def table(ident:str,name:str,date:str|None=None,security:str|None=None,side:str|None=None,query:str|None=None,
              eligible:bool|None=None,selected:bool|None=None,offset:int=0,limit:int=100):
        allowed = {"equity","benchmark_equity","trades","orders","positions","journal","fees","events","exits","universe",
                   "selection","order_events","terminal_settlements","diagnostics"}
        if name not in allowed:
            raise HTTPException(404,"请选择已保存的结果表")
        path = run_path(ident) / (name + ".parquet")
        if not path.exists():
            return {"rows":[],"total":0,"columns":[]}
        schema = pq.read_schema(path).names
        filters = []
        for key,value in [("date",date),("security",security),("side",side),("eligible",eligible),("selected",selected)]:
            if value is not None and value != "" and key in schema:
                filters.append((key,"==",value))
        if name=="selection" and date is None:
            date = read_json(run_path(ident)/"summary.json")["warmup"]["last_decision"]
            filters.append(("date","==",date))
        frame = pd.read_parquet(path,filters=filters or None)
        if query and "security" in frame:
            frame = frame[frame.security.str.contains(query,regex=False,na=False)]
        if name == "selection" and len(frame):
            frame = frame.sort_values(["eligible","rank","security"],ascending=[False,True,True])
        if name in {"trades","orders","journal","fees","events"}:
            frame = frame.iloc[::-1]
        offset,limit = max(0,offset),min(3000,max(1,limit))
        return {"rows":records(frame.iloc[offset:offset+limit]),"total":len(frame),"columns":list(frame.columns)}

    @app.get("/api/runs/{ident}/trade/{trade_id}")
    def explain(ident:str,trade_id:str):
        return explain_trade(run_path(ident),trade_id)

    @app.get("/api/runs/{ident}/file/{filename}")
    def file(ident:str,filename:str):
        path = contained(run_path(ident),filename)
        if path.suffix not in {".md",".png",".csv",".parquet",".json",".jsonl"} or not path.exists():
            raise HTTPException(404,"文件将在运行完成后提供")
        return FileResponse(path)

    @app.get("/api/event-catalog")
    def event_catalog():
        return read_json(root / "research/event_catalog.json",[])

    dist = root / "web" / "dist"
    if dist.exists():
        app.mount("/",StaticFiles(directory=dist,html=True),name="workbench")
    return app


def explain_trade(directory,trade_id):
    path = directory / "trades.parquet"
    if not path.exists():
        raise HTTPException(404,"成交明细将在运行完成后提供")
    trades = pd.read_parquet(path,filters=[("trade_id","==",trade_id)])
    if not len(trades):
        raise HTTPException(404,"请核对成交编号")
    trade = records(trades)[0]
    order = pd.read_parquet(directory/"orders.parquet",filters=[("order_id","==",trade["order_id"])])
    journal = pd.read_parquet(directory/"journal.parquet",filters=[("source_event","==",trade["order_id"])])
    selection = pd.read_parquet(directory/"selection.parquet",filters=[("date","==",trade["decision_date"]),("security","==",trade["security"])])
    events = pd.read_parquet(directory/"order_events.parquet",filters=[("order_id","==",trade["order_id"])])
    return {"trade":trade,"order":records(order),"journal":records(journal),"selection":records(selection),"events":records(events)}
