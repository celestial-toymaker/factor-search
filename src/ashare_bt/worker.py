from pathlib import Path
import json
import os
import sys
import traceback

from .config import Config
from .engine import run
from .utils import Cancelled, atomic_json


def execute(path):
    path = Path(path)
    job = json.loads(path.read_text())
    cancel = path.with_suffix(".cancel")
    job.update(status="running", pid=os.getpid(), stage="data", message="准备数据", progress=0)
    atomic_json(path,job)

    def progress(**record):
        job.update(record)
        weights = {"data":(.02,.10),"factors":(.12,.08),"status":(.20,.15),"backtest":(.35,.60),"report":(.95,.05)}
        offset,span = weights.get(record.get("stage"),(.0,.0))
        job["progress"] = offset + span * record.get("done",0) / max(record.get("total",1),1)
        atomic_json(path,job)

    try:
        result = run(Config.from_dict(job["config"],Path(job["root"])),progress=progress,cancel_file=cancel,run_id=job["id"])
        job.update(status="completed",message="回测与报告已完成",progress=1,summary=result.summary)
    except Cancelled:
        job.update(status="cancelled",message="任务已取消，已完成结果保留在运行目录")
    except Exception as exc:
        job.update(status="failed",message=str(exc),error=traceback.format_exc())
    atomic_json(path,job)


if __name__ == "__main__":
    execute(sys.argv[1])
