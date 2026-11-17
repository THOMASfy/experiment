#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""串行调度多个 run：前一个 run 的 aggregate.json 出现后，自动启动下一个。

每个 stage 仍由 schedule_runner.py 管理，所以北京时间禁跑窗口、强制暂停、
--resume、异常重试等规则全部复用。

用法:
    python run_chain.py \
      --run-dir results/mem0_bge_s/mem0_20261112-143726 \
      --run-dir results/mem0_bge_m/mem0_20261112-XXXXXX \
      --poll 15

后台启动(无窗口):
    start "" "...pythonw.exe" run_chain.py --run-dir ... --run-dir ...
"""

import argparse
import json
import logging
import msvcrt
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SCHEDULER = PROJECT_DIR / "schedule_runner.py"
BEIJING = timezone(timedelta(hours=8))
CREATE_NO_WINDOW = 0x08000000


def now_bj() -> datetime:
    return datetime.now(BEIJING)


def eprint(msg: str) -> None:
    stream = sys.stderr or sys.stdout
    if stream is not None:
        print(msg, file=stream)


def setup_logger() -> logging.Logger:
    log_dir = PROJECT_DIR / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mem0_chain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_dir / "chain.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if sys.stderr is not None:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+", encoding="utf-8")
    f.write("chain lock\n")
    f.flush()
    f.seek(0)
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        f.close()
        return None
    return f


def done(run_dir: Path) -> bool:
    return (run_dir / "aggregate.json").exists()


def main() -> int:
    ap = argparse.ArgumentParser(description="按顺序跑多个 LongMemEval run")
    ap.add_argument("--run-dir", action="append", required=True, help="stage 目录，可重复，按顺序执行")
    ap.add_argument("--poll", type=int, default=15, help="传给 schedule_runner.py 的轮询秒数")
    ap.add_argument("--retry-delay", type=int, default=300, help="stage scheduler 异常退出后的重试秒数")
    args = ap.parse_args()

    stages = [Path(p).resolve() for p in args.run_dir]
    if not SCHEDULER.exists():
        eprint(f"schedule_runner.py 不存在: {SCHEDULER}")
        return 2
    for stage in stages:
        if not (stage / "meta.json").exists():
            eprint(f"stage 缺少 meta.json: {stage}")
            return 2

    lock = acquire_lock(PROJECT_DIR / "results" / "chain.lock")
    if lock is None:
        eprint("已有另一个 chain 在运行")
        return 2

    logger = setup_logger()
    logger.info(
        "chain start | stages=%s",
        " -> ".join(str(s) for s in stages),
    )
    try:
        for idx, stage in enumerate(stages, 1):
            if done(stage):
                logger.info("stage %d/%d already done: %s", idx, len(stages), stage)
                continue

            logger.info("stage %d/%d start: %s", idx, len(stages), stage)
            attempt = 0
            while not done(stage):
                attempt += 1
                cmd = [
                    sys.executable,
                    "-u",
                    str(SCHEDULER),
                    "--run-dir",
                    str(stage),
                    "--find-existing",
                    "--poll",
                    str(args.poll),
                    "--kill-on-exit",
                ]
                logger.info("launch scheduler: %s", subprocess.list2cmdline(cmd))
                rc = subprocess.call(cmd, cwd=str(PROJECT_DIR))

                if done(stage):
                    logger.info("stage %d/%d done: %s", idx, len(stages), stage)
                    break
                if rc == 130:
                    logger.warning("scheduler interrupted; chain stops")
                    return 130

                delay = min(args.retry_delay * (2 ** min(attempt - 1, 5)), 1800)
                logger.warning(
                    "scheduler exited rc=%s without aggregate; retry stage in %ss",
                    rc,
                    delay,
                )
                time.sleep(delay)

        logger.info("all stages done")
        return 0
    except KeyboardInterrupt:
        logger.warning("chain interrupted")
        return 130
    finally:
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
