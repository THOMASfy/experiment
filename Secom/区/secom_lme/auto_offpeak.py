# -*- coding: utf-8 -*-
"""Run the remaining LongMemEval/SeCom questions only in DeepSeek off-peak.

DeepSeek peak windows (UTC, Mon-Fri):
    01:00-04:00 and 06:00-10:00

The script starts the resumable runner with a deadline shortly before the next
peak window.  If it reaches the deadline, it exits and sleeps until off-peak.
It can be started repeatedly by Windows Task Scheduler; a file lock prevents
duplicate supervisors.
"""

import argparse
import json
import msvcrt
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

WORK = Path(r"D:\longmemeval")
RUN = WORK / "results" / "secom_bge" / "auto_470"
PREFILTER_TOPN = 75
PREFILTER_KEEP_RAW = True
NO_JUDGE = True
RETRIEVAL_ONLY = True
PY = WORK / ".venv" / "Scripts" / "python.exe"
RUNNER = WORK / "run_secom_lme.py"
CFG = WORK / "run_local.cmd"
LOG = RUN / "auto_offpeak.log"
LOCK = RUN / "auto_offpeak.lock"

UTC = timezone.utc
HKT = timezone(timedelta(hours=8))
PEAK_WINDOWS = ((1.0, 4.0), (6.0, 10.0))


def log(message: str) -> None:
    line = "[{} HKT] {}".format(
        datetime.now(HKT).strftime("%Y-%m-%d %H:%M:%S"), message
    )
    if sys.stdout is not None:
        try:
            print(line, flush=True)
        except Exception:
            pass
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_cfg() -> None:
    if CFG.exists():
        text = CFG.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            match = re.match(r'\s*set "([^"]+)=(.*)"\s*$', line)
            if not match:
                continue
            key, value = match.group(1), match.group(2)
            if key in ("OPENAI_API_KEY", "OPENAI_API_BASE", "SEGMENT_MODEL"):
                os.environ[key] = value

    env_file = Path.home() / "dot_env" / "openai.env"
    if env_file.exists() and not os.environ.get("OPENAI_API_KEY"):
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

    os.environ.setdefault("OPENAI_API_BASE", "https://api.deepseek.com/v1")
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["SECOM_USAGE_PATH"] = str(RUN / "usage.jsonl")


def read_meta() -> dict:
    path = RUN / "meta.json"
    if not path.exists():
        raise SystemExit("missing {}".format(path))
    return json.loads(path.read_text(encoding="utf-8"))


def count_done() -> int:
    path = RUN / "predictions.jsonl"
    if not path.exists():
        return 0
    qids = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(item.get("question_id", ""))
            if qid:
                qids.add(qid)
    return len(qids)


def is_peak(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    hour = now.hour + now.minute / 60.0 + now.second / 3600.0
    return any(start <= hour < end for start, end in PEAK_WINDOWS)


def next_peak_start(now: datetime) -> datetime:
    for offset in range(0, 10):
        day = (now + timedelta(days=offset)).date()
        if day.weekday() >= 5:
            continue
        for hour in (1, 6):
            candidate = datetime(
                day.year, day.month, day.day, hour, 0, 0, tzinfo=UTC
            )
            if candidate > now:
                return candidate
    raise RuntimeError("could not find next peak")


def next_offpeak_start(now: datetime) -> datetime:
    if not is_peak(now):
        return now
    hour = now.hour + now.minute / 60.0
    if 1.0 <= hour < 4.0:
        return now.replace(hour=4, minute=0, second=0, microsecond=0)
    if 6.0 <= hour < 10.0:
        return now.replace(hour=10, minute=0, second=0, microsecond=0)
    return now


def fmt(now: datetime) -> str:
    return now.astimezone(HKT).strftime("%Y-%m-%d %H:%M %Z")


def run_runner(deadline: datetime, meta: dict) -> int:
    cmd = [
        str(PY),
        "-u",
        str(RUNNER),
        "--resume",
        str(RUN),
        "--data",
        str(WORK / "longmemeval_m_cleaned.json"),
        "--topk",
        "10",
        "--compress-rate",
        "0.75",
        "--limit",
        "0",
        "--skip-abs",
        "--cache",
        str(RUN / "cache.sqlite"),
        "--seg-workers",
        str(int(meta.get("seg_workers", 8))),
        "--judge-model",
        str(meta.get("judge_model", "deepseek-flash")),
        "--deadline-epoch",
        str(int(deadline.timestamp())),
    ]
    if PREFILTER_TOPN > 0:
        cmd.extend(["--prefilter-topn", str(PREFILTER_TOPN)])
    if PREFILTER_KEEP_RAW:
        cmd.append("--prefilter-keep-raw")
    if NO_JUDGE:
        cmd.append("--no-judge")
    if RETRIEVAL_ONLY:
        cmd.append("--retrieval-only")
    log("launch runner, deadline={} ({})".format(fmt(deadline), int(deadline.timestamp())))
    with LOG.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            cmd,
            cwd=str(WORK),
            env=os.environ.copy(),
            stdout=handle,
            stderr=handle,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    log("runner exit={}".format(completed.returncode))
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-min", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RUN.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    load_cfg()
    meta = read_meta()
    target = int(meta.get("n_planned", 470))

    if not PY.exists():
        log("missing python: {}".format(PY))
        return 2
    if not RUNNER.exists():
        log("missing runner: {}".format(RUNNER))
        return 2

    now = datetime.now(UTC)
    done = count_done()
    peak = is_peak(now)
    next_off = next_offpeak_start(now)
    next_peak = next_peak_start(now)
    deadline = next_peak - timedelta(minutes=args.buffer_min)

    if args.dry_run:
        log(
            "dry_run done={}/{} peak={} next_off={} next_peak={} deadline={}".format(
                done, target, peak, fmt(next_off), fmt(next_peak), fmt(deadline)
            )
        )
        return 0

    lock_file = LOCK.open("a+")
    try:
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        log("another supervisor already holds the lock; exit")
        lock_file.close()
        return 0

    try:
        log("supervisor start pid={} done={}/{}".format(os.getpid(), done, target))
        while True:
            done = count_done()
            if done >= target:
                log("DONE {}/{}".format(done, target))
                return 0

            now = datetime.now(UTC)
            if is_peak(now):
                wake = next_offpeak_start(now)
                wait = max(1, int((wake - now).total_seconds()) + 5)
                log("PEAK now={} sleep {}s until {}".format(fmt(now), wait, fmt(wake)))
                time.sleep(wait)
                continue

            peak = next_peak_start(now)
            deadline = peak - timedelta(minutes=args.buffer_min)
            if deadline <= now:
                wake = next_offpeak_start(peak + timedelta(minutes=1))
                wait = max(1, int((wake - now).total_seconds()) + 5)
                log(
                    "inside {}min peak buffer; sleep {}s until {}".format(
                        args.buffer_min, wait, fmt(wake)
                    )
                )
                time.sleep(wait)
                continue

            rc = run_runner(deadline, meta)
            done = count_done()
            if done >= target:
                log("DONE {}/{}".format(done, target))
                return 0

            if rc == 3:
                wake = next_offpeak_start(peak + timedelta(minutes=1))
                wait = max(1, int((wake - datetime.now(UTC)).total_seconds()) + 5)
                log("runner paused; sleep {}s until {}".format(wait, fmt(wake)))
                time.sleep(wait)
            elif rc != 0:
                log("runner failed rc={}; sleep 60s".format(rc))
                time.sleep(60)
            else:
                log("runner exited without pause; retry in 10s")
                time.sleep(10)
    finally:
        try:
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        lock_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
