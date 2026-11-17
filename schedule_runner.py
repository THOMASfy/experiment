#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按北京时间窗口自动暂停/恢复 run_mem0_lme.py，直到 run 完成。

规则:
    周一至周五 09:00-12:00、14:00-18:00 不跑;
    其余时间(含周末)跑。

工作方式:
    允许时段: 如果 runner 没在跑, 用 --resume 启动; 如果在跑, 接管并监控。
    禁止时段: 如果 runner 在跑, 强制结束; 等下一个允许时段再 --resume。
    完成标志: run_dir/aggregate.json 出现后退出调度器。

示例:
    # 先看它会做什么, 不动进程
    python schedule_runner.py --run-dir results/mem0_bge_s/mem0_20261112-143726 --dry-run --once

    # 接管当前正在跑的进程
    python schedule_runner.py --run-dir results/mem0_bge_s/mem0_20261112-143726 --adopt-pid 66636

    # 正常后台运行(自动查找当前 run 的 runner)
    python schedule_runner.py --run-dir results/mem0_bge_s/mem0_20261112-143726 --find-existing
"""

import argparse
import ctypes
import json
import logging
import msvcrt
import os
import subprocess
import sys
import time
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
RUNNER = PROJECT_DIR / "run_mem0_lme.py"
BEIJING = timezone(timedelta(hours=8))

# 周一至周五的禁跑时段 [start, end)
BLOCK_WINDOWS = (
    (dtime(9, 0), dtime(12, 0)),
    (dtime(14, 0), dtime(18, 0)),
)

# 只在这些时段跑(用于 DeepSeek 低峰价, 例如 00:30-08:30);
# None = 保持旧的"只屏蔽工作时段"行为。
RUN_WINDOW = None

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def now_bj() -> datetime:
    return datetime.now(BEIJING)


def is_blocked(dt: datetime) -> bool:
    """默认: 北京时间周一至周五 09:00-12:00 / 14:00-18:00 为禁跑。

    指定 --allow-window 后改为白名单模式: 只有白名单时段内允许跑
    (例如 00:30-08:30 对应 DeepSeek 低峰价), 其余时间一律视为 blocked。
    """
    if RUN_WINDOW is not None:
        start, end = RUN_WINDOW
        return not (start <= dt.time() < end)
    if dt.weekday() >= 5:  # 周六、周日
        return False
    t = dt.time()
    return any(start <= t < end for start, end in BLOCK_WINDOWS)


def next_transition(dt: datetime) -> datetime:
    """返回下一次 blocked 状态翻转的时间(分钟精度)。"""
    current = is_blocked(dt)
    probe = dt.replace(second=0, microsecond=0)
    for _ in range(8 * 24 * 60):
        probe += timedelta(minutes=1)
        if is_blocked(probe) != current:
            return probe
    return probe


def pid_alive(pid: int) -> bool:
    if not pid:
        return False
    kernel32 = ctypes.windll.kernel32
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(h, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(h)


def _powershell(script: str) -> str:
    cmd = [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + script,
    ]
    try:
        return subprocess.check_output(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
        ).strip()
    except Exception:
        return ""


def get_cmdline(pid: int) -> str:
    if not pid:
        return ""
    return _powershell(
        f"$p = Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\"; "
        "if ($p) { $p.CommandLine }"
    )


def is_runner(pid: int) -> bool:
    return "run_mem0_lme.py" in get_cmdline(pid)


def _norm_path(text: str) -> str:
    return os.path.normcase(text).replace("/", "\\")


def _path_variants(path: Path):
    yield _norm_path(str(path))
    try:
        yield _norm_path(os.path.relpath(path, PROJECT_DIR))
    except ValueError:
        pass


def _all_runner_processes():
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='pythonw.exe'\" | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    raw = _powershell(script)
    if not raw:
        return []
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(rows, dict):
        rows = [rows]
    return [
        (int(row["ProcessId"]), row.get("CommandLine") or "")
        for row in rows
        if "run_mem0_lme.py" in (row.get("CommandLine") or "")
    ]


def find_existing_runner(run_dir: Path):
    """找命令行里带 run_mem0_lme.py 且属于这个 run 的进程。

    兼容两种启动方式:
      - 调度器自己启动: --resume <run_dir 绝对路径>
      - 手工启动:       --out-root <run_dir 的父目录, 可能是相对路径>
    """
    run_keys = list(_path_variants(run_dir))
    parent_keys = list(_path_variants(run_dir.parent))
    found = []
    for pid, cmdline in _all_runner_processes():
        norm = _norm_path(cmdline)
        if any(key in norm for key in run_keys):
            found.append(pid)
            continue
        if any(f"--out-root {key}" in norm for key in parent_keys):
            found.append(pid)
    return sorted(set(found))


def kill_tree(pid: int, logger: logging.Logger) -> bool:
    logger.warning("stopping runner pid=%s (taskkill /T /F)", pid)
    try:
        subprocess.run(
            ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
        )
    except Exception as e:
        logger.error("taskkill failed: %r", e)
    for _ in range(20):
        if not pid_alive(pid):
            logger.info("runner pid=%s stopped", pid)
            return True
        time.sleep(0.5)
    logger.error("runner pid=%s is still alive after taskkill", pid)
    return False


def load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def build_resume_cmd(run_dir: Path, meta: dict) -> list:
    """根据 meta.json 重建与原 run 一致的 --resume 命令。"""
    cmd = [
        sys.executable,
        "-u",
        str(RUNNER),
        "--resume",
        str(run_dir),
        "--data",
        str(meta.get("data") or "longmemeval_s_cleaned.json"),
        "--topk",
        str(meta.get("topk", 10)),
        "--workers",
        str(meta.get("workers", 16)),
        "--retrieval",
        str(meta.get("retrieval") or "dense"),
        "--model",
        str(meta.get("llm_model") or "deepseek-flash"),
        "--max-tokens",
        str(meta.get("llm_max_tokens", 8192)),
        "--llm-thinking",
        str(meta.get("llm_thinking") or "disabled"),
        "--question-retries",
        str(meta.get("question_retries", 3)),
        "--embedder",
        str(meta.get("embedder") or "BAAI/bge-base-en-v1.5"),
        "--embedding-dims",
        str(meta.get("embedding_dims", 768)),
    ]
    cmd += ["--skip-abs"] if meta.get("skip_abs", True) else ["--no-skip-abs"]
    if meta.get("subset"):
        cmd += ["--subset", str(meta["subset"])]
    if meta.get("query_prefix") is not None:
        cmd += ["--query-prefix", str(meta["query_prefix"])]
    if meta.get("llm_reasoning_effort"):
        cmd += ["--llm-reasoning-effort", str(meta["llm_reasoning_effort"])]
    if meta.get("oversample"):
        cmd += ["--oversample", str(meta["oversample"])]
    if meta.get("delete_after_question"):
        cmd += ["--delete-after-question"]
    # [LME] 适配版摄入端参数 (profile=stock 时这些都不存在, 不传)
    if (meta.get("profile") or "stock") == "lme":
        cmd += ["--profile", "lme"]
        if meta.get("extract_cache"):
            cmd += ["--extract-cache", str(meta["extract_cache"])]
        if meta.get("lme_max_facts"):
            cmd += ["--lme-max-facts", str(meta["lme_max_facts"])]
        if meta.get("lme_cache_key"):
            cmd += ["--lme-cache-key", str(meta["lme_cache_key"])]
        if meta.get("lme_date_mode"):
            cmd += ["--lme-date-mode", str(meta["lme_date_mode"])]
        if meta.get("lme_prompt"):
            cmd += ["--lme-prompt", str(meta["lme_prompt"])]
        if meta.get("lme_fact_style"):
            cmd += ["--lme-fact-style", str(meta["lme_fact_style"])]
        if meta.get("lme_batch_sessions"):
            cmd += ["--lme-batch-sessions", str(meta["lme_batch_sessions"])]
        if meta.get("lme_prefilter_topk"):
            cmd += ["--lme-prefilter-topk", str(meta["lme_prefilter_topk"])]
    # 保证 resume 不会因为数据集变动而多跑题
    n_planned = meta.get("n_planned")
    if isinstance(n_planned, int) and n_planned > 0:
        cmd += ["--limit", str(n_planned)]
    return cmd


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("mem0_scheduler")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(run_dir / "scheduler.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # pythonw.exe 下 sys.stderr 为 None，不能再挂 StreamHandler
    if sys.stderr is not None:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def eprint(msg: str) -> None:
    stream = sys.stderr or sys.stdout
    if stream is not None:
        print(msg, file=stream)


def acquire_lock(path: Path):
    f = open(path, "a+", encoding="utf-8")
    f.write("scheduler lock\n")
    f.flush()
    f.seek(0)
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        f.close()
        return None
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description="按北京时间窗口调度 mem0 LongMemEval run")
    ap.add_argument("--run-dir", required=True, help="要接管的 run 目录")
    ap.add_argument("--poll", type=int, default=15, help="轮询秒数, 默认 15")
    ap.add_argument("--adopt-pid", type=int, default=0, help="接管已经在跑的具体 PID")
    ap.add_argument("--find-existing", action="store_true", help="自动查找当前 run 的 runner 进程")
    ap.add_argument("--dry-run", action="store_true", help="只打印决策, 不启动/不杀进程")
    ap.add_argument("--once", action="store_true", help="只执行一个决策周期后退出")
    ap.add_argument("--retry-delay", type=int, default=300, help="runner 异常退出后的重试基础秒数")
    ap.add_argument("--kill-on-exit", action="store_true", help="调度器退出时连 runner 一起停掉")
    ap.add_argument(
        "--allow-window",
        default="",
        help="白名单时段 HH:MM-HH:MM(北京时间), 只在此时段跑; 例如 DeepSeek 低峰 00:30-08:30",
    )
    args = ap.parse_args()

    global RUN_WINDOW
    if args.allow_window:
        try:
            a, b = args.allow_window.split("-")
            ah, am = [int(x) for x in a.strip().split(":")]
            bh, bm = [int(x) for x in b.strip().split(":")]
            RUN_WINDOW = (dtime(ah, am), dtime(bh, bm))
        except Exception:
            eprint(f"--allow-window 格式应为 HH:MM-HH:MM, 收到: {args.allow_window!r}")
            return 2

    run_dir = Path(args.run_dir).resolve()
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        eprint(f"meta.json 不存在: {meta_path}")
        return 2
    if not RUNNER.exists():
        eprint(f"runner 不存在: {RUNNER}")
        return 2

    lock = acquire_lock(run_dir / "scheduler.lock")
    if lock is None:
        eprint("已有另一个 scheduler 在管理这个 run")
        return 2

    logger = setup_logger(run_dir)
    state_path = run_dir / "scheduler_state.json"
    state = load_state(state_path)

    logger.info("scheduler start | run_dir=%s | dry_run=%s", run_dir, args.dry_run)
    if RUN_WINDOW is not None:
        logger.info(
            "RUN WINDOW MODE: only %s-%s (Asia/Shanghai) allowed; everything else blocked",
            RUN_WINDOW[0].strftime("%H:%M"),
            RUN_WINDOW[1].strftime("%H:%M"),
        )
    else:
        logger.info("blocked windows: Mon-Fri 09:00-12:00, 14:00-18:00 (Asia/Shanghai)")

    # 1) 恢复上次记录的 runner
    pid = int(state.get("pid") or 0)
    if pid and not (pid_alive(pid) and is_runner(pid)):
        logger.info("stale pid=%s in state, ignored", pid)
        pid = 0

    # 2) 显式接管
    if not pid and args.adopt_pid:
        if pid_alive(args.adopt_pid) and is_runner(args.adopt_pid):
            pid = args.adopt_pid
            logger.info("adopted runner pid=%s from --adopt-pid", pid)
        else:
            logger.error("--adopt-pid %s 不是存活的 run_mem0_lme.py 进程", args.adopt_pid)
            return 2

    # 3) 自动查找
    if not pid and args.find_existing:
        found = find_existing_runner(run_dir)
        if len(found) == 1:
            pid = found[0]
            logger.info("adopted runner pid=%s from --find-existing", pid)
        elif len(found) > 1:
            logger.error("找到多个 runner 进程 %s，请用 --adopt-pid 指定一个", found)
            return 2
        else:
            others = [p for p, _ in _all_runner_processes()]
            if others:
                logger.error(
                    "发现其它 run_mem0_lme.py 进程 %s，但无法确认属于本 run；"
                    "为避免重复启动，请用 --adopt-pid 指定，或先停掉它们",
                    others,
                )
                return 2

    if pid:
        state.update({"pid": pid, "adopted_at": now_bj().isoformat(timespec="seconds")})
        save_state(state_path, state)

    child = None
    fail_count = int(state.get("fail_count") or 0)
    next_start_at = 0.0
    dry_started = False
    last_key = None

    def log_once(key: str, level: int, msg: str, *a):
        nonlocal last_key
        if key != last_key:
            logger.log(level, msg, *a)
            last_key = key

    try:
        while True:
            now = now_bj()
            blocked = is_blocked(now)

            # 完成检查
            if (run_dir / "aggregate.json").exists():
                logger.info("aggregate.json found -> run finished; scheduler exits")
                if pid and args.kill_on_exit and not args.dry_run:
                    kill_tree(pid, logger)
                return 0

            # 刷新进程状态
            if pid > 0 and not pid_alive(pid):
                exit_code = child.poll() if child is not None else None
                logger.info(
                    "runner pid=%s exited (code=%s), aggregate.json not found",
                    pid,
                    exit_code,
                )
                pid = 0
                child = None
                fail_count += 1
                delay = min(args.retry_delay * (2 ** (fail_count - 1)), 1800)
                next_start_at = time.time() + delay
                logger.info("next start attempt in >= %ss", delay)

            nxt = next_transition(now)
            secs = max(1, int((nxt - now).total_seconds()))

            if blocked:
                dry_started = False
                if pid > 0:
                    log_once(
                        f"blocked-{pid}",
                        logging.WARNING,
                        "blocked window (until %s); stopping runner pid=%s",
                        nxt.strftime("%Y-%m-%d %H:%M"),
                        pid,
                    )
                    if not args.dry_run:
                        kill_tree(pid, logger)
                        pid = 0
                        child = None
                        state.update({"pid": 0, "fail_count": fail_count})
                        save_state(state_path, state)
                else:
                    log_once(
                        f"blocked-{nxt:%Y%m%d%H%M}",
                        logging.INFO,
                        "blocked window; runner is stopped, next allowed at %s",
                        nxt.strftime("%Y-%m-%d %H:%M"),
                    )
            else:
                if pid > 0 or dry_started:
                    log_once(
                        f"running-{pid}",
                        logging.INFO,
                        "allowed window; runner pid=%s is running (next block at %s)",
                        pid if pid > 0 else "(dry-run)",
                        nxt.strftime("%Y-%m-%d %H:%M"),
                    )
                else:
                    # 异常退避
                    if time.time() < next_start_at:
                        log_once(
                            "backoff",
                            logging.INFO,
                            "backoff before restart (fail_count=%d, %.0fs left)",
                            fail_count,
                            next_start_at - time.time(),
                        )
                    else:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        cmd = build_resume_cmd(run_dir, meta)
                        logger.info(
                            "allowed window; starting runner: %s",
                            subprocess.list2cmdline(cmd),
                        )
                        if args.dry_run:
                            logger.info("[dry-run] would start runner")
                            if args.once:
                                return 0
                            dry_started = True
                        else:
                            out_path = run_dir / "scheduler_child.out"
                            env = os.environ.copy()
                            env["PYTHONUNBUFFERED"] = "1"
                            env["PYTHONIOENCODING"] = "utf-8"
                            env["PYTHONUTF8"] = "1"
                            env["MEM0_TELEMETRY"] = "False"
                            with open(out_path, "ab", buffering=0) as out_f:
                                child = subprocess.Popen(
                                    cmd,
                                    cwd=str(PROJECT_DIR),
                                    stdin=subprocess.DEVNULL,
                                    stdout=out_f,
                                    stderr=subprocess.STDOUT,
                                    env=env,
                                    creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
                                    close_fds=True,
                                )
                            pid = child.pid
                            next_start_at = 0.0
                            state.update(
                                {
                                    "pid": pid,
                                    "started_at": now.isoformat(timespec="seconds"),
                                    "fail_count": fail_count,
                                }
                            )
                            save_state(state_path, state)
                            logger.info("runner started pid=%s", pid)
                            fail_count = 0

            if args.once:
                return 0

            # 在下一次状态翻转时提前醒来, 否则按 poll 轮询
            sleep_s = min(args.poll, secs)
            time.sleep(max(1, sleep_s))

    except KeyboardInterrupt:
        logger.warning("scheduler interrupted")
        if pid and pid > 0 and args.kill_on_exit and not args.dry_run:
            kill_tree(pid, logger)
        elif pid and pid > 0:
            logger.warning(
                "runner pid=%s left running; restart scheduler with --adopt-pid %s",
                pid,
                pid,
            )
        return 130
    finally:
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
