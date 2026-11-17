# -*- coding: utf-8 -*-
"""Run SeCom + BGE on LongMemEval M.

This is the harness that is not part of the official SeCom repository.  It
builds SeCom memory units, retrieves with BGE, asks a reader LLM, and records
both retrieval and QA metrics.  It is resumable and caches segmentation and
compression results by session content.

Typical smoke test:
    python run_secom_lme.py --data longmemeval_m_cleaned.json --limit 2 \
        --topk 10 --skip-abs

Typical full run:
    python run_secom_lme.py --data longmemeval_m_cleaned.json \
        --subset subset_m_500_ids.json --topk 10 --compress-rate 0.75 \
        --limit 0 --skip-abs --out-root results/secom_bge
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make lme_metrics importable when this file is launched from another cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from openai import OpenAI

from lme_metrics import aggregate_predictions, judge_qa, record_usage, retrieval_scores
from secom import SeCom
from secom.utils import extract_result


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            if stream is None:
                continue
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            if stream is None:
                continue
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            data = data["data"]
        else:
            data = list(data.values())
    if not isinstance(data, list):
        raise ValueError("data must be a JSON list or an object containing one")
    return data


def load_subset(path: Optional[str]) -> Optional[set]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return set(str(x) for x in value)
    if isinstance(value, dict):
        if isinstance(value.get("ids"), list):
            return set(str(x) for x in value["ids"])
        return set(str(x) for x in value.keys())
    raise ValueError("unsupported subset format")


def is_abstention_item(item: Dict[str, Any]) -> bool:
    return str(item.get("question_id", "")).endswith("_abs")


def dedupe_predictions(predictions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_qid: Dict[str, Dict[str, Any]] = {}
    for pred in predictions:
        qid = str(pred.get("question_id", ""))
        if qid:
            by_qid[qid] = pred
    return list(by_qid.values())


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print("[warn] ignored malformed prediction line", flush=True)
    return dedupe_predictions(rows)


def append_prediction(path: Path, pred: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(pred, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# Persistent cache: segmentation and compression are keyed by content.
# ---------------------------------------------------------------------------

class SegmentCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path), timeout=30)
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            pass
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS segments "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS compressed "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.commit()
        self.pending = 0

    def get_segments(self, key: str) -> Optional[List[List[str]]]:
        row = self.conn.execute(
            "SELECT value FROM segments WHERE key = ?", (key,)
        ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def put_segments(self, key: str, segments: List[List[str]]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO segments (key, value) VALUES (?, ?)",
            (key, json.dumps(segments, ensure_ascii=False)),
        )
        self._maybe_commit()

    def get_compressed(self, key: str) -> Optional[List[str]]:
        row = self.conn.execute(
            "SELECT value FROM compressed WHERE key = ?", (key,)
        ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def put_compressed(self, key: str, units: List[str]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO compressed (key, value) VALUES (?, ?)",
            (key, json.dumps(units, ensure_ascii=False)),
        )
        self._maybe_commit()

    def _maybe_commit(self) -> None:
        self.pending += 1
        if self.pending >= 20:
            self.conn.commit()
            self.pending = 0

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()


# ---------------------------------------------------------------------------
# SeCom segmentation / compression
# ---------------------------------------------------------------------------

def session_to_exchanges(session: Sequence[Any]) -> List[str]:
    """Convert LongMemEval turns into SeCom user-bot exchange strings."""
    exchanges: List[str] = []
    parts: List[Tuple[str, str]] = []

    def flush() -> None:
        nonlocal parts
        if parts:
            exchanges.append("\n".join("{}: {}".format(role, text) for role, text in parts))
            parts = []

    for turn in session:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            if parts:
                flush()
            parts.append(("User", content))
        elif role == "assistant":
            parts.append(("Assistant", content))
            flush()
        else:
            parts.append((role.title() or "Turn", content))
    flush()
    return exchanges


def segmentation_key(secom: SeCom, exchanges: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(str(getattr(secom, "segment_model", "")).encode("utf-8"))
    digest.update(b"\n")
    digest.update(secom.segment_prompt.encode("utf-8"))
    digest.update(b"\n")
    digest.update(json.dumps(list(exchanges), ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def compression_key(seg_key: str, rate: float, compress_model: str) -> str:
    raw = "{}|{}|{}".format(seg_key, rate, compress_model)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def fallback_segments(exchanges: Sequence[str]) -> List[List[str]]:
    return [list(exchanges[i:i + 3]) for i in range(0, len(exchanges), 3)]


def segment_one(secom: SeCom, exchanges: Sequence[str]) -> Tuple[Optional[List[List[str]]], Optional[str]]:
    """Segment one session without SeCom's stdout side effects.

    Returns (segments, error).  error is None on success.
    """
    try:
        text = secom.prefix_exchanges_with_idx(list(exchanges))
        prompt = secom.segment_prompt.format(text_to_be_segmented=text)
        response = secom.segmentor(prompt, max_tokens=1024)
        if not response:
            return None, "bad response: empty response"

        seg_jsonl, ok = extract_result(response, "segmentation")
        if not ok:
            return None, "bad response: {}".format(str(response)[:500])

        segments: List[List[str]] = []
        previous = 0
        for line in seg_jsonl.strip().splitlines():
            line = line.strip().strip(",")
            if not line:
                continue
            item = json.loads(line)
            count = int(item["num_exchanges"])
            if count <= 0:
                raise ValueError("num_exchanges <= 0")
            remaining = len(exchanges) - previous
            if remaining <= 0:
                break
            # The segmentor occasionally over-counts the final segment.  Clip
            # it instead of falling back to a fake 3-exchange split.
            count = min(count, remaining)
            segments.append(list(exchanges[previous:previous + count]))
            previous += count

        if not segments:
            raise ValueError("empty segmentation")
        if previous < len(exchanges):
            # Same for an under-count: keep the tail rather than fallback.
            segments.append(list(exchanges[previous:]))
        return segments, None
    except Exception as exc:  # noqa: BLE001
        return None, "not segmented: {}".format(exc)


def compress_segments(
    secom: SeCom,
    segments: List[List[str]],
    rate: float,
    comp_key: str,
    cache: Optional[SegmentCache],
) -> List[str]:
    if rate >= 1.0:
        return ["\n".join(segment) for segment in segments]

    if cache is not None:
        cached = cache.get_compressed(comp_key)
        if cached is not None:
            return cached

    # LLMLingua's tokenizer rejects literal special tokens such as
    # "<|endoftext|>" that occasionally appear in benchmark conversations.
    # Replace the token surface form before compression; the content remains
    # semantically equivalent for retrieval and reading.
    safe_segments = [
        [re.sub(r"<\|[^|]*\|>", "[special]", str(text)) for text in segment]
        for segment in segments
    ]
    units = secom.compress(safe_segments, compress_rate=rate)
    cleaned: List[str] = []
    for unit in units:
        if isinstance(unit, str):
            cleaned.append(unit)
        elif isinstance(unit, (list, tuple)):
            cleaned.append("\n".join(str(x) for x in unit))
        else:
            cleaned.append(str(unit))

    if cache is not None:
        cache.put_compressed(comp_key, cleaned)
    return cleaned


def prefilter_item(
    item: Dict[str, Any],
    question: str,
    embeddings: HuggingFaceEmbeddings,
    topn: int,
    keep_raw: bool = False,
) -> Tuple[Dict[str, Any], List[Document]]:
    """Keep only the top-N raw sessions by BGE similarity.

    This is an optional cost reduction.  It embeds raw sessions locally
    (no API calls) and runs segmentation only on the shortlist.  If keep_raw
    is true, the non-shortlisted sessions are still added to retrieval as raw
    session documents, so the gold session is not hard-filtered out.
    """
    if topn <= 0:
        return item, []
    sessions = item.get("haystack_sessions") or []
    ids = item.get("haystack_session_ids") or []
    dates = item.get("haystack_dates") or []
    if len(ids) != len(sessions):
        ids = ["session_{}".format(i) for i in range(len(sessions))]
    if len(dates) != len(sessions):
        dates = ["unknown"] * len(sessions)

    docs: List[Document] = []
    docs_by_index: Dict[int, Document] = {}
    for index, session in enumerate(sessions):
        text = "\n".join(session_to_exchanges(session))
        if text:
            doc = Document(
                page_content=text,
                metadata={"prefilter_index": index},
            )
            docs.append(doc)
            docs_by_index[index] = doc
    if len(docs) <= topn:
        return item, []

    store = FAISS.from_documents(docs, embeddings)
    hits = store.similarity_search(question, k=topn)
    keep = sorted({int(doc.metadata["prefilter_index"]) for doc in hits})
    keep_set = set(keep)
    filtered = dict(item)
    filtered["haystack_sessions"] = [sessions[i] for i in keep]
    filtered["haystack_session_ids"] = [ids[i] for i in keep]
    filtered["haystack_dates"] = [dates[i] for i in keep]

    raw_docs: List[Document] = []
    if keep_raw:
        for index in range(len(sessions)):
            if index in keep_set or index not in docs_by_index:
                continue
            raw_docs.append(
                Document(
                    page_content=docs_by_index[index].page_content,
                    metadata={
                        "session_id": str(ids[index]),
                        "date": str(dates[index]),
                    },
                )
            )
    return filtered, raw_docs


def build_memory(
    secom: SeCom,
    item: Dict[str, Any],
    topk: int,
    rate: float,
    cache: Optional[SegmentCache],
    executor: ThreadPoolExecutor,
    tlog: Any,
    compress_model: str,
) -> Tuple[List[Document], int, int, int]:
    sessions = item.get("haystack_sessions") or []
    session_ids = item.get("haystack_session_ids") or []
    dates = item.get("haystack_dates") or []
    n_sessions = len(sessions)

    if len(session_ids) != n_sessions:
        session_ids = ["session_{}".format(i) for i in range(n_sessions)]
    if len(dates) != n_sessions:
        dates = ["unknown"] * n_sessions

    results: List[Optional[Tuple[List[List[str]], bool, str]]] = [None] * n_sessions
    exchanges_list: List[Optional[List[str]]] = [None] * n_sessions
    futures: Dict[Any, Tuple[int, str, List[str]]] = {}

    for index, (session_id, date, session) in enumerate(zip(session_ids, dates, sessions)):
        exchanges = session_to_exchanges(session)
        exchanges_list[index] = exchanges
        if not exchanges:
            results[index] = ([], True, "")
            continue

        key = segmentation_key(secom, exchanges)
        cached = cache.get_segments(key) if cache is not None else None
        if cached is not None:
            results[index] = (cached, True, key)
        else:
            future = executor.submit(segment_one, secom, exchanges)
            futures[future] = (index, key, exchanges)

    for future in as_completed(futures):
        index, key, exchanges = futures[future]
        try:
            segments, error = future.result()
        except Exception as exc:  # noqa: BLE001
            segments, error = None, "worker exception: {}".format(exc)

        if error is not None:
            tlog("[seg] session={} {}".format(index, error[:500]))
            segments = fallback_segments(exchanges)
            results[index] = (segments, False, key)
        else:
            if cache is not None:
                cache.put_segments(key, segments or [])
            results[index] = (segments or [], True, key)

    documents: List[Document] = []
    failed = 0
    for index, (session_id, date) in enumerate(zip(session_ids, dates)):
        entry = results[index]
        if entry is None:
            continue
        segments, ok, key = entry
        if not ok:
            failed += 1
        if not segments:
            continue

        comp_key = compression_key(key, rate, compress_model) if key else ""
        units = compress_segments(secom, segments, rate, comp_key, cache)
        for segment, unit in zip(segments, units):
            documents.append(
                Document(
                    page_content=unit,
                    metadata={
                        "session_id": str(session_id),
                        "date": str(date),
                        "unit_index": len(documents),
                    },
                )
            )

    return documents, n_sessions, len(documents), failed


def retrieve(
    documents: Sequence[Document],
    embeddings: HuggingFaceEmbeddings,
    question: str,
    topk: int,
) -> List[Tuple[Document, float]]:
    if not documents:
        return []
    store = FAISS.from_documents(list(documents), embeddings)
    k = min(topk, len(documents))
    hits = store.similarity_search_with_score(question, k=k)
    return [(doc, float(score)) for doc, score in hits]


def build_reader_prompt(
    question: str,
    question_date: str,
    hits: Sequence[Tuple[Document, float]],
) -> str:
    blocks: List[str] = []
    for index, (doc, _score) in enumerate(hits, 1):
        date = str(doc.metadata.get("date", "unknown"))
        blocks.append(
            "### Memory {}:\nSession Date: {}\nSession Content:\n{}".format(
                index, date, doc.page_content
            )
        )
    history = "\n\n".join(blocks) if blocks else "(no relevant memory retrieved)"
    return (
        "I will give you several history chats between you and a user. "
        "Please answer the question based on the relevant chat history.\n\n"
        "History Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:"
    ).format(history, question_date, question)


def chat(
    client: OpenAI,
    model: str,
    messages: Sequence[Dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if "deepseek" in model.lower():
                kwargs["extra_body"] = {
                    "thinking": {"type": "disabled"},
                    "reasoning_effort": "none",
                }
            completion = client.chat.completions.create(**kwargs)
            record_usage("reader", model, completion)
            return (completion.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError("chat failed after retries: {}".format(last_error))


def unique_session_ids(hits: Sequence[Tuple[Document, float]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for doc, _score in hits:
        sid = str(doc.metadata.get("session_id", ""))
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SeCom + BGE on LongMemEval M")
    parser.add_argument("--data", default=None, help="longmemeval_m_cleaned.json")
    parser.add_argument("--subset", default=None, help="subset_m_500_ids.json")
    parser.add_argument("--config", default="SeCom/secom/configs/bge.yaml")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--compress-rate", type=float, default=0.75)
    parser.add_argument("--limit", type=int, default=0, help="0 means all")
    parser.add_argument("--skip-abs", action="store_true", help="skip abstention in retrieval metrics")
    parser.add_argument("--out-root", default="results/secom_bge")
    parser.add_argument("--resume", default=None, help="existing run directory")
    parser.add_argument("--reader-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seg-workers", type=int, default=8)
    parser.add_argument(
        "--prefilter-topn",
        type=int,
        default=0,
        help="segment only the top-N raw sessions by local BGE; 0 disables",
    )
    parser.add_argument(
        "--prefilter-keep-raw",
        action="store_true",
        help="also add non-shortlisted sessions to retrieval as raw documents",
    )
    parser.add_argument("--max-new", type=int, default=0, help="stop after N new questions")
    parser.add_argument("--deadline-epoch", type=float, default=0, help="stop before this Unix time")
    parser.add_argument("--cache", default="cache/secom_cache.sqlite")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="skip reader and judge; keep only retrieval metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    meta: Dict[str, Any] = {}
    if args.retrieval_only:
        args.no_judge = True
    if args.resume:
        run_dir = Path(args.resume).resolve()
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            raise SystemExit("resume directory has no meta.json: {}".format(run_dir))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        args.data = args.data or meta.get("data")
        args.subset = args.subset if args.subset is not None else meta.get("subset")
        args.topk = int(meta.get("topk", args.topk))
        args.compress_rate = float(meta.get("compress_rate", args.compress_rate))
        args.skip_abs = bool(meta.get("skip_abs", args.skip_abs))
        args.limit = int(meta.get("limit", args.limit))
        args.prefilter_topn = int(meta.get("prefilter_topn", args.prefilter_topn))
        args.prefilter_keep_raw = bool(
            meta.get("prefilter_keep_raw", args.prefilter_keep_raw)
        )
        if "no_judge" in meta:
            args.no_judge = bool(meta["no_judge"])
        if "retrieval_only" in meta:
            args.retrieval_only = bool(meta["retrieval_only"])
        if args.retrieval_only:
            args.no_judge = True
    else:
        if not args.data:
            raise SystemExit("--data is required for a new run")
        root = Path(args.out_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        base = root / "secom_{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
        run_dir = base
        suffix = 1
        while run_dir.exists():
            run_dir = Path(str(base) + "_{}".format(suffix))
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)

    if not args.data:
        raise SystemExit("no data path (pass --data or use a run with meta.json)")

    data_path = Path(args.data).resolve()
    subset_path = Path(args.subset).resolve() if args.subset else None
    config_path = Path(args.config).resolve()
    cache_path = Path(args.cache).resolve()

    log_path = run_dir / "run.log"
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_handle)
    sys.stderr = Tee(original_stderr, log_handle)

    started = time.time()
    cache: Optional[SegmentCache] = None
    executor: Optional[ThreadPoolExecutor] = None
    try:
        print("=" * 80, flush=True)
        print("run_dir:", run_dir, flush=True)
        print("data:", data_path, flush=True)
        print("subset:", subset_path, flush=True)
        print("config:", config_path, flush=True)

        data = load_data(str(data_path))
        print("load_data_sec:", round(time.time() - started, 1), flush=True)
        subset = load_subset(str(subset_path) if subset_path else None)
        if subset is not None:
            data = [item for item in data if str(item.get("question_id")) in subset]
        if args.skip_abs:
            data = [item for item in data if not is_abstention_item(item)]
            print("skip_abs: removed abstention questions", flush=True)
        if args.limit and args.limit > 0:
            data = data[: args.limit]
        n_planned = len(data)
        print("n_planned:", n_planned, flush=True)

        secom = SeCom(granularity="segment", config_path=str(config_path))
        print("secom_init_sec:", round(time.time() - started, 1), flush=True)
        segment_model = str(secom.config.segmentor.segment_model)
        reader_model = args.reader_model or segment_model
        judge_model = args.judge_model or reader_model
        compress_model = str(secom.config.compressor.compress_model)

        print("segment_model:", segment_model, flush=True)
        print("reader_model:", reader_model, flush=True)
        print("judge_model:", judge_model, flush=True)

        meta.update(
            {
                "run_id": run_dir.name,
                "data": str(data_path),
                "subset": str(subset_path) if subset_path else None,
                "topk": args.topk,
                "compress_rate": args.compress_rate,
                "skip_abs": bool(args.skip_abs),
                "limit": args.limit,
                "prefilter_topn": args.prefilter_topn,
                "prefilter_keep_raw": bool(args.prefilter_keep_raw),
                "no_judge": bool(args.no_judge),
                "retrieval_only": bool(args.retrieval_only),
                "n_planned": n_planned,
                "segment_model": segment_model,
                "reader_model": reader_model,
                "judge_model": judge_model,
                "retriever": args.embedding_model,
                "config": str(config_path),
                "seg_workers": args.seg_workers,
                "cmd": " ".join(sys.argv),
                "created_at": meta.get("created_at", now_iso()),
                "resumed_at": now_iso() if args.resume else None,
            }
        )
        (run_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        device = args.device
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            print("[warn] CUDA not available, falling back to CPU", flush=True)
            device = "cpu"
        print("embedding device:", device, flush=True)
        embeddings = HuggingFaceEmbeddings(
            model_name=args.embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE") or None
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is not set")
        client = OpenAI(api_key=api_key, base_url=api_base)

        if not args.no_cache:
            cache = SegmentCache(cache_path)
            print("cache:", cache_path, flush=True)

        log_lock = threading.Lock()

        def tlog(message: str) -> None:
            with log_lock:
                print(message, flush=True)

        executor = ThreadPoolExecutor(max_workers=max(1, args.seg_workers))
        pred_path = run_dir / "predictions.jsonl"
        predictions = load_predictions(pred_path)
        done = set(str(p.get("question_id")) for p in predictions)
        print("already done:", len(done), flush=True)

        consecutive_errors = 0
        new_count = 0
        paused = False
        for index, item in enumerate(data, 1):
            qid = str(item.get("question_id", ""))
            if qid in done:
                continue
            if args.max_new and new_count >= args.max_new:
                print("[pause] max_new reached", flush=True)
                paused = True
                break
            if args.deadline_epoch and time.time() >= args.deadline_epoch:
                print("[pause] deadline reached", flush=True)
                paused = True
                break
            new_count += 1

            question = str(item.get("question", ""))
            question_type = str(item.get("question_type", ""))
            question_date = str(item.get("question_date", ""))
            answer = str(item.get("answer", ""))
            gold_session_ids = [str(x) for x in (item.get("answer_session_ids") or [])]
            is_abstention = is_abstention_item(item)
            n_sessions_total = len(item.get("haystack_sessions") or [])

            item_start = time.time()
            prefilter_raw_docs: List[Document] = []
            n_segmented_sessions = n_sessions_total
            try:
                if args.prefilter_topn:
                    item, prefilter_raw_docs = prefilter_item(
                        item,
                        question,
                        embeddings,
                        args.prefilter_topn,
                        keep_raw=args.prefilter_keep_raw,
                    )
                documents, n_sessions, n_units, n_failed = build_memory(
                    secom=secom,
                    item=item,
                    topk=args.topk,
                    rate=args.compress_rate,
                    cache=cache,
                    executor=executor,
                    tlog=tlog,
                    compress_model=compress_model,
                )
                n_segmented_sessions = n_sessions
                if prefilter_raw_docs:
                    base = len(documents)
                    for offset, raw_doc in enumerate(prefilter_raw_docs):
                        raw_doc.metadata["unit_index"] = base + offset
                    documents.extend(prefilter_raw_docs)
                    n_units += len(prefilter_raw_docs)
                    n_sessions = n_sessions_total
                t_memory = time.time()
                hits = retrieve(documents, embeddings, question, args.topk)
                t_retrieve = time.time()
                t_reader = t_retrieve
                hypothesis = None
                if not args.retrieval_only:
                    prompt = build_reader_prompt(question, question_date, hits)
                    hypothesis = chat(
                        client,
                        reader_model,
                        [{"role": "user", "content": prompt}],
                        max_tokens=args.max_tokens,
                    )
                    t_reader = time.time()

                qa_correct = None
                if not args.no_judge and not args.retrieval_only:
                    qa_correct = judge_qa(
                        client=client,
                        model=judge_model,
                        question=question,
                        answer=answer,
                        hypothesis=hypothesis,
                        question_type=question_type,
                        abstention=is_abstention,
                    )
                t_judge = time.time()

                retrieved_session_ids = unique_session_ids(hits)
                retrieved_unit_ids = [int(doc.metadata.get("unit_index", -1)) for doc, _ in hits]
                scores = retrieval_scores(retrieved_session_ids, gold_session_ids, ks=(5, 10))

                pred = {
                    "question_id": qid,
                    "question_type": question_type,
                    "question": question,
                    "answer": answer,
                    "question_date": question_date,
                    "hypothesis": hypothesis,
                    "is_abstention": is_abstention,
                    "gold_session_ids": gold_session_ids,
                    "retrieved_session_ids": retrieved_session_ids,
                    "retrieved_unit_ids": retrieved_unit_ids,
                    "n_sessions": n_sessions,
                    "n_sessions_total": n_sessions_total,
                    "n_segmented_sessions": n_segmented_sessions,
                    "n_raw_sessions": len(prefilter_raw_docs),
                    "n_units": n_units,
                    "n_seg_failed": n_failed,
                    "qa_correct": qa_correct,
                    "elapsed_sec": round(time.time() - item_start, 3),
                    "stage_sec": {
                        "memory": round(t_memory - item_start, 3),
                        "retrieve": round(t_retrieve - t_memory, 3),
                        "reader": round(t_reader - t_retrieve, 3),
                        "judge": round(t_judge - t_reader, 3),
                    },
                }
                pred.update(scores)
                append_prediction(pred_path, pred)
                predictions.append(pred)
                done.add(qid)
                consecutive_errors = 0

                print(
                    "[{}/{}] {} type={} sessions={} units={} seg_fail={} "
                    "ret={} qa={} {:.1f}s".format(
                        len(done),
                        n_planned,
                        qid,
                        question_type,
                        n_sessions,
                        n_units,
                        n_failed,
                        len(retrieved_session_ids),
                        qa_correct,
                        pred["elapsed_sec"],
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                consecutive_errors += 1
                tlog("[ERROR] {}: {}".format(qid, exc))
                traceback.print_exc()
                pred = {
                    "question_id": qid,
                    "question_type": question_type,
                    "question": question,
                    "answer": answer,
                    "question_date": question_date,
                    "hypothesis": "",
                    "is_abstention": is_abstention,
                    "gold_session_ids": gold_session_ids,
                    "retrieved_session_ids": [],
                    "retrieved_unit_ids": [],
                    "n_sessions": len(item.get("haystack_sessions") or []),
                    "n_sessions_total": n_sessions_total,
                    "n_segmented_sessions": n_segmented_sessions,
                    "n_raw_sessions": len(prefilter_raw_docs),
                    "n_units": 0,
                    "n_seg_failed": 0,
                    "qa_correct": None,
                    "error": "{}: {}".format(type(exc).__name__, exc),
                    "elapsed_sec": round(time.time() - item_start, 3),
                }
                append_prediction(pred_path, pred)
                predictions.append(pred)
                done.add(qid)
                if consecutive_errors >= 5:
                    raise SystemExit("too many consecutive errors; stopping")

        predictions = dedupe_predictions(predictions)
        aggregate = aggregate_predictions(predictions, skip_abs=args.skip_abs)
        aggregate.update(
            {
                "run_id": run_dir.name,
                "topk": args.topk,
                "skip_abs": bool(args.skip_abs),
                "n_planned": n_planned,
                "n_done": len(predictions),
                "n_error": sum(1 for p in predictions if p.get("error")),
                "retrieval_only": bool(args.retrieval_only),
                "elapsed_sec": round(time.time() - started, 3),
            }
        )
        if args.retrieval_only:
            for key in (
                "n_qa_judged",
                "qa_accuracy",
                "qa_by_type",
                "n_abstention",
                "qa_abstention_accuracy",
            ):
                aggregate.pop(key, None)
        (run_dir / "aggregate.json").write_text(
            json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("=" * 80, flush=True)
        print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
        expected_planned = 470 if args.skip_abs else 500
        print("ACCEPTANCE_META_OK" if (
            int(meta.get("topk", -1)) == 10
            and bool(meta.get("skip_abs")) is True
            and int(meta.get("n_planned", -1)) == expected_planned
        ) else "ACCEPTANCE_META_CHECK", flush=True)
        if paused:
            print("[pause] exiting for off-peak supervisor", flush=True)
            raise SystemExit(3)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if cache is not None:
            cache.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.flush()
        log_handle.close()


if __name__ == "__main__":
    main()
