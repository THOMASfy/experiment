# -*- coding: utf-8 -*-
import json
import shutil
from datetime import datetime
from pathlib import Path

work = Path(r"D:\longmemeval")
run = work / "results" / "secom_bge" / "auto_470"
run.mkdir(parents=True, exist_ok=True)

q1_path = work / "results" / "two" / "secom_20261112-151205" / "predictions.jsonl"
q2_path = work / "results" / "two_fix" / "secom_20261112-152826" / "predictions.jsonl"

rows = []
for path, qid in ((q1_path, "7161e7e2"), (q2_path, "c4f10528")):
    found = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if str(item.get("question_id")) == qid:
                found = item
                break
    if found is None:
        raise SystemExit("missing {} in {}".format(qid, path))
    rows.append(found)

with (run / "predictions.jsonl").open("w", encoding="utf-8") as f:
    for item in rows:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

meta = {
    "run_id": run.name,
    "data": str(work / "longmemeval_m_cleaned.json"),
    "subset": None,
    "topk": 10,
    "compress_rate": 0.75,
    "skip_abs": True,
    "limit": 0,
    "n_planned": 470,
    "segment_model": "deepseek-flash",
    "reader_model": "deepseek-flash",
    "judge_model": "deepseek-flash",
    "retriever": "BAAI/bge-base-en-v1.5",
    "config": str(work / "SeCom" / "secom" / "configs" / "bge.yaml"),
    "seg_workers": 8,
    "cmd": "seeded from two-question run; remaining 468 will be added by auto_offpeak.py",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "resumed_at": None,
}
(run / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

cache_src = work / "results" / "two" / "cache.sqlite"
cache_dst = run / "cache.sqlite"
if cache_src.exists():
    shutil.copy2(cache_src, cache_dst)
print("seeded", run)
print("predictions", [x["question_id"] for x in rows])
print("cache", cache_dst, cache_dst.exists())
