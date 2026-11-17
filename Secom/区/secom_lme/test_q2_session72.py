# -*- coding: utf-8 -*-
import json
import time

from run_secom_lme import segment_one, session_to_exchanges
from secom import SeCom

with open(r"D:\longmemeval\two_questions.json", "r", encoding="utf-8") as f:
    data = json.load(f)
item = data[1]
exchanges = session_to_exchanges(item["haystack_sessions"][72])
print("qid", item["question_id"], "exchanges", len(exchanges), flush=True)
t0 = time.time()
secom = SeCom(granularity="segment", config_path=r"D:\longmemeval\SeCom\secom\configs\bge.yaml")
print("secom_init", round(time.time() - t0, 1), flush=True)
t1 = time.time()
segments, error = segment_one(secom, exchanges)
print("segment_sec", round(time.time() - t1, 2), "error", error, "n_segments", len(segments or []), flush=True)
print("coverage", sum(len(s) for s in (segments or [])), "/", len(exchanges), flush=True)
print(segments, flush=True)
