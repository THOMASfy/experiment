# -*- coding: utf-8 -*-
import json
from collections import Counter

path = r"D:\longmemeval\longmemeval_m_cleaned.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
if isinstance(data, dict):
    data = data.get("data", list(data.values()))
print("n=", len(data))
print("types=", Counter(x.get("question_type") for x in data))
print("abs=", sum(str(x.get("question_id", "")).endswith("_abs") for x in data))
for x in data[:8]:
    qid = x.get("question_id")
    sessions = x.get("haystack_sessions") or []
    turns = sum(len(s) for s in sessions)
    print(qid, x.get("question_type"), "sessions=", len(sessions), "turns=", turns,
          "qchars=", len(str(x.get("question", ""))), "abs=", str(qid).endswith("_abs"))
