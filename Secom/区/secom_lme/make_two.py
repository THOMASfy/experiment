# -*- coding: utf-8 -*-
import json

src = r"D:\longmemeval\longmemeval_m_cleaned.json"
dst = r"D:\longmemeval\two_questions.json"
with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)
if isinstance(data, dict):
    data = data.get("data", list(data.values()))
rows = [x for x in data if not str(x.get("question_id", "")).endswith("_abs")][:2]
with open(dst, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False)
print("wrote", dst, "n=", len(rows))
for x in rows:
    print(x["question_id"], x["question_type"], len(x.get("haystack_sessions") or []))
