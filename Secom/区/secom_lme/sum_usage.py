# -*- coding: utf-8 -*-
import json
from collections import defaultdict

path = r"D:\longmemeval\results\two\usage.jsonl"
agg = defaultdict(lambda: defaultdict(int))
models = defaultdict(int)
lines = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        lines += 1
        row = json.loads(line)
        src = row.get("source", "?")
        u = row.get("usage") or {}
        agg[src]["calls"] += 1
        agg[src]["prompt"] += int(u.get("prompt_tokens") or 0)
        agg[src]["completion"] += int(u.get("completion_tokens") or 0)
        agg[src]["total"] += int(u.get("total_tokens") or 0)
        agg[src]["cache_hit"] += int(u.get("prompt_cache_hit_tokens") or 0)
        agg[src]["cache_miss"] += int(u.get("prompt_cache_miss_tokens") or 0)
        models[row.get("model", "?")] += 1

print("lines", lines)
for src, v in agg.items():
    print(src, dict(v))
print("models", dict(models))
print("grand", {k: sum(v[k] for v in agg.values()) for k in ("calls", "prompt", "completion", "total", "cache_hit", "cache_miss")})
