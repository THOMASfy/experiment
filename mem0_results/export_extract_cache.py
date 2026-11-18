#!/usr/bin/env python3
"""Export the LongMemEval-S slice of a full mem0-lme extraction cache.

Why this exists: the full cache produced by the S + M runs is ~892MB, almost all
of it BGE vectors (626MB) plus entries that belong to the M dataset. Shipping
that in git is impossible, so this script keeps the expensive part -- the LLM
fact extraction, ~11MB of text -- and only for session keys reachable from
`longmemeval_s_cleaned.json`.

Session keys must match `lme_profile.session_cache_key(msgs, date, "auto")`
exactly, which is why this reuses `clean_turns_local` / `session_cache_key`
instead of re-implementing the hashing.

Usage
-----
    python mem0_results/export_extract_cache.py \
        --cache results/extract_cache/cache_resolve_auto.sqlite \
        --data longmemeval_s_cleaned.json \
        --out-dir mem0_results/mem0_lme_s/cache
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lme_profile import clean_turns_local, session_cache_key  # noqa: E402


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def s_session_keys(dataset: str) -> set:
    with open(dataset, encoding="utf-8") as handle:
        items = json.load(handle)
    keys = set()
    for item in items:
        sessions = item.get("haystack_sessions") or []
        dates = item.get("haystack_dates") or []
        for index, turns in enumerate(sessions):
            msgs = clean_turns_local(turns)
            if not msgs:
                continue
            date = dates[index] if index < len(dates) else ""
            keys.add(session_cache_key(msgs, str(date), "auto"))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="full extraction cache sqlite")
    parser.add_argument("--data", required=True, help="longmemeval_s_cleaned.json")
    parser.add_argument("--out-dir", required=True, help="where to write .sqlite.gz + manifest.json")
    args = parser.parse_args()

    keys = s_session_keys(args.data)
    os.makedirs(args.out_dir, exist_ok=True)
    plain = os.path.join(args.out_dir, "extract_cache_s_facts.sqlite")
    archive = plain + ".gz"
    if os.path.exists(plain):
        os.remove(plain)

    source = sqlite3.connect("file:" + args.cache.replace("\\", "/") + "?mode=ro", uri=True)
    target = sqlite3.connect(plain)
    target.execute("CREATE TABLE facts (key TEXT PRIMARY KEY, facts TEXT NOT NULL)")
    rows = []
    for key in keys:
        row = source.execute("select facts from facts where key=?", (key,)).fetchone()
        if row is not None:
            rows.append((key, row[0]))
    target.executemany("INSERT INTO facts (key, facts) VALUES (?, ?)", rows)
    target.commit()
    target.execute("VACUUM")
    target.close()
    source.close()

    with open(plain, "rb") as fin, gzip.open(archive, "wb", compresslevel=9) as fout:
        shutil.copyfileobj(fin, fout)
    os.remove(plain)

    manifest = {
        "kind": "mem0-lme extraction cache (facts only)",
        "source": args.cache,
        "source_sha256": sha256(args.cache),
        "source_size_bytes": os.path.getsize(args.cache),
        "dataset": os.path.basename(args.data),
        "dataset_sha256": sha256(args.data),
        "cache_key_mode": "auto (md5 of cleaned session turns, + |date when relative time words present)",
        "facts_rows": len(rows),
        "s_session_keys": len(keys),
        "missing_keys": len(keys) - len(rows),
        "vectors_included": False,
        "vectors_note": (
            "BGE-base vectors are omitted: ~250MB of float32 for S alone; the runner "
            "re-embeds any fact text missing from the vectors table."
        ),
        "file": os.path.basename(archive),
        "file_sha256": sha256(archive),
    }
    # keep the release pointer if a previous manifest recorded one
    previous = os.path.join(args.out_dir, "manifest.json")
    if os.path.exists(previous):
        try:
            with open(previous, encoding="utf-8") as handle:
                old = json.load(handle)
            if "full_cache_with_vectors" in old:
                manifest["full_cache_with_vectors"] = old["full_cache_with_vectors"]
        except (json.JSONDecodeError, OSError):
            pass
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
