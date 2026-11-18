#!/usr/bin/env python3
"""Restore the published LongMemEval-S mem0 extraction cache.

The archive in this directory (`cache/extract_cache_s_facts.sqlite.gz`) holds the
LLM fact-extraction results for every session that the S run touched, keyed the
same way `lme_profile.session_cache_key(..., key_mode="auto")` keys them.
Vectors are not included; the runner re-embeds any fact text it cannot find in
the `vectors` table, so a restored facts-only cache is fully usable.

Usage
-----
    # restore a fresh cache for an S run
    python mem0_results/restore_extract_cache.py \
        --out results/extract_cache/cache_resolve_auto.sqlite

    # add the S entries into an existing cache (keeps M entries)
    python mem0_results/restore_extract_cache.py --merge

Then point the runner at it:

    python run_mem0_lme.py ... --extract-cache <out> --lme-cache-key auto
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARCHIVE = os.path.join(HERE, "mem0_lme_s", "cache", "extract_cache_s_facts.sqlite.gz")
DEFAULT_MANIFEST = os.path.join(HERE, "mem0_lme_s", "cache", "manifest.json")
DEFAULT_OUT = os.path.join("results", "extract_cache", "cache_resolve_auto.sqlite")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="merge into an existing cache instead of overwriting it",
    )
    parser.add_argument("--skip-hash", action="store_true", help="skip archive checksum check")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if not args.skip_hash:
        actual = sha256(args.archive)
        if actual != manifest.get("file_sha256"):
            raise SystemExit(
                f"checksum mismatch for {args.archive}\n"
                f"  expected {manifest.get('file_sha256')}\n  actual   {actual}"
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        with gzip.open(args.archive, "rb") as fin, open(tmp, "wb") as fout:
            shutil.copyfileobj(fin, fout)

        if args.merge and os.path.exists(args.out):
            conn = sqlite3.connect(args.out)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS facts (key TEXT PRIMARY KEY, facts TEXT NOT NULL)"
            )
            conn.execute("ATTACH DATABASE ? AS src", (tmp,))
            before = conn.execute("select count(*) from facts").fetchone()[0]
            conn.execute("INSERT OR IGNORE INTO facts (key, facts) SELECT key, facts FROM src.facts")
            conn.commit()
            after = conn.execute("select count(*) from facts").fetchone()[0]
            conn.execute("DETACH DATABASE src")
            conn.close()
            print(f"merged into {args.out}: {before} -> {after} facts rows")
        else:
            if os.path.exists(args.out):
                os.remove(args.out)
            shutil.move(tmp, args.out)
            conn = sqlite3.connect(args.out)
            rows = conn.execute("select count(*) from facts").fetchone()[0]
            conn.close()
            print(f"restored {rows} facts rows to {args.out}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    print(
        "vectors are not included; the runner will re-embed missing fact texts "
        f"({manifest.get('vectors_note', '')})"
    )


if __name__ == "__main__":
    main()
