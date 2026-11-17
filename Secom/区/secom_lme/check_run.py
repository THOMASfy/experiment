# -*- coding: utf-8 -*-
import json
import re
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python check_run.py <run_dir>")
    run = Path(sys.argv[1])
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    print("topk      =", meta.get("topk"))
    print("skip_abs  =", meta.get("skip_abs"))
    print("n_planned =", meta.get("n_planned"))

    assert int(meta["topk"]) == 10, "topk != 10"
    assert meta.get("skip_abs") in (True, 1, "true", "True"), "skip_abs != true"
    expected = 470 if meta.get("skip_abs") in (True, 1, "true", "True") else 500
    assert int(meta["n_planned"]) == expected, "n_planned != {}".format(expected)

    log = (run / "run.log").read_text(encoding="utf-8", errors="ignore")
    bad = log.count("bad response")
    fallback = log.count("not segmented")
    units = sum(int(x) for x in re.findall(r"units=(\d+)", log))
    print("bad_response      =", bad)
    print("fallback_sessions =", fallback)
    print("units_reported    =", units)
    assert bad == 0, "run.log has bad response"
    assert fallback == 0, "run.log has segmentation fallback"
    print("ACCEPTANCE OK")


if __name__ == "__main__":
    main()
