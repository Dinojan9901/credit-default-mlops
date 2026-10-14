"""Replay rows from a CSV against the API to populate the Grafana dashboard.

    python scripts/generate_traffic.py --csv data/monitoring/current_nodrift.csv
    python scripts/generate_traffic.py --csv data/monitoring/current_drift.csv --rps 20

Standard library only. Mixes single and batch calls, plus a few invalid requests so the
4xx series isn't empty.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

FEATURES = [
    "limit_bal", "sex", "education", "marriage", "age",
    *[f"pay_{i}" for i in range(1, 7)],
    *[f"bill_amt{i}" for i in range(1, 7)],
    *[f"pay_amt{i}" for i in range(1, 7)],
]  # fmt: skip


def post(url: str, payload: Any) -> int:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as err:
        return int(err.code)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--rps", type=float, default=10.0)
    ap.add_argument("--duration", type=float, default=120.0, help="seconds")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with Path(args.csv).open(newline="") as f:
        rows = [{k: int(float(r[k])) for k in FEATURES} for r in csv.DictReader(f)]

    base = args.url.rstrip("/")
    end, sent, statuses = time.monotonic() + args.duration, 0, {}
    while time.monotonic() < end:
        roll = rng.random()
        if roll < 0.02:
            status = post(f"{base}/predict", {**rng.choice(rows), "age": 5})  # invalid
        elif roll < 0.12:
            status = post(f"{base}/predict-batch", {"records": rng.sample(rows, 20)})
        else:
            status = post(f"{base}/predict", rng.choice(rows))
        statuses[status] = statuses.get(status, 0) + 1
        sent += 1
        time.sleep(1.0 / args.rps)
    print(f"sent {sent} requests: {statuses}")


if __name__ == "__main__":
    main()
