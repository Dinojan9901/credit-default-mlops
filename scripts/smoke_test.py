"""Black-box smoke test for a running API (docker-compose, kind, CI).

Standard library only, so it runs anywhere Python does:
    python scripts/smoke_test.py --url http://localhost:30080 --timeout 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

EXAMPLE = {
    "limit_bal": 50000,
    "sex": 2,
    "education": 2,
    "marriage": 1,
    "age": 35,
    "pay_1": 0,
    "pay_2": 0,
    "pay_3": 0,
    "pay_4": 0,
    "pay_5": 0,
    "pay_6": 0,
    "bill_amt1": 30000,
    "bill_amt2": 28000,
    "bill_amt3": 26000,
    "bill_amt4": 24000,
    "bill_amt5": 22000,
    "bill_amt6": 20000,
    "pay_amt1": 2000,
    "pay_amt2": 2000,
    "pay_amt3": 2000,
    "pay_amt4": 2000,
    "pay_amt5": 2000,
    "pay_amt6": 2000,
}
RISKY = {
    **EXAMPLE,
    "pay_1": 3,
    "pay_2": 2,
    "pay_3": 2,
    "pay_amt1": 0,
    "pay_amt2": 0,
    "bill_amt1": 49000,
}


def call(url: str, payload: Any | None = None) -> tuple[int, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
            status = resp.status
    except urllib.error.HTTPError as err:
        body, status = err.read().decode(), err.code
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


def wait_ready(base: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        try:
            status, last = call(f"{base}/health")
            if status == 200:
                return
        except OSError as exc:
            last = exc
        time.sleep(2)
    raise SystemExit(f"API not ready after {timeout}s: {last}")


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()
    base = args.url.rstrip("/")

    wait_ready(base, args.timeout)
    print(f"API ready at {base}")

    status, info = call(f"{base}/model-info")
    check(status == 200 and "version" in info, f"/model-info -> {status}")
    print(json.dumps(info, indent=2))

    status, pred = call(f"{base}/predict", EXAMPLE)
    check(status == 200 and 0 <= pred["default_probability"] <= 1, f"/predict -> {pred}")

    status, batch = call(f"{base}/predict-batch", {"records": [EXAMPLE, RISKY]})
    check(status == 200 and batch["n_records"] == 2, f"/predict-batch -> {status}")
    p_safe, p_risky = (p["default_probability"] for p in batch["predictions"])
    check(p_risky > p_safe, f"late payer scored riskier ({p_risky:.3f} > {p_safe:.3f})")

    status, _ = call(f"{base}/predict", {**EXAMPLE, "age": 5})
    check(status == 422, f"invalid input rejected -> {status}")

    with urllib.request.urlopen(f"{base}/metrics", timeout=10) as resp:
        metrics = resp.read().decode()
    check("model_prediction_probability_count" in metrics, "/metrics exposes prediction metrics")
    print("Smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
