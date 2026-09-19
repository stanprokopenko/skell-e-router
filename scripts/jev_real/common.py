"""Shared helpers for the jev-real benchmark runners.

Small on purpose: a timed call wrapper, an append-only jsonl writer, a latency
summary, and the ``--run`` gate every runner in this folder uses. Nothing here
knows about routing; other runners reuse it unchanged.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def timed(fn, *args, **kwargs) -> dict:
    """Run ``fn`` and return its dict result plus elapsed_s, or an error row.

    The callable must return a dict. Errors never propagate: the caller scores
    a failure the same way production would, so the run finishes either way.
    Router errors keep their machine-readable code; anything else is recorded
    by exception type only, so provider messages (which can echo the prompt)
    stay out of the results file.
    """
    started = time.perf_counter()
    try:
        row = dict(fn(*args, **kwargs))
        row["error"] = None
    except Exception as exc:  # noqa: BLE001 - a failed call is a data point
        code = getattr(exc, "code", None)
        row = {"error": code if isinstance(code, str) and code else type(exc).__name__}
    row["elapsed_s"] = time.perf_counter() - started
    return row


class JsonlWriter:
    """Append-only jsonl sink that refuses to touch an existing file.

    Exclusive creation is deliberate: these runs cost money, so a rerun must
    not silently overwrite the evidence of the previous one.
    """

    def __init__(self, path: Path, *, overwrite_ok: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a" if overwrite_ok else "x", encoding="utf-8")

    def write(self, row: dict) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def latency_summary(seconds) -> dict:
    """median / p95 / mean over a sequence of call durations."""
    values = sorted(float(s) for s in seconds if s is not None)
    if not values:
        return {"n": 0, "median_s": None, "p95_s": None, "mean_s": None}
    p95_index = min(len(values) - 1, int(0.95 * len(values)))
    return {
        "n": len(values),
        "median_s": statistics.median(values),
        "p95_s": values[p95_index],
        "mean_s": statistics.fmean(values),
    }


def add_run_gate(parser, *, default_budget: float) -> None:
    """The flags every runner here shares: --run, --limit, --budget."""
    parser.add_argument("--run", action="store_true",
                        help="actually call the models; without it only the plan prints")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N cases (deterministic order)")
    parser.add_argument("--budget", type=float, default=default_budget,
                        help="hard cap in USD; the run refuses to start above it")


def gate(args, *, planned_calls: int, estimated_usd: float, detail: dict | None = None) -> bool:
    """Print the plan and say whether calls may proceed.

    Returns False when the caller should stop, either because ``--run`` is
    absent or because the estimate is over budget (which exits nonzero).
    """
    plan = {"planned_calls": planned_calls, "estimated_usd": round(estimated_usd, 4),
            "budget_usd": args.budget, "run": bool(args.run)}
    if detail:
        plan.update(detail)
    print(json.dumps(plan, indent=2), flush=True)
    if estimated_usd > args.budget:
        print("Estimate exceeds the budget cap. Lower --limit or raise --budget.", flush=True)
        raise SystemExit(2)
    if not args.run:
        print("Dry run. Re-run with --run to make the calls.", flush=True)
        return False
    return True
