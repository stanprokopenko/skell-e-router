"""Build a join bundle's feature rows from the feature runs it joins. No Jev calls.

Round two, third pass, step 4 of
``docs/superpowers/specs/2026-09-26-jev-roughcut-round-two-design.md``: f3 is
f1's eighteen questions plus f2's ``said_earlier`` and ``wrap_up``. f1 and f2
asked their questions over the same state, blocks and sentences, each question
answered independently of the others, so f3's rows are the f1 rows with the two
f2 columns added, joined by episode and sentence id. The bundle's
``joined_from`` (``roughcut_jev_prompts.py``) says which run supplies which
column; the first source is the base row.

Checks, all hard failures: both runs cover the same episodes in the same order
and the same sentence ids in the same order, the sentence text, the code
features and the v3 join agree cell for cell, and every question cell of the
joined row is answered (no ``None``).

Writes ``docs/jev-real/<prefix>-<split>-features.jsonl`` and
``-timing.json`` for ``split`` in fit and heldout; both refuse overwrite. The
timing file records zero requests and $0 for the join itself, and carries each
source run's own per-episode seconds and dollars so the combiner can report
them.

Usage::

  python scripts/jev_real/roughcut_jev_join.py --feature-version f3
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_DIR = ROOT / "docs" / "jev-real"
sys.path.insert(0, str(HERE))

from roughcut_jev_prompts import feature_prompts_for  # noqa: E402

SPLITS = ("fit", "heldout")


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def rel(path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def join_split(bundle, prefix_for, split):
    """``(rows, timing)`` for one split; raises SystemExit on any mismatch."""
    sources = list(bundle.joined_from)
    base = sources[0]
    runs = {}
    for source in sources:
        name = f"{prefix_for(source)}-{split}"
        features = OUT_DIR / f"{name}-features.jsonl"
        timing = OUT_DIR / f"{name}-timing.json"
        for path in (features, timing):
            if not path.exists():
                raise SystemExit(f"missing source file {path}")
        rows = read_jsonl(features)
        bad = [r["id"] for r in rows if r.get("feature_version") != source]
        if bad:
            raise SystemExit(f"{features.name}: rows not asked by bundle {source}: {bad[:5]}")
        runs[source] = {"name": name, "features": features, "timing_path": timing, "rows": rows,
                        "timing": json.loads(timing.read_text(encoding="utf-8"))}

    keys = [k for k, _t in bundle.questions]
    base_rows = runs[base]["rows"]
    joined, missing = [], []
    for source in sources[1:]:
        other = runs[source]["rows"]
        if len(other) != len(base_rows):
            raise SystemExit(f"{split}: {source} has {len(other)} rows, {base} has {len(base_rows)}")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for i, row in enumerate(base_rows):
        q = {}
        for source in sources:
            src = runs[source]["rows"][i]
            if (src["episode"], src["id"]) != (row["episode"], row["id"]):
                raise SystemExit(f"{split} row {i}: {source} is at {src['episode']} {src['id']}, "
                                 f"{base} at {row['episode']} {row['id']}")
            for field in ("text", "code", "v3"):
                if src[field] != row[field]:
                    raise SystemExit(f"{split} {row['episode']} {row['id']}: {field} differs "
                                     f"between {source} and {base}")
            for key in bundle.joined_from[source]:
                if key not in src["q"]:
                    raise SystemExit(f"{split} {row['episode']} {row['id']}: {source} has no {key}")
                q[key] = src["q"][key]
        ordered = {k: q[k] for k in keys}
        missing += [(row["episode"], row["id"], k) for k, v in ordered.items() if v is None]
        joined.append({"episode": row["episode"], "id": row["id"], "feature_version": bundle.version,
                       "q": ordered, "code": row["code"], "v3": row["v3"], "text": row["text"],
                       "joined_utc": stamp,
                       "source_recorded_utc": {s: runs[s]["rows"][i].get("recorded_utc") for s in sources}})
    if missing:
        raise SystemExit(f"{split}: {len(missing)} unanswered cells in the joined rows, first {missing[:5]}")

    episodes = list(dict.fromkeys(r["episode"] for r in joined))
    for source in sources:
        t_eps = runs[source]["timing"]["episodes"]
        absent = [e for e in episodes if e not in t_eps]
        if absent:
            raise SystemExit(f"{runs[source]['timing_path'].name} has no timing for {absent}")
    per_episode = {}
    for e in episodes:
        src = {s: runs[s]["timing"]["episodes"][e] for s in sources}
        per_episode[e] = {
            # Seconds: the measured wall clock of the source passes whose answers
            # the joined rows read, summed. Dollars: the join made no request.
            "wall_clock_s": round(sum(src[s]["wall_clock_s"] for s in sources), 3),
            "cost_usd": 0.0, "requests": 0, "errors": 0, "unanswered_cells": 0,
            "parts_per_block": {},
            "sentences": sum(1 for r in joined if r["episode"] == e),
            "v3_run": src[base].get("v3_run"),
            "source_runs": {s: {"run": runs[s]["name"], "wall_clock_s": src[s]["wall_clock_s"],
                                "cost_usd": src[s]["cost_usd"], "requests": src[s]["requests"]}
                            for s in sources},
        }
    timing = {
        "generated_utc": stamp, "script": "scripts/jev_real/roughcut_jev_join.py",
        "kind": "join", "feature_version": bundle.version, "questions": keys,
        "joined_from": bundle.joined_from, "base": base,
        "sources": {s: {"run": runs[s]["name"], "features": rel(runs[s]["features"]),
                        "features_md5": md5(runs[s]["features"]),
                        "timing": rel(runs[s]["timing_path"]),
                        "timing_md5": md5(runs[s]["timing_path"]),
                        "cost_usd": runs[s]["timing"]["totals"]["cost_usd"],
                        "requests": runs[s]["timing"]["totals"]["requests"]}
                    for s in sources},
        "new_jev_requests": 0,
        "episodes": per_episode,
        "totals": {"episodes": len(episodes), "sentences": len(joined), "requests": 0, "errors": 0,
                   "failed_parts": 0, "unanswered_cells": 0, "cost_usd": 0.0,
                   "source_cost_usd": {s: runs[s]["timing"]["totals"]["cost_usd"] for s in sources},
                   "source_wall_clock_s": {s: runs[s]["timing"]["totals"]["wall_clock_s"] for s in sources}},
    }
    return joined, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feature-version", default="f3", help="the join bundle to build")
    parser.add_argument("--prefix", default=None,
                        help="output prefix (default roughcut-jev-<version>); sources are "
                             "roughcut-jev-<source>-<split>")
    args = parser.parse_args()
    bundle = feature_prompts_for(args.feature_version)
    if not bundle.joined_from:
        parser.error(f"bundle {bundle.version} is not a join bundle")
    prefix = args.prefix or f"roughcut-jev-{bundle.version}"
    outputs = [OUT_DIR / f"{prefix}-{split}-{kind}" for split in SPLITS
               for kind in ("features.jsonl", "timing.json")]
    existing = [str(p) for p in outputs if p.exists()]
    if existing:
        parser.error(f"refusing to overwrite {existing}")

    started = time.perf_counter()
    results = {split: join_split(bundle, lambda s: f"roughcut-jev-{s}", split) for split in SPLITS}
    elapsed = time.perf_counter() - started
    summary = {}
    for split, (rows, timing) in results.items():
        timing["totals"]["wall_clock_s"] = round(elapsed, 3)
        features = OUT_DIR / f"{prefix}-{split}-features.jsonl"
        with features.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (OUT_DIR / f"{prefix}-{split}-timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        summary[split] = {"episodes": timing["totals"]["episodes"], "sentences": len(rows),
                          "questions": len(timing["questions"]), "unanswered_cells": 0,
                          "new_jev_requests": 0, "cost_usd": 0.0,
                          "source_cost_usd": timing["totals"]["source_cost_usd"]}
    print(json.dumps({"feature_version": bundle.version, "joined_from": bundle.joined_from,
                      "join_seconds": round(elapsed, 3), "splits": summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
