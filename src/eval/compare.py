#!/usr/bin/env python3
"""
compare.py - compare evaluated systems: mean and spread over runs, per slice.

Usage:
    python src/eval/compare.py [--results results] [--metric success] [--min-items 60]

Reads every results/gen_*_items.jsonl and recomputes the scores from the per-item
records, so runs made before a metric existed are scored the same way. A "system" is
the label in the file name (model, plus "+fewshot"); fine-tuned seeds are separate
systems (e.g. ra-qwen3-1.7b-s0) and are also pooled as "ra-qwen3-1.7b-s*".

For each system and slice it prints the mean over runs and the standard deviation
(over repeated eval runs of the same model, or over training seeds for the pool), and
the eval noise: the share of items whose success flips between two runs of one model.
A difference between systems smaller than these spreads is not a difference.

Slices: all, heldout (papers never used in fine-tuning: the slice the conclusion uses),
dev, test. Runs on fewer than --min-items items are listed but not aggregated.
"""
import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_examples import PAPER_SPLIT  # noqa: E402
from items import ROOT, item_split, load_items, paper_splits  # noqa: E402

SLICES = ("all", "heldout", "dev", "test")


def success(r):
    s = r["scores"]
    if r["answerable"]:
        return bool(s["rules_ok"] and s["answered"] and s["gold_cited"])
    return bool(s["clean_abstain"])


def load_runs(results):
    items = {i["item_id"]: i for i in load_items()}
    splits = paper_splits()
    runs = []
    for f in sorted(results.glob("gen_*_items.jsonl")):
        m = re.match(r"gen_(.+)_(\d{8}-\d{6})_items\.jsonl$", f.name)
        if not m:
            continue
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        rows = [r for r in rows if r["item_id"] in items]
        for r in rows:
            it = items[r["item_id"]]
            r["success"] = success(r)
            r["slices"] = {"all", item_split(it, splits),
                           *(["heldout"] if not set(it["paper_ids"]) & set(PAPER_SPLIT) else [])}
        runs.append({"system": m.group(1), "stamp": m.group(2), "file": f.name, "rows": rows})
    return runs


def rate(rows, key, sl):
    vals = [r[key] if key == "success" else r["scores"].get(key) for r in rows if sl in r["slices"]]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def flip_rate(runs):
    """Share of shared items whose success differs, averaged over pairs of runs."""
    rates = []
    for a, b in combinations(runs, 2):
        sa = {r["item_id"]: r["success"] for r in a["rows"]}
        sb = {r["item_id"]: r["success"] for r in b["rows"]}
        common = sa.keys() & sb.keys()
        if common:
            rates.append(sum(sa[i] != sb[i] for i in common) / len(common))
    return st.mean(rates) if rates else None


def fmt(vals):
    if not vals:
        return "-"
    m = st.mean(vals)
    return f"{m:.2f}" + (f" ± {st.stdev(vals):.2f}" if len(vals) > 1 else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--metric", default="success",
                    help="success, or a per-item score such as rules_ok, gold_cited, clean_abstain")
    ap.add_argument("--min-items", type=int, default=60)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    runs = load_runs(args.results)
    print("runs:")
    for r in runs:
        used = "" if len(r["rows"]) >= args.min_items else "  (partial, not aggregated)"
        print(f"  {r['system']:28} {r['stamp']}  {len(r['rows'])} items{used}")
    full = [r for r in runs if len(r["rows"]) >= args.min_items]
    groups = defaultdict(list)
    for r in full:
        groups[r["system"]].append(r)
        if re.search(r"-s\d+$", r["system"]):
            groups[re.sub(r"-s\d+$", "-s*", r["system"])].append(r)

    n = {sl: rate(full[0]["rows"], args.metric, sl)[1] for sl in SLICES} if full else {}
    print(f"\n{args.metric}: mean ± sd over runs (n items per slice: "
          + ", ".join(f"{sl} {n.get(sl, 0)}" for sl in SLICES) + ")\n")
    print(f"{'system':30}{'runs':>5}" + "".join(f"{sl:>14}" for sl in SLICES) + f"{'eval flips':>12}")
    for name, rs in sorted(groups.items()):
        cols = [fmt([v for v, _ in (rate(r["rows"], args.metric, sl) for r in rs) if v is not None])
                for sl in SLICES]
        same_model = len({r["system"] for r in rs}) == 1
        fl = flip_rate(rs) if same_model else None
        print(f"{name:30}{len(rs):>5}" + "".join(f"{c:>14}" for c in cols)
              + f"{(f'{fl:.2f}' if fl is not None else '-'):>12}")
    print("\nsd for one model = spread between repeated eval runs; for *-s* = spread over "
          "training seeds (each seed counted once per eval run).")


if __name__ == "__main__":
    main()
