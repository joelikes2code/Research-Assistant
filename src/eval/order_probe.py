#!/usr/bin/env python3
"""
order_probe.py - does the model choose passages by content or by position?

Usage:
    python src/eval/order_probe.py [--probe results/probe] [--base results/igpu]

Each probe run (eval_generation.py --order reversed|shuffled) gets the SAME five
passages as the matching base run (same system, retrieved order, same GPU backend),
only presented in a different order. Per system and order, over answerable items:

    success         base vs probe, with the exact McNemar p on the paired items
    same cites      share of items answered in both runs whose cited chunk SET is identical
    cited rank 1    share of citations to the passage the RETRIEVER ranked first
    cited pos 1     share of citations to the passage SHOWN first in the prompt

A content-driven model keeps "same cites" high and "cited rank 1" stable across orders;
a position-driven model keeps "cited pos 1" stable instead (under reversal the
retriever's first passage is shown last). Mean rank/position (1-5) is printed too.

Reads per-item files only; prints no paper text.
"""
import argparse
import re
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import load_runs  # noqa: E402
from items import ROOT  # noqa: E402

ORDER_RE = re.compile(r"\+(reversed|shuffled)$")


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def cite_positions(row, base_ctx):
    """(retriever rank, shown position) of every citation that is in the context."""
    shown = row["context_ids"]
    return [(base_ctx.index(c) + 1, shown.index(c) + 1) for c in row["cited"]
            if c in shown and c in base_ctx]


def stats(pairs):
    """pairs: [(base_row, probe_row)] for one system/order/slice."""
    b = sum(x["success"] and not y["success"] for x, y in pairs)
    c = sum(y["success"] and not x["success"] for x, y in pairs)
    both = [(x, y) for x, y in pairs if x["cited"] and y["cited"]]
    same = sum(set(x["cited"]) == set(y["cited"]) for x, y in both)
    out = {"n": len(pairs), "succ_base": sum(x["success"] for x, _ in pairs),
           "succ_probe": sum(y["success"] for _, y in pairs), "base_only": b, "probe_only": c,
           "p": mcnemar_exact(b, c), "same_cites": f"{same}/{len(both)}"}
    for tag in ("base", "probe"):
        pos = [p for x, y in pairs for p in cite_positions(x if tag == "base" else y,
                                                          x["context_ids"])]
        n = max(1, len(pos))
        out[f"{tag}_rank1"] = sum(r == 1 for r, _ in pos) / n
        out[f"{tag}_pos1"] = sum(s == 1 for _, s in pos) / n
        out[f"{tag}_mean_rank"] = sum(r for r, _ in pos) / n
        out[f"{tag}_mean_pos"] = sum(s for _, s in pos) / n
        out[f"{tag}_cites"] = len(pos)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--probe", type=Path, default=ROOT / "results" / "probe")
    ap.add_argument("--base", type=Path, default=ROOT / "results" / "igpu")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    base = {r["system"]: r for r in load_runs(args.base.resolve())}
    print("| system | order | slice | n | success base -> probe | base-only / probe-only | p "
          "| same cites | cited rank 1 base -> probe | cited pos 1 base -> probe "
          "| mean rank base -> probe | mean pos base -> probe |")
    print("|---" * 12 + "|")
    for run in sorted(load_runs(args.probe.resolve()), key=lambda r: r["system"]):
        m = ORDER_RE.search(run["system"])
        system = run["system"][:m.start()] if m else None
        if system not in base:
            print(f"skip {run['file']}: no base run for {system}", file=sys.stderr)
            continue
        brows = {x["item_id"]: x for x in base[system]["rows"]}
        for sl in ("all", "heldout"):
            pairs = [(brows[y["item_id"]], y) for y in run["rows"]
                     if y["answerable"] and sl in y["slices"] and y["item_id"] in brows]
            for x, y in pairs:  # the probe must only reorder, never change the passages
                assert set(x["context_ids"]) == set(y["context_ids"]), y["item_id"]
            s = stats(pairs)
            print(f"| {system} | {m.group(1)} | {sl} | {s['n']} | {s['succ_base']} -> "
                  f"{s['succ_probe']} | {s['base_only']} / {s['probe_only']} | {s['p']:.3f} | "
                  f"{s['same_cites']} | {s['base_rank1']:.2f} -> {s['probe_rank1']:.2f} | "
                  f"{s['base_pos1']:.2f} -> {s['probe_pos1']:.2f} | "
                  f"{s['base_mean_rank']:.2f} -> {s['probe_mean_rank']:.2f} | "
                  f"{s['base_mean_pos']:.2f} -> {s['probe_mean_pos']:.2f} |")


if __name__ == "__main__":
    main()
