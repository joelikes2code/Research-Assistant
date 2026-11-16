#!/usr/bin/env python3
"""
eval_retrieval.py - how often does retrieval put the gold chunks in front of the model?

Usage:
    python src/eval/eval_retrieval.py [--modes hybrid dense bm25] [--ks 1 3 5 10]
                                      [--split all|dev|test] [--show-misses 5]

Answerable items only. Retrieval mirrors generate.py: qa searches the whole corpus
(no paper filter); compare searches each paper separately.

Metrics (mean over items):
    recall@k  fraction of evidence groups with a gold chunk in the top k
    full@k    1 if EVERY group has a gold chunk in the top k
    MRR       mean over groups of 1/rank of the first gold chunk (0 if not in top max-k)
Writes results/retrieval_<timestamp>.json (scores and missed item IDs only, no text).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from items import ROOT, item_split, load_chunks, load_items, paper_splits, resolve_gold  # noqa: E402


def rankings(retriever, item, mode, depth):
    """paper_id or None -> ranked chunk IDs, as generate.py would retrieve them."""
    if item["task"] == "compare":
        return {p: [h["chunk_id"] for h in retriever.search(item["question"], depth, [p], mode)]
                for p in item["paper_ids"]}
    return {None: [h["chunk_id"] for h in retriever.search(item["question"], depth, None, mode)]}


def group_rank(group, ranks):
    """1-based rank of the first gold chunk of this group, or None."""
    paper = next(iter(group)).split("_c")[0]
    ranked = ranks.get(paper, ranks.get(None, []))
    for i, cid in enumerate(ranked, 1):
        if cid in group:
            return i
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modes", nargs="+", default=["hybrid", "dense", "bm25"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    ap.add_argument("--show-misses", type=int, default=5, metavar="K",
                    help="list hybrid items with a group missing from the top K")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from retrieve import Retriever
    retriever = Retriever()
    chunks, splits = load_chunks(), paper_splits()
    items = [i for i in load_items() if i["answerable"]
             and (args.split == "all" or item_split(i, splits) == args.split)]
    depth = max(args.ks)
    gold = {i["item_id"]: resolve_gold(i, chunks) for i in items}

    report = {"n_items": len(items), "split": args.split, "index": retriever.meta, "modes": {}}
    for mode in args.modes:
        per_item = {}
        for it in items:
            ranks = rankings(retriever, it, mode, depth)
            per_item[it["item_id"]] = [group_rank(g, ranks) for g in gold[it["item_id"]]]
        res = {"MRR": sum(sum(1 / r for r in rs if r) / len(rs) for rs in per_item.values())
               / len(items)}
        for k in args.ks:
            res[f"recall@{k}"] = sum(sum(bool(r and r <= k) for r in rs) / len(rs)
                                     for rs in per_item.values()) / len(items)
            res[f"full@{k}"] = sum(all(r and r <= k for r in rs)
                                   for rs in per_item.values()) / len(items)
        for sp in ("dev", "test"):
            ids = [i["item_id"] for i in items if item_split(i, splits) == sp]
            if ids and args.split == "all":
                res[f"{sp}_full@{args.show_misses}"] = sum(
                    all(r and r <= args.show_misses for r in per_item[i]) for i in ids) / len(ids)
        res["misses"] = {i: rs for i, rs in per_item.items()
                         if not all(r and r <= args.show_misses for r in rs)}
        report["modes"][mode] = res

    cols = ["MRR"] + [f"recall@{k}" for k in args.ks] + [f"full@{k}" for k in args.ks]
    print(f"{len(items)} answerable items, split={args.split}\n")
    print(f"{'mode':8}" + "".join(f"{c:>11}" for c in cols))
    for mode, res in report["modes"].items():
        print(f"{mode:8}" + "".join(f"{res[c]:>11.3f}" for c in cols))
    if args.split == "all":
        k = args.show_misses
        print("\nfull@%d by split: " % k + "  ".join(
            f"{m}: dev {r.get(f'dev_full@{k}', 0):.3f} / test {r.get(f'test_full@{k}', 0):.3f}"
            for m, r in report["modes"].items()))
    misses = report["modes"].get("hybrid", {}).get("misses", {})
    if misses:
        print(f"\nhybrid misses at k={args.show_misses} (rank of each evidence group, None = not in top {depth}):")
        q = {i["item_id"]: i["question"] for i in items}
        for iid, rs in misses.items():
            print(f"  {iid}  ranks={rs}  {q[iid][:80]}")

    out = ROOT / "results" / f"retrieval_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
