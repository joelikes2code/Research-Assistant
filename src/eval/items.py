#!/usr/bin/env python3
"""
items.py - load the evaluation items and resolve their gold chunks.

Usage (check that every evidence quote still matches the current chunks):
    python src/eval/items.py

Item files (JSONL):
    data/eval/items.jsonl          committed; papers that may be quoted in the repo
    data/eval/items_private.jsonl  gitignored; P001 (exclusive licence, keep out of repo)

Item fields:
    item_id, task ("qa" | "compare"), paper_ids, question, answerable (bool),
    evidence: list of GROUPS; each group is a list of alternative exact quotes from the
              paper, and ANY one of them supports that part of the answer. Every group
              must be found for full support (compare items have one group per paper).
    reference: short reference answer, for human review (not scored automatically).

Gold is stored as quotes, not chunk IDs, so the eval set survives re-chunking:
gold chunk IDs are recomputed from chunks.jsonl on every run. The split of an item
(dev/test) comes from papers.csv, never from the item, so there is one source of truth.
"""
import csv
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
from validate_examples import norm  # noqa: E402

ITEM_FILES = [ROOT / "data" / "eval" / "items.jsonl",
              ROOT / "data" / "eval" / "items_private.jsonl"]
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
PAPERS = ROOT / "data" / "papers.csv"


def load_items(paths=None):
    items = []
    for p in paths or ITEM_FILES:
        p = Path(p)
        if p.exists():
            items += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    ids = [i["item_id"] for i in items]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate item_ids: {sorted(dupes)}")
    return items


def load_chunks(path=CHUNKS):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def paper_splits(path=PAPERS):
    """paper_id -> 'test' | 'dev' (test_candidate counts as test; everything else dev)."""
    with open(path, encoding="utf-8", newline="") as f:
        return {r["paper_id"]: ("test" if r["split"].startswith("test") else "dev")
                for r in csv.DictReader(f)}


def item_split(item, splits):
    """An item is test if ANY of its papers is a test paper (keeps test text out of dev)."""
    return "test" if any(splits.get(p) == "test" for p in item["paper_ids"]) else "dev"


def resolve_gold(item, chunks):
    """List of sets of chunk IDs, one set per evidence group. Raises if a group matches
    no chunk of the item's papers (quote typo, or the chunker changed the text)."""
    pool = [(c["chunk_id"], norm(c["text"])) for c in chunks if c["paper_id"] in item["paper_ids"]]
    groups = []
    for g, alts in enumerate(item["evidence"]):
        hit = {cid for cid, text in pool for q in alts if norm(q) in text}
        if not hit:
            raise LookupError(f"{item['item_id']} evidence group {g} matches no chunk: {alts}")
        groups.append(hit)
    return groups


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    items, chunks, splits = load_items(), load_chunks(), paper_splits()
    bad = 0
    for it in items:
        try:
            groups = resolve_gold(it, chunks)
        except LookupError as e:
            print("FAIL", e)
            bad += 1
            continue
        if it["answerable"] and not groups:
            print("FAIL", it["item_id"], "answerable but has no evidence")
            bad += 1
        print(f"ok   {it['item_id']:5} {item_split(it, splits):4} {it['task']:7} "
              f"{'answerable' if it['answerable'] else 'unanswerable':12} "
              f"gold={[sorted(g) for g in groups]}")
    n_test = sum(item_split(i, splits) == "test" for i in items)
    print(f"\n{len(items)} items ({len(items) - n_test} dev, {n_test} test), {bad} failing")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
