#!/usr/bin/env python3
"""
build_examples.py - turn drafted seeds into training records, validate, split.

Usage:
    python src/build_examples.py [--drafts data/train/drafts] [--out data/train/build] [-k 5]

Seeds (data/train/drafts/*.jsonl), written by the teacher from the frozen chunks:
    {"seed_id", "task": "qa"|"compare"|"analysis", "paper_ids", "question",
     "answer": str | {field: {"value", "evidence"}} | null,
     "abstain": true  (optional: build ONLY a should-abstain record for this question),
     "same_paper": true (with abstain: the paper never states the answer, so passages
                         are retrieved from the paper itself instead of other papers)}
Gold chunks are the chunk IDs the answer cites; the answer is the training target.

For each seed the passages are retrieved exactly as generate.py would retrieve them
(same k, same per-task logic), but only from the allowed papers of that split; gold
chunks missing from the top k replace the lowest-ranked non-gold passages, at a
seeded random position. Abstain seeds get passages from the OTHER papers of the
split only, so the answer is truly absent, and the answer is validate_examples.ABSTAIN.

Papers: train = P003 P004 P005 P008, val = P007. P001 is private (never uploaded) and
P002/P006 are test papers: seeds about them are rejected.

Outputs (OUT): examples.jsonl, accepted.jsonl / rejected.jsonl (from
validate_examples.py), train.jsonl, val.jsonl, and a printed report.
"""
import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "eval"))
from validate_examples import ABSTAIN, cited_ids, jaccard  # noqa: E402

SPLITS = {"train": ["P003", "P004", "P005", "P008"], "val": ["P007"]}
PAPER_SPLIT = {p: s for s, ps in SPLITS.items() for p in ps}


def split_of(seed):
    splits = {PAPER_SPLIT.get(p) for p in seed["paper_ids"]}
    if None in splits:
        bad = [p for p in seed["paper_ids"] if p not in PAPER_SPLIT]
        raise ValueError(f"{seed['seed_id']}: papers {bad} may not be used for training")
    if len(splits) != 1:
        raise ValueError(f"{seed['seed_id']}: mixes train and val papers")
    return splits.pop()


def gold_of(seed):
    a = seed["answer"]
    if isinstance(a, dict):
        return sorted({i for v in a.values() for i in v.get("evidence", [])})
    return sorted(set(cited_ids(a))) if isinstance(a, str) else []


def force_in(hits, gold, rng):
    """Make sure every gold chunk is in the passage list, keeping its length."""
    hits = list(hits)
    missing = [g for g in gold if g not in hits]
    for g in missing:
        drop = next((i for i in range(len(hits) - 1, -1, -1) if hits[i] not in gold), None)
        if drop is None:
            hits.append(g)
            continue
        hits.pop(drop)
        hits.insert(rng.randrange(len(hits) + 1), g)
    return hits


def build(seed, retriever, chunks, k):
    import generate
    split = split_of(seed)
    pool = SPLITS[split]
    rng = random.Random(seed["seed_id"])
    task, q = seed["task"], seed.get("question") or "Summarize the paper."
    if seed.get("abstain"):
        others = [p for p in pool if p not in seed["paper_ids"]]
        if seed.get("same_paper"):
            # hard abstain: passages from the paper itself; the teacher asserts that the
            # answer is nowhere in its main text (reviewers: check these first)
            scope = seed["paper_ids"]
        elif others:
            scope = others
        else:
            raise ValueError(f"{seed['seed_id']}: no other paper in split '{split}'; "
                             "an abstain seed here needs \"same_paper\": true")
        ids = [h["chunk_id"] for h in retriever.search(q, k, scope)]
        answer, gold = ABSTAIN, []
    else:
        scope = pool if task == "qa" else seed["paper_ids"]
        ids = [c["id"] for c in generate.get_context(retriever, task, q, k, scope)]
        gold = gold_of(seed)
        ids = force_in(ids, gold, rng)
        answer = seed["answer"]
    return {"item_id": seed["seed_id"] + ("-abs" if seed.get("abstain") else ""),
            "paper_ids": seed["paper_ids"], "task": task, "question": q,
            "context": [{"id": i, "text": chunks[i]} for i in ids],
            "gold_chunks": gold, "answer": answer, "split": split}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--drafts", type=Path, default=ROOT / "data" / "train" / "drafts")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "train" / "build")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from items import load_items
    from retrieve import Retriever
    retriever = Retriever()
    chunks_path = ROOT / "data" / "processed" / "chunks.jsonl"
    chunks = {json.loads(l)["chunk_id"]: json.loads(l)["text"]
              for l in open(chunks_path, encoding="utf-8")}

    seeds = []
    for f in sorted(args.drafts.glob("*.jsonl")):
        seeds += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    ids = [s["seed_id"] for s in seeds]
    if len(ids) != len(set(ids)):
        sys.exit(f"duplicate seed_ids: {[i for i, n in Counter(ids).items() if n > 1]}")

    eval_qs = [i["question"] for i in load_items()]
    overlap = [(s["seed_id"], round(max(jaccard(s.get("question") or "", e) for e in eval_qs), 2))
               for s in seeds if s.get("question")]
    overlap = [o for o in overlap if o[1] >= 0.5]

    args.out.mkdir(parents=True, exist_ok=True)
    recs, errors = [], []
    for s in seeds:
        try:
            recs.append(build(s, retriever, chunks, args.k))
        except (ValueError, KeyError) as e:
            errors.append(f"{s['seed_id']}: {e}")
    ex_path = args.out / "examples.jsonl"
    with open(ex_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(seeds)} seeds -> {len(recs)} records ({len(errors)} build errors)")
    for e in errors:
        print("  build error:", e)
    v = subprocess.run([sys.executable, str(SRC / "validate_examples.py"), str(ex_path),
                        str(chunks_path), str(args.out)], capture_output=True, text=True,
                       encoding="utf-8")
    print(v.stdout.strip() or v.stderr.strip())

    from prompt import build_messages
    accepted = [json.loads(l) for l in open(args.out / "accepted.jsonl", encoding="utf-8")]
    leaked = sorted({c["id"] for r in accepted for c in r["context"]
                     if c["id"].split("_c")[0] not in PAPER_SPLIT})
    if leaked:
        sys.exit(f"passages from papers outside the training splits: {leaked[:5]}")
    for split in SPLITS:
        rows = [r for r in accepted if r["split"] == split]
        with open(args.out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # chat format for fine-tuning: the exact prompt generate.py sends, plus the answer
        with open(args.out / f"{split}_chat.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                target = (json.dumps(r["answer"], ensure_ascii=False)
                          if isinstance(r["answer"], dict) else r["answer"])
                msgs = build_messages(r["task"], r["question"], r["context"])
                msgs.append({"role": "assistant", "content": target})
                f.write(json.dumps({"item_id": r["item_id"], "messages": msgs},
                                   ensure_ascii=False) + "\n")
        kinds = Counter("abstain" if r["answer"] == ABSTAIN else r["task"] for r in rows)
        print(f"{split}: {len(rows)} accepted  {dict(sorted(kinds.items()))}")
    if overlap:
        print(f"seeds similar to eval questions (Jaccard >= 0.5): {overlap}")


if __name__ == "__main__":
    main()
