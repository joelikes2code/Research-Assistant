#!/usr/bin/env python3
"""
eval_generation.py - run the full pipeline (retrieve -> Ollama -> citation check) on
the eval items and score it.

Usage:
    python src/eval/eval_generation.py [--model qwen3:1.7b] [-k 5] [--split all|dev|test]
                                       [--limit N] [--items E001 U003 ...]

Scored automatically (mean over items):
    rules_ok        no citation-rule violation (check_citations.py)
    answerable items:
      answered      did not abstain
      gold_cited    cites at least one gold chunk
      gold_cited_all  cites a gold chunk for EVERY evidence group
      gold_in_ctx   every evidence group had a gold chunk in the retrieved context
    unanswerable items:
      abstained     the answer contains the abstain sentence
      clean_abstain the answer is exactly the abstain sentence
NOT scored: whether the answer is actually correct. Compare it with each item's
"reference" by hand (or with a judge model later); rules_ok and gold_cited only show
the answer is grounded in the right place.

Writes results/gen_<model>_<timestamp>.json (summary) and ..._items.jsonl (per item:
answer, cited IDs, context IDs, errors; no passage text). The _items.jsonl files are
gitignored because answers can quote P001.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from items import ROOT, item_split, load_chunks, load_items, paper_splits, resolve_gold  # noqa: E402
from validate_examples import ABSTAIN, cited_ids  # noqa: E402


def cited(answer):
    if isinstance(answer, dict):
        return [i for v in answer.values() if isinstance(v, dict) for i in v.get("evidence", [])]
    return cited_ids(answer) if isinstance(answer, str) else []


def score(item, rec, groups):
    ans = rec["answer"]
    ids, ctx = set(cited(ans)), {c["id"] for c in rec["context"]}
    s = {"rules_ok": not rec["citation_errors"]}
    abstained = isinstance(ans, str) and ABSTAIN in ans
    if item["answerable"]:
        s["answered"] = not abstained
        s["gold_cited"] = any(ids & g for g in groups)
        s["gold_cited_all"] = all(ids & g for g in groups)
        s["gold_in_ctx"] = all(ctx & g for g in groups)
    else:
        s["abstained"] = abstained
        s["clean_abstain"] = isinstance(ans, str) and ans.strip() == ABSTAIN
    return s


def summarize(rows):
    out = {"n": len(rows)}
    for key in ("rules_ok", "answered", "gold_cited", "gold_cited_all", "gold_in_ctx",
                "abstained", "clean_abstain"):
        vals = [r["scores"][key] for r in rows if key in r["scores"]]
        if vals:
            out[key] = round(sum(vals) / len(vals), 3)
            out[key + "_n"] = len(vals)
    out["errors"] = dict(Counter(e.split(":")[0] for r in rows for e in r["citation_errors"]
                                 ).most_common())
    out["mean_seconds"] = round(sum(r["seconds"] for r in rows) / max(1, len(rows)), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=None, help="Ollama model (default: generate.py's)")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--items", nargs="*", help="only these item IDs")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import generate
    from retrieve import Retriever
    model = args.model or generate.DEFAULT_MODEL
    retriever = Retriever()
    chunks, splits = load_chunks(), paper_splits()
    items = [i for i in load_items()
             if (args.split == "all" or item_split(i, splits) == args.split)
             and (not args.items or i["item_id"] in args.items)][:args.limit]

    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = ROOT / "results" / f"gen_{re.sub(r'[^A-Za-z0-9.]+', '-', model)}_{stamp}"
    rows = []
    with open(f"{base}_items.jsonl", "w", encoding="utf-8") as f:
        for n, it in enumerate(items, 1):
            groups = resolve_gold(it, chunks)
            t0 = time.time()
            rec = generate.answer(it["question"], it["task"], args.k,
                                  it["paper_ids"] if it["task"] == "compare" else None,
                                  model, retriever=retriever)
            row = {"item_id": it["item_id"], "split": item_split(it, splits),
                   "task": it["task"], "answerable": it["answerable"],
                   "question": it["question"], "answer": rec["answer"],
                   "cited": cited(rec["answer"]),
                   "context_ids": [c["id"] for c in rec["context"]],
                   "gold": [sorted(g) for g in groups],
                   "citation_errors": rec["citation_errors"],
                   "seconds": round(time.time() - t0, 1)}
            row["scores"] = score(it, rec, groups)
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            flags = " ".join(k for k, v in row["scores"].items() if not v) or "all ok"
            print(f"[{n}/{len(items)}] {it['item_id']:5} {row['seconds']:6.1f}s  missing: {flags}",
                  flush=True)

    summary = {"model": model, "k": args.k, "split": args.split, "index": retriever.meta,
               "all": summarize(rows)}
    for sp in ("dev", "test"):
        part = [r for r in rows if r["split"] == sp]
        if part:
            summary[sp] = summarize(part)
    Path(f"{base}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nmodel={model}  items={len(rows)}")
    for name in ("all", "dev", "test"):
        if name in summary:
            s = summary[name]
            print(f"  {name:4} " + "  ".join(f"{k}={s[k]:.2f}" for k in
                  ("rules_ok", "answered", "gold_cited", "gold_cited_all", "gold_in_ctx",
                   "abstained", "clean_abstain") if k in s) + f"  ({s['mean_seconds']}s/item)")
    print("  errors:", summary["all"]["errors"])
    print(f"wrote {Path(base).relative_to(ROOT)}.json and _items.jsonl")


if __name__ == "__main__":
    main()
