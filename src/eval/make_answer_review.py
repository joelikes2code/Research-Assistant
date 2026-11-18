#!/usr/bin/env python3
"""
make_answer_review.py - blind hand-check of answer correctness (the part no metric checks).

Usage:
    python src/eval/make_answer_review.py build RUN_items.jsonl [RUN_items.jsonl ...]
                                          --out results/review/batch1.md [--n 20 --seed 0]
    python src/eval/make_answer_review.py tally results/review/batch1.md

build: picks N held-out answerable items with a fixed seed (the pick depends only on the
eval items, never on the runs, so every batch reviews the SAME items). For each item it
shows the question, the reference answer, and every run's answer in a shuffled order
labelled A, B, C..., with the text of the chunks each answer cites. Which run is which
label goes to <out>.key.json; do not open it until you have finished the sheet.

Fill in each `verdict X:` line with one letter:
    c  correct: agrees with the reference, and the cited chunks support it
    p  partly correct: right but incomplete, or one of several claims is wrong
    w  wrong: contradicts the reference, or states things the cited chunks do not say
    a  abstained (declined to answer an answerable question)

tally: reads the filled sheet and its key and prints the counts per run.

Only answerable held-out items are sampled: unanswerable items are already scored
exactly by eval_generation.py (the abstain sentence matches or not). The sheets quote
paper text, including private P001, so results/review/ is gitignored.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from items import load_chunks, load_items  # noqa: E402  (also puts src/ on sys.path)
from build_examples import PAPER_SPLIT  # noqa: E402  papers in the fine-tuning data

# one letter, optionally followed by "# note"; [ \t] so an empty verdict never reads the next line
VERDICT = re.compile(r"^verdict ([A-Z]) \((\w+)\):[ \t]*([cpwa]?)[ \t]*(?:#.*)?$", re.I | re.M)


def run_name(path):
    return re.sub(r"^gen_|_items$", "", Path(path).stem)


def pick_items(n, seed):
    items = [i for i in load_items()
             if i["answerable"] and not set(i["paper_ids"]) & set(PAPER_SPLIT)]
    items.sort(key=lambda i: i["item_id"])
    return sorted(random.Random(seed).sample(items, min(n, len(items))),
                  key=lambda i: i["item_id"])


def build(args):
    runs = {run_name(p): {r["item_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
            for p in args.runs}
    chunks = {c["chunk_id"]: c["text"] for c in load_chunks()}
    picked = pick_items(args.n, args.seed)
    rng = random.Random(args.seed + 1)
    key, out = {}, [f"# Answer review ({len(runs)} systems, {len(picked)} items)\n",
                    "Codes: c correct | p partly | w wrong or unsupported | a abstained. "
                    "Judge against the reference and the cited chunks.\n"]
    for item in picked:
        iid = item["item_id"]
        order = list(runs)
        rng.shuffle(order)
        key[iid] = {chr(65 + k): name for k, name in enumerate(order)}
        out += [f"\n---\n## {iid}  ({', '.join(item['paper_ids'])})\n",
                f"**Question:** {item['question']}\n",
                f"**Reference:** {item['reference']}\n"]
        for label, name in key[iid].items():
            rec = runs[name].get(iid)
            if rec is None:
                sys.exit(f"{name} has no answer for {iid}")
            out.append(f"\n### {label}\n\n{rec['answer'].strip()}\n")
            for cid in rec["cited"]:
                text = chunks.get(cid, "(no such chunk: the model invented this ID)")
                out.append(f"<details><summary>{cid}</summary>\n\n{text}\n\n</details>\n")
            out.append(f"\nverdict {label} ({iid}): \n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    Path(args.out + ".key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"wrote {args.out} ({len(picked)} items x {len(runs)} systems) and its .key.json")


def tally(args):
    key = json.loads(Path(args.sheet + ".key.json").read_text(encoding="utf-8"))
    counts, missing = {}, []
    for label, iid, v in VERDICT.findall(Path(args.sheet).read_text(encoding="utf-8")):
        name = key[iid][label.upper()]
        if not v:
            missing.append(f"{iid}/{label}")
            continue
        counts.setdefault(name, {c: 0 for c in "cpwa"})[v.lower()] += 1
    for name, c in sorted(counts.items()):
        n = sum(c.values())
        print(f"{name:32s} n={n:2d}  correct={c['c']:2d}  partly={c['p']:2d}  "
              f"wrong={c['w']:2d}  abstained={c['a']:2d}  correct_rate={c['c'] / n:.2f}")
    if missing:
        print(f"{len(missing)} verdicts still empty: {', '.join(missing[:10])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("runs", nargs="+")
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=20)
    b.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("tally")
    t.add_argument("sheet")
    args = ap.parse_args()
    build(args) if args.cmd == "build" else tally(args)


if __name__ == "__main__":
    main()
