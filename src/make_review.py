#!/usr/bin/env python3
"""
make_review.py - write a hand-review sheet for the built training examples.

Usage:
    python src/make_review.py [--build data/train/build] [--train-frac 0.25] [--seed 0]

Writes BUILD/review.md with ALL validation examples and a sample of the training
examples (every compare, analysis and same-paper abstain example, plus a seeded random
sample of the rest up to --train-frac of the training set). For each example it shows
the question, the answer, and the text of each cited passage, so a claim can be
checked against its evidence without opening chunks.jsonl. For abstain examples it
shows the start of every passage: check that none of them answers the question.

Mark each example by replacing "[ ]" with "[x]" (fine) or "[!]" (wrong) and add a note;
fix wrong ones in data/train/drafts/ and re-run build_examples.py.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_examples import ABSTAIN  # noqa: E402


def show(r, out):
    is_abs = r["answer"] == ABSTAIN
    ctx = {c["id"]: c["text"] for c in r["context"]}
    out.append(f"### [ ] {r['item_id']}  ({r['task']}{', abstain' if is_abs else ''})\n")
    out.append(f"**Q:** {r['question']}\n")
    if isinstance(r["answer"], dict):
        for f, v in r["answer"].items():
            out.append(f"- **{f}:** {v['value']}  `{', '.join(v['evidence']) or '-'}`")
        out.append("")
    else:
        out.append(f"**A:** {r['answer']}\n")
    if is_abs:
        out.append("Passages given (none should answer the question):\n")
        for i, t in ctx.items():
            out.append(f"- `{i}` {t[:260]} ...")
    else:
        for i in r["gold_chunks"]:
            out.append(f"> `{i}` {ctx[i]}\n")
    out.append("\n---\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", type=Path, default=ROOT / "data" / "train" / "build")
    ap.add_argument("--train-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    load = lambda n: [json.loads(l) for l in open(args.build / n, encoding="utf-8")]
    train, val = load("train.jsonl"), load("val.jsonl")
    must = [r for r in train if r["task"] != "qa"
            or (r["answer"] == ABSTAIN and r["item_id"].split("-")[1][0] == "h")]
    rest = [r for r in train if r not in must]
    n_more = max(0, round(args.train_frac * len(train)) - len(must))
    sample = must + random.Random(args.seed).sample(rest, min(n_more, len(rest)))
    sample.sort(key=lambda r: r["item_id"])

    out = [f"# Review sheet\n\nValidation: all {len(val)} examples. Training: {len(sample)} of "
           f"{len(train)} ({100 * len(sample) / len(train):.0f}%).\n\nFor each example check: "
           "(1) the answer is true to the quoted passages, (2) each sentence's citation is the "
           "passage that supports it, (3) for abstain examples, no passage answers the question.\n",
           "## Validation\n"]
    for r in val:
        show(r, out)
    out.append("## Training sample\n")
    for r in sample:
        show(r, out)
    path = args.build / "review.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}: {len(val)} validation + {len(sample)} training examples")


if __name__ == "__main__":
    main()
