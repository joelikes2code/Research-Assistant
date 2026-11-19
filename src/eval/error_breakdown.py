#!/usr/bin/env python3
"""
error_breakdown.py - where do answers fail: retrieval or generation?

Usage:
    python src/eval/error_breakdown.py [--results results/igpu]
                                       [--review results/review/batch1.md results/review/batch2.md]

Part 1 (automatic, every answerable held-out item, every run in --results). Each item
falls in exactly one class:
    success              answered, cited a gold chunk, no rule broken
    R  retrieval miss    no gold chunk among the retrieved passages, so the metric
                         cannot be met:  R-abstain (declined) | R-answer (answered
                         from other passages; may still be true, see part 2)
    G  generation error  a gold chunk WAS retrieved:  G-abstain (declined anyway, no
                         citations; content + decline sentence counts as answered) |
                         G-wrong-cite (answered, cited only non-gold passages) |
                         G-rules (cited gold but broke a citation rule)
"Gold retrieved" means at least one gold chunk of any evidence group is in context
(the same condition success needs), not eval_generation's stricter gold_in_ctx.

Part 2 (human labels): the blind hand-check verdicts (c/p/w/a) crossed with the same
classes, on the runs the sheets were built from (CPU runs in results/). This shows how
often the automatic metric and human correctness disagree, and in which direction.

Part 3 (position): for G-wrong-cite answers, the rank (1-5) of the cited passage vs the
rank of the gold passage in the context. A preview of the passage-order probe.

Reads per-item files only; writes nothing. Per-item files can quote P001, so they stay
gitignored; this script's printed output contains no paper text.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import load_runs  # noqa: E402
from items import ROOT  # noqa: E402
from make_answer_review import VERDICT  # noqa: E402

CLASSES = ["success", "R-abstain", "R-answer", "G-abstain", "G-wrong-cite", "G-rules"]


def classify(r):
    s, ctx, cited = r["scores"], set(r["context_ids"]), set(r["cited"])
    gold = [set(g) for g in r["gold"]]
    if s["rules_ok"] and s["answered"] and s["gold_cited"]:
        return "success"
    # "declined" = no citations and not answered; an answer with content AND the decline
    # sentence appended counts as answered (it breaks a rule: abstention_has_extra_text)
    declined = not s["answered"] and not cited
    if not any(ctx & g for g in gold):
        return "R-abstain" if declined else "R-answer"
    if declined:
        return "G-abstain"
    if not any(cited & g for g in gold):
        return "G-wrong-cite"
    return "G-rules"


def label(system):
    return {"qwen3-1.7b": "B1 zero-shot", "qwen3-1.7b+fewshot": "B3 few-shot",
            "qwen3-4b-q4-K-M": "B2 qwen3:4b"}.get(system, system.replace("ra-qwen3-1.7b-", "FT "))


def table(rows_by_name, title):
    print(f"\n{title}\n")
    print(f"| system | n | {' | '.join(CLASSES)} | R total | G total |")
    print("|---" * (len(CLASSES) + 4) + "|")
    for name, rows in rows_by_name.items():
        c = Counter(classify(r) for r in rows)
        R, G = c["R-abstain"] + c["R-answer"], c["G-abstain"] + c["G-wrong-cite"] + c["G-rules"]
        print(f"| {name} | {len(rows)} | {' | '.join(str(c[k]) for k in CLASSES)} | {R} | {G} |")


def part1(results):
    runs = load_runs(results)
    by = {f"{label(r['system'])} ({r['stamp'][9:13]})":
          [x for x in r["rows"] if "heldout" in x["slices"] and x["answerable"]]
          for r in sorted(runs, key=lambda r: r["system"])}
    table(by, f"Part 1: answerable held-out items, runs in {results.relative_to(ROOT)}")
    return runs


def part2(sheets):
    pairs = []                                    # (human verdict, class, run, row)
    for sheet in sheets:
        key = json.loads(Path(f"{sheet}.key.json").read_text(encoding="utf-8"))
        files = {}
        for lab, iid, v in VERDICT.findall(Path(sheet).read_text(encoding="utf-8")):
            run = key[iid][lab.upper()]
            if run not in files:
                p = ROOT / "results" / f"gen_{run}_items.jsonl"
                files[run] = {r["item_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
            row = files[run][iid]
            pairs.append((v.lower(), classify(row), run.rsplit("_", 1)[0], row))
    print(f"\nPart 2: human verdict x class ({len(pairs)} hand-checked answers, CPU runs)\n")
    print("| class | n | correct | partly | wrong | abstained |")
    print("|---|---|---|---|---|---|")
    for k in CLASSES:
        c = Counter(v for v, cl, _, _ in pairs if cl == k)
        if sum(c.values()):
            print(f"| {k} | {sum(c.values())} | {c['c']} | {c['p']} | {c['w']} | {c['a']} |")
    succ = [v for v, cl, _, _ in pairs if cl == "success"]
    fail = [v for v, cl, _, _ in pairs if cl != "success"]
    print(f"\n- metric success but human not 'correct': {sum(v != 'c' for v in succ)}/{len(succ)}")
    print(f"- metric failure but human 'correct':     {sum(v == 'c' for v in fail)}/{len(fail)}")
    agree = sum((cl == "success") == (v == "c") for v, cl, _, _ in pairs)
    print(f"- metric and human agree on correct/not:  {agree}/{len(pairs)}")
    wrong = [(cl, run) for v, cl, run, _ in pairs if v == "w"]
    rw = sum(cl.startswith("R") for cl, _ in wrong)
    print(f"- human 'wrong' answers: {len(wrong)}; gold not retrieved {rw}, gold retrieved "
          f"{len(wrong) - rw} ({Counter(cl for cl, _ in wrong if not cl.startswith('R'))})")
    return pairs


def part3(runs):
    pos = Counter()
    for r in runs:
        for x in r["rows"]:
            if "heldout" in x["slices"] and x["answerable"] and classify(x) == "G-wrong-cite":
                ctx = x["context_ids"]
                gold_ranks = sorted(ctx.index(c) + 1 for g in x["gold"] for c in g if c in ctx)
                for c in x["cited"]:
                    if c in ctx:
                        pos[(ctx.index(c) + 1, gold_ranks[0])] += 1
    if not pos:
        return
    print("\nPart 3: G-wrong-cite citations: rank of cited passage vs best gold rank (all runs)\n")
    print("| cited rank \\ gold rank | " + " | ".join(str(g) for g in range(1, 6)) + " |")
    print("|---" * 6 + "|")
    for c in range(1, 6):
        print(f"| {c} | " + " | ".join(str(pos[(c, g)]) for g in range(1, 6)) + " |")
    above = sum(n for (c, g), n in pos.items() if c < g)
    print(f"\n- cited passage ranked ABOVE the gold one: {above}/{sum(pos.values())} citations")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "igpu")
    ap.add_argument("--review", nargs="*", type=Path,
                    default=[ROOT / "results" / "review" / f"batch{i}.md" for i in (1, 2)])
    args = ap.parse_args()
    runs = part1(args.results.resolve())
    if all(p.exists() for p in args.review):
        part2(args.review)
    part3(runs)


if __name__ == "__main__":
    main()
