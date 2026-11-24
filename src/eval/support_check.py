#!/usr/bin/env python3
"""
support_check.py - does each cited passage actually SUPPORT the sentence that cites it?

Usage (needs the optional dependencies: uv sync --group nli):
    python src/eval/support_check.py [--runs results/igpu results/probe] [--limit N]

An automatic, content-level check that the eval metric lacks: the metric only asks
whether a gold chunk was cited. Here every answer sentence (string answers) or analysis
field value is the hypothesis, the text of each chunk it cites is the premise, and an
NLI model gives P(entailment). A sentence is SUPPORTED if any one cited chunk entails
it with P >= 0.5. An answer is fully supported if all its sentences are supported;
a sentence with no citation counts as unsupported. Abstentions are skipped.

Fixed before seeing any result (do not tune on the hand-check labels):
    model     MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (commit logged)
    threshold 0.5, max over cited chunks (no chunk concatenation)
Known biases: claims that need two passages together, and sentences whose subject is
only in an earlier sentence ("It uses ..."), are under-supported; premises longer than
512 tokens are truncated (counted in the output).

Scores every gen_*_items.jsonl in --runs plus the CPU runs named in the hand-check keys
(results/review/*.key.json), skipping runs already scored. Per-sentence results go to
results/support/<run>_support_items.jsonl (gitignored: they quote answers). Prints:
    1. per run: share of sentences supported, share of answers fully supported
    2. support x human verdict on the 120 hand-checked answers (validates the checker)
    3. support x error class (does a "wrong passage" citation still support the claim?)
"""
import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from items import ROOT, load_chunks  # noqa: E402  (also puts src/ on sys.path)
from validate_examples import ABSTAIN, CITE_RE, NOT_STATED, SENT_SPLIT, cited_ids, norm  # noqa: E402
from error_breakdown import CLASSES, classify  # noqa: E402
from make_answer_review import VERDICT  # noqa: E402

MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
THRESHOLD = 0.5
OUT = ROOT / "results" / "support"
REVIEW = ROOT / "results" / "review"


def claims(answer):
    """[(hypothesis, cited chunk IDs)] for one answer; [] for an abstention."""
    if isinstance(answer, dict):                                  # analysis
        return [(str(v.get("value", "")), list(v.get("evidence", [])))
                for v in answer.values()
                if isinstance(v, dict) and norm(str(v.get("value", ""))) != NOT_STATED]
    if not isinstance(answer, str) or not answer.strip() or answer.strip() == ABSTAIN:
        return []
    out = []
    for sent in SENT_SPLIT.split(answer.replace(ABSTAIN, "").strip()):
        hyp = re.sub(r"\s+([.,;:!?])", r"\1", CITE_RE.sub("", sent)).strip()
        if hyp:
            out.append((hyp, cited_ids(sent)))
    return out


class NLI:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail = labels["entailment"]
        self.commit = getattr(self.model.config, "_commit_hash", None)
        self.cache, self.truncated = {}, 0

    def score(self, pairs, batch=4):
        """pairs: [(premise, hypothesis)] -> P(entailment) each, cached."""
        todo = sorted({p for p in pairs if p not in self.cache}, key=lambda p: len(p[0]))
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            enc = self.tok([p for p, _ in part], [h for _, h in part], truncation="only_first",
                           max_length=512, padding=True, return_tensors="pt",
                           return_overflowing_tokens=False)
            self.truncated += sum(len(self.tok(p, h)["input_ids"]) > 512 for p, h in part)
            with self.torch.inference_mode():
                probs = self.model(**enc).logits.softmax(-1)[:, self.entail].tolist()
            self.cache.update(zip(part, probs))
        return [self.cache[p] for p in pairs]


def score_run(path, nli, chunk_text, limit=None):
    out = OUT / f"{path.stem.removeprefix('gen_').removesuffix('_items')}_support_items.jsonl"
    if out.exists():
        return out
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()][:limit]
    t0, recs = time.time(), []
    for r in rows:
        sents = []
        for hyp, ids in claims(r["answer"]):
            known = [i for i in ids if i in chunk_text]
            probs = nli.score([(chunk_text[i], hyp) for i in known])
            best = max(probs, default=0.0)
            sents.append({"hyp": hyp, "cited": ids, "p": [round(p, 4) for p in probs],
                          "supported": best >= THRESHOLD})
        recs.append({"item_id": r["item_id"], "sentences": sents,
                     "fully_supported": bool(sents) and all(s["supported"] for s in sents)})
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(out)                    # only complete files count as scored
    print(f"scored {path.name}: {sum(len(r['sentences']) for r in recs)} sentences, "
          f"{time.time() - t0:.0f}s", flush=True)
    return out


def load_support(out):
    return {r["item_id"]: r for r in map(json.loads, open(out, encoding="utf-8"))}


def report_runs(scored):
    print("\n1. Support per run (answered items only; heldout = answerable held-out)\n")
    print("| run | slice | answers | sentences | supported | answers fully supported |")
    print("|---|---|---|---|---|---|")
    for path, out in scored:
        sup = load_support(out)
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        for sl, keep in (("all", lambda r: True),
                         ("heldout", lambda r: r["heldout"] and r["answerable"])):
            recs = [sup[r["item_id"]] for r in rows if keep(r) and sup[r["item_id"]]["sentences"]]
            sents = [s for rec in recs for s in rec["sentences"]]
            if recs:
                print(f"| {path.parent.name}/{path.stem.removeprefix('gen_').removesuffix('_items')} "
                      f"| {sl} | {len(recs)} | {len(sents)} | "
                      f"{sum(s['supported'] for s in sents) / len(sents):.2f} | "
                      f"{sum(r['fully_supported'] for r in recs) / len(recs):.2f} |")


def human_pairs():
    """[(verdict, run name, item_id)] from every filled hand-check sheet."""
    pairs = []
    for key_path in sorted(REVIEW.glob("*.md.key.json")):
        key = json.loads(key_path.read_text(encoding="utf-8"))
        sheet = Path(str(key_path).removesuffix(".key.json")).read_text(encoding="utf-8")
        for lab, iid, v in VERDICT.findall(sheet):
            if v:
                pairs.append((v.lower(), key[iid][lab.upper()], iid))
    return pairs


def report_human(pairs):
    table = defaultdict(Counter)       # verdict -> Counter(fully supported / not / abstained)
    by_class = defaultdict(Counter)
    for v, run, iid in pairs:
        rec = load_support(OUT / f"{run}_support_items.jsonl")[iid]
        row = next(r for r in map(json.loads, open(ROOT / "results" / f"gen_{run}_items.jsonl",
                                                   encoding="utf-8")) if r["item_id"] == iid)
        s = "none" if not rec["sentences"] else "full" if rec["fully_supported"] else "partial"
        table[v][s] += 1
        by_class[classify(row)][(s, v)] += 1
    print(f"\n2. NLI support x human verdict ({len(pairs)} hand-checked answers, CPU runs)\n")
    print("| human | n | fully supported | not fully | no claims (abstained) |")
    print("|---|---|---|---|---|")
    for v in "cpwa":
        c = table[v]
        print(f"| {v} | {sum(c.values())} | {c['full']} | {c['partial']} | {c['none']} |")
    answered = [(v, s) for v in "cpw" for s in ("full", "partial") for _ in range(table[v][s])]
    agree = sum((s == "full") == (v == "c") for v, s in answered)
    print(f"\n- NLI 'fully supported' vs human 'correct' agree on {agree}/{len(answered)} answered")
    full = [v for v, s in answered if s == "full"]
    correct = [s for v, s in answered if v == "c"]
    print(f"- fully supported answers judged correct: {full.count('c')}/{len(full)}")
    print(f"- correct answers judged fully supported: {correct.count('full')}/{len(correct)}")
    print("\n3. Error class x support x human (same answers)\n")
    print("| class | n | full & c | full & not c | partial & c | partial & not c |")
    print("|---|---|---|---|---|---|")
    for k in CLASSES:
        c = by_class[k]
        n = sum(c.values())
        if n:
            full_c = c[("full", "c")]
            full_n = sum(c[("full", v)] for v in "pwa")
            part_c = c[("partial", "c")]
            print(f"| {k} | {n} | {full_c} | {full_n} | {part_c} | {n - full_c - full_n - part_c} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="*", type=Path,
                    default=[ROOT / "results" / "igpu", ROOT / "results" / "probe"])
    ap.add_argument("--limit", type=int, help="score only the first N items per run (testing)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    pairs = human_pairs()
    # hand-checked CPU runs first: they validate the checker, so they matter most if time runs out
    files = [ROOT / "results" / f"gen_{run}_items.jsonl" for run in sorted({r for _, r, _ in pairs})]
    files += [f for d in args.runs if d.exists() for f in sorted(d.glob("gen_*_items.jsonl"))]
    chunk_text = {c["chunk_id"]: c["text"] for c in load_chunks()}
    todo = [f for f in files if not (OUT / f"{f.stem.removeprefix('gen_').removesuffix('_items')}"
                                     f"_support_items.jsonl").exists()]
    if todo:
        nli = NLI()
        print(f"model {MODEL} @ {nli.commit}, threshold {THRESHOLD}, {len(todo)} runs to score",
              flush=True)
        for f in todo:
            score_run(f, nli, chunk_text, args.limit)
        print(f"premises truncated to 512 tokens: {nli.truncated} pairs", flush=True)
        (OUT / "model.json").write_text(json.dumps(
            {"model": MODEL, "commit": nli.commit, "threshold": THRESHOLD,
             "truncated_pairs": nli.truncated}, indent=2), encoding="utf-8")
    scored = [(f, OUT / f"{f.stem.removeprefix('gen_').removesuffix('_items')}_support_items.jsonl")
              for f in files]
    report_runs(scored)
    if pairs:
        report_human(pairs)


if __name__ == "__main__":
    main()
