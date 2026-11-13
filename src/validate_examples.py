#!/usr/bin/env python3
"""
validate_examples.py - automatic checks for teacher-drafted training examples.

Usage:
    python validate_examples.py examples.jsonl chunks.jsonl OUT_DIR

Writes OUT_DIR/accepted.jsonl, rejected.jsonl (with reasons) and prints a report.
Passing these checks does NOT prove an example is good. You must still review a
sample by hand (all of validation, >=25% of training).

Citation convention: put the citation BEFORE the final period:
    "The authors evaluate on X [P012_c07]."
Multiple: [P012_c07, P012_c08]

Record fields:
    item_id, paper_ids, task ("qa" | "compare" | "analysis"), question,
    context: [{"id","text"}], gold_chunks: [ids],  answer
For task "analysis", answer is a dict: {field: {"value": str, "evidence": [ids]}}.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ABSTAIN = "Insufficient evidence in the provided passages."
ANALYSIS_FIELDS = ["objective", "method", "data", "metrics", "main_result", "limitations"]
NOT_STATED = "not stated"

CITE_RE = re.compile(r"\[([A-Za-z0-9]+_c\d+(?:\s*,\s*[A-Za-z0-9]+_c\d+)*)\]")
QUOTE_RE = re.compile(r"[\"\u201c]([^\"\u201d]{20,})[\"\u201d]")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?%?")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def norm(s: str) -> str:
    s = s.lower()
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2010", "-")
    return re.sub(r"\s+", " ", s).strip()


def cited_ids(text):
    ids = []
    for m in CITE_RE.finditer(text):
        ids += [x.strip() for x in m.group(1).split(",")]
    return ids


def check_support(text, ids, chunk_text, errors, label=""):
    """Quoted spans and numbers in `text` must appear in the cited chunks."""
    evidence = norm(" ".join(chunk_text[i] for i in ids if i in chunk_text))
    plain = CITE_RE.sub("", text)
    for m in QUOTE_RE.finditer(plain):
        if norm(m.group(1)) not in evidence:
            errors.append(f"{label}quote_not_in_cited_chunk")
    for m in NUM_RE.finditer(plain):
        tok = m.group(0)
        digits = re.sub(r"\D", "", tok)
        if len(digits) >= 2 or "." in tok or "%" in tok:     # skip tiny ints
            variants = {tok, tok.replace(",", ""), tok.rstrip("%")}
            if not any(v in evidence for v in variants):
                errors.append(f"{label}number_not_in_cited_chunk:{tok}")


def validate_qa(rec, chunk_text, errors):
    ctx_ids = {c["id"] for c in rec["context"]}
    ans = rec["answer"]
    if not isinstance(ans, str):
        errors.append("answer_not_string")
        return
    gold = set(rec.get("gold_chunks", []))
    is_abstain = ans.strip().startswith(ABSTAIN)
    if is_abstain and gold:
        errors.append("abstains_but_has_gold")
    if not is_abstain and not gold:
        errors.append("answers_but_no_gold")
    if is_abstain:
        if cited_ids(ans):
            errors.append("abstention_has_citation")
        return
    ids = cited_ids(ans)
    if not ids:
        errors.append("no_citations")
    if set(ids) - ctx_ids:
        errors.append("cites_chunk_not_in_context")
    if gold and not (gold & set(ids)):
        errors.append("cites_no_gold_chunk")
    for sent in SENT_SPLIT.split(ans.strip()):
        if sent.strip() and not CITE_RE.search(sent):
            errors.append("sentence_without_citation")
            break
    check_support(ans, [i for i in ids if i in ctx_ids], chunk_text, errors)
    if rec.get("task") == "compare":
        for pid in rec.get("paper_ids", []):
            if not any(i.startswith(pid + "_c") for i in ids):
                errors.append(f"compare_missing_paper:{pid}")


def validate_analysis(rec, chunk_text, errors):
    ctx_ids = {c["id"] for c in rec["context"]}
    ans = rec["answer"]
    if not isinstance(ans, dict):
        errors.append("analysis_not_object")
        return
    for f in ANALYSIS_FIELDS:
        if f not in ans:
            errors.append(f"missing_field:{f}")
            continue
        val, ev = ans[f].get("value", ""), ans[f].get("evidence", [])
        if norm(val) == NOT_STATED:
            if ev:
                errors.append(f"{f}:not_stated_with_evidence")
            continue
        if not ev:
            errors.append(f"{f}:no_evidence")
        if set(ev) - ctx_ids:
            errors.append(f"{f}:evidence_not_in_context")
        check_support(val, [i for i in ev if i in ctx_ids], chunk_text, errors, f"{f}:")


def jaccard(a, b):
    a, b = set(norm(a).split()), set(norm(b).split())
    return len(a & b) / max(1, len(a | b))


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    ex_path, ch_path, out_dir = sys.argv[1:]
    chunk_text = {}
    for line in open(ch_path, encoding="utf-8"):
        c = json.loads(line)
        chunk_text[c["chunk_id"]] = c["text"]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    acc, rej, reasons = [], [], Counter()
    seen_q = {}                                           # paper -> [questions]
    for line in open(ex_path, encoding="utf-8"):
        rec = json.loads(line)
        errors = []
        for k in ("item_id", "task", "context", "answer"):
            if k not in rec:
                errors.append(f"missing_key:{k}")
        if not errors:
            bad = [c["id"] for c in rec["context"] if c["id"] not in chunk_text]
            if bad:
                errors.append("context_id_unknown")
            elif any(norm(c["text"]) != norm(chunk_text[c["id"]]) for c in rec["context"]):
                errors.append("context_text_differs_from_chunk")
        if not errors:
            if rec["task"] == "analysis":
                validate_analysis(rec, chunk_text, errors)
            else:
                validate_qa(rec, chunk_text, errors)
                key = ",".join(rec.get("paper_ids", []))
                q = rec.get("question", "")
                if any(jaccard(q, p) > 0.8 for p in seen_q.get(key, [])):
                    errors.append("near_duplicate_question")
                seen_q.setdefault(key, []).append(q)
        if errors:
            rec["_errors"] = sorted(set(errors))
            rej.append(rec)
            reasons.update(set(e.split(":")[0] if e.startswith("number") else e for e in errors))
        else:
            acc.append(rec)

    for name, rows in (("accepted", acc), ("rejected", rej)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = len(acc) + len(rej)
    print(f"total={total} accepted={len(acc)} rejected={len(rej)} "
          f"({100*len(rej)/max(1,total):.0f}% rejected)")
    for reason, n in reasons.most_common():
        print(f"  {n:4d}  {reason}")


if __name__ == "__main__":
    main()
