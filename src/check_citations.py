"""
check_citations.py - check a MODEL answer against the passages it was given, using the
same rules as validate_examples.py (which checks teacher-written training examples
and also needs gold chunks; this needs none).

    from check_citations import check_answer
    problems = check_answer(task, answer, context)    # context: [{"id", "text"}]
    # [] means no rule was broken. It does not mean the answer is correct.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_examples import (ABSTAIN, ANALYSIS_FIELDS, CITE_RE, NOT_STATED,  # noqa: E402
                               SENT_SPLIT, check_support, cited_ids, norm)


def check_answer(task, answer, context, paper_ids=()):
    chunk_text = {c["id"]: c["text"] for c in context}
    ctx_ids = set(chunk_text)
    errors = []
    if task == "analysis":
        if not isinstance(answer, dict):
            return ["analysis_not_object"]
        for f in ANALYSIS_FIELDS:
            if not isinstance(answer.get(f), dict):
                errors.append(f"missing_field:{f}")
                continue
            val, ev = answer[f].get("value", ""), answer[f].get("evidence", [])
            if norm(val) == NOT_STATED:
                if ev:
                    errors.append(f"{f}:not_stated_with_evidence")
                continue
            if not ev:
                errors.append(f"{f}:no_evidence")
            if set(ev) - ctx_ids:
                errors.append(f"{f}:evidence_not_in_context")
            check_support(val, [i for i in ev if i in ctx_ids], chunk_text, errors, f"{f}:")
        return sorted(set(errors))

    if not isinstance(answer, str):
        return ["answer_not_string"]
    ans = answer.strip()
    if ABSTAIN in ans:
        if ans != ABSTAIN:
            errors.append("abstention_has_extra_text")
        if cited_ids(ans):
            errors.append("abstention_has_citation")
        return errors
    ids = cited_ids(ans)
    if not ids:
        errors.append("no_citations")
    if set(ids) - ctx_ids:
        errors.append("cites_chunk_not_in_context:" + ",".join(sorted(set(ids) - ctx_ids)))
    for sent in SENT_SPLIT.split(ans):
        if sent.strip() and not CITE_RE.search(sent):
            errors.append("sentence_without_citation")
            break
    check_support(ans, [i for i in ids if i in ctx_ids], chunk_text, errors)
    if task == "compare":
        for pid in paper_ids:
            if not any(i.startswith(pid + "_c") for i in ids):
                errors.append(f"compare_missing_paper:{pid}")
    return sorted(set(errors))
