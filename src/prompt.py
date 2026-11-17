"""
prompt.py - prompts for the three tasks, sharing the citation rules that
validate_examples.py enforces (ABSTAIN text, [P012_c07] citations before the final
period, analysis fields with evidence).

    from prompt import build_messages
    msgs = build_messages("qa", question, passages)     # passages: [{"id", "text"}]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_examples import ABSTAIN, ANALYSIS_FIELDS, NOT_STATED  # noqa: E402

TASKS = ("qa", "compare", "analysis")

RULES = f"""You answer questions about research papers using ONLY the passages provided.
Rules:
1. Use only facts stated in the passages. Do not use outside knowledge.
2. End every sentence with the ID of the passage that supports it, in square brackets,
   BEFORE the final period. Example: "The authors evaluate on AgentDojo [P012_c07]."
   Several passages: [P012_c07, P012_c08].
3. Cite only IDs that appear in the passages. Copy numbers exactly as written there.
4. If the passages do not answer the question, reply with exactly this sentence and
   nothing else, with no citation: {ABSTAIN}
5. Be concise: 2-5 sentences."""

COMPARE_RULES = """6. Cover every paper the question asks about, citing at least one passage from
   each. Say explicitly where the papers agree or differ."""

ANALYSIS_RULES = f"""You extract a structured summary of ONE research paper using ONLY the passages provided.
Return a JSON object with exactly these keys: {", ".join(ANALYSIS_FIELDS)}.
Each value is an object {{"value": <one or two sentences>, "evidence": [<passage IDs>]}}.
- objective: the problem or goal. method: the approach. data: datasets, benchmarks or
  environments used. metrics: how results are measured. main_result: the headline
  finding, with numbers copied exactly. limitations: weaknesses the authors state.
- evidence lists the IDs of the passages that support the value.
- If the passages do not state a field, use {{"value": "{NOT_STATED}", "evidence": []}}.
- Use only facts stated in the passages. Do not use outside knowledge."""

# One retrieval query per analysis field (retrieval for a whole-paper summary).
ANALYSIS_QUERIES = {
    "objective": "problem addressed, goal and contributions of this work",
    "method": "proposed approach, method, framework or system design",
    "data": "datasets, benchmarks, environments and tasks used in the experiments",
    "metrics": "evaluation metrics such as attack success rate, utility, accuracy",
    "main_result": "main experimental results and findings",
    "limitations": "limitations, threats to validity and future work",
}


# Few-shot baseline: two training examples shown as earlier turns, one cited answer and
# one abstention whose passages come from the right paper but lack the answer. Training
# papers only (never eval or test papers); each demo keeps 2 passages to limit length.
DEMO_IDS = ("P003-q26", "P004-a05-abs")
DEMO_PASSAGES = 2
_demos = None


def load_demos(path=None):
    global _demos
    if _demos is None:
        path = path or Path(__file__).resolve().parent.parent / "data/train/build/train.jsonl"
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} missing: run src/build_examples.py first")
        rows = {}
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            rows[r["item_id"]] = r
        _demos = []
        for i in DEMO_IDS:
            r = rows[i]
            gold = [c for c in r["context"] if c["id"] in r["gold_chunks"]]
            rest = [c for c in r["context"] if c["id"] not in r["gold_chunks"]]
            _demos.append((r["question"], (gold + rest)[:DEMO_PASSAGES], r["answer"]))
    return _demos


def format_passages(passages) -> str:
    return "\n\n".join(f"[{p['id']}]\n{p['text']}" for p in passages)


def build_messages(task: str, question: str, passages, demos=()) -> list[dict]:
    """demos: [(question, passages, answer)] shown as earlier turns (qa/compare only)."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if task == "analysis":
        system = ANALYSIS_RULES
        user = f"Passages:\n\n{format_passages(passages)}\n\nSummarize the paper as JSON."
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
    system = RULES + ("\n" + COMPARE_RULES if task == "compare" else "")
    msgs = [{"role": "system", "content": system}]
    for q, ps, a in demos:
        msgs += [{"role": "user", "content": f"Passages:\n\n{format_passages(ps)}\n\nQuestion: {q}"},
                 {"role": "assistant", "content": a}]
    user = f"Passages:\n\n{format_passages(passages)}\n\nQuestion: {question}"
    return msgs + [{"role": "user", "content": user}]


def analysis_schema(context_ids) -> dict:
    """JSON schema for Ollama structured output; evidence is restricted to context IDs."""
    field = {"type": "object",
             "properties": {"value": {"type": "string"},
                            "evidence": {"type": "array",
                                         "items": {"type": "string", "enum": list(context_ids)}}},
             "required": ["value", "evidence"]}
    return {"type": "object", "properties": {f: field for f in ANALYSIS_FIELDS},
            "required": list(ANALYSIS_FIELDS)}
