#!/usr/bin/env python3
"""
generate.py - retrieve passages, prompt a local Ollama model, check the citations.

Usage:
    python src/generate.py "How does MELON detect indirect prompt injection?"
    python src/generate.py "How do CaMeL and MELON differ?" --task compare --papers P005 P008
    python src/generate.py "" --task analysis --papers P007
    python src/generate.py QUESTION [-k 5] [--model qwen3:4b] [--json]

Needs Ollama running (http://localhost:11434, or set OLLAMA_HOST) and the model
pulled (`ollama pull qwen3:1.7b`). Temperature 0 and a fixed seed make runs mostly,
but not exactly, repeatable: Ollama on this machine is not bit-deterministic, and
about 1 in 3 answers tested differed between identical runs. Repeat an eval to measure
that noise before comparing models. Qwen3's thinking mode is off by default (--think).

--json prints one record in validate_examples.py's format (item_id, task, paper_ids,
question, context, answer) plus "model" and "citation_errors".
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_citations import check_answer  # noqa: E402
from prompt import ANALYSIS_QUERIES, analysis_schema, build_messages, load_demos  # noqa: E402

DEFAULT_MODEL = "qwen3:1.7b"
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA
NUM_CTX = 8192          # 8 passages x <=512 tokens + instructions fits comfortably
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def ollama_chat(messages, model=DEFAULT_MODEL, fmt=None, think=False, timeout=600):
    body = {"model": model, "messages": messages, "stream": False, "think": think,
            "options": {"temperature": 0, "seed": 0, "num_ctx": NUM_CTX}}
    if fmt is not None:
        body["format"] = fmt
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama error {e.code}: {e.read().decode(errors='replace')}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach Ollama at {OLLAMA} ({e.reason}); is it running?") from e
    content = THINK_RE.sub("", out["message"]["content"])
    # Thinking-only variants (e.g. some qwen3:4b builds) ignore think=false and emit
    # their reasoning with only a closing tag: keep what follows the last </think>.
    return content.rsplit("</think>", 1)[-1].strip()


def get_context(retriever, task, question, k, paper_ids):
    """compare: k passages per paper. analysis: top 2 per field query, max 2k."""
    if task == "compare":
        hits = [h for pid in paper_ids for h in retriever.search(question, k, [pid])]
    elif task == "analysis":
        seen, hits = set(), []
        for q in ANALYSIS_QUERIES.values():
            for h in retriever.search(q, 2, paper_ids):
                if h["chunk_id"] not in seen:
                    seen.add(h["chunk_id"])
                    hits.append(h)
        hits = hits[:2 * k]
    else:
        hits = retriever.search(question, k, paper_ids)
    return [{"id": h["chunk_id"], "text": h["text"]} for h in hits]


def answer(question, task="qa", k=5, paper_ids=None, model=DEFAULT_MODEL,
           think=False, retriever=None, fewshot=False):
    if task == "compare" and len(paper_ids or []) < 2:
        raise ValueError("--task compare needs --papers with at least two IDs")
    if task == "analysis" and len(paper_ids or []) != 1:
        raise ValueError("--task analysis needs --papers with exactly one ID")
    if retriever is None:
        from retrieve import Retriever
        retriever = Retriever()
    context = get_context(retriever, task, question, k, paper_ids)
    demos = load_demos() if fewshot and task != "analysis" else ()
    msgs = build_messages(task, question, context, demos)
    if task == "analysis":
        raw = ollama_chat(msgs, model, fmt=analysis_schema([c["id"] for c in context]),
                          think=think)
        try:
            ans = json.loads(raw)
        except json.JSONDecodeError:
            ans = raw                                   # check_answer reports it
    else:
        ans = ollama_chat(msgs, model, think=think)
    return {"task": task, "paper_ids": list(paper_ids or []), "question": question,
            "context": context, "answer": ans, "model": model,
            "citation_errors": check_answer(task, ans, context, paper_ids or [])}


def main():
    ap = argparse.ArgumentParser(description="Answer a question about the papers.")
    ap.add_argument("question")
    ap.add_argument("--task", default="qa", choices=["qa", "compare", "analysis"])
    ap.add_argument("--papers", nargs="*", help="restrict retrieval to these paper IDs")
    ap.add_argument("-k", type=int, default=5, help="passages (per paper for compare)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--think", action="store_true", help="enable Qwen3 thinking mode")
    ap.add_argument("--fewshot", action="store_true", help="prepend two training examples")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        rec = answer(args.question, args.task, args.k, args.papers, args.model, args.think,
                     fewshot=args.fewshot)
    except (ValueError, RuntimeError) as e:
        sys.exit(str(e))
    if args.json:
        print(json.dumps(rec, ensure_ascii=False))
        return
    print("passages:", ", ".join(c["id"] for c in rec["context"]))
    print()
    if isinstance(rec["answer"], dict):
        for f, v in rec["answer"].items():
            print(f"{f}: {v.get('value')}  {v.get('evidence')}")
    else:
        print(rec["answer"])
    print()
    errs = rec["citation_errors"]
    print("citation check:", "OK (rules followed; not proof of correctness)" if not errs
          else "; ".join(errs))


if __name__ == "__main__":
    main()
