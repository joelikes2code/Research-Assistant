#!/usr/bin/env python3
"""
chat_format.py - render chat messages EXACTLY as Ollama's qwen3 template does with
think=false, so the fine-tuning text matches the prompt the model sees at inference.

Ollama's template differs from Hugging Face's Qwen3 apply_chat_template: it appends
" /no_think" to the last user turn and opens the answer with "<think>\n</think>\n"
(HF: no suffix, "<think>\n\n</think>\n\n"). Training on the HF format would teach the
model a prompt it never sees when served by Ollama.

    from chat_format import render_prompt, render_training
    render_prompt(messages)    # what Ollama feeds the model (ends where the answer starts)
    render_training(messages)  # prompt + assistant answer + "<|im_end|>\n"

Verify against a running Ollama: the chat API and the raw rendered prompt must give the
same prompt token count (a wrong newline or missing " /no_think" changes it). Outputs
are compared too but only reported: Ollama is not bit-deterministic here even at
temperature 0 (the same raw prompt twice can differ), so output equality is not a test.
    python src/chat_format.py [--n 5] [--model qwen3:1.7b]
Checked 2026-11-18 on qwen3:1.7b: 5/5 identical token counts.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

ASSISTANT_START = "<|im_start|>assistant\n<think>\n</think>\n"


def render_prompt(messages):
    """messages: [system?, user] (an assistant message at the end is ignored)."""
    system = "".join(m["content"] for m in messages if m["role"] == "system")
    users = [m["content"] for m in messages if m["role"] == "user"]
    if len(users) != 1:
        raise ValueError("expected exactly one user message")
    out = f"<|im_start|>system\n\n{system}<|im_end|>\n" if system else ""
    out += f"<|im_start|>user\n{users[0]} /no_think<|im_end|>\n"
    return out + ASSISTANT_START


def render_training(messages):
    answer = messages[-1]
    if answer["role"] != "assistant":
        raise ValueError("last message must be the assistant answer")
    return render_prompt(messages[:-1]) + answer["content"] + "<|im_end|>\n"


def _post(path, body):
    req = urllib.request.Request(f"http://localhost:11434{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description="Check render_prompt against Ollama.")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default="qwen3:1.7b")
    ap.add_argument("--data", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data/train/build/val_chat.jsonl")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    opts = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 64}
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")][:args.n]
    ok = 0
    for r in rows:
        msgs = r["messages"][:-1]
        chat = _post("/api/chat", {"model": args.model, "messages": msgs, "stream": False,
                                   "think": False, "options": opts})
        raw = _post("/api/generate", {"model": args.model, "prompt": render_prompt(msgs),
                                      "raw": True, "stream": False, "options": opts})
        same_n = chat["prompt_eval_count"] == raw["prompt_eval_count"]
        same_out = chat["message"]["content"].strip() == raw["response"].strip()
        ok += same_n
        print(f"{r['item_id']:12} prompt tokens chat={chat['prompt_eval_count']} "
              f"raw={raw['prompt_eval_count']}  same output={same_out}")
    print(f"{ok}/{len(rows)} identical prompt token counts")
    sys.exit(0 if ok == len(rows) else 1)


if __name__ == "__main__":
    main()
