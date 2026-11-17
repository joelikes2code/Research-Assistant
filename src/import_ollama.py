#!/usr/bin/env python3
"""
import_ollama.py - register a fine-tuned GGUF with Ollama, served exactly like the base.

Usage:
    python src/import_ollama.py path/to/ra-qwen3-1.7b-s0.Q4_K_M.gguf ra-qwen3-1.7b-s0
                                [--base qwen3:1.7b]

Copies the base model's Modelfile (TEMPLATE, stop tokens, parameters) and swaps only
the FROM line, so the fine-tuned model sees the same prompt format and sampling
settings as the baseline it is compared with. Then checks the format with
chat_format.py (identical prompt token counts for the chat API and the training text).
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gguf", type=Path)
    ap.add_argument("name")
    ap.add_argument("--base", default="qwen3:1.7b")
    args = ap.parse_args()
    if not args.gguf.is_file():
        sys.exit(f"not found: {args.gguf}")

    base = subprocess.run(["ollama", "show", args.base, "--modelfile"], capture_output=True,
                          text=True, encoding="utf-8", check=True).stdout
    lines = base.splitlines()
    n_from = sum(l.startswith("FROM ") for l in lines)
    if n_from != 1:
        sys.exit(f"expected one FROM line in the {args.base} Modelfile, found {n_from}")
    out = [f"FROM {args.gguf.resolve()}" if l.startswith("FROM ") else l for l in lines]
    modelfile = args.gguf.with_suffix(".Modelfile")
    modelfile.write_text("\n".join(out) + "\n", encoding="utf-8")

    subprocess.run(["ollama", "create", args.name, "-f", str(modelfile)], check=True)
    check = Path(__file__).resolve().parent / "chat_format.py"
    sys.exit(subprocess.run([sys.executable, str(check), "--model", args.name, "--n", "3"]).returncode)


if __name__ == "__main__":
    main()
