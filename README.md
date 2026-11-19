# Research Assistant

Answers questions about a set of agentic-AI security papers, citing a passage for every
sentence, and declines when the passages do not contain the answer. Everything runs
locally on a CPU with [Ollama](https://ollama.com); fine-tuning uses a free cloud GPU.

The project is also an experiment: can LoRA fine-tuning teach a small model
(`qwen3:1.7b`) to cite and decline correctly, compared with a bigger model and with
few-shot prompting? See [docs/finetune_experiment.md](docs/finetune_experiment.md).

## Pipeline

```
PDFs ─► chunk_papers.py ─► chunks.jsonl ─► index.py ─► Chroma + BM25
                                                          │
question ─► retrieve.py (hybrid search) ─► prompt.py ─► Ollama ─► check_citations.py
                                                          │
                              app/streamlit_app.py  ◄─────┴─────►  src/eval/
```

| Stage | File | What it does |
|---|---|---|
| Chunk | `src/chunk_papers.py` | Section-aware chunks of the main text (Abstract to References), ≤ 512 tokens, stable IDs like `P003_c12`. **Frozen at v0.10** (tag `chunker-v0.10-frozen`): training data cites these IDs. |
| Index | `src/index.py` | Embeds chunks with `bge-small-en-v1.5` into Chroma; records fingerprints so a stale index is refused. |
| Retrieve | `src/retrieve.py` | Dense + BM25 search merged with reciprocal-rank fusion. |
| Generate | `src/generate.py`, `src/prompt.py` | Builds the prompt, calls Ollama (temperature 0, thinking off). Tasks: `qa`, `compare`, `analysis`. |
| Check | `src/check_citations.py` | Every sentence cited, only given passages cited, numbers and quotes present in the cited passage, exact decline sentence. |

## Setup

Requires Python 3.13, [uv](https://docs.astral.sh/uv/) and Ollama.

```
uv sync
ollama pull qwen3:1.7b
uv run python src/chunk_papers.py data/pdfs data/processed   # only if PDFs changed
uv run python src/index.py
```

PDFs are not in the repository (see Data). Put them in `data/pdfs/` as `P001.pdf` ... `P008.pdf`.

## Use

```
uv run streamlit run app/streamlit_app.py
uv run python src/generate.py "How does MELON detect indirect prompt injection?"
uv run python src/generate.py "How do CaMeL and MELON differ?" --task compare --papers P005 P008
uv run python src/generate.py "" --task analysis --papers P007
```

A passing citation check means the answer followed the rules, not that it is true:
the app shows the passages so each claim can be checked.

## Data and licences

`data/papers.csv` is the record for every paper: licence, how it was verified, split and
first public date.

- **P001** has an exclusive publisher licence: used locally only, never in training data,
  never uploaded, its eval items are in a gitignored file.
- **P002, P006** are test papers (first public after Qwen3's release): never in training data.
- **P003, P004, P005, P008** train, **P007** validation: CC BY 4.0.
- Built training files contain paper text and are gitignored; rebuild them with
  `src/build_examples.py`.

## Evaluation

```
uv run python src/eval/items.py             # check every gold quote still matches a chunk
uv run python src/eval/eval_retrieval.py    # recall@k / MRR for hybrid, dense, BM25
uv run python src/eval/eval_generation.py --model qwen3:1.7b [--fewshot]
uv run python src/eval/compare.py           # mean ± sd over runs, per slice
```

- **69 items** (`data/eval/items.jsonl` + 4 private): answerable, compare and unanswerable
  questions. Gold evidence is stored as exact quotes, resolved to chunk IDs at run time,
  so the eval set survives re-chunking.
- **Primary metric `success`:** answerable item answered with a gold citation and no rule
  broken; unanswerable item declined with exactly the decline sentence. A model that
  always declines scores 0.12 overall.
- **Held-out slice (43 items):** papers never used in fine-tuning. Only this slice measures
  generalization; many dev items share facts with the training data.
- **Noise:** Ollama is not bit-deterministic on CPU even at temperature 0, so evals are
  repeated and compared against their run-to-run spread.

### Results

| System | Eval runs | success, all (69) | success, held-out (43) | hand-checked correct (20) |
|---|---|---|---|---|
| always decline (trivial) | – | 0.12 | 0.14 | – |
| `qwen3:1.7b` zero-shot | 2 | 0.13 | 0.14 | 11/20 |
| `qwen3:1.7b` few-shot | 2 | 0.45 | 0.42 | 13/20 |
| `qwen3:4b` zero-shot | 1 | 0.57 | 0.51 | 14/20 |
| `qwen3:1.7b` + LoRA (3 seeds) | 1 each | 0.70 ± 0.01 | 0.64 ± 0.01 | 14, 12, 11 /20 |

Fine-tuning makes the 1.7B model follow the citation rules (rules_ok ≈ 1.0) and decline
exactly when it should, which the automatic metric rewards. **Its answers are not more
often correct:** in a blind hand-check of the same 20 held-out questions every system is
55-70% correct, with no detectable difference. The common error is a true fact cited
to the wrong passage. Retrieval also caps every system: the gold chunk is in the top 5
for only 78% of answerable items (hybrid; BM25 83%, dense only 67%). Details, seeds and
significance tests: [docs/finetune_experiment.md](docs/finetune_experiment.md).

## Fine-tuning

1. `uv run python src/build_examples.py` builds 371 examples from the drafts in
   `data/train/drafts/` (315 train, 56 validation), validates them, and writes
   `data/train/build/{train,val}_chat.jsonl`. `src/make_review.py` writes a review sheet.
2. Run `notebooks/finetune_lora.ipynb` on Kaggle or Colab (T4 GPU) with those two files,
   once per seed (0, 1, 2). It exports a Q4_K_M GGUF.
3. `uv run python src/import_ollama.py <file>.gguf ra-qwen3-1.7b-s0` registers it with
   Ollama using the base model's template, then checks the prompt format.
4. Evaluate with `eval_generation.py --model ra-qwen3-1.7b-s0` and compare with `compare.py`.

The training text is rendered exactly as Ollama's qwen3 template renders prompts with
thinking off (`src/chat_format.py`), which differs from Hugging Face's Qwen3 template.

## Known limitations

- Tables and equations extract poorly; text inside figures is kept as body text.
- The eval items, training examples and gold quotes were drafted by an LLM (Claude) and
  checked mechanically and by a second pass; a human spot-check is still advised.
- The held-out slice is small (43 items): one item is about 2 points.
- Answer correctness beyond citation rules is not scored automatically.

## Layout

```
app/            Streamlit UI
data/           papers.csv, eval items, training drafts (PDFs and built files gitignored)
docs/           experiment plan
notebooks/      fine-tuning notebook
results/        eval summaries (per-item outputs gitignored)
src/            pipeline; src/eval/ evaluation
```
