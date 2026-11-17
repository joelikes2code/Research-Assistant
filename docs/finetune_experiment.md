# Fine-tuning experiment plan

## Hypothesis
LoRA fine-tuning `qwen3:1.7b` on 315 cited-answer examples raises **held-out success**
from 0.13 to at least 0.40 (mean of 3 training seeds), and beats both cheaper
alternatives below: a bigger model without training, and few-shot prompting.
Because the failures are mostly about format and about when to decline, which
examples can teach directly.

**Refuted if:** the mean held-out success is below 0.40, or it is not higher than the
few-shot baseline by more than the spread between seeds.

## Metric
**Primary: `success`, per item** (`src/eval/eval_generation.py`):
- answerable item: the model answers, cites at least one gold chunk, and breaks no
  citation rule;
- unanswerable item: the answer is exactly the abstain sentence.

Why not the old metrics alone: `abstained` and `rules_ok` reward a model that always
declines. Under `success`, always-abstain scores 0.12, the same as `qwen3:1.7b` today.

**Supporting:** `rules_ok`, `gold_cited`, `gold_in_ctx`, `answered`, `clean_abstain`, and
error counts by type. **Not measured automatically:** whether answers are true. Hand-check
20 answers per model against each item's `reference`.

## Data and leakage
- **Training:** P003, P004, P005, P008 (315 examples). **Validation:** P007 (56), used
  only to choose the epoch. P001, P002, P006 never appear (the notebook checks this).
- **The dev eval slice is not a generalization test after fine-tuning.** 22 of 31
  answerable dev items have a gold chunk cited in training, so the model could score by
  recall. Dev numbers are reported but not used for the conclusion.
- **Held-out slice:** items whose papers are not in the fine-tuning data (P001, P002,
  P006). Currently **15 items**, so one item is about 7 points. Expand it to about 40
  (more P002/P006 items) **before** the final evaluation. The test papers are evaluated
  once, at the end.

## Baselines (same retrieval, prompt, k=5, eval code, Q4_K_M in Ollama)
| ID | System | Status | Held-out success |
|---|---|---|---|
| B0 | always abstain | computed | 0.13 (2/15) |
| B1 | `qwen3:1.7b`, zero-shot | 1 eval run | 0.13 (2/15) |
| B2 | `qwen3:4b`, zero-shot (bigger model, no training) | to run, ~2 h | |
| B3 | `qwen3:1.7b` with 2 training examples in the prompt (few-shot) | needs a `--fewshot` option | |

## Runs
| ID | Config | Seeds | Where |
|---|---|---|---|
| FT | LoRA r=16, lr 2e-4, 3 epochs, best epoch by P007 val loss | 0, 1, 2 | Kaggle T4, ~30 min each (estimate) |
| A1 (optional) | FT without abstain examples | 0, 1, 2 | tests whether abstain data causes over-declining |

Report mean and standard deviation over seeds for each slice.

## Evaluation noise
Ollama is not bit-deterministic on this machine at temperature 0: about 1 in 3 answers
tested differed between identical runs. So run the B1 eval twice more and FT seed 0 twice
to measure the run-to-run spread. A model difference smaller than that spread is no
difference.

## Order
1. Expand the held-out items to about 40 (dev decisions never look at them).
2. B1 repeats, B2, B3: about 5 h of local machine time.
3. FT seeds 0-2 on Kaggle, import each with `src/import_ollama.py`, evaluate.
4. Hand-check 20 answers per system. Then evaluate on the test papers once.
5. A1 only if FT works, to explain why.

## Logging
Each training run saves `<run>.json`: seed, git commit, data hashes, hyperparameters,
validation loss before training and per epoch, library versions, GPU. Each eval run
saves a summary JSON (committed) and per-item JSONL (gitignored, can quote P001).
