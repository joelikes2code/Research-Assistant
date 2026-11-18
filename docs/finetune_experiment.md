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
  P006). Expanded from 15 to **43 items** (37 answerable, 6 unanswerable) before any
  fine-tuned model was evaluated; one item is about 2.3 points.

## Baselines (same retrieval, prompt, k=5, eval code, Q4_K_M in Ollama)
| ID | System | Eval runs | Held-out success (n=43) |
|---|---|---|---|
| B0 | always abstain | computed | 0.14 (6/43) |
| B1 | `qwen3:1.7b`, zero-shot | 2 | 0.14, 0.14 |
| B2 | `qwen3:4b-q4_K_M`, zero-shot (bigger model, no training) | 1 | 0.51 |
| B3 | `qwen3:1.7b` with 2 training examples in the prompt (`--fewshot`) | 2 | 0.42, 0.42 |

B2 note: the first B2 attempt used the installed `qwen3:4b` tag, which turned out to be
Qwen3-4B-Thinking-2507. It reasons before every answer and ignores `think=false` (about
300 s per item; it timed out at item 16). That run is discarded. B2 uses
`qwen3:4b-q4_K_M`, the original hybrid Qwen3 4B: same generation and chat template as
`qwen3:1.7b`, so model size is the only difference from B1.

## Runs
| ID | Config | Seeds | Where |
|---|---|---|---|
| FT | LoRA r=16, lr 2e-4, 3 epochs, best epoch by P007 val loss | 0, 1, 2 | Kaggle T4, 20-30 min each |
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

## Results (2026-11-18; hand-check batch 2 pending)

**Training** (`results/ra-qwen3-1.7b-s*.json`): validation loss 4.002 before training for
every seed (LoRA starts as the identity); best 0.481 / 0.476 / 0.477, kept epoch 3 / 2 / 2.
Same data hashes in all runs.

**Held-out slice, automatic metrics** (`results/gen_*.json`):

| System | success | rules_ok | gold_cited | clean_abstain |
|---|---|---|---|---|
| B1 zero-shot (2 runs) | 0.14 | 0.21 | 0.49 | 0.17 |
| B3 few-shot (2 runs) | 0.42 | 0.56 | 0.62 | 0.67 |
| B2 qwen3:4b zero-shot | 0.51 | 0.86 | 0.59 | 0.50 |
| FT seed 0 / 1 / 2 | 0.65 / 0.65 / 0.63 (mean 0.64) | 1.00 / 1.00 / 0.98 | 0.59 / 0.59 / 0.57 | 1.00 |

- Both pre-registered criteria hold: mean 0.64 >= 0.40, and the margin over few-shot
  (+0.22) is much larger than the seed spread (0.02).
- Paired on the same 43 items (added analysis, not pre-registered): FT succeeds where B3
  fails on 12 items, the reverse on 2-3; exact McNemar p = 0.013 / 0.013 / 0.035 per seed.
  These are not independent tests (same items, same B3 run).
- Against B2 (4B, no training): FT-only 9 items, B2-only 3-4; McNemar p = 0.15 / 0.15 /
  0.27. The fine-tuned 1.7B scores higher (0.64 vs 0.51) but this sample cannot
  separate it from the 2.4x larger model. B2 follows the rules better than few-shot
  (rules_ok 0.86) but abstains correctly on only half the unanswerable items.
- The gain is format and abstention, not evidence choice: `gold_cited` does not improve
  over few-shot. Retrieval caps it: the gold chunk is in context for only 78% of items.

**Hand-check batch 1** (`make_answer_review.py`, 20 answerable held-out items, blind):
B1 11/20 correct, B3 13/20 (1 partly). Paired: B3-only 3, B1-only 1, McNemar p = 0.62.
Automatic success on the same 20 items was 1/20 (B1) and 6/20 (B3): the metric mostly
measures compliance with the citation rules, and few-shot's gain was format, not
correctness. One answer passed automatically but was judged wrong. Batch 2 (B2 + the
three FT seeds, same 20 items) decides whether fine-tuning improves correctness or only
format.

**Eval noise is small:** between repeated runs, B1 changed 1 of 69 answers and B3 none,
far below the "1 in 3" measured earlier. So the second FT seed 0 eval run from the plan
was skipped: the differences between systems (and between seeds) are not run-to-run noise.

## Logging
Each training run saves `<run>.json`: seed, git commit, data hashes, hyperparameters,
validation loss before training and per epoch, library versions, GPU. Each eval run
saves a summary JSON (committed) and per-item JSONL (gitignored, can quote P001).
