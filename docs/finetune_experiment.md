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

## Results (2026-11-18, hand-check batch 2 2026-11-19)

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
correctness. One answer passed automatically but was judged wrong.

**Hand-check batch 2** (B2 + the three FT seeds, the same 20 items, blind): drafted by
Claude with a one-line reason per verdict, then checked by the user, who changed 3 of 80
(E018: one FT answer w to p, one c to p; E040: the B2 answer p to c).

| System | Graded by | correct | partly | wrong | abstained |
|---|---|---|---|---|---|
| B1 zero-shot | user | 11 | 0 | 9 | 0 |
| B3 few-shot | user | 13 | 1 | 6 | 0 |
| B2 qwen3:4b | Claude, user-checked | 14 | 2 | 3 | 1 |
| FT seed 0 / 1 / 2 | Claude, user-checked | 14 / 12 / 11 (mean 12.3) | 0 / 4 / 2 | 5 / 3 / 5 | 1 / 1 / 2 |

- Each FT seed vs B2, paired: FT-only correct 3 / 1 / 1 items, B2-only 3 / 3 / 4; exact
  McNemar p = 1.00 / 0.62 / 0.38.
- **Conclusion: fine-tuning improved format and abstention, not correctness.** All six
  systems are 55-70% correct, no difference this sample can detect (n = 20); if anything
  the untrained 4B model is as correct as the best seed. The large automatic gain (0.64
  vs 0.42 few-shot) is rule compliance and clean declining.
- The most common error in every system is a correct fact cited to the wrong chunk
  (X001, X003, E018, E051). E052 was wrong for all four systems and E047 was declined
  by all four (its gold chunk is likely not retrieved).
- Limits: n = 20; two graders across the batches (B1/B3 by the user, the rest drafted by
  Claude, who also wrote the training data, so a bias toward FT was possible; the user's
  three changes all moved FT seeds down or B2 up).

**Second inference backend (added 2026-11-19, not pre-registered).** All six systems
were re-run with Ollama on the integrated GPU (Vulkan, `OLLAMA_IGPU_ENABLE=1`) instead of
the CPU; summaries in `results/igpu/`. Retrieval was identical (same passages for all 69
items). Held-out success CPU -> GPU: FT 0.65/0.65/0.63 -> 0.60/0.67/0.60 (mean 0.64 ->
0.63 ± 0.04), B2 0.51 -> 0.53, B3 0.42 -> 0.42, B1 0.14 -> 0.09. Same ranking; FT vs B3
paired p = 0.039 / 0.003 / 0.039, FT vs B2 p = 0.55 / 0.07 / 0.55 (still unresolved).
The backend reworded 23-28 of 69 FT answers, 24 for B2, 41 for B3 and 56 for B1, but
changed held-out success on only 1-4 items per system: less constrained systems are
moved more. Backend change is a larger perturbation than repeating a run (0-1 of 69),
so every comparison uses runs from one backend; the planned follow-up analyses use the
GPU runs.

**Error breakdown (added 2026-11-19, exploratory; `src/eval/error_breakdown.py`).**
Each answerable held-out item is classed as success, retrieval miss (no gold chunk among
the 5 passages) or generation error with a gold chunk available (declined, cited only
non-gold passages, or cited gold but broke a rule). GPU runs, n = 37:

| System | success | retrieval miss | wrong passage | rule broken | declined |
|---|---|---|---|---|---|
| B1 zero-shot | 3 | 7 | 14 | 13 | 0 |
| B3 few-shot | 14 | 7 | 8 | 8 | 0 |
| B2 qwen3:4b | 19 | 7 | 8 | 3 | 0 |
| FT s0 / s1 / s2 | 20 / 23 / 20 | 7 | 10 / 7 / 8 | 0 / 0 / 1 | 0 / 0 / 1 |

- Retrieval misses are the same 7 items for every system (identical search).
- Fine-tuning removed rule breaks (13 -> 8 -> 3 -> 0) but not wrong-passage citations
  (about 8 for every system except zero-shot). CPU runs show the same pattern.

Crossed with the 120 hand-checked answers (CPU runs): metric success was correct 35/40,
partly 5, wrong 0; but 40 of 80 metric failures were judged correct (25 "wrong passage",
10 "rule broken", 5 answered after a retrieval miss). The metric is conservative: it
never passed a wrong answer but caught only 35 of 75 correct ones (47%), mainly because
the gold lists are incomplete (the answer is often also in another passage, e.g. a later
section restating the abstract). Of the 31 answers judged wrong, 13 were retrieval misses
and 17 cited a wrong passage although a gold one was in context.

Position: in wrong-passage cases the gold passage was usually ranked first and the model
cited a lower-ranked one; it cited a passage ranked above the gold one in only 14 of 72
citations, so there is no simple primacy bias (to be tested by the passage-order probe).

**Passage-order probe (added 2026-11-20, exploratory, incomplete; `src/eval/order_probe.py`).**
Same five passages as the GPU base run, shown in reversed or shuffled order
(`eval_generation.py --order`, `--outdir results/probe`). Only FT seed 0 reversed has
finished; B2 reversed, s0/B2 shuffled and s1/s2 reversed are still to run. Answerable
held-out items, n = 37, one run:

| System | success base -> reversed | McNemar p | same cited set | cites retriever rank 1 | cites shown position 1 |
|---|---|---|---|---|---|
| FT s0 | 20 -> 23 | 0.38 | 25/35 | 0.42 -> 0.43 | 0.42 -> 0.11 |

- Citations follow the passage, not the slot: the share citing the retriever's top
  passage is unchanged, while the share citing whatever is shown first collapses. This
  agrees with the error breakdown (no primacy bias), but it is one seed and one order.

**NLI support check (implemented, not yet run; `src/eval/support_check.py`).** Scores
whether each cited chunk entails the sentence citing it (DeBERTa-v3-large NLI, threshold
0.5 and max over cited chunks, both fixed in advance), and validates the checker against
the 120 hand-checked answers. Install with `uv sync --group nli`. A test pair took about
17 s on the CPU, so a full run is an overnight job. This is the automatic test of the
"format, not correctness" conclusion above.

**Eval noise is small:** between repeated runs, B1 changed 1 of 69 answers and B3 none,
far below the "1 in 3" measured earlier. So the second FT seed 0 eval run from the plan
was skipped: the differences between systems (and between seeds) are not run-to-run noise.

## Logging
Each training run saves `<run>.json`: seed, git commit, data hashes, hyperparameters,
validation loss before training and per epoch, library versions, GPU. Each eval run
saves a summary JSON (committed) and per-item JSONL (gitignored, can quote P001).
