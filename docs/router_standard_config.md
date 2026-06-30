# Standard Router Configuration

This document freezes the selected local router configuration after the
v001-small and v002-small LoRA comparisons.

## Runtime

- model: `Qwen/Qwen3.5-4B`
- adapter: none
- model family: `multimodal-lm`, text-only
- system prompt: `prompts/router_system_v3.md`
- few-shot messages: `prompts/router_fewshot_v3.jsonl`
- output schema: `schemas/router_output.schema.json`
- schema repair: once, only after parse or schema failure
- vocabulary source: `docs/router_canonical_vocabulary.md`
- vocabulary validation: strict
- vocabulary repair: once, only after schema-valid vocabulary failure
- maximum new tokens: 256
- temperature: 0
- model loading: local files only
- saved output: final-only JSON; do not retain internal reasoning

The processing order is JSON parse, schema validation, optional schema repair,
vocabulary validation, optional vocabulary repair, then schema and vocabulary
revalidation. A result that remains invalid must not be dispatched.

## Selected Baselines

Eval inputs:

- `evals/router_eval_001_canonical.jsonl`
- `evals/router_eval_holdout_001.jsonl`

Selected base results:

- `eval_results/qwen35_4b_prompt_v3_canonical_vocab_repair_eval_001.json`
- `eval_results/qwen35_4b_prompt_v3_canonical_vocab_repair_predictions_001.jsonl`
- `eval_results/qwen35_4b_prompt_v3_holdout_vocab_repair_eval_001.json`
- `eval_results/qwen35_4b_prompt_v3_holdout_vocab_repair_predictions_001.jsonl`

The canonical baseline is 50/50 schema-valid, 50/50 vocabulary-valid, 45 mode
matches, 39 tool matches, 17 verification matches, and 3 risk underestimates.
The holdout baseline is 30/30 schema-valid, 30/30 vocabulary-valid, 28 mode
matches, 24 tool matches, 8 verification matches, and 8 risk underestimates.

## Non-Selected Adapters

- `adapters/qwen35-4b-router-lora-v001-small`
- `adapters/qwen35-4b-router-lora-v002-small`

These adapters are comparison artifacts. They must not be loaded by the
standard router configuration. Their directories remain git-ignored.

## Operational Boundary

The local router chooses a route and verification contract; it does not by
itself authorize external transmission, destructive code execution, or a
high-risk engineering decision. External dispatch, RAG retrieval, human
approval, and deterministic checks remain separate controlled stages.
