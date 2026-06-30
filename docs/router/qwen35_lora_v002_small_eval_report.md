# Qwen3.5 4B Router LoRA v002-small Evaluation

M17 evaluates the v002-small adapter with prompt-router v3, JSON schema
repair, strict canonical vocabulary validation, and one vocabulary repair
attempt. Generation used local model files, 256 maximum new tokens, and
temperature zero.

## Canonical 50

| Metric | Base M14 | LoRA v001 M14 | LoRA v002 M17 | v002 vs base |
| --- | ---: | ---: | ---: | ---: |
| json_valid_count | 50 | 50 | 50 | 0 |
| schema_valid_count | 50 | 50 | 50 | 0 |
| vocab_valid_count | 50 | 50 | 49 | -1 |
| expected_mode_match_count | 45 | 45 | 45 | 0 |
| must_tools_contained_count | 39 | 39 | 39 | 0 |
| must_verification_contained_count | 17 | 16 | 18 | +1 |
| risk_underestimated_count | 3 | 4 | 4 | +1 (worse) |
| fusion_policy_match_count | 48 | 48 | 48 | 0 |
| request_more_info_count | 8 | 8 | 8 | 0 |
| vocab_repair_attempted_count | 3 | 3 | 3 | 0 |
| vocab_repair_success_count | 3 | 3 | 2 | -1 |

The remaining vocabulary failure is `router_eval_001_039`. Its repair placed
the canonical verification label `backup_check` in `needed_tools`, where it
is not an allowed tool. The strict evaluator therefore exited nonzero after
saving the complete result and prediction files. The run was not repeated or
selectively replaced.

## Holdout 30

The latest available v001 holdout reference is the M12 prompt-v3 run. It
predates M14 vocabulary retry, so its vocabulary metrics are not available.

| Metric | Base M14 | LoRA v001 M12 | LoRA v002 M17 | v002 vs base |
| --- | ---: | ---: | ---: | ---: |
| json_valid_count | 30 | 30 | 30 | 0 |
| schema_valid_count | 30 | 30 | 30 | 0 |
| vocab_valid_count | 30 | n/a | 30 | 0 |
| expected_mode_match_count | 28 | 28 | 28 | 0 |
| must_tools_contained_count | 24 | 26 | 26 | +2 |
| must_verification_contained_count | 8 | 8 | 8 | 0 |
| risk_underestimated_count | 8 | 8 | 8 | 0 |
| fusion_policy_match_count | 29 | 29 | 29 | 0 |
| request_more_info_count | 6 | 6 | 6 | 0 |
| vocab_repair_attempted_count | 2 | n/a | 0 | -2 |
| vocab_repair_success_count | 2 | n/a | 0 | -2 |

## Adoption Decision

**Keep base standard and retain v002 as reference.**

V002 does not clearly outperform the standard base router. Canonical
verification improved only from 17 to 18, below the target of 23, while risk
underestimation worsened from 3 to 4 and strict vocabulary validity fell from
50 to 49. On holdout, tools improved from 24 to 26, but verification stayed
at 8 rather than the target 12, and risk underestimation stayed at 8 rather
than the target maximum 6. Mode, fusion, and clarification behavior did not
regress materially, but these isolated gains do not offset the contract and
risk regressions.

The standard remains Qwen3.5 4B base with prompt-router v3 and schema/vocab
repair. LoRA v001-small and v002-small remain comparison artifacts and are not
selected adapters.

## Artifacts

- canonical result: `eval_results/qwen35_lora_v002_small_prompt_v3_canonical_vocab_repair_eval_001.json`
- canonical predictions: `eval_results/qwen35_lora_v002_small_prompt_v3_canonical_vocab_repair_predictions_001.jsonl`
- holdout result: `eval_results/qwen35_lora_v002_small_prompt_v3_holdout_vocab_repair_eval_001.json`
- holdout predictions: `eval_results/qwen35_lora_v002_small_prompt_v3_holdout_vocab_repair_predictions_001.jsonl`

Operational logs and adapter weights are git-ignored.
