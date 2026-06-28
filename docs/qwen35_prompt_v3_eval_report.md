# Qwen3.5 Prompt-Router v3 Evaluation

M12 evaluates canonical vocabulary repair without additional training. All v3
runs used local `Qwen/Qwen3.5-4B` files, temperature zero, 256 maximum new
tokens, schema repair enabled, and the v3 system/few-shot prompts.

## Existing 50-case Eval

| Metric | Base prompt v2 | Base prompt v3 | Delta |
| --- | ---: | ---: | ---: |
| expected_mode_match_count | 45 | 45 | 0 |
| must_tools_contained_count | 37 | 39 | +2 |
| must_verification_contained_count | 7 | 0 | -7 |
| risk_underestimated_count | 3 | 3 | 0 |
| request_more_info_count | 8 | 8 | 0 |

The verification decrease is an eval-contract artifact: all 86 expected labels
in `evals/router_eval_001.jsonl` are legacy natural-language phrases, while v3
deliberately emits canonical identifiers. This metric must not be interpreted
as a model regression. Mode, risk, and request-more-info behavior stayed
stable; tool containment improved by two net cases.

## Base Holdout

| Metric | Base prompt v2 | Base prompt v3 | Delta |
| --- | ---: | ---: | ---: |
| expected_mode_match_count | 27 | 28 | +1 |
| must_tools_contained_count | 22 | 24 | +2 |
| must_verification_contained_count | 0 | 8 | +8 |
| risk_underestimated_count | 8 | 8 | 0 |
| request_more_info_count | 7 | 6 | -1 |

The holdout uses canonical expectations in all 30 cases. Prompt v3 therefore
produced a real strict-verification improvement from 0/30 to 8/30. It also
corrected holdout case 010 from `request_more_info` to `external_expert` and
improved tool containment.

## LoRA v001-small Holdout

| Metric | LoRA prompt v2 | LoRA prompt v3 | Delta |
| --- | ---: | ---: | ---: |
| expected_mode_match_count | 28 | 28 | 0 |
| must_tools_contained_count | 22 | 26 | +4 |
| must_verification_contained_count | 0 | 8 | +8 |
| risk_underestimated_count | 8 | 8 | 0 |
| request_more_info_count | 6 | 6 | 0 |

Under prompt v3, LoRA and base both reached 8 strict verification matches and
28 mode matches. LoRA reached 26 tool matches versus base's 24. Neither prompt
repair nor the existing adapter improved the eight holdout risk
underestimations.

## Decision

Prompt-router v3 is the correct contract for future Qwen3.5 4B work because it
substantially improves canonical label emission and holdout verification
without a weight update. It is not sufficient by itself: exact verification
coverage remains 8/30 and risk calibration remains unchanged.

Continue with `Qwen/Qwen3.5-4B` on RTX5080 16GB. Qwen3.5 9B is outside the
project scope. Before proposing v002 training, migrate or version the original
50-case eval to canonical expectations and add whitelist validation for model
outputs.

## Results

- `eval_results/qwen35_4b_prompt_v3_eval_001.json`
- `eval_results/qwen35_4b_prompt_v3_predictions_001.jsonl`
- `eval_results/qwen35_4b_prompt_v3_holdout_eval_001.json`
- `eval_results/qwen35_4b_prompt_v3_holdout_predictions_001.jsonl`
- `eval_results/qwen35_lora_v001_small_prompt_v3_holdout_eval_001.json`
- `eval_results/qwen35_lora_v001_small_prompt_v3_holdout_predictions_001.jsonl`

No training or adapter update occurred.

M13 subsequently migrated the original 50-case eval to canonical expectations
and reran base and LoRA v001-small. See
`docs/qwen35_prompt_v3_canonical_eval_report.md`; the legacy 50-case
verification score in this document remains non-comparable by design.
