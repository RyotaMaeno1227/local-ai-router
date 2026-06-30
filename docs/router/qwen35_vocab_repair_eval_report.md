# Qwen3.5 Vocabulary Repair Eval Report

M14 adds inference-time canonical vocabulary validation and one repair attempt
after schema validation. No model weights or adapters were changed.

## Base Canonical 50

| Metric | M13 v3 | M14 v3 + vocab repair | Delta |
| --- | ---: | ---: | ---: |
| vocab_valid_count | 47 | 50 | +3 |
| vocab_invalid_count | 3 | 0 | -3 |
| vocab_repair_attempted_count | 0 | 3 | +3 |
| vocab_repair_success_count | 0 | 3 | +3 |
| unknown_needed_tools_count | 2 | 0 | -2 |
| unknown_verification_checks_count | 1 | 0 | -1 |
| must_verification_contained_count | 17 | 17 | 0 |
| must_tools_contained_count | 39 | 39 | 0 |
| risk_underestimated_count | 3 | 3 | 0 |
| expected_mode_match_count | 45 | 45 | 0 |

The repaired cases were 020, 032, and 037. All final predictions passed both
the in-process strict check and `scripts/validate_router_vocab.py --strict`.

## Base Holdout 30

| Metric | M13 v3 | M14 v3 + vocab repair | Delta |
| --- | ---: | ---: | ---: |
| vocab_valid_count | 28 | 30 | +2 |
| vocab_invalid_count | 2 | 0 | -2 |
| vocab_repair_attempted_count | 0 | 2 | +2 |
| vocab_repair_success_count | 0 | 2 | +2 |
| unknown_needed_tools_count | 1 | 0 | -1 |
| unknown_verification_checks_count | 1 | 0 | -1 |
| must_verification_contained_count | 8 | 8 | 0 |
| must_tools_contained_count | 24 | 24 | 0 |
| risk_underestimated_count | 8 | 8 | 0 |
| expected_mode_match_count | 28 | 28 | 0 |

The repaired cases were holdout 006 and 020. Final vocabulary validity reached
30/30 without changing measured router behavior.

## LoRA v001-small Canonical 50

| Metric | M13 v3 | M14 v3 + vocab repair | Delta |
| --- | ---: | ---: | ---: |
| vocab_valid_count | 47 | 50 | +3 |
| vocab_invalid_count | 3 | 0 | -3 |
| vocab_repair_attempted_count | 0 | 3 | +3 |
| vocab_repair_success_count | 0 | 3 | +3 |
| unknown_needed_tools_count | 1 | 0 | -1 |
| unknown_verification_checks_count | 2 | 0 | -2 |
| must_verification_contained_count | 16 | 16 | 0 |
| must_tools_contained_count | 39 | 39 | 0 |
| risk_underestimated_count | 4 | 4 | 0 |
| expected_mode_match_count | 45 | 45 | 0 |

The repaired cases were 020, 021, and 037. LoRA remains a comparison adapter,
not the selected router: it is still weaker than base on canonical
verification containment and risk calibration.

## Repair Behavior

Observed repairs included:

- tool `code_exec` removed while retaining `log_parser`
- tool `convergence_check` removed while retaining `log_parser`
- check `diff_check` replaced with `destructive_action_review`
- check `tolerance_check` replaced with `dimension_check`
- check `mesh_checker` replaced with an allowed verification identifier

Vocabulary repair guarantees whitelist membership only. It does not guarantee
the closest semantic replacement. One base repair changed tool
`code_context_check` to `text_compare`; this remained score-neutral because the
required tool was `pytest`, but it shows why downstream router metrics and
human-readable audit fields must remain visible.

The runtime order is JSON parse, schema validation, optional schema repair,
vocabulary validation, optional vocabulary repair, then schema and vocabulary
revalidation. Only final JSON is stored; internal reasoning is not retained.

## Decision

Keep Qwen3.5 4B base plus prompt-router v3 and strict vocabulary retry as the
standard local router candidate on RTX5080 16GB. The retry eliminates observed
unknown vocabulary without affecting mode, tools containment, verification
containment, or risk metrics. It is a contract-enforcement layer, not a quality
improvement for risk or verification selection.

Qwen3.5 9B remains out of scope. M14 ran no fine-tuning, LoRA dry-run, adapter
update, API access, package installation, or full 150-row training.

## Results

- `eval_results/qwen35_4b_prompt_v3_canonical_vocab_repair_eval_001.json`
- `eval_results/qwen35_4b_prompt_v3_canonical_vocab_repair_predictions_001.jsonl`
- `eval_results/qwen35_4b_prompt_v3_holdout_vocab_repair_eval_001.json`
- `eval_results/qwen35_4b_prompt_v3_holdout_vocab_repair_predictions_001.jsonl`
- `eval_results/qwen35_lora_v001_small_prompt_v3_canonical_vocab_repair_eval_001.json`
- `eval_results/qwen35_lora_v001_small_prompt_v3_canonical_vocab_repair_predictions_001.jsonl`
