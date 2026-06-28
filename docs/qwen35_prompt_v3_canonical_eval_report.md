# Qwen3.5 Prompt v3 Canonical Eval Report

M13 evaluates base `Qwen/Qwen3.5-4B` and LoRA v001-small with prompt-router v3
against the migrated canonical 50-case eval. No weight update occurred.

## Canonical 50-case Comparison

| Metric | Qwen3.5 base v3 | LoRA v001-small v3 | LoRA delta |
| --- | ---: | ---: | ---: |
| json_valid_count | 50 | 50 | 0 |
| schema_valid_count | 50 | 50 | 0 |
| expected_mode_match_count | 45 | 45 | 0 |
| must_tools_contained_count | 39 | 39 | 0 |
| must_verification_contained_count | 17 | 16 | -1 |
| risk_underestimated_count | 3 | 4 | +1 worse |
| fusion_policy_match_count | 48 | 48 | 0 |
| request_more_info_count | 8 | 8 | 0 |

The adapter did not improve this canonical eval. It lost verification
containment on case 049 and underestimated risk on case 013 where base did not.
Base is therefore the stronger canonical 50-case result, although both still
miss most complete verification sets.

## Canonical Holdout Reference

| Metric | Qwen3.5 base v3 | LoRA v001-small v3 | LoRA delta |
| --- | ---: | ---: | ---: |
| expected_mode_match_count | 28 | 28 | 0 |
| must_tools_contained_count | 24 | 26 | +2 |
| must_verification_contained_count | 8 | 8 | 0 |
| risk_underestimated_count | 8 | 8 | 0 |
| fusion_policy_match_count | 29 | 29 | 0 |
| request_more_info_count | 6 | 6 | 0 |

LoRA improves holdout tool containment but not mode, verification, fusion, or
risk. Taken together, the current adapter has mixed and narrow effects rather
than a general router improvement.

## Vocabulary Whitelist Audit

`scripts/validate_router_vocab.py` found no structural errors. It reported the
following unknown vocabulary without modifying predictions:

| Prediction set | Unknown occurrences | Values |
| --- | ---: | --- |
| base canonical 50 | 3 | tool `code_context_check`, tool `code_exec`, check `mesh_checker` |
| LoRA canonical 50 | 3 | tool `code_exec`, check `mesh_checker`, check `tolerance_check` |
| base holdout 30 | 2 | tool `convergence_check`, check `diff_check` |
| LoRA holdout 30 | 0 | none |

These are field-placement or invented-label errors. They are not aliases to be
silently accepted. Non-strict mode records them for analysis; strict mode
correctly exits nonzero. Future inference should validate and retry on
whitelist failure before using a router decision.

## Legacy GPT-OSS Reference

GPT-OSS prompt-router v2 scored 44 mode matches, 42 tool matches, 7 legacy
verification matches, 16 risk underestimates, and 9 request-more-info outputs
on the original 50-case eval. Its verification count uses legacy phrases and
is not directly comparable to canonical v3. It remains a historical reference,
not a canonical benchmark.

## Decision

Continue with Qwen3.5 4B and prompt-router v3 on RTX5080 16GB. Do not move to
Qwen3.5 9B. Before any new training proposal, add inference-time whitelist
retry and address risk calibration. M13 performed no fine-tuning, LoRA dry-run,
adapter update, API access, or full 150-row training.

## Results

- `eval_results/qwen35_4b_prompt_v3_canonical_eval_001.json`
- `eval_results/qwen35_4b_prompt_v3_canonical_predictions_001.jsonl`
- `eval_results/qwen35_lora_v001_small_prompt_v3_canonical_eval_001.json`
- `eval_results/qwen35_lora_v001_small_prompt_v3_canonical_predictions_001.jsonl`
