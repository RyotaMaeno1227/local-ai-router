# Prompt Router V2 Eval Report

## Scope

M7b evaluates `openai/gpt-oss-20b` as an inference-only router on RTX5080. No fine-tuning, LoRA dry-run, adapter training, package reinstall, or external model API route was used.

Prompt-router v2 uses:

- `prompts/router_system_v2.md`
- `prompts/router_fewshot_v2.jsonl`
- one schema repair attempt only when JSON parsing or schema validation fails
- `--local-files-only`
- local cached MXFP4 kernel loading

## Result Files

- Smoke eval: `eval_results/prompt_router_v2_smoke_001.json`
- Full eval: `eval_results/prompt_router_v2_eval_001.json`
- Full predictions: `eval_results/prompt_router_v2_predictions_001.jsonl`
- Operational logs: `logs/prompt_router_v2_smoke_001.md`, `logs/prompt_router_v2_eval_001.md` (ignored by git)

## M5 vs M7b Metrics

| Metric | M5 base | M7b prompt-router v2 | Delta |
|---|---:|---:|---:|
| json_valid_count | 50 | 50 | 0 |
| schema_valid_count | 44 | 50 | +6 |
| expected_mode_match_count | 16 | 44 | +28 |
| must_tools_contained_count | 9 | 42 | +33 |
| must_verification_contained_count | 0 | 7 | +7 |
| risk_underestimated_count | 35 | 16 | -19 |
| fusion_policy_match_count | 46 | 48 | +2 |
| request_more_info_count | 24 | 9 | -15 |
| final_channel_found_count | 49 | 50 | +1 |
| repair_attempted_count | 0 | 1 | +1 |
| repair_success_count | 0 | 1 | +1 |

`risk_underestimated_count` is lower-is-better. `request_more_info_count` is not always lower-is-better, but M7b reduced the M5 overuse pattern and matched the eval set's expected request-more-info count of 9.

## Summary

Prompt-router v2 substantially improved schema conformance, mode routing, tool selection, and risk calibration without any weight update.

Main improvements:

- Schema valid outputs improved from 44/50 to 50/50.
- Expected mode matches improved from 16/50 to 44/50.
- Required tool coverage improved from 9/50 to 42/50.
- Risk underestimation decreased from 35/50 to 16/50.
- `request_more_info` overuse decreased from 24 outputs to 9 outputs.

Remaining weakness:

- Required verification phrase coverage is still low at 7/50. The outputs contain concrete verification checks, but exact eval phrase containment remains strict.

## Operational Notes

The first offline smoke attempt showed that Transformers' MXFP4 kernel wrapper tries to verify the kernel publisher through the Hub. `scripts/eval_router.py` now resolves the local cached kernel snapshot directly when `--local-files-only --trust-remote-code` are used, avoiding Hub access during local eval.

Full eval completed with `--max-new-tokens 256`. A shorter 192-token setting caused truncation for at least one convergence-check JSON, and 96 tokens caused schema repair overuse.
