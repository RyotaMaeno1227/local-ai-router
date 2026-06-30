# Qwen3.5 4B Router Eval Report

M9 evaluated `Qwen/Qwen3.5-4B` as a base router model with no fine-tuning,
LoRA dry-run, or adapter training.

## Run Summary

- Start commit: `51f8457 add router model candidate survey`
- Model: `Qwen/Qwen3.5-4B`
- Loader path: `AutoProcessor` + `AutoModelForMultimodalLM`
- Prompt setup: `prompts/router_system_v2.md` +
  `prompts/router_fewshot_v2.jsonl`
- Repair policy: one schema repair attempt enabled, but not needed
- Model download: Qwen model download was allowed and completed during smoke
- GPT-OSS download: not run
- Fine-tuning: not run

Saved outputs:

- `eval_results/qwen35_4b_eval_smoke_001.json`
- `eval_results/qwen35_4b_eval_001.json`
- `eval_results/qwen35_4b_predictions_001.jsonl`
- `logs/qwen35_disk_check_001.md`
- `logs/qwen35_4b_smoke_infer_001.md`
- `logs/qwen35_4b_eval_smoke_001.md`
- `logs/qwen35_4b_eval_001.md`

Operational note: Qwen initially emitted internal thinking text. The eval path now
requests `enable_thinking=False` when supported and strips `<think>...</think>`
blocks before JSON extraction. Raw internal reasoning is not saved in prediction
outputs.

Prompt note: `prompts/router_system_v2.md` still uses the existing
`needed_models=["openai/gpt-oss-20b"]` convention. M9 compares router control
behavior and does not score `needed_models` identity.

## Metrics Comparison

Comparison target:

- GPT-OSS prompt-router v2: `eval_results/prompt_router_v2_eval_001.json`
- Qwen3.5 4B base router eval: `eval_results/qwen35_4b_eval_001.json`

| Metric | GPT-OSS prompt-router v2 | Qwen3.5 4B | Delta |
| --- | ---: | ---: | ---: |
| total | 50 | 50 | 0 |
| json_valid_count | 50 | 50 | 0 |
| schema_valid_count | 50 | 50 | 0 |
| expected_mode_match_count | 44 | 45 | +1 |
| must_tools_contained_count | 42 | 37 | -5 |
| must_verification_contained_count | 7 | 7 | 0 |
| risk_underestimated_count | 16 | 3 | -13 |
| fusion_policy_match_count | 48 | 48 | 0 |
| request_more_info_count | 9 | 8 | -1 |
| final_channel_found_count | 50 | 0 | -50 |
| repair_attempted_count | 1 | 0 | -1 |
| repair_success_count | 1 | 0 | -1 |

`final_channel_found_count` is not directly comparable. GPT-OSS uses
Harmony-style final channel markers, while Qwen emits ordinary decoded assistant
text. Qwen predictions are still final-only JSON after the reasoning stripping
step.

## Findings

Qwen3.5 4B is a viable base router candidate for the current 50-case eval:

- JSON validity and schema validity were both 50/50.
- Expected mode matching was slightly higher than GPT-OSS prompt-router v2.
- Risk underestimation was much lower than GPT-OSS prompt-router v2.
- Tool containment was weaker and should be the first prompt/data target if Qwen
  is promoted to a fine-tuned router candidate.
- Verification containment remains low for both models, so prompt and dataset
  work should keep emphasizing concrete verification terms.

## Next Gate

Do not start LoRA, QLoRA, or adapter training from this result alone. The next
approval gate should decide whether to:

1. tune the Qwen prompt to improve `needed_tools` containment,
2. run a Qwen fine-tuning feasibility dry-run, or
3. keep Qwen as a base router candidate and use GPT-OSS as verifier.
