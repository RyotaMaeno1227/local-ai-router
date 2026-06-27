# Qwen3.5 4B Router LoRA v001-small Evaluation

M11a trained one LoRA adapter on a fixed 50-row subset of
`data/router_sft_001.jsonl`. This is the first approved small training run, not
the prohibited 150-row full training run.

## Training

Configuration:

- base model: `Qwen/Qwen3.5-4B`
- model family: `multimodal-lm`, text-only
- local files only: enabled
- training rows: 50
- maximum sequence length: 256
- batch size: 1
- gradient accumulation: 4
- epochs: 1
- LoRA: `r=4`, `alpha=8`, targets `q_proj` and `v_proj`
- model and training dtype: bfloat16

The run completed 13 optimizer steps in 51.34 seconds. It exposed 458,752
trainable parameters, reported `train_loss=2.746`, produced no non-finite
loss, and saved the adapter to
`adapters/qwen35-4b-router-lora-v001-small`. No CUDA OOM occurred.

## Existing 50-case Eval

All models used prompt-router v2 on `evals/router_eval_001.jsonl`. The Qwen
base and LoRA runs used the same generation settings and no schema repair was
needed.

| Metric | Qwen3.5 base | Qwen3.5 LoRA v001-small | Delta | GPT-OSS prompt-router v2 |
| --- | ---: | ---: | ---: | ---: |
| json_valid_count | 50 | 50 | 0 | 50 |
| schema_valid_count | 50 | 50 | 0 | 50 |
| expected_mode_match_count | 45 | 45 | 0 | 44 |
| must_tools_contained_count | 37 | 39 | +2 | 42 |
| must_verification_contained_count | 7 | 7 | 0 | 7 |
| risk_underestimated_count | 3 | 3 | 0 | 16 |
| fusion_policy_match_count | 48 | 48 | 0 | 48 |
| request_more_info_count | 8 | 8 | 0 | 9 |

The two tool-containment improvements were `router_eval_001_015` and
`router_eval_001_039`. No measured regression occurred on this eval, but the
small run did not improve mode classification, canonical verification terms,
or risk calibration. GPT-OSS still has higher tool containment, while Qwen
retains substantially lower risk underestimation.

## Holdout Eval

The independent 30-row synthetic holdout contains five cases each for FEM
fundamentals, nonlinear FEM, contact, code review, paper/novelty review, and
RAG/API/fusion routing. Its prompts are not present in either the 150-row SFT
source or the original 50-row eval.

| Metric | LoRA v001-small holdout |
| --- | ---: |
| total | 30 |
| json_valid_count | 30 |
| schema_valid_count | 30 |
| expected_mode_match_count | 28 |
| must_tools_contained_count | 22 |
| must_verification_contained_count | 0 |
| risk_underestimated_count | 8 |
| fusion_policy_match_count | 29 |
| request_more_info_count | 6 |

The two mode mismatches were holdout cases 020 and 022. Tool misses were most
visible in code review (3), nonlinear FEM (2), and contact (2). Risk was
underestimated in 8 cases, concentrated in contact (3), nonlinear FEM (2),
and code review (2).

No Qwen base holdout run exists, so the holdout numbers measure the adapter's
absolute behavior and must not be interpreted as a base-to-LoRA improvement.
The 0/30 exact verification containment result shows that canonical check-name
generalization remains the highest-priority gap. A later training proposal
should improve data diversity and verify the scoring vocabulary before any
larger run is approved.

## Artifacts

- training subset: `data/router_sft_train_050.jsonl`
- holdout: `evals/router_eval_holdout_001.jsonl`
- existing-eval result: `eval_results/qwen35_lora_v001_small_eval_001.json`
- existing-eval predictions: `eval_results/qwen35_lora_v001_small_predictions_001.jsonl`
- holdout result: `eval_results/qwen35_lora_v001_small_holdout_eval_001.json`
- holdout predictions: `eval_results/qwen35_lora_v001_small_holdout_predictions_001.jsonl`

The adapter and operational logs remain git-ignored.
