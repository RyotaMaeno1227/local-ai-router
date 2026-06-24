# Qwen3.5 4B Eval Failure Analysis

Input files:

- `eval_results/qwen35_4b_eval_001.json`
- `eval_results/qwen35_4b_predictions_001.jsonl`

M9 evaluated `Qwen/Qwen3.5-4B` as a base router with prompt-router v2. The
model produced schema-valid JSON for all 50 cases, so M10 dry-run should focus
on whether a LoRA training loop can run safely and whether future SFT should
target router behavior gaps rather than JSON syntax.

## Aggregate Failures

| Failure type | Count | Category breakdown |
| --- | ---: | --- |
| expected_mode mismatch | 5 | FEM基礎: 1, 接触解析: 1, 論文・新規性確認: 3 |
| must_tools missing | 13 | FEM基礎: 2, 非線形FEM: 4, 接触解析: 2, コード確認: 4, 論文・新規性確認: 1 |
| must_verification missing | 43 | FEM基礎: 6, 非線形FEM: 9, 接触解析: 9, コード確認: 10, 論文・新規性確認: 9 |
| risk underestimated | 3 | 非線形FEM: 2, 接触解析: 1 |
| request_more_info predictions | 8 | FEM基礎: 2, 非線形FEM: 1, 接触解析: 1, コード確認: 3, 論文・新規性確認: 1 |

## Mode Mismatches

- `router_eval_001_008`: expected `local_answer`, predicted `code_exec`.
- `router_eval_001_024`: expected `web_or_rag_required`, predicted `local_rag`.
- `router_eval_001_043`: expected `managed_fusion`, predicted `local_rag`.
- `router_eval_001_044`: expected `request_more_info`, predicted `external_expert`.
- `router_eval_001_045`: expected `local_answer`, predicted `local_rag`.

The main mode weakness is research and novelty routing. Qwen tends to choose a
retrieval route where the expected behavior is either managed fusion,
clarification, or local conceptual answering.

## Tool Selection Gaps

`must_tools` failures are concentrated in code and nonlinear FEM checks:

- FEM基礎: 2 misses
- 非線形FEM: 4 misses
- 接触解析: 2 misses
- コード確認: 4 misses
- 論文・新規性確認: 1 miss

Common pattern: Qwen selects plausible but narrower tools, such as `log_parser`,
`mesh_checker`, or `static_analysis`, while missing one or more required
verification tools from the eval row.

## Verification Gaps

`must_verification_contained_count` was 7/50. This is the largest measured gap.
Many predictions include reasonable checks, but they do not contain the exact
required verification terms used by the eval set. Future SFT examples should
prefer explicit canonical check names such as:

- `symbol_definition`
- `dimension_check`
- `assumption_check`
- `residual_definition_check`
- `tangent_consistency_check`
- `convergence_check`
- `active_set_check`
- `complementarity_check`
- `sign_convention_check`
- `compile`
- `run_tests`
- `matrix_size_check`
- `boundary_condition_check`
- `citation_check`
- `overclaim_check`
- `compare_existing_methods`

## Risk Underestimation

Risk underestimation occurred in 3 cases:

- `router_eval_001_014`: 非線形FEM, expected at least `medium`, predicted `low`.
- `router_eval_001_017`: 非線形FEM, expected at least `high`, predicted `medium`.
- `router_eval_001_023`: 接触解析, expected at least `medium`, predicted `low`.

This is much better than GPT-OSS prompt-router v2, but nonlinear/contact cases
still need conservative risk calibration.

## request_more_info Examples

Qwen predicted `request_more_info` in 8 cases:

- `router_eval_001_003`
- `router_eval_001_010`
- `router_eval_001_012`
- `router_eval_001_022`
- `router_eval_001_033`
- `router_eval_001_037`
- `router_eval_001_039`
- `router_eval_001_050`

Most are appropriate missing-input or approval-gate cases. One notable mismatch
is `router_eval_001_044`, where the expected mode was `request_more_info` but
Qwen escalated to `external_expert`.

## GPT-OSS Prompt-Router V2 Comparison

| Metric | GPT-OSS prompt-router v2 | Qwen3.5 4B | Qwen delta |
| --- | ---: | ---: | ---: |
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

Strengths versus GPT-OSS prompt-router v2:

- Lower risk underestimation.
- Slightly better expected mode matching.
- No schema repair was needed.

Weaknesses versus GPT-OSS prompt-router v2:

- Lower `must_tools` containment.
- Same low `must_verification` containment.
- No Harmony-style final channel marker; this is expected for Qwen and is not a
  router quality failure.

## LoRA Priority

M10 LoRA dry-run is only a training-loop feasibility check. If a later approved
training phase is run, the first behavioral targets should be:

1. canonical verification check names,
2. richer `needed_tools` selection for code/nonlinear/contact cases,
3. research novelty routing between `local_rag`, `managed_fusion`,
   `web_or_rag_required`, and `request_more_info`,
4. conservative risk calibration for nonlinear FEM and contact analysis.
