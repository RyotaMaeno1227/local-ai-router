# Base Eval Failure Analysis

M6 analysis input:

- Base eval report: `eval_results/base_eval_001.json`
- Base predictions: `eval_results/base_predictions_001.jsonl`
- Prediction rows: 50

## Overall Metrics

```json
{
  "expected_mode_match_count": 16,
  "final_channel_found_count": 49,
  "fusion_policy_match_count": 46,
  "json_valid_count": 50,
  "must_tools_contained_count": 9,
  "must_verification_contained_count": 0,
  "request_more_info_count": 24,
  "risk_underestimated_count": 35,
  "schema_valid_count": 44,
  "total": 50
}
```

## Category Failure Counts

| Category | mode mismatch | tools missing | verification missing | risk underestimated | schema invalid |
|---|---:|---:|---:|---:|---:|
| FEM基礎 | 5 | 7 | 10 | 5 | 0 |
| コード確認 | 8 | 9 | 10 | 3 | 0 |
| 接触解析 | 8 | 8 | 10 | 10 | 1 |
| 論文・新規性確認 | 6 | 9 | 10 | 8 | 3 |
| 非線形FEM | 7 | 8 | 10 | 9 | 2 |

## Expected Mode Mismatch By Category

- FEM基礎: 5 (router_eval_001_001, router_eval_001_004, router_eval_001_005, router_eval_001_006, router_eval_001_007)
- コード確認: 8 (router_eval_001_031, router_eval_001_032, router_eval_001_033, router_eval_001_034, router_eval_001_035, router_eval_001_036, router_eval_001_038, router_eval_001_040)
- 接触解析: 8 (router_eval_001_021, router_eval_001_024, router_eval_001_025, router_eval_001_026, router_eval_001_027, router_eval_001_028, router_eval_001_029, router_eval_001_030)
- 論文・新規性確認: 6 (router_eval_001_041, router_eval_001_043, router_eval_001_046, router_eval_001_047, router_eval_001_048, router_eval_001_049)
- 非線形FEM: 7 (router_eval_001_011, router_eval_001_013, router_eval_001_015, router_eval_001_016, router_eval_001_017, router_eval_001_018, router_eval_001_020)

## Needed Tools Missing By Category

- FEM基礎: 7 (router_eval_001_001, router_eval_001_003, router_eval_001_004, router_eval_001_005, router_eval_001_006, router_eval_001_007, router_eval_001_010)
- コード確認: 9 (router_eval_001_031, router_eval_001_032, router_eval_001_033, router_eval_001_034, router_eval_001_036, router_eval_001_037, router_eval_001_038, router_eval_001_039...)
- 接触解析: 8 (router_eval_001_021, router_eval_001_022, router_eval_001_024, router_eval_001_025, router_eval_001_026, router_eval_001_028, router_eval_001_029, router_eval_001_030)
- 論文・新規性確認: 9 (router_eval_001_041, router_eval_001_042, router_eval_001_043, router_eval_001_044, router_eval_001_046, router_eval_001_047, router_eval_001_048, router_eval_001_049...)
- 非線形FEM: 8 (router_eval_001_011, router_eval_001_012, router_eval_001_013, router_eval_001_015, router_eval_001_016, router_eval_001_017, router_eval_001_019, router_eval_001_020)

## Verification Missing By Category

- FEM基礎: 10 (router_eval_001_001, router_eval_001_002, router_eval_001_003, router_eval_001_004, router_eval_001_005, router_eval_001_006, router_eval_001_007, router_eval_001_008...)
- コード確認: 10 (router_eval_001_031, router_eval_001_032, router_eval_001_033, router_eval_001_034, router_eval_001_035, router_eval_001_036, router_eval_001_037, router_eval_001_038...)
- 接触解析: 10 (router_eval_001_021, router_eval_001_022, router_eval_001_023, router_eval_001_024, router_eval_001_025, router_eval_001_026, router_eval_001_027, router_eval_001_028...)
- 論文・新規性確認: 10 (router_eval_001_041, router_eval_001_042, router_eval_001_043, router_eval_001_044, router_eval_001_045, router_eval_001_046, router_eval_001_047, router_eval_001_048...)
- 非線形FEM: 10 (router_eval_001_011, router_eval_001_012, router_eval_001_013, router_eval_001_014, router_eval_001_015, router_eval_001_016, router_eval_001_017, router_eval_001_018...)

## Risk Underestimated Examples

- `router_eval_001_001` FEM基礎: expected `code_exec`, predicted `local_answer`, risk `low`
- `router_eval_001_003` FEM基礎: expected `request_more_info`, predicted `request_more_info`, risk `low`
- `router_eval_001_004` FEM基礎: expected `code_exec`, predicted `local_rag`, risk `low`
- `router_eval_001_006` FEM基礎: expected `external_expert`, predicted `request_more_info`, risk `medium`
- `router_eval_001_010` FEM基礎: expected `request_more_info`, predicted `request_more_info`, risk `low`
- `router_eval_001_011` 非線形FEM: expected `code_exec`, predicted `local_answer`, risk `low`
- `router_eval_001_012` 非線形FEM: expected `request_more_info`, predicted `request_more_info`, risk `medium`
- `router_eval_001_013` 非線形FEM: expected `code_exec`, predicted `local_answer`, risk `medium`

## request_more_info Overuse Examples

- `router_eval_001_005` FEM基礎: expected `local_answer`, min_risk `low`
- `router_eval_001_006` FEM基礎: expected `external_expert`, min_risk `critical`
- `router_eval_001_007` FEM基礎: expected `local_rag`, min_risk `low`
- `router_eval_001_016` 非線形FEM: expected `web_or_rag_required`, min_risk `medium`
- `router_eval_001_020` 非線形FEM: expected `external_expert`, min_risk `critical`
- `router_eval_001_024` 接触解析: expected `web_or_rag_required`, min_risk `medium`
- `router_eval_001_025` 接触解析: expected `external_expert`, min_risk `critical`
- `router_eval_001_026` 接触解析: expected `code_exec`, min_risk `medium`
- `router_eval_001_031` コード確認: expected `code_exec`, min_risk `medium`
- `router_eval_001_032` コード確認: expected `code_exec`, min_risk `low`

## Schema Invalid Examples

- `router_eval_001_011` 非線形FEM: router_eval_001_011: needed_tools must be array, got null
- `router_eval_001_019` 非線形FEM: router_eval_001_019: needed_tools must be array, got null
- `router_eval_001_028` 接触解析: router_eval_001_028: needed_tools must be array, got null
- `router_eval_001_044` 論文・新規性確認: router_eval_001_044: needed_tools must be array, got null
- `router_eval_001_045` 論文・新規性確認: router_eval_001_045: needed_tools must be array, got null
- `router_eval_001_047` 論文・新規性確認: router_eval_001_047: task_type is required

## Fine-Tuning Priorities

1. Reduce risk underestimation, especially safety-critical and high-risk engineering requests.
2. Penalize unnecessary `request_more_info`; only use it when code/files/comparison targets are absent.
3. Teach tool selection for `code_exec`, including `python`, `regression_check`, `mesh_checker`, `log_parser`, and `json_schema_validator`.
4. Make `verification.checks` concrete and domain-specific instead of empty or generic.
5. Keep `fusion_policy.enabled` aligned with managed fusion and self-fusion cases.
6. Preserve strict schema conformance for nested `fusion_policy` and `final_answer_policy` fields.
