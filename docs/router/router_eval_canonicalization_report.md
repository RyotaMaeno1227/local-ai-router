# Router Eval Canonicalization Report

M13 migrated the legacy verification expectations in
`evals/router_eval_001.jsonl` to
`evals/router_eval_001_canonical.jsonl`. The source file was not modified.

## Summary

- rows: 50
- legacy `must_verification` occurrences: 86
- distinct legacy labels: 84
- normalized legacy occurrences: 86
- canonical occurrences after one-to-many expansion and row deduplication: 89
- distinct canonical labels: 31
- unknown or unresolved labels: 0

The output has three more occurrences because several legacy phrases contained
two independent requirements. For example, `compare options and uncertainty`
became both `assumption_check` and `overclaim_check`.

## Canonical Label Counts

| Canonical label | Count |
| --- | ---: |
| `assumption_check` | 13 |
| `convergence_check` | 11 |
| `overclaim_check` | 10 |
| `missing_file_check` | 5 |
| `citation_check` | 5 |
| `matrix_size_check` | 4 |
| `boundary_condition_check` | 3 |
| `safety_approval_check` | 3 |
| `run_tests` | 3 |
| `approval_check` | 3 |
| `positive_definiteness_check` | 2 |
| `dimension_check` | 2 |
| `material_source_check` | 2 |
| `terminology_check` | 2 |
| `warning_extraction` | 2 |
| `backup_check` | 2 |
| `code_context_check` | 2 |
| `compare_existing_methods` | 2 |
| remaining 13 canonical labels | 1 each |

## Examples

| Legacy `must_verification` | Canonical `must_verification` |
| --- | --- |
| `symmetry check`, `smallest eigenvalue check` | `symmetry_check`, `positive_definiteness_check` |
| `missing boundary conditions` | `missing_file_check`, `boundary_condition_check` |
| `residual trend extraction`, `iteration count check` | `residual_definition_check`, `convergence_check` |
| `collect independent judgments`, `judge final recommendation` | `approval_check`, `overclaim_check` |
| `destructive command review`, `backup requirement` | `destructive_action_review`, `backup_check` |
| `require cited prior work`, `avoid unsupported novelty claim` | `citation_check`, `overclaim_check` |

All mappings are explicit in `scripts/canonicalize_router_eval.py`; unresolved
labels cause a nonzero exit rather than being guessed.

## Compatibility

The original and canonical 50-case evals share prompts, mode expectations,
risk thresholds, tools, and fusion expectations, but their verification
scores are not interchangeable. Prompt-router v2 was evaluated against legacy
natural-language labels. Prompt-router v3 is evaluated against canonical
identifiers. Historical v2 verification counts must remain labeled as legacy
reference values.
