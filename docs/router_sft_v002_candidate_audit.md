# Router SFT v002 Candidate Audit

Candidate: `data/router_sft_v002_candidate.jsonl`

## Summary

- total rows: 90
- schema valid rows: 90/90
- vocabulary strict valid rows: 90/90
- controlled pair/triplet groups: 35
- controlled pair/triplet rows: 90
- exact duplicate candidate prompts: 0 groups
- exact eval prompt matches: 0
- exact prior-SFT prompt matches: 0
- near-match candidates at threshold 0.82: 3
- request_more_info policy errors: 0
- fusion policy errors: 0

## Area Distribution

| Value | Count |
| --- | ---: |
| `code_review_tools_safety` | 15 |
| `contact_analysis_risk` | 15 |
| `fem_fundamentals` | 10 |
| `nonlinear_fem_risk_verification` | 15 |
| `paper_novelty_citation` | 15 |
| `rag_api_fusion` | 10 |
| `request_more_info_boundary` | 10 |

## Mode Distribution

| Value | Count |
| --- | ---: |
| `code_exec` | 23 |
| `external_expert` | 9 |
| `local_answer` | 23 |
| `local_rag` | 4 |
| `managed_fusion` | 6 |
| `request_more_info` | 19 |
| `self_fusion_lite` | 2 |
| `web_or_rag_required` | 4 |

## Risk Distribution

| Value | Count |
| --- | ---: |
| `critical` | 6 |
| `high` | 25 |
| `low` | 18 |
| `medium` | 41 |

## Needed Tools Distribution

| Value | Count |
| --- | ---: |
| `contact_checker` | 3 |
| `filesystem_audit` | 3 |
| `git_diff` | 5 |
| `gpu_monitor` | 1 |
| `human_review` | 33 |
| `linear_algebra_check` | 2 |
| `literature_search_placeholder` | 3 |
| `local_docs` | 4 |
| `log_parser` | 2 |
| `pytest` | 7 |
| `python` | 20 |
| `regression_check` | 3 |
| `static_analysis` | 1 |
| `symbolic_check` | 2 |
| `text_compare` | 6 |
| `vendor_docs` | 1 |

## Verification Checks Distribution

| Value | Count |
| --- | ---: |
| `active_set_check` | 3 |
| `approval_check` | 8 |
| `assumption_check` | 32 |
| `backup_check` | 3 |
| `boundary_condition_check` | 7 |
| `citation_check` | 10 |
| `code_context_check` | 5 |
| `command_line_check` | 1 |
| `compare_existing_methods` | 4 |
| `comparison_target_check` | 7 |
| `compile` | 5 |
| `complementarity_check` | 2 |
| `convergence_check` | 13 |
| `destructive_action_review` | 1 |
| `dimension_check` | 7 |
| `limiting_case_check` | 4 |
| `material_source_check` | 5 |
| `matrix_size_check` | 9 |
| `missing_file_check` | 19 |
| `overclaim_check` | 19 |
| `positive_definiteness_check` | 2 |
| `provenance_check` | 6 |
| `residual_definition_check` | 7 |
| `resource_conflict_check` | 2 |
| `run_tests` | 9 |
| `safety_approval_check` | 6 |
| `schema_required_key_check` | 2 |
| `sign_convention_check` | 5 |
| `symbol_definition` | 7 |
| `symmetry_check` | 1 |
| `tangent_consistency_check` | 4 |
| `terminology_check` | 10 |
| `warning_extraction` | 1 |

## Boundary And Fusion Counts

- empty `needed_tools`: 24 (`router_sft_v002_candidate_001`, `router_sft_v002_candidate_004`, `router_sft_v002_candidate_010`, `router_sft_v002_candidate_013`, `router_sft_v002_candidate_016`, `router_sft_v002_candidate_019`, `router_sft_v002_candidate_022`, `router_sft_v002_candidate_029`, `router_sft_v002_candidate_033`, `router_sft_v002_candidate_036`, `router_sft_v002_candidate_039`, `router_sft_v002_candidate_045`, `router_sft_v002_candidate_047`, `router_sft_v002_candidate_056`, `router_sft_v002_candidate_061`, `router_sft_v002_candidate_063`, `router_sft_v002_candidate_065`, `router_sft_v002_candidate_067`, `router_sft_v002_candidate_070`, `router_sft_v002_candidate_076`, `router_sft_v002_candidate_079`, `router_sft_v002_candidate_084`, `router_sft_v002_candidate_088`, `router_sft_v002_candidate_090`)
- empty `verification.checks`: 0 (None)
- `request_more_info`: 19 (`router_sft_v002_candidate_006`, `router_sft_v002_candidate_008`, `router_sft_v002_candidate_018`, `router_sft_v002_candidate_024`, `router_sft_v002_candidate_026`, `router_sft_v002_candidate_032`, `router_sft_v002_candidate_038`, `router_sft_v002_candidate_041`, `router_sft_v002_candidate_044`, `router_sft_v002_candidate_048`, `router_sft_v002_candidate_051`, `router_sft_v002_candidate_053`, `router_sft_v002_candidate_060`, `router_sft_v002_candidate_074`, `router_sft_v002_candidate_081`, `router_sft_v002_candidate_083`, `router_sft_v002_candidate_085`, `router_sft_v002_candidate_087`, `router_sft_v002_candidate_089`)
- `managed_fusion`: 6 (`router_sft_v002_candidate_009`, `router_sft_v002_candidate_030`, `router_sft_v002_candidate_049`, `router_sft_v002_candidate_057`, `router_sft_v002_candidate_075`, `router_sft_v002_candidate_080`)
- `self_fusion_lite`: 2 (`router_sft_v002_candidate_029`, `router_sft_v002_candidate_076`)

## Exact Duplicate Audit

- within candidate: None
- against `evals/router_eval_001_canonical.jsonl`: None
- against `evals/router_eval_holdout_001.jsonl`: None
- against `data/router_sft_001.jsonl`: None
- against `data/router_sft_002_canonical.jsonl`: None

## Near-Duplicate Candidates

These are review candidates only; similarity does not make them duplicates.

| Score | Candidate | Reference | Candidate prompt | Reference prompt |
| ---: | --- | --- | --- | --- |
| 0.844 | `router_sft_v002_candidate_009` | `evals/router_eval_holdout_001.jsonl:router_eval_holdout_001_014` | Two specialists disagree whether a frictional interface model is conservative enough for a high-consequence report. | Two specialists disagree whether a frictional seal model is conservative enough for a high-risk decision. |
| 0.838 | `router_sft_v002_candidate_072` | `evals/router_eval_holdout_001.jsonl:router_eval_holdout_001_027` | Determine whether a newly announced solver feature exists when local documentation may be outdated. | Determine whether a newly announced external solver feature exists when the local knowledge base may be outdated. |
| 0.832 | `router_sft_v002_candidate_055` | `evals/router_eval_001_canonical.jsonl:router_eval_001_049` | Review a safety-critical numerical claim in a synthetic paper for publication readiness. | Assess a safety-critical engineering claim in a synthetic paper abstract for publication readiness. |

## Validation Errors

- schema errors: 0
- vocabulary errors: 0
- request_more_info policy errors: 0
- fusion policy errors: 0

## Decision

This file is candidate training data only. No fine-tuning or adapter update was run in M16.
Near-duplicate candidates remain listed above for human review before any training approval.
