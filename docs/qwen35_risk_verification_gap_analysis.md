# Qwen3.5 Risk and Verification Gap Analysis

M15 analyzes the M14 base router results without loading a model. Inputs are
the final schema-valid and vocabulary-valid predictions for canonical 50 and
holdout 30.

## Summary

| Gap | Canonical 50 | Holdout 30 |
| --- | ---: | ---: |
| risk underestimated | 3 | 8 |
| exact verification set incomplete | 34 | 22 |
| required tools missing | 11 | 6 |
| mode mismatch | 5 | 2 |
| request_more_info false positive | 0 | 1 |
| request_more_info false negative | 1 | 0 |

The published canonical-50 metric reports 17/50 verification containment, but
exact membership in `verification.checks` is 16/50. Case 025 is counted by the
current scorer because `approval_check` is a substring of
`safety_approval_check` in the combined checks/reason text. This analysis uses
exact canonical label membership and does not treat that substring as a hit.

## Risk Underestimation

Canonical 50 cases:

| Case | Category | Minimum | Predicted | Related missing verification |
| --- | --- | --- | --- | --- |
| `router_eval_001_014` | 非線形FEM | medium | low | `assumption_check`, `convergence_check` |
| `router_eval_001_017` | 非線形FEM | high | medium | `matrix_size_check`, `positive_definiteness_check` |
| `router_eval_001_023` | 接触解析 | medium | low | `limiting_case_check` |

Holdout 30 cases:

| Case | Category | Minimum | Predicted | Related missing verification/tools |
| --- | --- | --- | --- | --- |
| `router_eval_holdout_001_007` | 非線形FEM | medium | low | `tangent_consistency_check` |
| `router_eval_holdout_001_008` | 非線形FEM | high | medium | `run_tests`; tool `git_diff` |
| `router_eval_holdout_001_011` | 接触解析 | high | medium | `sign_convention_check`; tool `python` |
| `router_eval_holdout_001_012` | 接触解析 | medium | low | `assumption_check`, `limiting_case_check` |
| `router_eval_holdout_001_013` | 接触解析 | medium | low | verification complete |
| `router_eval_holdout_001_016` | コード確認 | high | medium | `boundary_condition_check`, `run_tests` |
| `router_eval_holdout_001_018` | コード確認 | medium | low | `boundary_condition_check` |
| `router_eval_holdout_001_029` | RAG/API/Fusion判断 | medium | low | `assumption_check`, `convergence_check` |

Risk errors cluster around technical tasks that look executable or
explanatory but carry engineering consequences. The model often identifies a
plausible mode while assigning one risk level too low.

## Missing Verification Labels

Combined frequency across both evals:

| Label | Misses |
| --- | ---: |
| `assumption_check` | 15 |
| `overclaim_check` | 6 |
| `convergence_check` | 5 |
| `matrix_size_check` | 5 |
| `terminology_check` | 5 |
| `citation_check` | 4 |
| `run_tests` | 4 |
| `boundary_condition_check` | 3 |
| `sign_convention_check` | 3 |
| `approval_check` | 2 |
| `comparison_target_check` | 2 |
| `dimension_check` | 2 |
| `limiting_case_check` | 2 |
| `provenance_check` | 2 |
| `tangent_consistency_check` | 2 |

The central problem is incomplete sets, not invalid vocabulary. Predictions
usually contain one relevant check but omit companion checks needed for a
complete engineering verification path.

## Missing Tools

Combined frequency:

| Tool | Misses |
| --- | ---: |
| `python` | 10 |
| `filesystem_audit` | 2 |
| `human_review` | 2 |
| `static_analysis` | 1 |
| `symbolic_check` | 1 |
| `git_diff` | 1 |

The dominant miss is `python` on deterministic numerical inspection. Safety
and destructive workflows also need explicit audit/review tools rather than a
generic execution route.

## Domain Trends

- **Nonlinear FEM:** verification was incomplete in 9/10 canonical cases and
  3/5 holdout cases. Five risk underestimates across both sets involve missing
  convergence, tangent, assumptions, or code-test evidence.
- **Contact analysis:** canonical verification missed 8/10 and holdout risk was
  underestimated in 3/5. Training must pair active-set and complementarity
  checks with sign convention, assumptions, and risk floors.
- **Code review:** canonical tools missed 4/10; holdout verification missed 4/5
  and risk was underestimated in 2/5. `compile`, `run_tests`,
  `boundary_condition_check`, and appropriate diff/audit tools should appear
  together where required.
- **Research novelty:** risk calibration was acceptable, but canonical mode
  mismatched in 3/10 and verification missed in 5/10. The model confuses local
  retrieval, managed fusion, clarification, and local summarization. Citation,
  overclaim, comparison, and approval checks need explicit mode contrasts.
- **FEM fundamentals:** no risk underestimate occurred, but exact verification
  missed in 5/10 canonical and 4/5 holdout cases. Symbol, dimension,
  assumption, boundary, and limiting-case bundles need broader coverage.
- **RAG/API/fusion:** holdout verification missed in 5/5 and one
  self-fusion case was rated low instead of medium.

## Mode and Clarification Boundaries

Canonical mode mismatches were cases 008, 024, 043, 044, and 045. Holdout
mismatches were 020 and 022. The clarification boundary fails in both
directions:

- canonical 044 expected `request_more_info` but predicted `external_expert`,
- holdout 020 expected `external_expert` but predicted `request_more_info`.

v002 needs paired examples where the same domain differs only by whether a
required artifact/comparison target is absent. High risk alone must not trigger
clarification, and missing evidence must not be replaced by escalation.

## v002 Training Signals

The highest-value new examples are:

1. medium/high risk floors for nonlinear, contact, and code execution tasks,
2. complete canonical verification bundles rather than single plausible checks,
3. deterministic numerical tasks requiring `python` plus domain tools,
4. code changes combining diff/test/boundary or matrix verification,
5. novelty examples contrasting local answer, retrieval, fusion, and missing input,
6. paired positive and negative `request_more_info` examples.

No training or model loading occurred in M15. Detailed machine-readable output
is generated by `scripts/analyze_router_quality_gaps.py`.
