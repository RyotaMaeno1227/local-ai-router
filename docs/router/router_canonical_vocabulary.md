# Router Canonical Vocabulary

This document defines the stable output vocabulary for router datasets,
prompts, evaluation, and future training. Schema-constrained values are closed
sets. Tool and verification values are strings in the current schema, but new
data should use the preferred identifiers below instead of free-form phrases.

## Modes

- `local_answer`: answer from stable local reasoning without retrieval or execution
- `local_rag`: retrieve from indexed local project material
- `code_exec`: inspect files or run deterministic code/tools
- `external_expert`: require accountable human or external specialist judgment
- `managed_fusion`: combine multiple specialist judgments with an explicit judge
- `self_fusion_lite`: compare a small number of internal alternatives without external calls
- `web_or_rag_required`: current or source-backed evidence is required
- `request_more_info`: required input or comparison target is absent

## Risks

- `low`: explanatory or reversible work with limited consequence
- `medium`: quantitative or engineering work requiring explicit verification
- `high`: consequential, destructive, nonlinear, contact, or research-claim work
- `critical`: safety, certification, or release approval requiring human authority

Risk is a minimum safety classification. Do not lower risk because a task can
be executed locally.

## needed_tools

Preferred tool identifiers currently observed in the router datasets:

- execution and analysis: `python`, `pytest`, `linear_algebra_check`, `regression_check`, `symbolic_check`
- FEM and logs: `mesh_checker`, `contact_checker`, `log_parser`
- code and schema: `git_diff`, `static_analysis`, `json_schema_validator`
- retrieval and comparison: `local_docs`, `local_rag`, `vendor_docs`, `literature_search_placeholder`, `text_compare`
- operations and safety: `filesystem_audit`, `gpu_monitor`, `repro_checklist`, `human_review`

Use the smallest sufficient set, but do not leave `needed_tools` empty when
the route requires retrieval, execution, source comparison, or approval.
These identifiers describe routing capabilities; they do not authorize an API
call or destructive action.

## verification.checks

All new verification labels must use lowercase snake_case. Do not emit natural
language variants such as `citation check`, `dimension consistency`, or
`tangent consistency check` in the `checks` array. Explanatory prose belongs in
`verification.reason`.

Preferred canonical identifiers:

- FEM fundamentals: `symbol_definition`, `dimension_check`, `assumption_check`, `limiting_case_check`
- nonlinear FEM: `residual_definition_check`, `tangent_consistency_check`, `convergence_check`
- contact: `active_set_check`, `complementarity_check`, `sign_convention_check`
- matrices and boundaries: `matrix_size_check`, `symmetry_check`, `positive_definiteness_check`, `boundary_condition_check`
- mesh: `topology_check`, `nonmanifold_edge_check`
- code: `compile`, `run_tests`, `code_context_check`, `warning_extraction`
- research: `citation_check`, `terminology_check`, `overclaim_check`, `compare_existing_methods`
- missing inputs and reproduction: `missing_file_check`, `comparison_target_check`, `material_source_check`, `command_line_check`, `expected_failure_check`
- safety and provenance: `safety_approval_check`, `backup_check`, `provenance_check`, `destructive_action_review`
- schema: `schema_required_key_check`, `enum_check`, `nested_object_check`
- operations: `active_process_check`, `resource_conflict_check`, `approval_check`

Canonical alias examples for migration:

| Non-canonical output | Canonical label |
| --- | --- |
| `citation check` | `citation_check` |
| `dimension consistency` | `dimension_check` |
| `tangent consistency check` | `tangent_consistency_check` |
| `sign convention check` | `sign_convention_check` |
| `active set consistency` | `active_set_check` |
| `missing attachment check` | `missing_file_check` |
| `comparison target check` | `comparison_target_check` |
| `backup review` | `backup_check` |

## fusion_policy

Every output contains:

- `enabled`: boolean
- `type`: string or null
- `reason`: concise decision reason
- `panel_size`: integer or null
- `judge_required`: boolean

Use `enabled=true` only for `managed_fusion` or `self_fusion_lite`.
`managed_fusion` requires a judge and is reserved for high-risk research
comparison, novelty assessment, conflicting expert review, or genuinely
multidisciplinary judgment. `self_fusion_lite` is for bounded comparison of
plausible local alternatives.

## request_more_info

Use `request_more_info` only when work cannot start because:

- code review was requested but no code or file was supplied,
- an attachment-dependent task lacks the attachment,
- a comparison task lacks a concrete comparison target or required identifier.

Do not use it merely because a task is difficult, risky, quantitative, or
would benefit from more context. Route available work and state uncertainty in
the final-answer policy instead.
