# Router SFT v002 Design

This is a data-design proposal only. M15 does not create a new SFT JSONL, run
training, update an adapter, or train the full 150-row dataset.

## Objective

Start with a reviewed 90-row canonical candidate set, within the approved
80-100 range. The objective is behavioral calibration: conservative risk,
complete verification sets, sufficient tools, and correct clarification
boundaries. It is not knowledge injection.

## Proposed Allocation

| Area | Rows | Primary targets |
| --- | ---: | --- |
| contact analysis risk calibration | 15 | medium/high floors; active set, complementarity, sign, assumptions |
| nonlinear FEM risk and tangent/residual verification | 15 | residual, tangent, convergence; solver/code evidence |
| code review tools and safety | 15 | python, pytest, git diff, static analysis, audit/review |
| paper/novelty overclaim and citation | 15 | citation, overclaim, comparison, approval; mode contrasts |
| FEM fundamentals | 10 | symbol, dimension, boundary, matrix, limiting case |
| RAG/API/fusion routing | 10 | local RAG vs web/RAG vs managed/self fusion |
| request_more_info positive/negative | 10 | five valid clarification and five near-miss negatives |
| **Total** | **90** | |

## Example Design Rules

- Every assistant output must satisfy schema and strict canonical vocabulary.
- Put only canonical IDs in `verification.checks`; prose belongs in `reason`.
- Encode the minimum defensible risk even when the task is locally executable.
- For medium/high engineering work, include all required companion checks, not
  one representative check.
- Include the smallest sufficient tool set, with explicit `python`, testing,
  diff, retrieval, audit, or human review capabilities where evidence requires
  them.
- Do not use `request_more_info` for difficulty, risk, or uncertainty alone.
  Use it only for absent code/files/attachments or comparison targets.
- Keep chain-of-thought out of all messages.

## Contrast Pairs

At least 30 rows should be organized as controlled pairs or triplets:

- conceptual contact explanation: low/medium local answer versus quantitative
  contact result inspection: medium/high code execution,
- supplied nonlinear log versus missing log,
- supplied code patch versus code-review request without code,
- supplied manuscript text versus missing abstract/related-work target,
- local project fact versus current external novelty claim,
- expert disagreement requiring fusion versus missing evidence requiring
  clarification.

These contrasts reduce shortcut learning from domain words alone.

## Verification Bundles

Examples should repeatedly demonstrate complete bundles:

- nonlinear: `residual_definition_check`, `tangent_consistency_check`,
  `convergence_check`,
- contact: `active_set_check`, `complementarity_check`,
  `sign_convention_check`, `assumption_check`,
- code: `compile`, `run_tests`, `matrix_size_check`,
  `boundary_condition_check`,
- novelty: `citation_check`, `overclaim_check`,
  `compare_existing_methods`,
- fundamentals: `symbol_definition`, `dimension_check`, `assumption_check`,
  `limiting_case_check`.

Not every row needs every label in a bundle, but omissions must be justified by
the task rather than caused by output brevity.

## Risk Calibration

- Oversample the eleven observed underestimation patterns using new prompts,
  not paraphrases of eval cases.
- Include adjacent-risk contrasts: low versus medium conceptual engineering,
  medium versus high executable review, and high versus critical approval.
- Treat contact, nonlinear stability, boundary-condition code, destructive
  changes, and safety claims conservatively.
- Include counterexamples so harmless explanations are not inflated to high
  risk.

## Data Separation and Acceptance Gates

- Do not copy or closely paraphrase canonical 50 or holdout prompts.
- Run exact and near-duplicate checks against both eval sets before approval.
- Reserve a separate validation slice; do not select examples based on final
  holdout outputs after training begins.
- Require zero schema errors, zero vocabulary errors, no chain-of-thought, and
  no empty required verification.
- Report mode/risk/tool/check distributions before any approved training.
- Score canonical verification by exact set membership, not substring matching.

## Candidate IDs

`data/router_sft_v002_candidate_ids.txt` is intentionally not created in M15.
The majority of v002 examples must be newly authored contrast cases, so IDs
should be assigned only after prompt drafting, deduplication, and human review.

The next milestone should create and audit candidate JSONL. Training remains a
separate approval gate.
