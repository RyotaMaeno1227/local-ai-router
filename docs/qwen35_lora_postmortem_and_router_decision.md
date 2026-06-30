# Qwen3.5 Router LoRA Postmortem And Decision

## Decision

The standard local router remains **Qwen/Qwen3.5-4B base with prompt-router
v3, one schema repair, strict vocabulary validation, and one vocabulary
repair**. No LoRA adapter is selected. V001-small and v002-small remain local
reference artifacts only.

## Canonical 50 Comparison

| Metric | Base M14 | LoRA v001-small M14 | LoRA v002-small M17 |
| --- | ---: | ---: | ---: |
| json_valid_count | 50 | 50 | 50 |
| schema_valid_count | 50 | 50 | 50 |
| vocab_valid_count | 50 | 50 | 49 |
| expected_mode_match_count | 45 | 45 | 45 |
| must_tools_contained_count | 39 | 39 | 39 |
| must_verification_contained_count | 17 | 16 | 18 |
| risk_underestimated_count | 3 | 4 | 4 |
| fusion_policy_match_count | 48 | 48 | 48 |
| request_more_info_count | 8 | 8 | 8 |
| vocab repair attempted/succeeded | 3/3 | 3/3 | 3/2 |

## Holdout 30 Comparison

The latest comparable v001 holdout is the M12 prompt-v3 run. It predates M14
vocabulary retry, so its vocabulary repair fields are unavailable.

| Metric | Base M14 | LoRA v001-small M12 | LoRA v002-small M17 |
| --- | ---: | ---: | ---: |
| json_valid_count | 30 | 30 | 30 |
| schema_valid_count | 30 | 30 | 30 |
| vocab_valid_count | 30 | n/a | 30 |
| expected_mode_match_count | 28 | 28 | 28 |
| must_tools_contained_count | 24 | 26 | 26 |
| must_verification_contained_count | 8 | 8 | 8 |
| risk_underestimated_count | 8 | 8 | 8 |
| fusion_policy_match_count | 29 | 29 | 29 |
| request_more_info_count | 6 | 6 | 6 |
| vocab repair attempted/succeeded | 2/2 | n/a | 0/0 |

## Training Summary

V001-small used 50 rows for one epoch and 13 optimizer steps. It exposed
458,752 trainable parameters and completed with `train_loss=2.746`.

V002-small used the frozen 90-row canonical candidate with SHA-256
`f451b8007cf9992ab4310625bd5d01078049e5bcb919d3ef6647b94873ec2332`.
It used bfloat16, sequence length 256, batch size 1, gradient accumulation 4,
one epoch, LoRA `r=4`, `alpha=8`, and `q_proj`/`v_proj`. The run completed 23
optimizer steps in 85.0 seconds with 458,752 trainable parameters and
`train_loss=2.637`. There was no CUDA OOM or non-finite loss, and the adapter
was saved successfully.

## Strict Vocabulary Failure

V002 canonical case `router_eval_001_039` remained vocabulary-invalid after
its single repair attempt. The model placed `backup_check`, a canonical
verification label, in `needed_tools`. The strict evaluator exited nonzero
after saving all 50 predictions. The run was not repeated or selectively
replaced, and the failure counts against v002 adoption.

This is a model output contract failure, not a corrupt evaluation run. It also
shows that inference repair is not guaranteed to recover every field-placement
error after weight updates.

## Why Base Remains Standard

- Base has the best canonical risk result: 3 underestimates versus 4 for both
  adapters.
- Base is the only configuration with 50/50 canonical vocabulary validity in
  the selected comparison without an adapter regression.
- Neither adapter improves canonical mode, tools, fusion, or clarification.
- V002 gains one canonical verification match, but 18 remains below the target
  of 23 and comes with worse risk and vocabulary results.
- Both adapters gain two holdout tool matches, but holdout verification remains
  8 and risk underestimation remains 8. The central quality gaps are unchanged.
- Base avoids adapter lifecycle, provenance, loading, and rollback complexity.

## Why The Adapters Are Not Selected

V001-small produced a narrow holdout tool gain but regressed canonical
verification and risk. V002-small recovered one verification match over base
but repeated the risk regression and introduced a strict vocabulary failure.
Neither demonstrated a broad, repeatable improvement across canonical and
holdout sets.

Additional small LoRA runs are paused because two iterations have not changed
the limiting factors. The remaining problems depend on evidence availability,
retrieval quality, source grounding, and deterministic verification more than
on another small weight update. Continuing now would add overfitting and
selection risk without a stronger independent eval or a new training signal.

## Next Focus

Development moves to:

1. an external API dispatcher with explicit routing, privacy, cost, and audit
   boundaries;
2. local RAG for project documents, FEM references, and implementation facts;
3. hallucination guards for citations, source support, and overclaim control;
4. deterministic FEM verification bundles for theory, nonlinear analysis,
   contact, and code execution;
5. evaluation of the complete evidence-to-answer workflow, not another adapter
   in isolation.

No additional training, model load, adapter update, or API call was performed
in M18.
