# Router SFT v002 Candidate Freeze Note

M16b freezes `data/router_sft_v002_candidate.jsonl` as the reviewed candidate
for the next training approval gate. The frozen file contains 90 rows in 35
controlled pair/triplet groups.

## Validation Status

- router schema valid: 90/90
- strict canonical vocabulary valid: 90/90
- exact duplicate prompts within the candidate: 0
- exact matches against canonical 50-case and holdout 30-case evals: 0
- exact matches against prior SFT datasets: 0
- near-duplicate candidates at the audit threshold of 0.82: 0
- request-more-info policy errors: 0
- fusion-policy consistency errors: 0
- SHA-256: `f451b8007cf9992ab4310625bd5d01078049e5bcb919d3ef6647b94873ec2332`

## Near-Duplicate Cleanup

The M16 audit identified three review candidates. Their user prompts were
rewritten without changing area, pair/triplet membership, risk, mode, tools,
verification checks, or fusion policy:

- `router_sft_v002_candidate_009`: changed from the eval-like conservative
  friction wording to adjudication of evidence sufficiency between contact
  and tribology reviewers.
- `router_sft_v002_candidate_055`: changed from publication-readiness wording
  to independent review of an unsupported numerical safety assertion.
- `router_sft_v002_candidate_072`: changed from the newly-announced-feature
  pattern to deciding whether a claimed contact-stabilization option change
  requires current vendor documentation instead of archived local notes.

The updated audit is recorded in
`docs/router_sft_v002_candidate_audit.md`. It reports no remaining near-match
candidates at the configured threshold.

## Approval Boundary

No fine-tuning, LoRA dry-run, adapter update, model load, or 150-row training
was performed in M16b. This frozen dataset is an approved candidate for the
next explicit training approval gate; freeze status does not authorize or
automatically start training.
