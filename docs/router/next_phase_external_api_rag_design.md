# Next Phase: External API, RAG, And Verification Design

## Scope

The next phase connects the standard local router to controlled evidence and
expert paths for:

- current paper and novelty research;
- FEM fundamentals, derivations, and implementation facts;
- general numerical analysis methods;
- hallucination suppression through source and claim verification.

M18 defines the design only. It performs no API call and stores no credentials.

## Dispatch Conditions

External dispatch is eligible only when the validated router output requests
`web_or_rag_required`, `external_expert`, or `managed_fusion`, and local
evidence cannot satisfy the verification contract. Typical triggers are a
current novelty claim, missing current vendor behavior, cross-domain expert
disagreement, or a high-risk claim requiring independent review.

Do not dispatch externally when:

- `local_answer`, `local_rag`, or local `code_exec` can satisfy the request;
- the router returns `request_more_info` because an artifact or comparison
  target is missing;
- schema or strict vocabulary validation remains invalid after one repair;
- the payload may contain confidential, proprietary, personal, export
  controlled, or otherwise restricted data;
- required human approval for a high/critical-risk route is absent.

The dispatcher must re-check route, risk, allowed provider, payload policy,
budget, and approval immediately before transmission. Router output alone is
not authorization.

## External Expert Roles

Provider names below are policy slots, not verified availability or model
specifications. Exact model IDs, terms, data handling, and pricing must be
reviewed at implementation time.

- **GPT-5.5 slot:** primary structured synthesis, tool-aware analysis, and
  consolidation of retrieved evidence.
- **Opus slot:** independent critique, ambiguity detection, and overclaim or
  contradiction review for research and high-risk prose.
- **GLM-family slot:** cost-sensitive independent perspective, multilingual
  source assistance, and diversity for managed fusion.

`managed_fusion` should use independently constructed requests, retain each
source-backed answer separately, and pass them to a judge policy. It must not
present agreement between models as evidence by itself.

## Local RAG Role

Local RAG is the first retrieval path for project facts, indexed FEM notes,
solver configuration, internal implementation documentation, approved
references, and reproducibility records. Retrieval output must include stable
document IDs, locations, version or date, and bounded excerpts.

Local RAG must not be treated as current external evidence for novelty,
recent publications, or vendor changes unless its index freshness is proven.
Conflicting or stale sources should cause escalation or an explicit uncertainty
result, not silent source selection.

## Hallucination Guard

Every evidence-backed answer should produce a claim-to-source map and run:

- `citation_check`: each material factual claim has an accessible source;
- `overclaim_check`: conclusion strength does not exceed the evidence;
- `source_quote_check`: proposed next-phase guard that confirms a short source
  excerpt supports the attributed claim.

`source_quote_check` is not currently part of the canonical router vocabulary.
It must be reviewed and added through the schema/vocabulary process before a
router is allowed to emit it. Until then it belongs to the downstream guard,
not `verification.checks`.

The guard should reject nonexistent citations, mismatched titles or IDs,
unsupported quotations, inaccessible evidence, and claims whose cited excerpt
does not entail them. Rejection returns a structured failure to the router or
requests human review; it must not invent a replacement source.

## FEM Verification Bundles

Use deterministic bundles selected by task type:

- FEM fundamentals: `symbol_definition`, `dimension_check`,
  `assumption_check`, `boundary_condition_check`, `limiting_case_check`;
- nonlinear FEM: `residual_definition_check`, `tangent_consistency_check`,
  `convergence_check`, `material_source_check`;
- contact analysis: `active_set_check`, `complementarity_check`,
  `sign_convention_check`, `convergence_check`;
- code and implementation: `compile`, `run_tests`, `matrix_size_check`,
  `boundary_condition_check`, `code_context_check`;
- research claims: `citation_check`, `compare_existing_methods`,
  `terminology_check`, `overclaim_check`, `provenance_check`.

Passing a language-model review does not replace numerical checks, tests,
mesh or convergence studies, or human approval for safety-significant work.

## Cost And Audit Logging

Each external attempt should append one structured record containing:

```json
{
  "request_id": "opaque-local-id",
  "timestamp_utc": "ISO-8601",
  "route_mode": "external_expert",
  "risk": "high",
  "provider_slot": "opus",
  "model_id": "deployment-resolved-id",
  "purpose": "independent_overclaim_review",
  "payload_hash": "sha256",
  "source_ids": ["local-doc-id"],
  "approval_id": "human-approval-reference",
  "input_tokens": 0,
  "output_tokens": 0,
  "cached_tokens": 0,
  "latency_ms": 0,
  "estimated_cost": 0.0,
  "currency": "deployment-configured",
  "retry_index": 0,
  "status": "success_or_failure",
  "error_class": null
}
```

Do not log raw prompts, credentials, internal reasoning, or full confidential
documents. Store final expert text and source excerpts separately under a
retention policy, linked by `request_id` and content hash.

## Data Assumption

The first implementation assumes **no confidential data is sent to external
APIs**. The dispatcher should default-deny external transmission unless the
payload has been classified non-confidential and provider policy, human
approval, and budget checks all pass. Local-only processing is the fallback.

## Implementation Sequence

1. define provider-neutral request, response, cost, and error schemas;
2. implement a dry local dispatcher with no network transport;
3. add local RAG retrieval with source IDs and excerpts;
4. implement citation, quote-support, overclaim, and FEM verification guards;
5. add redaction and approval gates plus append-only cost/audit logs;
6. enable one external provider only after a separate approval and test with
   non-confidential synthetic requests;
7. evaluate end-to-end grounded answer quality and failure containment before
   adding managed multi-provider fusion.
