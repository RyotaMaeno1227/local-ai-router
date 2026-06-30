# Router Documentation

This directory is the single documentation home for router, Qwen3.5, and
GPT-OSS evaluation and decision records.

## Current Standard

The selected router is **Qwen/Qwen3.5-4B base + prompt-router v3 + one schema
repair + strict vocabulary validation + one vocabulary repair**. No adapter is
loaded by the standard configuration.

LoRA v001-small and v002-small are reference artifacts only. Neither adapter
is selected for production routing.

## Primary Index

- [Standard router configuration](router_standard_config.md)
- [LoRA postmortem and router decision](qwen35_lora_postmortem_and_router_decision.md)
- [Next-phase external API and RAG design](next_phase_external_api_rag_design.md)
- [Vocabulary repair evaluation](qwen35_vocab_repair_eval_report.md)
- [SFT v002 freeze note](router_sft_v002_freeze_note.md)
- [LoRA v002-small evaluation](qwen35_lora_v002_small_eval_report.md)

## Supporting History

- [Base evaluation failure analysis](base_eval_failure_analysis.md)
- [Prompt-router v2 evaluation](prompt_router_v2_eval_report.md)
- [Router model candidates](router_model_candidates.md)
- [Qwen3.5 4B base evaluation](qwen35_4b_router_eval_report.md)
- [Canonical vocabulary](router_canonical_vocabulary.md)
- [Prompt-router v3 canonical evaluation](qwen35_prompt_v3_canonical_eval_report.md)
- [Risk and verification gap analysis](qwen35_risk_verification_gap_analysis.md)
- [SFT v002 design](router_sft_v002_design.md)
- [SFT v002 candidate audit](router_sft_v002_candidate_audit.md)
- [LoRA v002-small training](qwen35_lora_v002_small_training_report.md)

Evaluation data and generated results remain under `evals/` and
`eval_results/`; they are not moved into this documentation directory.
Adapter files under `adapters/` and operational logs under `logs/` remain
outside git management except for their existing placeholder files.

All 23 requested M18b source documents existed and were moved into this
directory. No candidate document was skipped.
