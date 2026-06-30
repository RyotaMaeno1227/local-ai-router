# Router Vocabulary Repair Report

M12 repairs the prompt-level contract for `verification.checks` without any
weight update. Source SFT files remain unchanged.

## SFT Normalization

`scripts/normalize_router_vocabulary.py` parsed every assistant router JSON,
checked labels against an explicit canonical whitelist, applied known aliases,
and wrote new JSONL files.

| Input | Output | Rows | Labels normalized | Unknown labels |
| --- | --- | ---: | ---: | ---: |
| `data/router_sft_001.jsonl` | `data/router_sft_002_canonical.jsonl` | 150 | 0 | 0 |
| `data/router_sft_train_050.jsonl` | `data/router_sft_train_050_canonical.jsonl` | 50 | 0 | 0 |

Both source datasets already used the same 39 canonical verification labels.
The generated files are therefore content-normalized copies used to make the
M12 contract explicit. No unresolved SFT labels remain.

## Prompt Repair

The inconsistency was in prompt-router v2. Its system prompt requested plain
phrases and its few-shot examples demonstrated non-canonical labels.

| v2 label | v3 canonical label |
| --- | --- |
| `symmetry check` | `symmetry_check` |
| `smallest eigenvalue check` | `positive_definiteness_check` |
| `symbol definition` | `symbol_definition` |
| `missing boundary term check` | `boundary_condition_check` |
| `source documentation check` | `citation_check` |
| `current literature needed` | `compare_existing_methods` |
| `collect independent judgments` | `approval_check` |
| `judge final recommendation` | `overclaim_check` |
| `missing boundary conditions` | `boundary_condition_check` |
| `request missing file` | `missing_file_check` |

The reviewer-disagreement example was not a direct lexical conversion.
`approval_check` represents the accountable decision gate and
`overclaim_check` constrains the conservatism claim. No new ambiguous label was
invented.

`prompts/router_system_v3.md` now requires lowercase snake_case identifiers,
forbids natural-language labels, synonyms, and unknown labels, lists the
canonical set, and reserves explanatory prose for `verification.reason`.
`prompts/router_fewshot_v3.jsonl` uses canonical labels in all five examples.

## Eval Vocabulary Compatibility

The original 50-case eval contains 86 expected verification label occurrences;
all 86 are legacy natural-language phrases. The 30-case holdout contains 68
occurrences; all 68 are canonical. Therefore strict verification scores from
prompt v2 and v3 are directly comparable on the holdout, but not on the
original 50-case eval.

Prompt v3 emitted almost entirely canonical labels:

- base, original 50 cases: 91/92 labels canonical; unknown `mesh_checker` once
- base, holdout: 54/55 labels canonical; unknown `diff_check` once
- LoRA v001-small, holdout: 56/56 labels canonical

The two unknown outputs show that prompt constraints improve but do not
guarantee vocabulary membership. A future schema or post-generation validator
should enforce the whitelist. They were not silently normalized during eval.

## Training Status

No fine-tuning, LoRA dry-run, adapter update, or full 150-row training was run
in M12. The purpose was to isolate prompt and vocabulary effects before any
new weight update is considered.
