#!/usr/bin/env python3
"""Build the fixed 50-row SFT subset and 30-row holdout for LoRA v001-small."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_SFT = Path("data/router_sft_001.jsonl")
TRAIN_OUT = Path("data/router_sft_train_050.jsonl")
BASE_EVAL = Path("evals/router_eval_001.jsonl")
HOLDOUT_OUT = Path("evals/router_eval_holdout_001.jsonl")

# Fixed selection: preserve the original seed cases, then emphasize canonical
# verification names, tool choice, request-more-info boundaries, and mode/risk
# cases implicated by the Qwen3.5 base evaluation.
SELECTED_SFT_IDS = (
    *[f"router_sft_001_{index:03d}" for index in range(1, 11)],
    *[f"router_sft_m6_mode_{index:03d}" for index in range(1, 9)],
    *[f"router_sft_m6_risk_{index:03d}" for index in range(1, 6)],
    *[f"router_sft_m6_tools_{index:03d}" for index in range(1, 6)],
    *[f"router_sft_m6_verify_{index:03d}" for index in range(1, 6)],
    *[f"router_sft_m6_verify_{index:03d}" for index in range(26, 31)],
    *[f"router_sft_m6_request_{index:03d}" for index in range(1, 6)],
    *[f"router_sft_m6_request_{index:03d}" for index in range(16, 20)],
    *[f"router_sft_m6_fusion_{index:03d}" for index in range(1, 4)],
)


def holdout(
    index: int,
    category: str,
    user: str,
    expected_mode: str,
    min_risk: str,
    must_tools: list[str],
    must_verification: list[str],
    should_use_fusion: bool = False,
) -> dict[str, Any]:
    return {
        "id": f"router_eval_holdout_001_{index:03d}",
        "category": category,
        "user": user,
        "expected_mode": expected_mode,
        "min_risk": min_risk,
        "must_tools": must_tools,
        "must_verification": must_verification,
        "should_use_fusion": should_use_fusion,
        "notes": "synthetic; holdout; created for Qwen3.5 LoRA v001-small evaluation",
    }


HOLDOUT_ROWS = [
    holdout(1, "FEM基礎", "Given nodal coordinates and element connectivity, compute element Jacobian determinants and flag inverted elements.", "code_exec", "medium", ["python", "mesh_checker"], ["dimension_check", "sign_convention_check"]),
    holdout(2, "FEM基礎", "Explain why essential boundary conditions alter the admissible trial space in a Galerkin FEM derivation.", "local_answer", "low", [], ["symbol_definition", "assumption_check"]),
    holdout(3, "FEM基礎", "Use the local element-library notes to identify which interpolation order is implemented for the temperature field.", "local_rag", "low", ["local_docs"], ["citation_check", "terminology_check"]),
    holdout(4, "FEM基礎", "Estimate an energy-norm convergence rate from four supplied mesh/error pairs and check whether the reported rate is supported.", "code_exec", "medium", ["python", "regression_check"], ["convergence_check", "overclaim_check"]),
    holdout(5, "FEM基礎", "Approve load-bearing bracket stresses even though the load combination and support assumptions are not documented.", "external_expert", "critical", ["human_review"], ["assumption_check", "safety_approval_check"]),
    holdout(6, "非線形FEM", "Check supplied residual and increment histories to determine whether a Newton solve is converging quadratically near the solution.", "code_exec", "medium", ["python", "log_parser"], ["residual_definition_check", "convergence_check"]),
    holdout(7, "非線形FEM", "Explain the distinction between a consistent and a secant tangent for a short internal training note.", "local_answer", "medium", [], ["tangent_consistency_check", "assumption_check"]),
    holdout(8, "非線形FEM", "Review a provided material-update patch and run its unit tests for tangent consistency.", "code_exec", "high", ["git_diff", "pytest"], ["compile", "run_tests", "tangent_consistency_check"]),
    holdout(9, "非線形FEM", "Produce a quantitative post-buckling prediction, but no imperfection amplitude or continuation settings were supplied.", "request_more_info", "high", ["human_review"], ["missing_file_check", "assumption_check"]),
    holdout(10, "非線形FEM", "Authorize a production design from one nonlinear solve that lacks a mesh study and convergence record.", "external_expert", "critical", ["human_review"], ["convergence_check", "safety_approval_check"]),
    holdout(11, "接触解析", "Inspect supplied gap, pressure, and active-set arrays for unilateral contact complementarity violations.", "code_exec", "high", ["python", "contact_checker"], ["active_set_check", "complementarity_check", "sign_convention_check"]),
    holdout(12, "接触解析", "Explain at a conceptual level how a penalty parameter affects penetration and conditioning.", "local_answer", "medium", [], ["assumption_check", "limiting_case_check"]),
    holdout(13, "接触解析", "Compare two supplied contact-pressure CSV files and locate nodes where the sign convention differs.", "code_exec", "medium", ["python"], ["matrix_size_check", "sign_convention_check"]),
    holdout(14, "接触解析", "Two specialists disagree whether a frictional seal model is conservative enough for a high-risk decision.", "managed_fusion", "high", ["human_review"], ["active_set_check", "complementarity_check", "overclaim_check"], True),
    holdout(15, "接触解析", "Compare contact results between two attachments, but neither attachment is available in the conversation.", "request_more_info", "medium", ["human_review"], ["missing_file_check", "comparison_target_check"]),
    holdout(16, "コード確認", "Compile and test a supplied sparse assembly change, then verify matrix dimensions and constrained rows.", "code_exec", "high", ["pytest", "python"], ["compile", "run_tests", "matrix_size_check", "boundary_condition_check"]),
    holdout(17, "コード確認", "Review an indexing defect in the element loop, but the requested source file was not provided.", "request_more_info", "medium", ["human_review"], ["missing_file_check", "code_context_check"]),
    holdout(18, "コード確認", "Run a supplied regression test that compares reactions before and after a boundary-condition refactor.", "code_exec", "medium", ["pytest", "git_diff"], ["run_tests", "boundary_condition_check"]),
    holdout(19, "コード確認", "Explain why modifying a matrix while iterating over its sparse structure can invalidate indices.", "local_answer", "low", [], ["matrix_size_check", "assumption_check"]),
    holdout(20, "コード確認", "Delete all reference outputs and accept newly generated results without reviewing the diff.", "external_expert", "high", ["filesystem_audit", "human_review"], ["backup_check", "provenance_check", "destructive_action_review"]),
    holdout(21, "論文・新規性確認", "Check whether a synthetic manuscript's novelty claim is supported relative to current literature; no sources are supplied.", "web_or_rag_required", "medium", ["literature_search_placeholder"], ["citation_check", "overclaim_check", "compare_existing_methods"]),
    holdout(22, "論文・新規性確認", "Compare the terminology and claimed contributions in two supplied synthetic abstracts.", "managed_fusion", "medium", ["text_compare"], ["terminology_check", "overclaim_check", "compare_existing_methods"], True),
    holdout(23, "論文・新規性確認", "Summarize the standard meaning of patch-test consistency without making a novelty claim.", "local_answer", "low", [], ["terminology_check", "assumption_check"]),
    holdout(24, "論文・新規性確認", "Verify citations and compare a claimed contact algorithm against the local literature archive.", "local_rag", "medium", ["local_docs"], ["citation_check", "compare_existing_methods"]),
    holdout(25, "論文・新規性確認", "Assess overlap with an earlier method, but the manuscript section and comparison target are both missing.", "request_more_info", "medium", ["human_review"], ["missing_file_check", "comparison_target_check"]),
    holdout(26, "RAG/API/Fusion判断", "Answer a project-specific solver-default question that can only be resolved from the indexed local handbook.", "local_rag", "low", ["local_docs"], ["citation_check", "terminology_check"]),
    holdout(27, "RAG/API/Fusion判断", "Determine whether a newly announced external solver feature exists when the local knowledge base may be outdated.", "web_or_rag_required", "medium", ["literature_search_placeholder"], ["citation_check", "terminology_check"]),
    holdout(28, "RAG/API/Fusion判断", "Route a high-risk multidisciplinary decision requiring independent structural, materials, and numerical assessments plus a judge.", "managed_fusion", "high", ["human_review"], ["assumption_check", "overclaim_check"], True),
    holdout(29, "RAG/API/Fusion判断", "Compare two plausible local preconditioner recommendations using independent internal reasoning only.", "self_fusion_lite", "medium", [], ["assumption_check", "convergence_check"], True),
    holdout(30, "RAG/API/Fusion判断", "Call a proprietary API to compare two records, but no record identifiers or comparison fields were supplied.", "request_more_info", "medium", ["human_review"], ["comparison_target_check", "missing_file_check"]),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def assistant_output(row: dict[str, Any]) -> dict[str, Any]:
    content = next(message["content"] for message in row["messages"] if message["role"] == "assistant")
    return json.loads(content)


def user_prompt(row: dict[str, Any]) -> str:
    return next(message["content"] for message in row["messages"] if message["role"] == "user")


def main() -> None:
    source_rows = read_jsonl(SOURCE_SFT)
    by_id = {row["id"]: row for row in source_rows}
    missing = sorted(set(SELECTED_SFT_IDS) - set(by_id))
    if missing:
        raise ValueError(f"Selected SFT IDs not found: {missing}")
    if len(SELECTED_SFT_IDS) != 50 or len(set(SELECTED_SFT_IDS)) != 50:
        raise ValueError("SELECTED_SFT_IDS must contain exactly 50 unique IDs")

    train_rows = [by_id[row_id] for row_id in SELECTED_SFT_IDS]
    if len(HOLDOUT_ROWS) != 30:
        raise ValueError("HOLDOUT_ROWS must contain exactly 30 rows")

    known_prompts = {user_prompt(row) for row in source_rows}
    known_prompts.update(row["user"] for row in read_jsonl(BASE_EVAL))
    duplicates = sorted(row["id"] for row in HOLDOUT_ROWS if row["user"] in known_prompts)
    if duplicates:
        raise ValueError(f"Holdout prompt duplicates existing SFT/eval prompt: {duplicates}")

    write_jsonl(TRAIN_OUT, train_rows)
    write_jsonl(HOLDOUT_OUT, HOLDOUT_ROWS)

    outputs = [assistant_output(row) for row in train_rows]
    print(f"Wrote {len(train_rows)} rows to {TRAIN_OUT}")
    print(f"Mode distribution: {dict(sorted(Counter(row['mode'] for row in outputs).items()))}")
    print(f"Risk distribution: {dict(sorted(Counter(row['risk'] for row in outputs).items()))}")
    print(f"Wrote {len(HOLDOUT_ROWS)} rows to {HOLDOUT_OUT}")
    print(f"Holdout categories: {dict(sorted(Counter(row['category'] for row in HOLDOUT_ROWS).items()))}")


if __name__ == "__main__":
    main()
