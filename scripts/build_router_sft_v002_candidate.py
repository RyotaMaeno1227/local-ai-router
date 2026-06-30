#!/usr/bin/env python3
"""Build the reviewed 90-row canonical router SFT v002 candidate."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


OUTPUT = Path("data/router_sft_v002_candidate.jsonl")
SYSTEM = (
    "You are a local scientific AI router/verifier. Output only compact JSON with keys "
    "task_type, domain, risk, mode, needed_tools, needed_models, verification, "
    "fusion_policy, final_answer_policy. Use canonical vocabulary and no chain-of-thought."
)
EXPECTED_AREAS = {
    "contact_analysis_risk": 15,
    "nonlinear_fem_risk_verification": 15,
    "code_review_tools_safety": 15,
    "paper_novelty_citation": 15,
    "fem_fundamentals": 10,
    "rag_api_fusion": 10,
    "request_more_info_boundary": 10,
}


def fusion_policy(mode: str) -> dict[str, Any]:
    if mode == "managed_fusion":
        return {
            "enabled": True,
            "type": "expert_panel",
            "reason": "Independent experts and a judge are required.",
            "panel_size": 3,
            "judge_required": True,
        }
    if mode == "self_fusion_lite":
        return {
            "enabled": True,
            "type": "self_consistency",
            "reason": "Compare bounded local alternatives.",
            "panel_size": 2,
            "judge_required": False,
        }
    return {
        "enabled": False,
        "type": None,
        "reason": "A single route is sufficient.",
        "panel_size": None,
        "judge_required": False,
    }


def add(
    rows: list[dict[str, Any]],
    area: str,
    pair_id: str,
    user: str,
    task_type: str,
    domain: str,
    risk: str,
    mode: str,
    tools: list[str],
    checks: list[str],
) -> None:
    index = len(rows) + 1
    output = {
        "task_type": task_type,
        "domain": domain,
        "risk": risk,
        "mode": mode,
        "needed_tools": tools,
        "needed_models": ["openai/gpt-oss-20b"],
        "verification": {
            "required": True,
            "checks": checks,
            "reason": "Use the complete canonical verification path for this request.",
        },
        "fusion_policy": fusion_policy(mode),
        "final_answer_policy": {
            "format": "json_only",
            "include_uncertainty": risk != "low",
            "include_sources": mode in {"local_rag", "web_or_rag_required"},
        },
    }
    rows.append(
        {
            "id": f"router_sft_v002_candidate_{index:03d}",
            "area": area,
            "pair_id": pair_id,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": json.dumps(output, ensure_ascii=False, separators=(",", ":"))},
            ],
        }
    )


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    a = "contact_analysis_risk"
    add(rows, a, "contact_penalty", "Explain qualitatively how contact penalty stiffness affects penetration and conditioning; no result files are involved.", "contact_penalty_explanation", "contact analysis", "medium", "local_answer", [], ["assumption_check", "limiting_case_check"])
    add(rows, a, "contact_penalty", "Inspect the supplied penetration and reaction-force tables to quantify sensitivity to three penalty stiffness values.", "contact_penalty_sensitivity", "contact analysis", "high", "code_exec", ["python"], ["dimension_check", "convergence_check", "assumption_check"])
    add(rows, a, "contact_penalty", "Decide whether a penalty-contact result is acceptable for a safety-significant seal when conditioning evidence is incomplete.", "contact_safety_decision", "contact analysis", "critical", "external_expert", ["human_review"], ["assumption_check", "safety_approval_check", "overclaim_check"])
    add(rows, a, "contact_active_set", "Describe the purpose of an active set in unilateral contact for an internal theory note.", "active_set_explanation", "contact analysis", "low", "local_answer", [], ["symbol_definition", "assumption_check"])
    add(rows, a, "contact_active_set", "Analyze the provided gap, pressure, and active-set arrays for complementarity violations and inconsistent normal signs.", "contact_array_verification", "contact analysis", "high", "code_exec", ["python", "contact_checker"], ["active_set_check", "complementarity_check", "sign_convention_check"])
    add(rows, a, "contact_active_set", "Check contact activity changes from an attached solver log, but the log attachment is absent.", "missing_contact_log", "contact analysis", "medium", "request_more_info", ["human_review"], ["missing_file_check", "active_set_check"])
    add(rows, a, "contact_friction", "Compare supplied slip and traction CSV data using the documented friction coefficient.", "friction_result_check", "contact analysis", "high", "code_exec", ["python", "contact_checker"], ["complementarity_check", "sign_convention_check", "assumption_check"])
    add(rows, a, "contact_friction", "Evaluate a frictional contact result from an attached material card, but the card was not provided.", "missing_friction_card", "contact analysis", "high", "request_more_info", ["human_review"], ["missing_file_check", "material_source_check"])
    add(rows, a, "contact_friction", "A contact analyst and a tribology reviewer reach different publication recommendations after examining the same interface evidence; adjudicate its sufficiency.", "friction_expert_dispute", "contact analysis", "high", "managed_fusion", ["human_review"], ["assumption_check", "overclaim_check", "approval_check"])
    add(rows, a, "contact_sign", "Explain the usual normal-gap and pressure sign conventions without assessing a specific simulation.", "contact_sign_explanation", "contact analysis", "medium", "local_answer", [], ["symbol_definition", "sign_convention_check"])
    add(rows, a, "contact_sign", "Compare two supplied nodal contact-pressure files and flag sign or node-count inconsistencies.", "contact_file_comparison", "contact analysis", "medium", "code_exec", ["python", "text_compare"], ["matrix_size_check", "sign_convention_check"])
    add(rows, a, "contact_sign", "Approve a certification claim based on contact pressures whose normal orientation was not documented.", "contact_certification_review", "contact analysis", "critical", "external_expert", ["human_review"], ["sign_convention_check", "assumption_check", "safety_approval_check"])
    add(rows, a, "contact_mesh", "Explain why contact pressure peaks may vary with surface mesh refinement.", "contact_mesh_explanation", "contact analysis", "medium", "local_answer", [], ["assumption_check", "limiting_case_check"])
    add(rows, a, "contact_mesh", "Fit the supplied contact-pressure peak values across four mesh levels and inspect active-set stability.", "contact_mesh_study", "contact analysis", "high", "code_exec", ["python", "regression_check", "contact_checker"], ["convergence_check", "active_set_check", "assumption_check"])
    add(rows, a, "contact_mesh", "Release a load-bearing contact design from a single mesh result without a sensitivity study.", "contact_release_review", "contact analysis", "critical", "external_expert", ["human_review"], ["convergence_check", "overclaim_check", "safety_approval_check"])

    a = "nonlinear_fem_risk_verification"
    add(rows, a, "newton_log", "Explain what residual stagnation means in a Newton iteration history without diagnosing a particular run.", "newton_stagnation_explanation", "nonlinear FEM", "medium", "local_answer", [], ["residual_definition_check", "assumption_check"])
    add(rows, a, "newton_log", "Parse the supplied Newton log and determine whether residual reduction is linear, quadratic, stagnant, or divergent.", "newton_log_analysis", "nonlinear FEM", "medium", "code_exec", ["python", "log_parser"], ["residual_definition_check", "convergence_check", "warning_extraction"])
    add(rows, a, "newton_log", "Diagnose a reported nonlinear convergence failure, but no solver log or residual history is attached.", "missing_newton_log", "nonlinear FEM", "medium", "request_more_info", ["human_review"], ["missing_file_check", "residual_definition_check"])
    add(rows, a, "consistent_tangent", "Explain why a consistent material tangent affects local Newton convergence.", "tangent_explanation", "nonlinear FEM", "medium", "local_answer", [], ["tangent_consistency_check", "assumption_check"])
    add(rows, a, "consistent_tangent", "Review the supplied constitutive-update patch and execute tangent finite-difference tests.", "tangent_code_review", "nonlinear FEM", "high", "code_exec", ["git_diff", "pytest", "python"], ["compile", "run_tests", "tangent_consistency_check"])
    add(rows, a, "consistent_tangent", "Authorize a production nonlinear material model when only one unreviewed tangent test is available.", "material_model_approval", "nonlinear FEM", "critical", "external_expert", ["human_review"], ["tangent_consistency_check", "run_tests", "safety_approval_check"])
    add(rows, a, "buckling_path", "Summarize how geometric imperfections influence a nonlinear buckling path at a conceptual level.", "buckling_concept", "nonlinear FEM", "medium", "local_answer", [], ["assumption_check", "limiting_case_check"])
    add(rows, a, "buckling_path", "Compare supplied load-displacement paths for three imperfection amplitudes and identify unstable branches.", "buckling_path_analysis", "nonlinear FEM", "high", "code_exec", ["python"], ["convergence_check", "assumption_check", "overclaim_check"])
    add(rows, a, "buckling_path", "Predict a post-buckling margin, but the required imperfection and continuation run-configuration file is absent.", "missing_buckling_inputs", "nonlinear FEM", "high", "request_more_info", ["human_review"], ["missing_file_check", "assumption_check"])
    add(rows, a, "material_update", "Inspect supplied stress-update output against a reference table and check state-variable dimensions.", "material_update_check", "nonlinear FEM", "high", "code_exec", ["python", "pytest"], ["run_tests", "matrix_size_check", "tangent_consistency_check"])
    add(rows, a, "material_update", "Run a nonlinear material verification using an attached parameter sheet that is not present.", "missing_material_sheet", "nonlinear FEM", "high", "request_more_info", ["human_review"], ["missing_file_check", "material_source_check"])
    add(rows, a, "material_update", "Approve safety margins from a nonlinear material model whose parameter provenance is undocumented.", "material_safety_review", "nonlinear FEM", "critical", "external_expert", ["human_review"], ["material_source_check", "provenance_check", "safety_approval_check"])
    add(rows, a, "continuation_choice", "Evaluate two supplied continuation histories and compare convergence robustness.", "continuation_history_check", "nonlinear FEM", "high", "code_exec", ["python", "log_parser"], ["residual_definition_check", "convergence_check", "assumption_check"])
    add(rows, a, "continuation_choice", "Choose between arc-length and displacement control for a bounded synthetic example using two internal alternatives.", "continuation_option_comparison", "nonlinear FEM", "medium", "self_fusion_lite", [], ["assumption_check", "convergence_check"])
    add(rows, a, "continuation_choice", "Independent reviewers disagree on continuation settings for a high-risk snap-through assessment.", "continuation_expert_fusion", "nonlinear FEM", "high", "managed_fusion", ["human_review"], ["convergence_check", "assumption_check", "approval_check"])

    a = "code_review_tools_safety"
    add(rows, a, "boundary_patch", "Compile and test the supplied boundary-condition patch, then inspect constrained matrix rows.", "boundary_patch_review", "FEM code review", "high", "code_exec", ["git_diff", "pytest", "python"], ["compile", "run_tests", "boundary_condition_check", "matrix_size_check"])
    add(rows, a, "boundary_patch", "Review a boundary-condition implementation for an indexing defect, but no source or diff is supplied.", "missing_boundary_code", "FEM code review", "medium", "request_more_info", ["human_review"], ["missing_file_check", "code_context_check"])
    add(rows, a, "boundary_patch", "Explain why constrained rows must be handled consistently in a sparse linear system.", "boundary_code_explanation", "FEM code review", "low", "local_answer", [], ["boundary_condition_check", "matrix_size_check"])
    add(rows, a, "destructive_script", "Inspect a supplied cleanup script and dry-run its file selection before any deletion.", "cleanup_script_review", "scientific code safety", "high", "code_exec", ["git_diff", "filesystem_audit"], ["code_context_check", "backup_check", "provenance_check"])
    add(rows, a, "destructive_script", "Authorize a recursive rewrite of reference outputs without a backup or review trail.", "destructive_change_review", "scientific code safety", "high", "external_expert", ["filesystem_audit", "human_review"], ["destructive_action_review", "backup_check", "approval_check"])
    add(rows, a, "destructive_script", "Explain why a non-destructive manifest should precede bulk cleanup of simulation outputs.", "cleanup_safety_explanation", "scientific code safety", "low", "local_answer", [], ["provenance_check", "backup_check"])
    add(rows, a, "sparse_assembly", "Run tests on the supplied sparse assembly refactor and verify dimensions, symmetry, and boundary rows.", "assembly_refactor_test", "FEM code review", "high", "code_exec", ["git_diff", "pytest", "python", "linear_algebra_check"], ["compile", "run_tests", "matrix_size_check", "boundary_condition_check"])
    add(rows, a, "sparse_assembly", "Investigate a sparse assembly regression, but the changed source file is absent.", "missing_assembly_source", "FEM code review", "medium", "request_more_info", ["human_review"], ["missing_file_check", "code_context_check"])
    add(rows, a, "sparse_assembly", "Describe how duplicate element contributions are accumulated in sparse assembly.", "assembly_concept", "FEM code review", "low", "local_answer", [], ["matrix_size_check", "assumption_check"])
    add(rows, a, "regression_baseline", "Compare supplied reaction-force outputs against the approved regression baseline and run targeted tests.", "reaction_regression", "FEM code review", "high", "code_exec", ["pytest", "python", "text_compare"], ["run_tests", "boundary_condition_check", "overclaim_check"])
    add(rows, a, "regression_baseline", "Compare a new result file to the approved baseline, but the baseline file is missing.", "missing_regression_baseline", "FEM code review", "medium", "request_more_info", ["human_review"], ["missing_file_check", "comparison_target_check"])
    add(rows, a, "regression_baseline", "Approve replacement of regression references after unexplained numerical differences.", "regression_release_review", "FEM code review", "high", "external_expert", ["filesystem_audit", "human_review"], ["provenance_check", "run_tests", "approval_check"])
    add(rows, a, "gpu_config", "Review the supplied GPU solver configuration diff and run a small deterministic regression test.", "gpu_config_review", "scientific code operations", "medium", "code_exec", ["git_diff", "pytest", "gpu_monitor"], ["compile", "run_tests", "resource_conflict_check"])
    add(rows, a, "gpu_config", "Review a claimed GPU configuration regression, but no configuration or command line is supplied.", "missing_gpu_config", "scientific code operations", "medium", "request_more_info", ["human_review"], ["missing_file_check", "command_line_check"])
    add(rows, a, "gpu_config", "Explain why GPU memory measurements should accompany performance claims.", "gpu_measurement_explanation", "scientific code operations", "low", "local_answer", [], ["resource_conflict_check", "overclaim_check"])

    a = "paper_novelty_citation"
    add(rows, a, "novelty_evidence", "Check a synthetic method's novelty against current published work beyond the local archive.", "current_novelty_check", "research novelty", "medium", "web_or_rag_required", ["literature_search_placeholder"], ["citation_check", "overclaim_check", "compare_existing_methods"])
    add(rows, a, "novelty_evidence", "Summarize the explicit contribution claims in the supplied synthetic introduction without judging novelty.", "claim_summary", "research writing", "low", "local_answer", [], ["terminology_check", "overclaim_check"])
    add(rows, a, "novelty_evidence", "Assess novelty for a manuscript, but its abstract, method summary, and related-work target are absent.", "missing_novelty_material", "research novelty", "medium", "request_more_info", ["human_review"], ["missing_file_check", "comparison_target_check"])
    add(rows, a, "abstract_comparison", "Compare two supplied synthetic abstracts for overlapping claims using independent reviewer views and a judge.", "abstract_fusion_review", "research novelty", "medium", "managed_fusion", ["text_compare"], ["terminology_check", "overclaim_check", "compare_existing_methods"])
    add(rows, a, "abstract_comparison", "Extract terminology differences from two supplied abstracts without making a publication decision.", "abstract_text_comparison", "research writing", "low", "code_exec", ["text_compare"], ["terminology_check", "overclaim_check"])
    add(rows, a, "abstract_comparison", "Compare two manuscript abstracts, but only the first abstract was provided.", "missing_second_abstract", "research novelty", "medium", "request_more_info", ["human_review"], ["missing_file_check", "comparison_target_check"])
    add(rows, a, "citation_provenance", "Verify the supplied bibliography entries against the indexed local literature archive.", "local_citation_audit", "research writing", "medium", "local_rag", ["local_docs"], ["citation_check", "terminology_check"])
    add(rows, a, "citation_provenance", "Complete a citation audit, but the reference list attachment is missing.", "missing_reference_list", "research writing", "medium", "request_more_info", ["human_review"], ["missing_file_check", "citation_check"])
    add(rows, a, "citation_provenance", "Approve invented citations to strengthen a related-work narrative.", "citation_fabrication_review", "research integrity", "high", "external_expert", ["human_review"], ["citation_check", "provenance_check", "approval_check"])
    add(rows, a, "safety_claim", "Escalate an unsupported numerical safety assertion from a synthetic manuscript for independent engineering review.", "safety_claim_review", "engineering publication", "critical", "external_expert", ["human_review"], ["safety_approval_check", "overclaim_check", "assumption_check"])
    add(rows, a, "safety_claim", "Summarize a supplied safety discussion while preserving its stated limitations.", "safety_text_summary", "engineering publication", "medium", "local_answer", [], ["terminology_check", "overclaim_check", "assumption_check"])
    add(rows, a, "safety_claim", "Two domain reviewers disagree whether evidence supports a high-consequence manuscript claim.", "safety_claim_fusion", "engineering publication", "high", "managed_fusion", ["human_review"], ["overclaim_check", "approval_check", "assumption_check"])
    add(rows, a, "paper_convergence", "Fit the supplied manuscript data points and verify the plotted convergence rate.", "paper_convergence_check", "research verification", "medium", "code_exec", ["python", "regression_check"], ["convergence_check", "overclaim_check", "dimension_check"])
    add(rows, a, "paper_convergence", "Check whether a convergence claim is new relative to current external literature.", "convergence_novelty_check", "research novelty", "medium", "web_or_rag_required", ["literature_search_placeholder"], ["citation_check", "compare_existing_methods", "overclaim_check"])
    add(rows, a, "paper_convergence", "Verify a convergence figure, but neither plotted data nor an extraction table is available.", "missing_plot_data", "research verification", "medium", "request_more_info", ["human_review"], ["missing_file_check", "convergence_check"])

    a = "fem_fundamentals"
    add(rows, a, "weak_form", "Explain the test and trial functions in a scalar diffusion weak form.", "weak_form_explanation", "FEM fundamentals", "low", "local_answer", [], ["symbol_definition", "assumption_check", "boundary_condition_check"])
    add(rows, a, "weak_form", "Inspect the supplied weak-form derivation for omitted boundary terms and dimension inconsistencies.", "weak_form_review", "FEM fundamentals", "medium", "local_answer", ["symbolic_check"], ["symbol_definition", "dimension_check", "boundary_condition_check"])
    add(rows, a, "matrix_property", "Explain why essential constraints can make an elasticity stiffness matrix positive definite.", "matrix_property_explanation", "FEM fundamentals", "low", "local_answer", [], ["assumption_check", "boundary_condition_check", "positive_definiteness_check"])
    add(rows, a, "matrix_property", "Check the supplied constrained sparse matrix for dimensions, symmetry, and positive definiteness.", "matrix_property_execution", "FEM fundamentals", "medium", "code_exec", ["python", "linear_algebra_check"], ["matrix_size_check", "symmetry_check", "positive_definiteness_check"])
    add(rows, a, "heat_units", "Explain the units of density, heat capacity, conductivity, and source in a heat equation.", "heat_units_explanation", "FEM fundamentals", "low", "local_answer", [], ["symbol_definition", "dimension_check", "assumption_check"])
    add(rows, a, "heat_units", "Check a supplied table of thermal coefficients and source units for dimensional consistency.", "thermal_table_check", "FEM fundamentals", "medium", "code_exec", ["python", "symbolic_check"], ["dimension_check", "matrix_size_check", "assumption_check"])
    add(rows, a, "mesh_rate", "Explain why at least several refinement levels are needed to estimate convergence order.", "mesh_rate_explanation", "FEM fundamentals", "low", "local_answer", [], ["convergence_check", "assumption_check"])
    add(rows, a, "mesh_rate", "Estimate the rate from five supplied mesh-error pairs and check the asymptotic range.", "mesh_rate_execution", "FEM fundamentals", "medium", "code_exec", ["python", "regression_check"], ["convergence_check", "dimension_check", "assumption_check"])
    add(rows, a, "element_docs", "Retrieve the interpolation and quadrature choices for an in-house element from indexed project notes.", "element_local_lookup", "FEM implementation", "low", "local_rag", ["local_docs"], ["citation_check", "terminology_check"])
    add(rows, a, "element_docs", "Explain generally how polynomial degree determines quadrature order without using project-specific notes.", "quadrature_explanation", "FEM fundamentals", "low", "local_answer", [], ["symbol_definition", "assumption_check", "limiting_case_check"])

    a = "rag_api_fusion"
    add(rows, a, "local_vs_current", "Find the project's default nonlinear tolerance in the indexed local solver handbook.", "local_solver_lookup", "project configuration", "low", "local_rag", ["local_docs"], ["citation_check", "terminology_check"])
    add(rows, a, "local_vs_current", "Decide whether a claimed change in a vendor solver's contact stabilization option requires current vendor documentation rather than archived local notes.", "current_solver_feature", "solver documentation", "medium", "web_or_rag_required", ["vendor_docs"], ["citation_check", "terminology_check"])
    add(rows, a, "api_records", "Compare two supplied API response snapshots locally; do not call the external service.", "api_snapshot_comparison", "data integration", "medium", "code_exec", ["python", "text_compare"], ["schema_required_key_check", "comparison_target_check"])
    add(rows, a, "api_records", "Compare proprietary API records, but no record identifiers or snapshots were supplied.", "missing_api_records", "data integration", "medium", "request_more_info", ["human_review"], ["missing_file_check", "comparison_target_check"])
    add(rows, a, "fusion_scope", "Three specialists disagree on structural, material, and numerical evidence for a high-risk decision.", "multidisciplinary_fusion", "engineering review", "high", "managed_fusion", ["human_review"], ["assumption_check", "overclaim_check", "approval_check"])
    add(rows, a, "fusion_scope", "Compare two bounded preconditioner choices using internal alternatives and no external evidence.", "preconditioner_self_fusion", "scientific computing", "medium", "self_fusion_lite", [], ["assumption_check", "convergence_check"])
    add(rows, a, "archive_recency", "Answer which constitutive model is enabled using the indexed project configuration archive.", "configuration_archive_lookup", "project configuration", "low", "local_rag", ["local_docs"], ["citation_check", "schema_required_key_check"])
    add(rows, a, "archive_recency", "Assess a current research novelty claim that cannot be resolved from the local archive alone.", "external_novelty_route", "research novelty", "medium", "web_or_rag_required", ["literature_search_placeholder"], ["citation_check", "compare_existing_methods", "overclaim_check"])
    add(rows, a, "single_vs_panel", "Explain a standard residual norm definition that does not require retrieval or expert judgment.", "residual_norm_answer", "nonlinear FEM", "low", "local_answer", [], ["residual_definition_check", "symbol_definition"])
    add(rows, a, "single_vs_panel", "Resolve conflicting high-consequence recommendations from structural, contact, and materials reviewers.", "high_consequence_panel", "engineering review", "high", "managed_fusion", ["human_review"], ["assumption_check", "overclaim_check", "approval_check"])

    a = "request_more_info_boundary"
    add(rows, a, "request_code", "Review a nonlinear assembly function for an indexing bug, but no source code or file is present.", "missing_code_review", "FEM code review", "medium", "request_more_info", ["human_review"], ["missing_file_check", "code_context_check"])
    add(rows, a, "request_code", "Review the supplied nonlinear assembly function and run its focused indexing tests.", "supplied_code_review", "FEM code review", "high", "code_exec", ["pytest", "python", "static_analysis"], ["compile", "run_tests", "code_context_check"])
    add(rows, a, "request_attachment", "Extract convergence history from an attached run log, but the attachment is absent.", "missing_log_attachment", "nonlinear FEM", "medium", "request_more_info", ["human_review"], ["missing_file_check", "residual_definition_check"])
    add(rows, a, "request_attachment", "Explain what information a convergence log normally contains without analyzing a specific attachment.", "log_content_explanation", "nonlinear FEM", "low", "local_answer", [], ["residual_definition_check", "terminology_check"])
    add(rows, a, "request_comparison", "Compare two displacement result files, but only one comparison target was supplied.", "missing_result_target", "FEM verification", "medium", "request_more_info", ["human_review"], ["missing_file_check", "comparison_target_check"])
    add(rows, a, "request_comparison", "Compare the two supplied displacement tables and quantify norm differences.", "supplied_result_comparison", "FEM verification", "medium", "code_exec", ["python", "text_compare"], ["matrix_size_check", "dimension_check"])
    add(rows, a, "request_material", "Use the attached material card for a safety review, but the card is not available.", "missing_material_attachment", "engineering safety", "high", "request_more_info", ["human_review"], ["missing_file_check", "material_source_check"])
    add(rows, a, "request_material", "Explain at a general level why material parameter provenance matters for nonlinear analysis.", "material_provenance_explanation", "nonlinear FEM", "medium", "local_answer", [], ["material_source_check", "provenance_check", "assumption_check"])
    add(rows, a, "request_manuscript", "Judge overlap with prior work, but the manuscript abstract and comparison paper are missing.", "missing_manuscript_targets", "research novelty", "medium", "request_more_info", ["human_review"], ["missing_file_check", "comparison_target_check"])
    add(rows, a, "request_manuscript", "Explain the distinction between novelty and validation without assessing a particular manuscript.", "novelty_validation_explanation", "research novelty", "low", "local_answer", [], ["terminology_check", "overclaim_check"])
    return rows


def user_prompt(row: dict[str, Any]) -> str:
    return next(message["content"] for message in row["messages"] if message["role"] == "user")


def reference_prompts() -> set[str]:
    prompts: set[str] = set()
    for path in (Path("evals/router_eval_001_canonical.jsonl"), Path("evals/router_eval_holdout_001.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prompts.add(json.loads(line)["user"])
    for path in (Path("data/router_sft_001.jsonl"), Path("data/router_sft_002_canonical.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prompts.add(user_prompt(json.loads(line)))
    return prompts


def main() -> None:
    rows = build_rows()
    areas = Counter(row["area"] for row in rows)
    prompts = [user_prompt(row) for row in rows]
    paired_rows = sum(count for count in Counter(row["pair_id"] for row in rows).values() if count >= 2)
    if len(rows) != 90:
        raise ValueError(f"Expected 90 rows, got {len(rows)}")
    if dict(areas) != EXPECTED_AREAS:
        raise ValueError(f"Unexpected area distribution: {dict(areas)}")
    if len(prompts) != len(set(prompts)):
        raise ValueError("Candidate contains duplicate user prompts")
    if set(prompts).intersection(reference_prompts()):
        raise ValueError("Candidate prompt exactly duplicates an eval or prior SFT prompt")
    if paired_rows < 30:
        raise ValueError(f"Expected at least 30 controlled-pair rows, got {paired_rows}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {OUTPUT}")
    print(f"Area distribution: {dict(areas)}")
    print(f"Controlled-pair/triplet rows: {paired_rows}")


if __name__ == "__main__":
    main()
