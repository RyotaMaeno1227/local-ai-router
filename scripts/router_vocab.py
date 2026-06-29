#!/usr/bin/env python3
"""Shared canonical router vocabulary and output validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_MODES = {
    "local_answer", "local_rag", "code_exec", "external_expert", "managed_fusion",
    "self_fusion_lite", "web_or_rag_required", "request_more_info",
}
ALLOWED_RISKS = {"low", "medium", "high", "critical"}
ALLOWED_TOOLS = {
    "contact_checker", "filesystem_audit", "git_diff", "gpu_monitor", "human_review",
    "json_schema_validator", "linear_algebra_check", "literature_search_placeholder",
    "local_docs", "local_rag", "log_parser", "mesh_checker", "pytest", "python",
    "regression_check", "repro_checklist", "static_analysis", "symbolic_check",
    "text_compare", "vendor_docs",
}
CANONICAL_LABELS = {
    "active_process_check", "active_set_check", "approval_check", "assumption_check",
    "backup_check", "boundary_condition_check", "citation_check", "code_context_check",
    "command_line_check", "compare_existing_methods", "comparison_target_check", "compile",
    "complementarity_check", "convergence_check", "destructive_action_review", "dimension_check",
    "enum_check", "expected_failure_check", "limiting_case_check", "material_source_check",
    "matrix_size_check", "missing_file_check", "nested_object_check", "nonmanifold_edge_check",
    "overclaim_check", "positive_definiteness_check", "provenance_check", "residual_definition_check",
    "resource_conflict_check", "run_tests", "safety_approval_check", "schema_required_key_check",
    "sign_convention_check", "symbol_definition", "symmetry_check", "tangent_consistency_check",
    "terminology_check", "topology_check", "warning_extraction",
}
FUSION_FIELDS = {"enabled", "type", "reason", "panel_size", "judge_required"}
ALLOWED_FUSION_TYPES = {None, "expert_panel", "self_consistency", "self_fusion_lite"}


@dataclass(frozen=True)
class VocabValidation:
    errors: tuple[str, ...]
    unknown_needed_tools: tuple[str, ...]
    unknown_verification_checks: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_vocab_document(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    documented = ALLOWED_MODES | ALLOWED_RISKS | ALLOWED_TOOLS | CANONICAL_LABELS | FUSION_FIELDS
    missing = sorted(value for value in documented if f"`{value}`" not in text)
    if missing:
        raise ValueError(f"{path}: canonical vocabulary entries missing from document: {', '.join(missing)}")


def validate_router_output_vocab(output: Any) -> VocabValidation:
    errors: list[str] = []
    unknown_tools: list[str] = []
    unknown_checks: list[str] = []
    if not isinstance(output, dict):
        return VocabValidation(("router output must be an object",), (), ())

    if output.get("mode") not in ALLOWED_MODES:
        errors.append(f"unknown mode: {output.get('mode')!r}")
    if output.get("risk") not in ALLOWED_RISKS:
        errors.append(f"unknown risk: {output.get('risk')!r}")

    tools = output.get("needed_tools")
    if not isinstance(tools, list):
        errors.append("needed_tools must be a list")
    else:
        for tool in tools:
            if tool not in ALLOWED_TOOLS:
                unknown_tools.append(str(tool))
                errors.append(f"unknown needed_tools: {tool!r}")

    verification = output.get("verification")
    checks = verification.get("checks") if isinstance(verification, dict) else None
    if not isinstance(checks, list):
        errors.append("verification.checks must be a list")
    else:
        for label in checks:
            if label not in CANONICAL_LABELS:
                unknown_checks.append(str(label))
                errors.append(f"unknown verification.checks: {label!r}")

    fusion = output.get("fusion_policy")
    if not isinstance(fusion, dict):
        errors.append("fusion_policy must be an object")
    else:
        for field in sorted(set(fusion) - FUSION_FIELDS):
            errors.append(f"unknown fusion_policy field: {field!r}")
        for field in sorted(FUSION_FIELDS - set(fusion)):
            errors.append(f"missing fusion_policy field: {field}")
        if fusion.get("type") not in ALLOWED_FUSION_TYPES:
            errors.append(f"unknown fusion_policy.type: {fusion.get('type')!r}")
        if "enabled" in fusion and not isinstance(fusion["enabled"], bool):
            errors.append("fusion_policy.enabled must be boolean")
        if "judge_required" in fusion and not isinstance(fusion["judge_required"], bool):
            errors.append("fusion_policy.judge_required must be boolean")
        panel_size = fusion.get("panel_size")
        if panel_size is not None and (not isinstance(panel_size, int) or isinstance(panel_size, bool)):
            errors.append("fusion_policy.panel_size must be integer or null")
        if "reason" in fusion and not isinstance(fusion["reason"], str):
            errors.append("fusion_policy.reason must be string")

    return VocabValidation(tuple(errors), tuple(unknown_tools), tuple(unknown_checks))
