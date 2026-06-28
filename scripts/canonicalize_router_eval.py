#!/usr/bin/env python3
"""Migrate legacy eval must_verification phrases to canonical identifiers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from normalize_router_vocabulary import CANONICAL_LABELS


LEGACY_ALIASES: dict[str, tuple[str, ...]] = {
    "OOM trace extraction": ("warning_extraction",),
    "VRAM usage review": ("resource_conflict_check",),
    "acceptance criteria check": ("approval_check",),
    "active set change count": ("active_set_check",),
    "assumption traceability": ("assumption_check",),
    "avoid fabricated software specifics": ("terminology_check",),
    "avoid guessing proprietary settings": ("assumption_check",),
    "avoid unsupported novelty claim": ("overclaim_check",),
    "backup decision": ("backup_check",),
    "backup requirement": ("backup_check",),
    "certification evidence required": ("safety_approval_check",),
    "check plotted values": ("overclaim_check",),
    "check polynomial degree": ("assumption_check",),
    "cite local note section if available": ("citation_check",),
    "collect independent judgments": ("approval_check",),
    "compare options and uncertainty": ("assumption_check", "overclaim_check"),
    "compare reviewer criteria": ("approval_check",),
    "condition estimate": ("matrix_size_check",),
    "confirm no missing boundary term": ("boundary_condition_check",),
    "connectivity index check": ("matrix_size_check",),
    "current literature needed": ("compare_existing_methods",),
    "destructive command review": ("destructive_action_review",),
    "diff inspection": ("code_context_check",),
    "dimension consistency": ("dimension_check",),
    "do not assert external facts without source": ("citation_check", "overclaim_check"),
    "do not fabricate citations": ("citation_check",),
    "do not fabricate parameters": ("assumption_check",),
    "eigenvalue sign check": ("positive_definiteness_check",),
    "enum check": ("enum_check",),
    "error table validation": ("dimension_check",),
    "evidence sufficiency": ("overclaim_check",),
    "extract claims only from supplied text": ("overclaim_check",),
    "fit slope": ("convergence_check",),
    "identify targeted tests": ("run_tests",),
    "independent comparison": ("compare_existing_methods",),
    "independent review": ("approval_check",),
    "iteration count check": ("convergence_check",),
    "judge evidence gap": ("overclaim_check",),
    "judge final recommendation": ("overclaim_check",),
    "judge overlap": ("overclaim_check",),
    "load path validation needed": ("assumption_check",),
    "log-log slope fit": ("convergence_check",),
    "mention instability and convergence context": ("convergence_check", "assumption_check"),
    "mesh sensitivity needed": ("convergence_check",),
    "minimum three refinement levels": ("assumption_check",),
    "missing boundary conditions": ("missing_file_check", "boundary_condition_check"),
    "missing friction data": ("material_source_check",),
    "missing material constants": ("material_source_check",),
    "monotonicity check": ("convergence_check",),
    "node list extraction": ("matrix_size_check",),
    "non-destructive manifest": ("provenance_check",),
    "normal orientation check": ("sign_convention_check",),
    "parse CSV": ("matrix_size_check",),
    "parse package versions": ("terminology_check",),
    "proof gap checklist": ("overclaim_check",),
    "read code context if supplied": ("code_context_check",),
    "recommend mesh study": ("convergence_check",),
    "recommend verification if quantitative claim is made": ("convergence_check",),
    "request abstract": ("missing_file_check",),
    "request command line": ("command_line_check",),
    "request input size": ("missing_file_check",),
    "request missing file": ("missing_file_check",),
    "request real sources": ("missing_file_check",),
    "request related work": ("comparison_target_check",),
    "require cited prior work": ("citation_check",),
    "require source documentation": ("citation_check",),
    "required key check": ("schema_required_key_check",),
    "residual trend extraction": ("residual_definition_check",),
    "safety claim review": ("safety_approval_check",),
    "safety-critical approval": ("safety_approval_check",),
    "single-run insufficiency": ("convergence_check",),
    "smallest eigenvalue check": ("positive_definiteness_check",),
    "state dependency on pressure gradient": ("assumption_check",),
    "state numerical risk": ("assumption_check",),
    "state sensitivity": ("assumption_check",),
    "state tradeoffs and limitations": ("limiting_case_check",),
    "state update invariant check": ("tangent_consistency_check",),
    "surface tag check": ("boundary_condition_check",),
    "symmetry check": ("symmetry_check",),
    "tolerance change identification": ("assumption_check",),
    "tolerance comparison": ("convergence_check",),
    "unit test suggestion": ("run_tests",),
    "unit test target identification": ("run_tests",),
    "warning extraction": ("warning_extraction",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must differ")

    rows = read_jsonl(input_path)
    normalized: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    total_before = 0
    total_after = 0
    for row in rows:
        migrated: list[str] = []
        for label in row.get("must_verification", []):
            total_before += 1
            replacements = (label,) if label in CANONICAL_LABELS else LEGACY_ALIASES.get(label)
            if replacements is None:
                unknown[label] += 1
                replacements = (label,)
            elif replacements != (label,):
                normalized[f"{label} -> {', '.join(replacements)}"] += 1
            for replacement in replacements:
                if replacement not in migrated:
                    migrated.append(replacement)
        row["must_verification"] = migrated
        total_after += len(migrated)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(rows),
        "labels_before": total_before,
        "labels_after": total_after,
        "normalized_label_counts": dict(sorted(normalized.items())),
        "normalized_label_total": sum(normalized.values()),
        "unknown_label_counts": dict(sorted(unknown.items())),
        "unknown_label_total": sum(unknown.values()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if unknown:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
