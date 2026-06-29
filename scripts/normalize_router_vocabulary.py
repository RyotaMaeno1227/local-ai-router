#!/usr/bin/env python3
"""Normalize SFT assistant verification labels without modifying the source."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from router_vocab import CANONICAL_LABELS

ALIASES = {
    "active set consistency": "active_set_check",
    "backup review": "backup_check",
    "citation check": "citation_check",
    "collect independent judgments": "approval_check",
    "comparison target check": "comparison_target_check",
    "current literature needed": "compare_existing_methods",
    "dimension consistency": "dimension_check",
    "enum check": "enum_check",
    "judge final recommendation": "approval_check",
    "missing attachment check": "missing_file_check",
    "missing boundary conditions": "boundary_condition_check",
    "missing boundary term check": "boundary_condition_check",
    "request missing file": "missing_file_check",
    "required key check": "schema_required_key_check",
    "sign convention check": "sign_convention_check",
    "smallest eigenvalue check": "positive_definiteness_check",
    "source documentation check": "citation_check",
    "symbol definition": "symbol_definition",
    "symmetry check": "symmetry_check",
    "tangent consistency check": "tangent_consistency_check",
    "unit test run": "run_tests",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Source SFT JSONL.")
    parser.add_argument("output", help="New normalized SFT JSONL.")
    return parser.parse_args()


def normalize_label(label: str) -> tuple[str, bool, bool]:
    if label in CANONICAL_LABELS:
        return label, False, False
    canonical = ALIASES.get(label.lower())
    if canonical:
        return canonical, canonical != label, False
    return label, False, True


def normalize_row(row: dict[str, Any], normalized: Counter[str], unknown: Counter[str]) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{row.get('id', '<unknown>')}: missing messages")

    copied = json.loads(json.dumps(row, ensure_ascii=False))
    for message in copied["messages"]:
        if message.get("role") != "assistant":
            continue
        output = json.loads(message["content"])
        verification = output.get("verification")
        checks = verification.get("checks") if isinstance(verification, dict) else None
        if not isinstance(checks, list):
            raise ValueError(f"{row.get('id', '<unknown>')}: verification.checks must be a list")
        canonical_checks: list[str] = []
        for label in checks:
            if not isinstance(label, str):
                raise ValueError(f"{row.get('id', '<unknown>')}: verification label must be a string")
            canonical, changed, unresolved = normalize_label(label)
            if changed:
                normalized[f"{label} -> {canonical}"] += 1
            if unresolved:
                unknown[label] += 1
            if canonical not in canonical_checks:
                canonical_checks.append(canonical)
        output["verification"]["checks"] = canonical_checks
        message["content"] = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
    return copied


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must differ; source files are never modified")

    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    normalized: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    output_rows = [normalize_row(row, normalized, unknown) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in output_rows),
        encoding="utf-8",
    )

    print(json.dumps({
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(rows),
        "normalized_label_counts": dict(sorted(normalized.items())),
        "normalized_label_total": sum(normalized.values()),
        "unknown_label_counts": dict(sorted(unknown.items())),
        "unknown_label_total": sum(unknown.values()),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
