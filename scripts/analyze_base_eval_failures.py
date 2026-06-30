#!/usr/bin/env python3
"""Analyze base router eval failures without loading any model."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-eval", default="eval_results/base_eval_001.json")
    parser.add_argument("--predictions", default="eval_results/base_predictions_001.jsonl")
    parser.add_argument("--output", default="docs/router/base_eval_failure_analysis.md")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be an object")
            rows.append(row)
    return rows


def ids_by(results: list[dict[str, Any]], predicate: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for result in results:
        if not result.get(predicate):
            grouped[str(result.get("category", "unknown"))].append(str(result["id"]))
    return dict(grouped)


def category_counts(results: list[dict[str, Any]], predicate: str, invert: bool = True) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in results:
        value = bool(result.get(predicate))
        if (not value) if invert else value:
            counts[str(result.get("category", "unknown"))] += 1
    return dict(sorted(counts.items()))


def markdown_list(grouped: dict[str, list[str]]) -> list[str]:
    lines: list[str] = []
    for category, ids in sorted(grouped.items()):
        lines.append(f"- {category}: {len(ids)} ({', '.join(ids[:8])}{'...' if len(ids) > 8 else ''})")
    if not lines:
        lines.append("- none")
    return lines


def example_lines(results: list[dict[str, Any]], predicate: str, limit: int = 8) -> list[str]:
    lines: list[str] = []
    for result in results:
        if result.get(predicate):
            lines.append(
                f"- `{result['id']}` {result.get('category')}: expected `{result.get('expected_mode')}`, "
                f"predicted `{result.get('predicted_mode')}`, risk `{result.get('predicted_risk')}`"
            )
        if len(lines) >= limit:
            break
    if not lines:
        lines.append("- none")
    return lines


def schema_invalid_lines(results: list[dict[str, Any]], limit: int = 8) -> list[str]:
    lines: list[str] = []
    for result in results:
        if result.get("schema_valid"):
            continue
        errors = result.get("schema_errors", [])
        first_error = errors[0] if errors else "unknown schema error"
        lines.append(f"- `{result['id']}` {result.get('category')}: {first_error}")
        if len(lines) >= limit:
            break
    if not lines:
        lines.append("- none")
    return lines


def request_more_info_overuse(results: list[dict[str, Any]], limit: int = 10) -> list[str]:
    lines: list[str] = []
    for result in results:
        if result.get("predicted_mode") == "request_more_info" and result.get("expected_mode") != "request_more_info":
            lines.append(
                f"- `{result['id']}` {result.get('category')}: expected `{result.get('expected_mode')}`, "
                f"min_risk `{result.get('min_risk')}`"
            )
        if len(lines) >= limit:
            break
    if not lines:
        lines.append("- none")
    return lines


def main() -> None:
    args = parse_args()
    report = load_json(Path(args.base_eval))
    predictions = read_jsonl(Path(args.predictions))
    results: list[dict[str, Any]] = report["results"]
    metrics = report["metrics"]

    mode_mismatch = category_counts(results, "expected_mode_match")
    tools_missing = category_counts(results, "must_tools_contained")
    verification_missing = category_counts(results, "must_verification_contained")
    risk_under = category_counts(results, "risk_underestimated", invert=False)
    schema_invalid = category_counts(results, "schema_valid")

    lines = [
        "# Base Eval Failure Analysis",
        "",
        "M6 analysis input:",
        "",
        f"- Base eval report: `{args.base_eval}`",
        f"- Base predictions: `{args.predictions}`",
        f"- Prediction rows: {len(predictions)}",
        "",
        "## Overall Metrics",
        "",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Category Failure Counts",
        "",
        "| Category | mode mismatch | tools missing | verification missing | risk underestimated | schema invalid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    categories = sorted({str(result.get("category", "unknown")) for result in results})
    for category in categories:
        lines.append(
            f"| {category} | {mode_mismatch.get(category, 0)} | {tools_missing.get(category, 0)} | "
            f"{verification_missing.get(category, 0)} | {risk_under.get(category, 0)} | {schema_invalid.get(category, 0)} |"
        )

    lines.extend(
        [
            "",
            "## Expected Mode Mismatch By Category",
            "",
            *markdown_list(ids_by(results, "expected_mode_match")),
            "",
            "## Needed Tools Missing By Category",
            "",
            *markdown_list(ids_by(results, "must_tools_contained")),
            "",
            "## Verification Missing By Category",
            "",
            *markdown_list(ids_by(results, "must_verification_contained")),
            "",
            "## Risk Underestimated Examples",
            "",
            *example_lines([r for r in results if r.get("risk_underestimated")], "risk_underestimated"),
            "",
            "## request_more_info Overuse Examples",
            "",
            *request_more_info_overuse(results),
            "",
            "## Schema Invalid Examples",
            "",
            *schema_invalid_lines(results),
            "",
            "## Fine-Tuning Priorities",
            "",
            "1. Reduce risk underestimation, especially safety-critical and high-risk engineering requests.",
            "2. Penalize unnecessary `request_more_info`; only use it when code/files/comparison targets are absent.",
            "3. Teach tool selection for `code_exec`, including `python`, `regression_check`, `mesh_checker`, `log_parser`, and `json_schema_validator`.",
            "4. Make `verification.checks` concrete and domain-specific instead of empty or generic.",
            "5. Keep `fusion_policy.enabled` aligned with managed fusion and self-fusion cases.",
            "6. Preserve strict schema conformance for nested `fusion_policy` and `final_answer_policy` fields.",
            "",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
