#!/usr/bin/env python3
"""Analyze router risk, verification, tool, mode, and clarification gaps."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RISK_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}
GAP_NAMES = (
    "risk_underestimated",
    "verification_miss",
    "tools_miss",
    "mode_mismatch",
    "request_more_info_false_positive",
    "request_more_info_false_negative",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=4,
        action="append",
        metavar=("NAME", "EVAL_JSONL", "RESULT_JSON", "PREDICTIONS_JSONL"),
        required=True,
        help="Add one named eval/result/prediction input set.",
    )
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--representative-per-category", type=int, default=2)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prediction_checks(prediction: dict[str, Any]) -> set[str]:
    verification = prediction.get("verification")
    checks = verification.get("checks") if isinstance(verification, dict) else None
    return {str(value) for value in checks} if isinstance(checks, list) else set()


def string_set(value: Any) -> set[str]:
    return {str(item) for item in value} if isinstance(value, list) else set()


def case_record(expected: dict[str, Any], result: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    expected_checks = string_set(expected.get("must_verification"))
    predicted_checks = prediction_checks(prediction)
    expected_tools = string_set(expected.get("must_tools"))
    predicted_tools = string_set(prediction.get("needed_tools"))
    expected_mode = expected.get("expected_mode")
    predicted_mode = prediction.get("mode")
    expected_risk = str(expected.get("min_risk", "low"))
    predicted_risk = str(prediction.get("risk", ""))
    return {
        "id": expected["id"],
        "category": expected.get("category"),
        "user": expected.get("user"),
        "expected_risk": expected_risk,
        "predicted_risk": predicted_risk,
        "risk_underestimated": RISK_ORDER.get(predicted_risk, 0) < RISK_ORDER.get(expected_risk, 1),
        "expected_mode": expected_mode,
        "predicted_mode": predicted_mode,
        "mode_mismatch": expected_mode != predicted_mode,
        "missing_verification": sorted(expected_checks - predicted_checks),
        "predicted_verification": sorted(predicted_checks),
        "missing_tools": sorted(expected_tools - predicted_tools),
        "predicted_tools": sorted(predicted_tools),
        "request_more_info_false_positive": expected_mode != "request_more_info"
        and predicted_mode == "request_more_info",
        "request_more_info_false_negative": expected_mode == "request_more_info"
        and predicted_mode != "request_more_info",
        "schema_valid": bool(result.get("schema_valid")),
        "vocab_valid": result.get("vocab_valid"),
    }


def summarize_cases(cases: list[dict[str, Any]], representative_per_category: int) -> dict[str, Any]:
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    missing_verification: Counter[str] = Counter()
    missing_tools: Counter[str] = Counter()
    for case in cases:
        category = str(case.get("category"))
        flags = {
            "risk_underestimated": case["risk_underestimated"],
            "verification_miss": bool(case["missing_verification"]),
            "tools_miss": bool(case["missing_tools"]),
            "mode_mismatch": case["mode_mismatch"],
            "request_more_info_false_positive": case["request_more_info_false_positive"],
            "request_more_info_false_negative": case["request_more_info_false_negative"],
        }
        for name, active in flags.items():
            if active:
                category_counts[category][name] += 1
        missing_verification.update(case["missing_verification"])
        missing_tools.update(case["missing_tools"])

    representatives: list[dict[str, Any]] = []
    selected_per_category: Counter[str] = Counter()
    for case in cases:
        if not (case["missing_verification"] or case["missing_tools"] or case["mode_mismatch"]):
            continue
        category = str(case.get("category"))
        if selected_per_category[category] >= representative_per_category:
            continue
        representatives.append(case)
        selected_per_category[category] += 1

    return {
        "total": len(cases),
        "gap_counts": {
            "risk_underestimated": sum(case["risk_underestimated"] for case in cases),
            "verification_miss": sum(bool(case["missing_verification"]) for case in cases),
            "tools_miss": sum(bool(case["missing_tools"]) for case in cases),
            "mode_mismatch": sum(case["mode_mismatch"] for case in cases),
            "request_more_info_false_positive": sum(case["request_more_info_false_positive"] for case in cases),
            "request_more_info_false_negative": sum(case["request_more_info_false_negative"] for case in cases),
        },
        "category_breakdown": {
            category: {name: counts.get(name, 0) for name in GAP_NAMES}
            for category, counts in sorted(category_counts.items())
        },
        "missing_verification_frequency": dict(missing_verification.most_common()),
        "missing_tools_frequency": dict(missing_tools.most_common()),
        "risk_underestimated_cases": [case for case in cases if case["risk_underestimated"]],
        "mode_mismatch_cases": [case for case in cases if case["mode_mismatch"]],
        "request_more_info_boundary_cases": [
            case
            for case in cases
            if case["request_more_info_false_positive"] or case["request_more_info_false_negative"]
        ],
        "representative_cases": representatives,
    }


def analyze_run(
    name: str,
    eval_path: Path,
    result_path: Path,
    predictions_path: Path,
    representative_per_category: int,
) -> dict[str, Any]:
    expected_rows = read_jsonl(eval_path)
    expected_by_id = {row["id"]: row for row in expected_rows}
    report = json.loads(result_path.read_text(encoding="utf-8"))
    result_by_id = {row["id"]: row for row in report["results"]}
    predictions = {row["id"]: row.get("prediction") or {} for row in read_jsonl(predictions_path)}
    expected_ids = set(expected_by_id)
    if expected_ids != set(result_by_id) or expected_ids != set(predictions):
        raise ValueError(f"{name}: input IDs do not match")
    cases = [case_record(row, result_by_id[row["id"]], predictions[row["id"]]) for row in expected_rows]
    return {
        "name": name,
        "eval_file": str(eval_path),
        "result_file": str(result_path),
        "predictions_file": str(predictions_path),
        "metrics": report.get("metrics", {}),
        "analysis": summarize_cases(cases, representative_per_category),
    }


def main() -> None:
    args = parse_args()
    if args.representative_per_category <= 0:
        raise ValueError("--representative-per-category must be positive")
    runs = [
        analyze_run(name, Path(eval_file), Path(result_file), Path(predictions_file), args.representative_per_category)
        for name, eval_file, result_file, predictions_file in args.run
    ]
    output = json.dumps({"runs": runs}, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
