#!/usr/bin/env python3
"""Validate router eval JSONL rows for M4."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_KEYS = {
    "id",
    "category",
    "user",
    "expected_mode",
    "min_risk",
    "must_tools",
    "must_verification",
    "should_use_fusion",
    "notes",
}
ALLOWED_MODES = {
    "local_answer",
    "local_rag",
    "code_exec",
    "external_expert",
    "managed_fusion",
    "self_fusion_lite",
    "web_or_rag_required",
    "request_more_info",
}
ALLOWED_RISKS = {"low", "medium", "high", "critical"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Router eval JSONL file to validate.")
    parser.add_argument("--quiet", action="store_true", help="Only print validation errors.")
    return parser.parse_args()


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no}: invalid JSONL: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"{path}:{line_no}: row must be an object")
                continue
            rows.append(row)
    return rows, errors


def validate_string_list(path: Path, row_idx: int, key: str, value: Any) -> list[str]:
    if not isinstance(value, list):
        return [f"{path}:{row_idx}: {key} must be a list"]
    errors: list[str] = []
    for item_idx, item in enumerate(value):
        if not isinstance(item, str) or not item:
            errors.append(f"{path}:{row_idx}: {key}[{item_idx}] must be a non-empty string")
    return errors


def validate_rows(path: Path, rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids: set[str] = set()
    for row_idx, row in enumerate(rows, start=1):
        missing = sorted(REQUIRED_KEYS - set(row))
        for key in missing:
            errors.append(f"{path}:{row_idx}: missing required key {key}")

        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            errors.append(f"{path}:{row_idx}: id must be a non-empty string")
        elif row_id in ids:
            errors.append(f"{path}:{row_idx}: duplicate id {row_id}")
        else:
            ids.add(row_id)

        for key in ("category", "user", "notes"):
            value = row.get(key)
            if not isinstance(value, str) or not value:
                errors.append(f"{path}:{row_idx}: {key} must be a non-empty string")

        expected_mode = row.get("expected_mode")
        if expected_mode not in ALLOWED_MODES:
            errors.append(f"{path}:{row_idx}: expected_mode must be one of {sorted(ALLOWED_MODES)}")

        min_risk = row.get("min_risk")
        if min_risk not in ALLOWED_RISKS:
            errors.append(f"{path}:{row_idx}: min_risk must be one of {sorted(ALLOWED_RISKS)}")

        errors.extend(validate_string_list(path, row_idx, "must_tools", row.get("must_tools")))
        errors.extend(validate_string_list(path, row_idx, "must_verification", row.get("must_verification")))

        if not isinstance(row.get("should_use_fusion"), bool):
            errors.append(f"{path}:{row_idx}: should_use_fusion must be boolean")
    return errors


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    rows, errors = read_jsonl(path)
    errors.extend(validate_rows(path, rows))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)

    if not args.quiet:
        categories = Counter(str(row["category"]) for row in rows)
        print(f"OK: {len(rows)} eval row(s) valid")
        for category, count in sorted(categories.items()):
            print(f"{category}: {count}")


if __name__ == "__main__":
    main()
