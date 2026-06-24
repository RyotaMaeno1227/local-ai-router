#!/usr/bin/env python3
"""Validate router output JSON against schemas/router_output.schema.json.

Supports:
- a single JSON router-output file
- a JSONL file containing one router output per line
- an SFT JSONL file with assistant.content containing router-output JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SCHEMA = Path("schemas/router_output.schema.json")


@dataclass(frozen=True)
class Candidate:
    label: str
    value: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="JSON, JSONL, or SFT messages JSONL files to validate.")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="Router output JSON Schema path.")
    parser.add_argument(
        "--mode",
        choices=("auto", "json", "jsonl", "sft-jsonl"),
        default="auto",
        help="Input interpretation. auto treats JSON as one object and JSONL as router-output or SFT rows.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print validation errors.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc


def read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return rows


def parse_assistant_content(label: str, content: Any) -> Candidate:
    if not isinstance(content, str):
        raise ValueError(f"{label}: assistant.content must be a JSON string")
    try:
        return Candidate(label=label, value=json.loads(content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}: assistant.content is not valid JSON: {exc}") from exc


def sft_candidates(path: Path, rows: Iterable[Any]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for row_idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{row_idx}: SFT row must be an object")
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"{path}:{row_idx}: missing messages array")
        found = False
        for msg_idx, message in enumerate(messages, start=1):
            if not isinstance(message, dict):
                raise ValueError(f"{path}:{row_idx}:messages[{msg_idx}]: message must be an object")
            if message.get("role") != "assistant":
                continue
            found = True
            label = f"{path}:{row_idx}:messages[{msg_idx}].content"
            candidates.append(parse_assistant_content(label, message.get("content")))
        if not found:
            raise ValueError(f"{path}:{row_idx}: no assistant message found")
    return candidates


def candidates_for_path(path: Path, mode: str) -> list[Candidate]:
    if mode == "json":
        return [Candidate(label=str(path), value=load_json(path))]

    if mode in {"jsonl", "sft-jsonl"}:
        rows = read_jsonl(path)
        if mode == "sft-jsonl":
            return sft_candidates(path, rows)
        return [Candidate(label=f"{path}:{idx}", value=row) for idx, row in enumerate(rows, start=1)]

    if path.suffix == ".json":
        return [Candidate(label=str(path), value=load_json(path))]

    rows = read_jsonl(path)
    if rows and all(isinstance(row, dict) and isinstance(row.get("messages"), list) for row in rows):
        return sft_candidates(path, rows)
    return [Candidate(label=f"{path}:{idx}", value=row) for idx, row in enumerate(rows, start=1)]


def schema_properties(schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("schema missing object properties")
    return properties


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def accepts_type(value: Any, expected: Any) -> bool:
    expected_types = expected if isinstance(expected, list) else [expected]
    return type_name(value) in expected_types


def validate_string_array(label: str, key: str, value: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, list):
        return [f"{label}: {key} must be an array"]
    seen: set[str] = set()
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item:
            errors.append(f"{label}: {key}[{idx}] must be a non-empty string")
            continue
        if item in seen:
            errors.append(f"{label}: {key}[{idx}] duplicates {item!r}")
        seen.add(item)
    return errors


def validate_object_fields(
    label: str,
    key: str,
    value: Any,
    spec: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"{label}: {key} must be an object"]

    required = spec.get("required", [])
    properties = schema_properties(spec)
    for required_key in required:
        if required_key not in value:
            errors.append(f"{label}: {key}.{required_key} is required")

    if spec.get("additionalProperties") is False:
        extra = sorted(set(value) - set(properties))
        for extra_key in extra:
            errors.append(f"{label}: {key}.{extra_key} is not allowed")

    for child_key, child_spec in properties.items():
        if child_key not in value:
            continue
        child_value = value[child_key]
        expected_type = child_spec.get("type")
        if expected_type is not None and not accepts_type(child_value, expected_type):
            errors.append(f"{label}: {key}.{child_key} must be {expected_type}, got {type_name(child_value)}")
            continue
        if child_spec.get("type") == "array":
            errors.extend(validate_string_array(label, f"{key}.{child_key}", child_value))
        if child_spec.get("minimum") is not None and isinstance(child_value, int):
            if child_value < child_spec["minimum"]:
                errors.append(f"{label}: {key}.{child_key} must be >= {child_spec['minimum']}")
    return errors


def validate_router_output(label: str, value: Any, schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"{label}: router output must be an object"]

    required = schema.get("required", [])
    properties = schema_properties(schema)
    for key in required:
        if key not in value:
            errors.append(f"{label}: {key} is required")

    if schema.get("additionalProperties") is False:
        extra = sorted(set(value) - set(properties))
        for key in extra:
            errors.append(f"{label}: {key} is not allowed")

    for key, spec in properties.items():
        if key not in value:
            continue
        item = value[key]
        expected_type = spec.get("type")
        if expected_type is not None and not accepts_type(item, expected_type):
            errors.append(f"{label}: {key} must be {expected_type}, got {type_name(item)}")
            continue
        enum = spec.get("enum")
        if enum is not None and item not in enum:
            errors.append(f"{label}: {key} must be one of {enum}, got {item!r}")
        if spec.get("type") == "array":
            errors.extend(validate_string_array(label, key, item))
        if spec.get("type") == "object":
            errors.extend(validate_object_fields(label, key, item, spec))
        if spec.get("minLength") and isinstance(item, str) and len(item) < spec["minLength"]:
            errors.append(f"{label}: {key} must be non-empty")
    return errors


def main() -> None:
    args = parse_args()
    schema_path = Path(args.schema)
    schema = load_json(schema_path)

    total = 0
    all_errors: list[str] = []
    for raw_path in args.paths:
        path = Path(raw_path)
        try:
            candidates = candidates_for_path(path, args.mode)
        except ValueError as exc:
            all_errors.append(str(exc))
            continue
        for candidate in candidates:
            total += 1
            all_errors.extend(validate_router_output(candidate.label, candidate.value, schema))

    if all_errors:
        for error in all_errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)

    if not args.quiet:
        print(f"OK: {total} router output object(s) valid")


if __name__ == "__main__":
    main()
