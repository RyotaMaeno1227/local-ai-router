#!/usr/bin/env python3
"""Whitelist validation for router predictions, SFT JSONL, and eval JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from router_vocab import (
    ALLOWED_MODES,
    ALLOWED_RISKS,
    ALLOWED_TOOLS,
    CANONICAL_LABELS,
    validate_router_output_vocab,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions")
    source.add_argument("--sft-file")
    source.add_argument("--eval-file")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero for unknown vocabulary.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: row must be an object")
        rows.append(row)
    return rows


def record_unknown(counter: Counter[str], field: str, value: Any) -> None:
    counter[f"{field}: {value!r}"] += 1


def validate_router(output: Any, unknown: Counter[str], structural: list[str], context: str) -> None:
    validation = validate_router_output_vocab(output)
    for error in validation.errors:
        if error.startswith("unknown "):
            unknown[error] += 1
        else:
            structural.append(f"{context}: {error}")


def prediction_output(row: dict[str, Any]) -> Any:
    prediction = row.get("prediction")
    if isinstance(prediction, dict):
        return prediction
    final_text = row.get("final_text")
    if isinstance(final_text, str):
        try:
            return json.loads(final_text)
        except json.JSONDecodeError:
            return None
    return None


def main() -> None:
    args = parse_args()
    path = Path(args.predictions or args.sft_file or args.eval_file)
    rows = read_jsonl(path)
    unknown: Counter[str] = Counter()
    structural: list[str] = []

    if args.predictions:
        for index, row in enumerate(rows, start=1):
            validate_router(prediction_output(row), unknown, structural, f"{path}:{index}")
    elif args.sft_file:
        for index, row in enumerate(rows, start=1):
            messages = row.get("messages")
            if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
                structural.append(f"{path}:{index}: messages must be a list of objects")
                continue
            assistants = [message for message in messages if message.get("role") == "assistant"]
            if not assistants:
                structural.append(f"{path}:{index}: missing assistant message")
                continue
            for message in assistants:
                try:
                    output = json.loads(message.get("content", ""))
                except json.JSONDecodeError as exc:
                    structural.append(f"{path}:{index}: invalid assistant JSON: {exc}")
                    continue
                validate_router(output, unknown, structural, f"{path}:{index}")
    else:
        for index, row in enumerate(rows, start=1):
            mode = row.get("expected_mode")
            risk = row.get("min_risk")
            if mode not in ALLOWED_MODES:
                record_unknown(unknown, "expected_mode", mode)
            if risk not in ALLOWED_RISKS:
                record_unknown(unknown, "min_risk", risk)
            tools = row.get("must_tools")
            if not isinstance(tools, list):
                structural.append(f"{path}:{index}: must_tools must be a list")
            else:
                for tool in tools:
                    if tool not in ALLOWED_TOOLS:
                        record_unknown(unknown, "must_tools", tool)
            checks = row.get("must_verification")
            if not isinstance(checks, list):
                structural.append(f"{path}:{index}: must_verification must be a list")
            else:
                for label in checks:
                    if label not in CANONICAL_LABELS:
                        record_unknown(unknown, "must_verification", label)
            if not isinstance(row.get("should_use_fusion"), bool):
                structural.append(f"{path}:{index}: should_use_fusion must be boolean")

    print(f"File: {path}")
    print(f"Rows: {len(rows)}")
    print(f"Unknown vocabulary occurrences: {sum(unknown.values())}")
    for item, count in sorted(unknown.items()):
        print(f"  {count} x {item}")
    print(f"Structural errors: {len(structural)}")
    for error in structural:
        print(f"  {error}", file=sys.stderr)

    if structural or (args.strict and unknown):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
