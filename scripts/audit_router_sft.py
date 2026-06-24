#!/usr/bin/env python3
"""Audit router SFT data before LoRA training.

This script does not load any model. It checks SFT JSONL structure, validates
assistant router JSON against the fixed schema, and compares SFT prompts with
eval prompts for exact and near-duplicate overlap.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_router_json import load_json, validate_router_output  # noqa: E402


DEFAULT_SCHEMA = Path("schemas/router_output.schema.json")
GENERIC_CHECK_TERMS = {
    "analysis",
    "analyze",
    "check",
    "confirm",
    "generic",
    "inspect",
    "review",
    "test",
    "tests",
    "validate",
    "validation",
    "verification",
    "verify",
}


@dataclass(frozen=True)
class SftAuditRow:
    line_no: int
    row_id: str
    user_prompt: str
    assistant_json: dict[str, Any] | None
    parse_error: str | None
    schema_errors: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sft_path", help="Router SFT JSONL file.")
    parser.add_argument("eval_path", help="Router eval JSONL file.")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="Router output schema path.")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.86,
        help="Prompt similarity ratio for near-duplicate candidates.",
    )
    parser.add_argument("--max-similar", type=int, default=20, help="Maximum near-duplicate candidates to show.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be an object")
            rows.append((line_no, row))
    return rows


def message_content(messages: Any, role: str) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content", "")
            return content if isinstance(content, str) else ""
    return ""


def assistant_content(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    assistant_messages = [
        message.get("content", "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    content = assistant_messages[-1] if assistant_messages else ""
    return content if isinstance(content, str) else ""


def focus_key(row_id: str, row: dict[str, Any]) -> str:
    explicit_focus = row.get("focus")
    if isinstance(explicit_focus, str) and explicit_focus:
        return explicit_focus
    parts = row_id.split("_")
    if len(parts) >= 5 and parts[:3] == ["router", "sft", "m6"]:
        return parts[3]
    return "seed_or_unspecified"


def normalize_check(check: str) -> list[str]:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", check.lower()).strip("_")
    return [token for token in normalized.split("_") if token]


def is_generic_only_check(check: str) -> bool:
    tokens = normalize_check(check)
    return not tokens or all(token in GENERIC_CHECK_TERMS for token in tokens)


def normalize_prompt(text: str) -> str:
    lowered = text.lower()
    return re.sub(r"\s+", " ", lowered).strip()


def audit_sft(path: Path, schema: dict[str, Any]) -> list[SftAuditRow]:
    audited: list[SftAuditRow] = []
    for line_no, row in read_jsonl(path):
        row_id = str(row.get("id", f"line_{line_no}"))
        messages = row.get("messages")
        user_prompt = message_content(messages, "user")
        content = assistant_content(messages)
        assistant_json: dict[str, Any] | None = None
        parse_error: str | None = None
        schema_errors: list[str] = []
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                assistant_json = parsed
                schema_errors = validate_router_output(f"{path}:{line_no}", parsed, schema)
            else:
                parse_error = "assistant.content JSON must be an object"
        except json.JSONDecodeError as exc:
            parse_error = f"assistant.content is not valid JSON: {exc}"
        audited.append(
            SftAuditRow(
                line_no=line_no,
                row_id=row_id,
                user_prompt=user_prompt,
                assistant_json=assistant_json,
                parse_error=parse_error,
                schema_errors=schema_errors,
            )
        )
    return audited


def eval_prompts(path: Path) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []
    for line_no, row in read_jsonl(path):
        row_id = row.get("id", f"line_{line_no}")
        user_prompt = row.get("user", "")
        if isinstance(row_id, str) and isinstance(user_prompt, str):
            prompts.append((row_id, user_prompt))
    return prompts


def id_list(rows: list[SftAuditRow], limit: int = 30) -> str:
    ids = [row.row_id for row in rows]
    if not ids:
        return "none"
    suffix = "..." if len(ids) > limit else ""
    return ", ".join(ids[:limit]) + suffix


def print_counter(title: str, counter: Counter[str]) -> None:
    print(f"\n## {title}")
    if not counter:
        print("- none")
        return
    for key, count in sorted(counter.items()):
        print(f"- {key}: {count}")


def prompt_overlap(
    sft_rows: list[SftAuditRow],
    eval_rows: list[tuple[str, str]],
    threshold: float,
    max_similar: int,
) -> tuple[list[tuple[str, str]], list[tuple[float, str, str]]]:
    exact: list[tuple[str, str]] = []
    similar: list[tuple[float, str, str]] = []
    normalized_eval = [(row_id, prompt, normalize_prompt(prompt)) for row_id, prompt in eval_rows]

    for sft_row in sft_rows:
        sft_norm = normalize_prompt(sft_row.user_prompt)
        for eval_id, _eval_prompt, eval_norm in normalized_eval:
            if not sft_norm or not eval_norm:
                continue
            if sft_norm == eval_norm:
                exact.append((sft_row.row_id, eval_id))
                continue
            ratio = SequenceMatcher(None, sft_norm, eval_norm).ratio()
            if ratio >= threshold:
                similar.append((ratio, sft_row.row_id, eval_id))

    similar.sort(reverse=True, key=lambda item: item[0])
    return exact, similar[:max_similar]


def main() -> None:
    args = parse_args()
    sft_path = Path(args.sft_path)
    eval_path = Path(args.eval_path)
    schema = load_json(Path(args.schema))

    sft_rows = audit_sft(sft_path, schema)
    raw_sft_rows = {line_no: row for line_no, row in read_jsonl(sft_path)}
    eval_rows = eval_prompts(eval_path)

    modes: Counter[str] = Counter()
    risks: Counter[str] = Counter()
    focus: Counter[str] = Counter()
    verification_empty: list[SftAuditRow] = []
    verification_generic_only: list[SftAuditRow] = []
    needed_tools_empty: list[SftAuditRow] = []

    for row in sft_rows:
        raw_row = raw_sft_rows.get(row.line_no, {})
        focus[focus_key(row.row_id, raw_row)] += 1
        if row.assistant_json is None:
            continue
        mode = row.assistant_json.get("mode")
        risk = row.assistant_json.get("risk")
        if isinstance(mode, str):
            modes[mode] += 1
        if isinstance(risk, str):
            risks[risk] += 1

        needed_tools = row.assistant_json.get("needed_tools")
        if needed_tools == []:
            needed_tools_empty.append(row)

        verification = row.assistant_json.get("verification")
        checks = verification.get("checks") if isinstance(verification, dict) else None
        if not isinstance(checks, list) or len(checks) == 0:
            verification_empty.append(row)
        elif all(isinstance(check, str) and is_generic_only_check(check) for check in checks):
            verification_generic_only.append(row)

    parse_error_rows = [row for row in sft_rows if row.parse_error]
    schema_invalid_rows = [row for row in sft_rows if row.schema_errors]
    schema_valid_count = len(sft_rows) - len(parse_error_rows) - len(schema_invalid_rows)
    request_more_info_count = modes.get("request_more_info", 0)
    local_rag_count = modes.get("local_rag", 0)
    exact, similar = prompt_overlap(sft_rows, eval_rows, args.similarity_threshold, args.max_similar)

    print("# Router SFT Audit")
    print(f"\nSFT file: `{sft_path}`")
    print(f"Eval file: `{eval_path}`")
    print(f"Similarity threshold: {args.similarity_threshold:.2f}")
    print("\n## Summary")
    print(f"- total_rows: {len(sft_rows)}")
    print(f"- schema_valid_count: {schema_valid_count}")
    print(f"- schema_invalid_count: {len(schema_invalid_rows)}")
    print(f"- parse_error_count: {len(parse_error_rows)}")
    print(f"- request_more_info_count: {request_more_info_count}")
    print(f"- local_rag_count: {local_rag_count}")
    print(f"- needed_tools_empty_count: {len(needed_tools_empty)}")
    print(f"- verification_empty_count: {len(verification_empty)}")
    print(f"- verification_generic_only_count: {len(verification_generic_only)}")

    print_counter("Mode Distribution", modes)
    print_counter("Risk Distribution", risks)
    print_counter("Focus Distribution", focus)

    print("\n## Audit Flags")
    print(f"- verification_empty_rows: {id_list(verification_empty)}")
    print(f"- verification_generic_only_rows: {id_list(verification_generic_only)}")
    print(f"- needed_tools_empty_rows: {id_list(needed_tools_empty)}")

    if schema_invalid_rows:
        print("\n## Schema Errors")
        for row in schema_invalid_rows[:20]:
            first_error = row.schema_errors[0] if row.schema_errors else row.parse_error
            print(f"- {row.row_id}: {first_error}")

    print("\n## Eval/SFT Prompt Overlap")
    print(f"- exact_match_count: {len(exact)}")
    for sft_id, eval_id in exact[:20]:
        print(f"  - {sft_id} == {eval_id}")
    print(f"- similar_candidate_count: {len(similar)}")
    for ratio, sft_id, eval_id in similar:
        print(f"  - {ratio:.3f}: {sft_id} ~= {eval_id}")

    if parse_error_rows or schema_invalid_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
