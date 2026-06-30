#!/usr/bin/env python3
"""Audit the 90-row router SFT v002 candidate dataset."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from router_vocab import validate_router_output_vocab
from validate_router_json import load_json, validate_router_output


EXPECTED_AREAS = {
    "contact_analysis_risk": 15,
    "nonlinear_fem_risk_verification": 15,
    "code_review_tools_safety": 15,
    "paper_novelty_citation": 15,
    "fem_fundamentals": 10,
    "rag_api_fusion": 10,
    "request_more_info_boundary": 10,
}
DEFAULT_EVAL_FILES = (
    Path("evals/router_eval_001_canonical.jsonl"),
    Path("evals/router_eval_holdout_001.jsonl"),
)
DEFAULT_SFT_FILES = (
    Path("data/router_sft_001.jsonl"),
    Path("data/router_sft_002_canonical.jsonl"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--schema", type=Path, default=Path("schemas/router_output.schema.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/router_sft_v002_candidate_audit.md"))
    parser.add_argument("--near-threshold", type=float, default=0.82)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: row must be an object")
        rows.append(row)
    return rows


def message_text(row: dict[str, Any], role: str) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def router_output(row: dict[str, Any]) -> dict[str, Any] | None:
    content = message_text(row, "assistant")
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def reference_prompts(path: Path) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []
    for index, row in enumerate(read_jsonl(path), start=1):
        prompt = row.get("user") if "user" in row else message_text(row, "user")
        if isinstance(prompt, str) and prompt.strip():
            prompts.append((str(row.get("id", index)), prompt.strip()))
    return prompts


def normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def similarity(left: str, right: str) -> float:
    left_norm = normalized_text(left)
    right_norm = normalized_text(right)
    if not left_norm or not right_norm:
        return 0.0
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, left_norm, right_norm).ratio()
    return max(jaccard, sequence)


def table(counter: Counter[str]) -> list[str]:
    lines = ["| Value | Count |", "| --- | ---: |"]
    lines.extend(f"| `{value}` | {count} |" for value, count in sorted(counter.items()))
    return lines


def bullet_ids(ids: list[str]) -> str:
    return ", ".join(f"`{item}`" for item in ids) if ids else "None"


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.near_threshold <= 1.0:
        raise ValueError("--near-threshold must be between 0 and 1")

    rows = read_jsonl(args.candidate)
    schema = load_json(args.schema)
    area_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    check_counts: Counter[str] = Counter()
    pair_members: dict[str, list[str]] = defaultdict(list)
    empty_tools: list[str] = []
    empty_checks: list[str] = []
    request_more_info: list[str] = []
    managed_fusion: list[str] = []
    self_fusion_lite: list[str] = []
    schema_errors: list[str] = []
    vocab_errors: list[str] = []
    clarification_policy_errors: list[str] = []
    fusion_policy_errors: list[str] = []
    prompt_to_ids: dict[str, list[str]] = defaultdict(list)
    candidate_prompts: list[tuple[str, str]] = []

    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id", f"row-{index}"))
        area_counts[str(row.get("area", "<missing>"))] += 1
        pair_id = row.get("pair_id")
        if isinstance(pair_id, str) and pair_id:
            pair_members[pair_id].append(row_id)

        prompt = message_text(row, "user").strip()
        candidate_prompts.append((row_id, prompt))
        prompt_to_ids[normalized_text(prompt)].append(row_id)
        output = router_output(row)
        if output is None:
            schema_errors.append(f"{row_id}: assistant.content is not a JSON object")
            vocab_errors.append(f"{row_id}: assistant.content is not a JSON object")
            continue

        schema_errors.extend(validate_router_output(row_id, output, schema))
        vocab = validate_router_output_vocab(output)
        vocab_errors.extend(f"{row_id}: {error}" for error in vocab.errors)
        mode = str(output.get("mode", "<missing>"))
        mode_counts[mode] += 1
        risk_counts[str(output.get("risk", "<missing>"))] += 1
        tools = output.get("needed_tools")
        if isinstance(tools, list):
            tool_counts.update(str(tool) for tool in tools)
            if not tools:
                empty_tools.append(row_id)
        checks = output.get("verification", {}).get("checks")
        if isinstance(checks, list):
            check_counts.update(str(check) for check in checks)
            if not checks:
                empty_checks.append(row_id)
        if mode == "request_more_info":
            request_more_info.append(row_id)
            if not isinstance(checks, list) or not {
                "missing_file_check", "comparison_target_check"
            }.intersection(checks):
                clarification_policy_errors.append(
                    f"{row_id}: request_more_info lacks a missing artifact or comparison-target check"
                )
        elif mode == "managed_fusion":
            managed_fusion.append(row_id)
        elif mode == "self_fusion_lite":
            self_fusion_lite.append(row_id)
        fusion = output.get("fusion_policy")
        if isinstance(fusion, dict):
            enabled = fusion.get("enabled")
            if mode in {"managed_fusion", "self_fusion_lite"} and enabled is not True:
                fusion_policy_errors.append(f"{row_id}: fusion mode must enable fusion_policy")
            if mode not in {"managed_fusion", "self_fusion_lite"} and enabled is not False:
                fusion_policy_errors.append(f"{row_id}: non-fusion mode must disable fusion_policy")
            if mode == "managed_fusion" and (
                fusion.get("type") != "expert_panel" or fusion.get("judge_required") is not True
            ):
                fusion_policy_errors.append(f"{row_id}: managed_fusion requires expert_panel and judge")
            if mode == "self_fusion_lite" and (
                fusion.get("type") != "self_consistency" or fusion.get("judge_required") is not False
            ):
                fusion_policy_errors.append(
                    f"{row_id}: self_fusion_lite requires self_consistency without judge"
                )

    internal_duplicates = [ids for key, ids in prompt_to_ids.items() if key and len(ids) > 1]
    controlled_groups = {key: ids for key, ids in pair_members.items() if len(ids) >= 2}
    controlled_rows = sum(len(ids) for ids in controlled_groups.values())

    reference_paths = (*DEFAULT_EVAL_FILES, *DEFAULT_SFT_FILES)
    exact_by_reference: dict[Path, list[tuple[str, str]]] = {}
    near_matches: list[tuple[float, str, Path, str, str, str]] = []
    candidate_norms = {row_id: normalized_text(prompt) for row_id, prompt in candidate_prompts}
    for path in reference_paths:
        exact: list[tuple[str, str]] = []
        for reference_id, reference_prompt in reference_prompts(path):
            reference_norm = normalized_text(reference_prompt)
            for candidate_id, candidate_prompt in candidate_prompts:
                if candidate_norms[candidate_id] == reference_norm:
                    exact.append((candidate_id, reference_id))
                    continue
                score = similarity(candidate_prompt, reference_prompt)
                if score >= args.near_threshold:
                    near_matches.append(
                        (score, candidate_id, path, reference_id, candidate_prompt, reference_prompt)
                    )
        exact_by_reference[path] = exact
    near_matches.sort(key=lambda item: (-item[0], item[1], str(item[2]), item[3]))

    lines = [
        "# Router SFT v002 Candidate Audit",
        "",
        f"Candidate: `{args.candidate}`",
        "",
        "## Summary",
        "",
        f"- total rows: {len(rows)}",
        f"- schema valid rows: {len(rows) - len({error.split(':', 1)[0] for error in schema_errors})}/{len(rows)}",
        f"- vocabulary strict valid rows: {len(rows) - len({error.split(':', 1)[0] for error in vocab_errors})}/{len(rows)}",
        f"- controlled pair/triplet groups: {len(controlled_groups)}",
        f"- controlled pair/triplet rows: {controlled_rows}",
        f"- exact duplicate candidate prompts: {len(internal_duplicates)} groups",
        f"- exact eval prompt matches: {sum(len(exact_by_reference[path]) for path in DEFAULT_EVAL_FILES)}",
        f"- exact prior-SFT prompt matches: {sum(len(exact_by_reference[path]) for path in DEFAULT_SFT_FILES)}",
        f"- near-match candidates at threshold {args.near_threshold:.2f}: {len(near_matches)}",
        f"- request_more_info policy errors: {len(clarification_policy_errors)}",
        f"- fusion policy errors: {len(fusion_policy_errors)}",
        "",
        "## Area Distribution",
        "",
        *table(area_counts),
        "",
        "## Mode Distribution",
        "",
        *table(mode_counts),
        "",
        "## Risk Distribution",
        "",
        *table(risk_counts),
        "",
        "## Needed Tools Distribution",
        "",
        *table(tool_counts),
        "",
        "## Verification Checks Distribution",
        "",
        *table(check_counts),
        "",
        "## Boundary And Fusion Counts",
        "",
        f"- empty `needed_tools`: {len(empty_tools)} ({bullet_ids(empty_tools)})",
        f"- empty `verification.checks`: {len(empty_checks)} ({bullet_ids(empty_checks)})",
        f"- `request_more_info`: {len(request_more_info)} ({bullet_ids(request_more_info)})",
        f"- `managed_fusion`: {len(managed_fusion)} ({bullet_ids(managed_fusion)})",
        f"- `self_fusion_lite`: {len(self_fusion_lite)} ({bullet_ids(self_fusion_lite)})",
        "",
        "## Exact Duplicate Audit",
        "",
        f"- within candidate: {internal_duplicates or 'None'}",
    ]
    for path in reference_paths:
        matches = exact_by_reference[path]
        rendered = ", ".join(f"`{candidate}` = `{reference}`" for candidate, reference in matches)
        lines.append(f"- against `{path}`: {rendered or 'None'}")

    lines.extend(["", "## Near-Duplicate Candidates", ""])
    if near_matches:
        lines.extend([
            "These are review candidates only; similarity does not make them duplicates.",
            "",
            "| Score | Candidate | Reference | Candidate prompt | Reference prompt |",
            "| ---: | --- | --- | --- | --- |",
        ])
        for score, candidate_id, path, reference_id, candidate_prompt, reference_prompt in near_matches:
            safe_candidate = candidate_prompt.replace("|", "\\|")
            safe_reference = reference_prompt.replace("|", "\\|")
            lines.append(
                f"| {score:.3f} | `{candidate_id}` | `{path}:{reference_id}` | {safe_candidate} | {safe_reference} |"
            )
    else:
        lines.append(f"None at threshold {args.near_threshold:.2f}.")

    lines.extend([
        "",
        "## Validation Errors",
        "",
        f"- schema errors: {len(schema_errors)}",
        f"- vocabulary errors: {len(vocab_errors)}",
        f"- request_more_info policy errors: {len(clarification_policy_errors)}",
        f"- fusion policy errors: {len(fusion_policy_errors)}",
    ])
    lines.extend(
        f"- `{error}`"
        for error in schema_errors + vocab_errors + clarification_policy_errors + fusion_policy_errors
    )
    lines.extend([
        "",
        "## Decision",
        "",
        "This file is candidate training data only. No fine-tuning or adapter update was run in M16.",
        "Near-duplicate candidates remain listed above for human review before any training approval.",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote audit report to {args.output}")
    print(f"Rows: {len(rows)}")
    print(f"Schema errors: {len(schema_errors)}")
    print(f"Vocabulary errors: {len(vocab_errors)}")
    print(f"Request-more-info policy errors: {len(clarification_policy_errors)}")
    print(f"Fusion policy errors: {len(fusion_policy_errors)}")
    print(f"Exact eval prompt matches: {sum(len(exact_by_reference[path]) for path in DEFAULT_EVAL_FILES)}")
    print(f"Near-match candidates: {len(near_matches)}")

    failures: list[str] = []
    if len(rows) != 90:
        failures.append(f"expected 90 rows, got {len(rows)}")
    if area_counts != Counter(EXPECTED_AREAS):
        failures.append(f"area distribution mismatch: {dict(area_counts)}")
    if controlled_rows < 30:
        failures.append(f"controlled pair/triplet rows below 30: {controlled_rows}")
    if schema_errors:
        failures.append(f"schema errors: {len(schema_errors)}")
    if vocab_errors:
        failures.append(f"vocabulary errors: {len(vocab_errors)}")
    if empty_checks:
        failures.append(f"empty verification checks: {len(empty_checks)}")
    if clarification_policy_errors:
        failures.append(f"request_more_info policy errors: {len(clarification_policy_errors)}")
    if fusion_policy_errors:
        failures.append(f"fusion policy errors: {len(fusion_policy_errors)}")
    if internal_duplicates:
        failures.append(f"internal exact duplicate groups: {len(internal_duplicates)}")
    exact_eval = sum(len(exact_by_reference[path]) for path in DEFAULT_EVAL_FILES)
    if exact_eval:
        failures.append(f"exact eval prompt matches: {exact_eval}")
    if failures:
        raise SystemExit("Audit failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
