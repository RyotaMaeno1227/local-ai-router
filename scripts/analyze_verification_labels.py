#!/usr/bin/env python3
"""Compare expected and predicted router verification labels."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


GENERIC_WORDS = {"check", "consistency", "inspection", "review", "verification", "verify"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_file", help="Eval JSONL containing must_verification labels.")
    parser.add_argument("results", nargs="+", help="One or more eval result JSON files.")
    parser.add_argument("--output", help="Optional Markdown output path.")
    parser.add_argument("--top", type=int, default=30, help="Maximum vocabulary rows per result.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verification_checks(result: dict[str, Any]) -> list[str]:
    prediction = result.get("prediction")
    if not isinstance(prediction, dict):
        return []
    verification = prediction.get("verification")
    if not isinstance(verification, dict):
        return []
    checks = verification.get("checks")
    if not isinstance(checks, list):
        return []
    return [str(label) for label in checks if isinstance(label, str) and label]


def words(label: str, *, drop_generic: bool) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", label.lower().replace("_", " ").replace("-", " "))
    if drop_generic:
        reduced = [token for token in tokens if token not in GENERIC_WORDS]
        return reduced or tokens
    return tokens


def alias_score(predicted: str, expected: str) -> float:
    predicted_words = words(predicted, drop_generic=True)
    expected_words = words(expected, drop_generic=True)
    predicted_set = set(predicted_words)
    expected_set = set(expected_words)
    if not predicted_set.intersection(expected_set):
        return 0.0
    union = predicted_set | expected_set
    jaccard = len(predicted_set & expected_set) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, " ".join(predicted_words), " ".join(expected_words)).ratio()
    return max(jaccard, sequence)


def alias_candidates(predicted: str, expected_vocabulary: set[str]) -> list[tuple[str, float]]:
    scored = sorted(
        ((expected, alias_score(predicted, expected)) for expected in expected_vocabulary),
        key=lambda item: (-item[1], item[0]),
    )
    return [(label, score) for label, score in scored[:3] if score >= 0.6]


def render(eval_path: Path, result_paths: list[Path], top: int) -> str:
    eval_rows = read_jsonl(eval_path)
    expected_by_id = {row["id"]: [str(label) for label in row["must_verification"]] for row in eval_rows}
    expected_counts = Counter(label for labels in expected_by_id.values() for label in labels)
    expected_vocabulary = set(expected_counts)

    lines = [
        "# Verification Label Analysis",
        "",
        f"Eval file: `{eval_path}`",
        f"Cases: {len(eval_rows)}",
        f"Expected vocabulary size: {len(expected_vocabulary)}",
        "",
        "## Expected must_verification labels",
        "",
        "| Label | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| `{label}` | {count} |" for label, count in expected_counts.most_common())

    for result_path in result_paths:
        report = json.loads(result_path.read_text(encoding="utf-8"))
        predicted_counts: Counter[str] = Counter()
        failures: list[tuple[str, list[str], list[str], list[str]]] = []
        for result in report["results"]:
            case_id = result["id"]
            expected = expected_by_id.get(case_id, [])
            predicted = verification_checks(result)
            predicted_counts.update(predicted)
            predicted_set = set(predicted)
            missing = [label for label in expected if label not in predicted_set]
            if missing:
                failures.append((case_id, expected, predicted, missing))

        unexpected = Counter(
            {label: count for label, count in predicted_counts.items() if label not in expected_vocabulary}
        )
        exact_successes = len(eval_rows) - len(failures)
        lines.extend(
            [
                "",
                f"## {result_path.name}",
                "",
                f"- exact label containment: {exact_successes}/{len(eval_rows)}",
                f"- unique predicted labels: {len(predicted_counts)}",
                f"- unique predicted labels outside expected vocabulary: {len(unexpected)}",
                "",
                "### Predicted verification labels",
                "",
                "| Label | Count | In expected vocabulary |",
                "| --- | ---: | --- |",
            ]
        )
        lines.extend(
            f"| `{label}` | {count} | {'yes' if label in expected_vocabulary else 'no'} |"
            for label, count in predicted_counts.most_common(top)
        )

        lines.extend(
            [
                "",
                "### Frequent predicted labels outside expected vocabulary",
                "",
                "| Predicted label | Count | Possible aliases |",
                "| --- | ---: | --- |",
            ]
        )
        for label, count in unexpected.most_common(top):
            candidates = alias_candidates(label, expected_vocabulary)
            rendered = ", ".join(f"`{candidate}` ({score:.2f})" for candidate, score in candidates) or "none"
            lines.append(f"| `{label}` | {count} | {rendered} |")

        lines.extend(
            [
                "",
                "### Exact match failures",
                "",
                "| Case | Missing expected labels | Predicted labels |",
                "| --- | --- | --- |",
            ]
        )
        for case_id, _expected, predicted, missing in failures:
            missing_text = ", ".join(f"`{label}`" for label in missing)
            predicted_text = ", ".join(f"`{label}`" for label in predicted) or "none"
            lines.append(f"| `{case_id}` | {missing_text} | {predicted_text} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.top <= 0:
        raise ValueError("--top must be positive")
    output = render(Path(args.eval_file), [Path(path) for path in args.results], args.top)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
