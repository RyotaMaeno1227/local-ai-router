#!/usr/bin/env python3
"""Check local Markdown document references in README.md and docs/**/*.md."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
ROOT_MARKDOWN_PATH = re.compile(r"(?<![A-Za-z0-9_./-])(?:README\.md|docs/[A-Za-z0-9_./-]+\.md)")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:")


def source_files() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").glob("**/*.md"))]


def clean_link_target(raw_target: str) -> str | None:
    target = raw_target.strip().strip("<>")
    if not target or target.startswith(EXTERNAL_PREFIXES) or target.startswith("#"):
        return None
    target = unquote(target.split("#", 1)[0].split("?", 1)[0])
    return target if target.endswith(".md") else None


def resolve_target(source: Path, target: str) -> Path:
    if target == "README.md" or target.startswith("docs/"):
        return ROOT / target
    return source.parent / target


def markdown_references(source: Path) -> list[tuple[int, str, Path]]:
    references: list[tuple[int, str, Path]] = []
    seen: set[tuple[int, str]] = set()
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        targets: list[str] = []
        for match in MARKDOWN_LINK.finditer(line):
            target = clean_link_target(match.group(1))
            if target:
                targets.append(target)
        targets.extend(match.group(0) for match in ROOT_MARKDOWN_PATH.finditer(line))
        for target in targets:
            key = (line_no, target)
            if key in seen:
                continue
            seen.add(key)
            references.append((line_no, target, resolve_target(source, target)))
    return references


def main() -> None:
    files = source_files()
    missing: list[str] = []
    reference_count = 0
    for source in files:
        for line_no, target, resolved in markdown_references(source):
            reference_count += 1
            if not resolved.is_file():
                relative_source = source.relative_to(ROOT)
                missing.append(f"{relative_source}:{line_no}: {target} -> {resolved.relative_to(ROOT)}")

    print(f"Checked Markdown files: {len(files)}")
    print(f"Checked local .md references: {reference_count}")
    print(f"Missing references: {len(missing)}")
    for item in missing:
        print(f"  {item}", file=sys.stderr)
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
