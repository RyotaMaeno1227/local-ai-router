# Codex Job Completion

Use this template for the final report of each Codex job. Replace every
placeholder and remove rows that do not apply.

## Commit

- hash: `<commit-hash>`
- message: `<commit-message>`
- branch: `<branch>`
- remote state: `<synchronized | ahead N | behind N | no remote>`

## Working Tree

- status: `<clean | list remaining changes>`
- `git status --short --branch`:

```text
<status output>
```

## Key Docs

- [Router documentation index](README.md)
- [Standard router configuration](router_standard_config.md)
- `<other key document and link>`

## Updated Docs

- `<path>`: `<what changed>`
- `<path>`: `<what changed>`

If no documentation changed, state `None`.

## Validation Summary

| Check | Command | Result |
| --- | --- | --- |
| Markdown links | `python scripts/check_markdown_links.py` | `<pass/fail and counts>` |
| Whitespace | `git diff --check` | `<pass/fail>` |
| Compile | `python -m compileall scripts` | `<pass/fail/not required>` |
| Task-specific validation | `<command>` | `<pass/fail and metrics>` |

Record known failures explicitly. Do not describe a partially passing check as
successful.

## Prohibited Actions Summary

| Action | Status |
| --- | --- |
| Fine-tuning or adapter training | `<prohibited and not run | approved and run>` |
| LoRA dry-run | `<prohibited and not run | approved and run>` |
| Model load or download | `<prohibited and not run | approved and run>` |
| External API connection | `<prohibited and not used | approved and used>` |
| Package installation | `<prohibited and not run | approved and run>` |
| `sudo` or environment deletion | `<prohibited and not run>` |
| Destructive git operation or force push | `<prohibited and not run>` |

Add any job-specific prohibition that is not covered above.

## Next Recommended Step

`<one concrete next step, including its approval boundary>`

Do not phrase an unapproved action as already authorized.
