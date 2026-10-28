#!/usr/bin/env python3
r"""Static GitHub Markdown math-compatibility checks for active ASTER documentation.

GitHub rejects \operatorname in rendered Markdown math. Use \mathrm or a
documented compatible alternative. These checks complement visual review;
they do not execute GitHub's frontend math renderer.
"""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "README.md",
    "docs/ASTER/README.md",
    "docs/ASTER/manuscript.zh.md",
    "docs/ASTER/module1-spec.zh.md",
    "docs/ASTER/experiment-plan.zh.md",
    "docs/ASTER/claims-and-falsification.zh.md",
    "docs/ASTER/implementation-status.zh.md",
)

BANNED = re.compile(r"\\operatorname\b")
TAG = re.compile(r"\\tag\{(\d+)\}")


def check(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    name = str(path.relative_to(ROOT))
    issues: list[str] = []

    for match in BANNED.finditer(source):
        line = source.count("\n", 0, match.start()) + 1
        issues.append(rf"{name}:{line}: GitHub blocks \operatorname; use \mathrm")

    if source.count("$$") % 2:
        issues.append(f"{name}: unmatched $$ math delimiter")

    if name == "docs/ASTER/manuscript.zh.md":
        numbers = [int(n) for n in TAG.findall(source)]
        expected = list(range(1, len(numbers) + 1))
        if numbers != expected:
            issues.append(f"{name}: formula tags {numbers} are not {expected}")

    return issues


def main() -> int:
    issues = []
    for relpath in FILES:
        issues.extend(check(ROOT / relpath))
    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        return 1
    print(f"ASTER math compatibility checks passed for {len(FILES)} active Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
