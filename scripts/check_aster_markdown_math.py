#!/usr/bin/env python3
r"""Static math-source checks for active ASTER Markdown documents.

GitHub renders math through MathJax. Fenced math blocks avoid conflicts with
Markdown parsing; equation numbers use visible math-text labels instead of
the tag macro, which can render incorrectly on GitHub. This script checks
source consistency, not the final browser-rendered appearance.
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
    "docs/ASTER/module2-supplement.zh.md",
    "docs/ASTER/experiment-plan.zh.md",
    "docs/ASTER/claims-and-falsification.zh.md",
    "docs/ASTER/implementation-status.zh.md",
)

BANNED = (
    (re.compile(r"\\operatorname\b"), r"use \mathrm or a documented equivalent in GitHub Markdown"),
    (re.compile(r"\\tag\b"), r"use \qquad\text{(n)}; \tag can misrender on GitHub"),
)
NUMBER = re.compile(r"\\text\{\((\d+)\)\}")
FENCE = chr(96) * 3


def check(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    name = str(path.relative_to(ROOT))
    issues: list[str] = []
    for pattern, message in BANNED:
        for match in pattern.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            issues.append(f"{name}:{line}: {message}")

    if source.count("$$") % 2:
        issues.append(f"{name}: unmatched $$ math delimiter")

    in_fence = False
    opener = ""
    n_math = 0
    for line_number, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith(FENCE):
            continue
        if not in_fence:
            in_fence = True
            opener = stripped[3:].strip()
            n_math += int(opener == "math")
        elif stripped == FENCE:
            in_fence = False
            opener = ""
        else:
            issues.append(f"{name}:{line_number}: closing code fence has unexpected text")
    if in_fence:
        issues.append(f"{name}: unclosed {opener or 'plain'} code fence")

    if name == "docs/ASTER/manuscript.zh.md":
        numbers = [int(n) for n in NUMBER.findall(source)]
        expected = list(range(1, 12))
        if numbers != expected:
            issues.append(f"{name}: equation labels {numbers} are not {expected}")
        if n_math == 0:
            issues.append(f"{name}: no protected math blocks")
    if name == "docs/ASTER/module2-supplement.zh.md" and n_math == 0:
        issues.append(f"{name}: no protected math blocks")
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
