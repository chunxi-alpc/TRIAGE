#!/usr/bin/env python3
r"""Check active ASTER Markdown math, module order, and local file links.

These are source checks, not a browser rendering or model validation test.
Historical documents are reached through immutable archive links and are not
included in the active-document checks.
"""
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "README.md",
    "docs/README.md",
    "docs/REPOSITORY_STATUS.md",
    "docs/ASTER/README.md",
    "docs/ASTER/manuscript.zh.md",
    "docs/ASTER/supplementary.zh.md",
    "docs/ASTER/experiments.zh.md",
    "docs/ASTER/references.zh.md",
    "docs/ASTER/archive/README.md",
    "src/aster/README.md",
)
BANNED = (
    (re.compile(r"\\operatorname\b"), r"use \mathrm or a documented equivalent in GitHub Markdown"),
    (re.compile(r"\\tag\b"), r"use \qquad\text{(n)}; \tag can misrender on GitHub"),
)
NUMBER = re.compile(r"\\text\{\((\d+)\)\}")
LINK = re.compile(r"!?\[[^\]]*\]\(([^\s)]+)\)")
FENCE = "```"
MODULES = (
    "### 4.2 模块一：对比适用性子空间学习",
    "### 4.3 模块二：处置盲适用性评估",
    "### 4.4 模块三：适用性门控的规则键编辑",
)


def check(path: Path) -> list[str]:
    name = str(path.relative_to(ROOT))
    if not path.is_file():
        return [f"{name}: active document missing"]
    source = path.read_text(encoding="utf-8")
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
    prose = []
    for line_number, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith(FENCE):
            if not in_fence:
                in_fence = True
                opener = stripped[3:].strip()
                n_math += int(opener == "math")
            elif stripped == FENCE:
                in_fence = False
                opener = ""
            else:
                issues.append(f"{name}:{line_number}: invalid closing fence")
        elif not in_fence:
            prose.append((line_number, line))
    if in_fence:
        issues.append(f"{name}: unclosed {opener or 'plain'} code fence")

    for line_number, line in prose:
        for target in LINK.findall(line):
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            relative = unquote(target.split("#", 1)[0])
            if relative and not (path.parent / relative).exists():
                issues.append(f"{name}:{line_number}: missing local link target {relative}")

    if name == "docs/ASTER/manuscript.zh.md":
        numbers = [int(n) for n in NUMBER.findall(source)]
        expected = list(range(1, 14))
        if numbers != expected:
            issues.append(f"{name}: equation labels {numbers} are not {expected}")
        for heading in MODULES:
            if source.count(heading) != 1:
                issues.append(f"{name}: expected one heading: {heading}")
        if all(heading in source for heading in MODULES):
            positions = [source.index(heading) for heading in MODULES]
            if positions != sorted(positions):
                issues.append(f"{name}: modules are out of order")
        for section in ("4.2", "4.3", "4.4"):
            headings = re.findall(rf"^#### {re.escape(section)}\.\d+ ", source, flags=re.M)
            if len(headings) != 3:
                issues.append(f"{name}: {section} must have exactly three subsections")
        if n_math == 0:
            issues.append(f"{name}: no protected math blocks")
    return issues


def main() -> int:
    issues = [issue for relpath in FILES for issue in check(ROOT / relpath)]
    if issues:
        print("\n".join(issues), file=sys.stderr)
        return 1
    print(f"ASTER source checks passed for {len(FILES)} active Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
