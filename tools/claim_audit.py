"""Claim Audit Tool - Enforces Law L1: Zero ungrounded numbers in documentation.

Extracts all numbers (percentages, decimals, 'N de M', milliseconds, dollars)
from README.md and docs/, and requires that each is within a block
<!-- GEN:<nombre> -->...<!-- /GEN --> produced by evals/generate_readme_tables.py
from a verified JSON report, or in tools/claim_allowlist.yaml.

Any orphan number outside of GEN blocks or the allowlist fails with exit code 1.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml


def load_allowlist(allowlist_path: Path) -> tuple[set[str], list[re.Pattern]]:
    if not allowlist_path.exists():
        raise FileNotFoundError(f"Allowlist file not found: {allowlist_path}")

    with allowlist_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    allowed_numbers = {str(n).strip() for n in data.get("allowed_numbers", [])}
    allowed_regexes = [re.compile(pattern) for pattern in data.get("allowed_regexes", [])]
    return allowed_numbers, allowed_regexes


def is_allowed(token: str, allowed_numbers: set[str], allowed_regexes: list[re.Pattern]) -> bool:
    clean_token = token.strip()
    if clean_token in allowed_numbers:
        return True

    # Strip %, $, ms, s, etc. for root number check
    stripped = re.sub(r"[^\d.]", "", clean_token)
    if stripped in allowed_numbers:
        return True

    for reg in allowed_regexes:
        if reg.match(clean_token) or (stripped and reg.match(stripped)):
            return True

    return False


def audit_markdown_file(
    file_path: Path, allowed_numbers: set[str], allowed_regexes: list[re.Pattern]
) -> list[dict[str, Any]]:
    content = file_path.read_text(encoding="utf-8")
    violations: list[dict[str, Any]] = []

    # 1. Strip out fenced code blocks ``` ... ```
    content_no_code = re.sub(r"```[\s\S]*?```", "", content)

    # 2. Strip out inline code ` ... `
    content_no_code = re.sub(r"`[^`\n]+`", "", content_no_code)

    # 3. Strip out markdown link URLs [text](URL) -> keep only [text]
    content_no_code = re.sub(r"\]\([^)]+\)", "]", content_no_code)

    # 4. Remove all generated blocks <!-- GEN:<name> --> ... <!-- /GEN -->
    content_outside_gen = re.sub(r"<!--\s*GEN:\w+\s*-->[\s\S]*?<!--\s*/GEN\s*-->", "", content_no_code)

    # 5. Find numbers outside GEN blocks
    # Patterns to catch:
    # - Percentages: e.g. 50.5%, 100%
    # - Currencies: e.g. $1.50, $0.88, $2.41
    # - Decimals: e.g. 0.5575, 4.511
    # - Ratios: e.g. 12 de 16, 50/79
    # - Latencies: e.g. 4511.1 ms, 22.9 s
    # - Plain integers
    combined_pattern = re.compile(
        r"(?P<currency>\$\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?|\$\s*\d+(?:\.\d+)?)"
        r"|(?P<percentage>\b\d+(?:\.\d+)?\s*%)"
        r"|(?P<latency>\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:ms|segundos|second|s\b)|\b\d+(?:\.\d+)?\s*(?:ms|segundos|second|s\b))"
        r"|(?P<ratio>\b\d+\s*(?:de|\/)\s*\d+\b)"
        r"|(?P<decimal>\b\d+\.\d+\b)"
        r"|(?P<integer>\b\d{1,3}(?:,\d{3})+\b|\b\d+\b)"
    )

    lines = content_outside_gen.splitlines()
    for line_num, line in enumerate(lines, 1):
        # Skip HTML comments entirely
        if line.strip().startswith("<!--") and line.strip().endswith("-->"):
            continue

        for match in combined_pattern.finditer(line):
            token = match.group(0).strip()
            pat_type = match.lastgroup or "unknown"
            if not is_allowed(token, allowed_numbers, allowed_regexes):
                violations.append(
                    {
                        "file": str(file_path),
                        "line": line_num,
                        "token": token,
                        "type": pat_type,
                        "context": line.strip()[:100],
                    }
                )

    return violations


def main() -> int:
    allowlist_path = Path("tools/claim_allowlist.yaml")
    allowed_numbers, allowed_regexes = load_allowlist(allowlist_path)

    files_to_check = [Path("README.md")]
    docs_dir = Path("docs")
    if docs_dir.exists():
        files_to_check.extend(docs_dir.glob("**/*.md"))

    all_violations: list[dict[str, Any]] = []
    for md_file in files_to_check:
        if md_file.exists():
            v = audit_markdown_file(md_file, allowed_numbers, allowed_regexes)
            all_violations.extend(v)

    if all_violations:
        print("\n" + "=" * 70, file=sys.stderr)
        print(f" CLAIM AUDIT FAILED: {len(all_violations)} ungrounded orphan numbers found!", file=sys.stderr)
        print(" Every number must be inside a <!-- GEN:<name> --> block from a report,", file=sys.stderr)
        print(" or registered in tools/claim_allowlist.yaml.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        for viol in all_violations:
            print(
                f"  [{viol['file']}:{viol['line']}] ({viol['type']}) '{viol['token']}' in: \"{viol['context']}\"",
                file=sys.stderr,
            )
        return 1

    print(f" Claim Audit Passed: All numbers in {[str(f) for f in files_to_check]} are properly grounded in GEN blocks or allowlisted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
