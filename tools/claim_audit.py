"""Claim Audit Tool - Enforces Law L1 & Law L4: Zero ungrounded numbers or non-existent models.

1. Law L1 (Zero ungrounded numbers):
   Extracts all numbers (percentages, decimals, 'N de M', milliseconds, dollars, multipliers, fractions)
   from all Markdown files in the repo, and requires that each is within a block
   <!-- GEN:<nombre> -->...<!-- /GEN --> produced by evals/generate_readme_tables.py
   from a verified JSON report, or in tools/claim_allowlist.yaml.

2. Law L4 (Nothing in README that does not exist):
   Extracts all model references (gemini-*, gpt-*, claude-*, llama-*, etc.)
   and requires that each model is explicitly declared in config/models.yaml
   or is an approved offline baseline.

Any orphan number, fraction, multiplier, or unknown model fails with exit code 1.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml


def load_allowlist(
    allowlist_path: Path, include_file_specific: bool = False
) -> tuple[set[str], list[re.Pattern]] | tuple[set[str], list[re.Pattern], dict[str, set[str]]]:
    if not allowlist_path.exists():
        raise FileNotFoundError(f"Allowlist file not found: {allowlist_path}")

    with allowlist_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    allowed_numbers = {str(n).strip() for n in data.get("allowed_numbers", [])}
    allowed_regexes = [re.compile(pattern) for pattern in data.get("allowed_regexes", [])]
    if include_file_specific:
        file_specific = {
            fname: {str(n).strip() for n in nums}
            for fname, nums in data.get("file_specific_allowlists", {}).items()
        }
        return allowed_numbers, allowed_regexes, file_specific
    return allowed_numbers, allowed_regexes



def get_allowed_models(models_yaml_path: Path) -> set[str]:
    allowed = {
        # Baseline/architectural components
        "stub-hash-768d",
        "stub-hash",
        "stub",
        "bm25",
        "flashrank",
        "vertex_ai",
        # Approved local candidate for Fase 1 retrieval evaluation
        "bge-small-en-v1.5",
    }
    if models_yaml_path.exists():
        with models_yaml_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for section in ["extraction_llm", "clause_llm", "alternative_llm", "embeddings", "reranker", "datagen_llm"]:
            sec_data = cfg.get(section, {})
            if isinstance(sec_data, dict):
                mid = sec_data.get("model_id")
                if mid:
                    allowed.add(mid.strip().lower())

    # Include official pricing models
    pricing_yaml_path = Path("config/pricing.yaml")
    if pricing_yaml_path.exists():
        with pricing_yaml_path.open("r", encoding="utf-8") as f:
            pcfg = yaml.safe_load(f) or {}
        for m in pcfg.get("models", {}):
            allowed.add(m.strip().lower())

    return allowed


def is_allowed(token: str, allowed_numbers: set[str], allowed_regexes: list[re.Pattern]) -> bool:
    clean_token = token.strip()
    if clean_token in allowed_numbers:
        return True

    return any(pattern.match(clean_token) for pattern in allowed_regexes)



def audit_markdown_file(
    file_path: Path,
    allowed_numbers: set[str],
    allowed_regexes: list[re.Pattern],
    file_specific_allowlists: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    if not file_path.exists():
        return []

    content = file_path.read_text(encoding="utf-8")
    violations: list[dict[str, Any]] = []

    # Combine global allowed numbers with per-file allowed numbers
    effective_allowed = set(allowed_numbers)
    if file_specific_allowlists:
        effective_allowed |= file_specific_allowlists.get(file_path.name, set())
        effective_allowed |= file_specific_allowlists.get(str(file_path), set())

    # 1. Strip out code blocks ``` ... ```
    content_no_code = re.sub(r"```[\s\S]*?```", "", content)

    # 2. Strip out inline code ` ... `
    content_no_code = re.sub(r"`[^`\n]+`", "", content_no_code)

    # 3. Strip out markdown link URLs [text](URL) -> keep only [text]
    content_no_code = re.sub(r"\]\([^)]+\)", "]", content_no_code)

    # 4. Remove all generated blocks <!-- GEN:<name> --> ... <!-- /GEN -->
    content_outside_gen = re.sub(r"<!--\s*GEN:\w+\s*-->[\s\S]*?<!--\s*/GEN\s*-->", "", content_no_code)

    # 5. Find numbers, multipliers outside GEN blocks
    combined_pattern = re.compile(
        r"(?P<currency>\$\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?|\$\s*\d+(?:\.\d+)?)"
        r"|(?P<percentage>\b\d+(?:\.\d+)?\s*%)"
        r"|(?P<latency>\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:ms|segundos|second|s\b)|\b\d+(?:\.\d+)?\s*(?:ms|segundos|second|s\b))"
        r"|(?P<ratio>\b\d+\s*(?:de|\/)\s*\d+\b)"
        r"|(?P<multiplier>\bx\s*\d+\b|\b\d+\s*x\b)"
        r"|(?P<semver>\b\d+\.\d+\.\d+\b)"
        r"|(?P<decimal>\b\d+\.\d+\b)"
        r"|(?P<integer>\b\d{1,3}(?:,\d{3})+\b|\b\d+\b)"
    )

    # Catch numbers, fractions and multipliers written in words in Spanish and English
    words_pattern = re.compile(
        r"(?i)\b(?:"
        r"(?:noventa|ochenta|setenta|sesenta|cincuenta|cuarenta|treinta|veinte|veinti[a-záéíóú]+)\s+y\s+(?:uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve)"
        r"|(?:cero|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|diecis[eé]is|diecisiete|dieciocho|diecinueve|veinte|veinti[a-záéíóú]+|treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa|cien|ciento|quinientos|mil|mill[oó]n|millones)(?:\s+y\s+[a-záéíóú]+)?\s+(?:por\s+ciento|percent)"
        r"|(?:por\s+ciento|percent)"
        r"|(?:de\s+cada|out\s+of)\s+(?:dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|cien|\d+)"
        r"|(?:(?:un|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve)\s+)?(?:cuarto|cuartos|tercio|tercios|quinto|quintos|sexto|sextos|s[eé]ptimo|s[eé]ptimos|octavo|octavos|noveno|novenos|d[eé]cimo|d[eé]cimos)"
        r"|(?:la|una)?\s*mitad"
        r"|(?:half|quarter|quarters|thirds?)(?!-party)"
        r"|(?:el|la|un|una)?\s*(?:doble|triple|cu[aá]druple|qu[ií]ntuple)"
        r"|(?:double|triple|quadruple)"
        r"|(?:x\s*\d+|\d+\s*x)"
        r")\b"
    )


    lines = content_outside_gen.splitlines()
    for line_num, line in enumerate(lines, 1):
        # Skip HTML comments entirely
        if line.strip().startswith("<!--") and line.strip().endswith("-->"):
            continue

        for match in combined_pattern.finditer(line):
            token = match.group(0).strip()
            pat_type = match.lastgroup or "unknown"
            if not is_allowed(token, effective_allowed, allowed_regexes):
                violations.append(
                    {
                        "file": str(file_path),
                        "line": line_num,
                        "token": token,
                        "type": pat_type,
                        "context": line.strip()[:100],
                    }
                )

        for match in words_pattern.finditer(line):
            token = match.group(0).strip()
            if not is_allowed(token, effective_allowed, allowed_regexes):
                violations.append(
                    {
                        "file": str(file_path),
                        "line": line_num,
                        "token": token,
                        "type": "number_in_words",
                        "context": line.strip()[:100],
                    }
                )

    return violations


def audit_markdown_models(file_path: Path, allowed_models: set[str]) -> list[dict[str, Any]]:
    content = file_path.read_text(encoding="utf-8")
    violations: list[dict[str, Any]] = []

    model_pattern = re.compile(
        r"\b(gemini-[a-zA-Z0-9.-]+|gpt-[a-zA-Z0-9.-]+|claude-[a-zA-Z0-9.-]+|llama-[a-zA-Z0-9.-]+|mistral-[a-zA-Z0-9.-]+|text-embedding-[a-zA-Z0-9.-]+|bge-[a-zA-Z0-9.-]+)\b",
        re.IGNORECASE,
    )

    for line_num, line in enumerate(content.splitlines(), 1):
        for match in model_pattern.finditer(line):
            model_name = match.group(0).lower()
            if model_name not in allowed_models:
                violations.append(
                    {
                        "file": str(file_path),
                        "line": line_num,
                        "model": model_name,
                        "context": line.strip()[:100],
                    }
                )
    return violations


def get_exempt_files(allowlist_path: Path) -> set[str]:
    if not allowlist_path.exists():
        return set()
    with allowlist_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {str(Path(f)).lower() for f in data.get("exempt_files", [])}


def discover_markdown_files(repo_root: Path, exempt_files: set[str]) -> list[Path]:
    md_files: list[Path] = []
    for p in repo_root.rglob("*.md"):
        try:
            rel = p.relative_to(repo_root)
        except ValueError:
            continue
        # Skip hidden directories, venv, caches, node_modules relative to repo root
        if any(
            part.startswith(".")
            or part in ["venv", ".venv", "site-packages", "node_modules", "brain", ".terraform"]
            for part in rel.parts[:-1]
        ):
            continue
        if str(rel).lower() in exempt_files or rel.name.lower() in exempt_files:
            continue
        md_files.append(rel)
    return sorted(md_files)


def main() -> int:
    allowlist_path = Path("tools/claim_allowlist.yaml")
    models_yaml_path = Path("config/models.yaml")
    allowed_numbers, allowed_regexes, file_allowlists = load_allowlist(allowlist_path, include_file_specific=True)
    allowed_models = get_allowed_models(models_yaml_path)
    exempt_files = get_exempt_files(allowlist_path)

    repo_root = Path.cwd()
    files_to_check = discover_markdown_files(repo_root, exempt_files)

    all_violations: list[dict[str, Any]] = []
    model_violations: list[dict[str, Any]] = []

    for md_file in files_to_check:
        if md_file.exists():
            v = audit_markdown_file(md_file, allowed_numbers, allowed_regexes, file_allowlists)
            all_violations.extend(v)
            mv = audit_markdown_models(md_file, allowed_models)
            model_violations.extend(mv)

    if all_violations or model_violations:
        print("\n" + "=" * 70, file=sys.stderr)
        print(" CLAIM AUDIT FAILED: Ungrounded numbers or unauthorized models detected!", file=sys.stderr)
        print(" Every number must be within a <!-- GEN:<name> --> block or in tools/claim_allowlist.yaml", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

        for v in all_violations:
            print(
                f"  ❌ [{v['file']}:{v['line']}] Ungrounded token '{v['token']}' ({v['type']}): \"{v['context']}\"",
                file=sys.stderr,
            )

        for mv in model_violations:
            print(
                f"  ❌ [{mv['file']}:{mv['line']}] Unauthorized model '{mv['model']}': \"{mv['context']}\"",
                file=sys.stderr,
            )

        return 1

    print(f" Claim Audit Passed: All numbers and models in {[str(f) for f in files_to_check]} are grounded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
