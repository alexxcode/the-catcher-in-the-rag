"""Regression test suite preventing divergence between README.md and empirical reports.

Enforces:
1. All benchmark tables published in README.md must match generate_readme_tables.py byte-for-byte.
2. No metric in generate_readme_tables.py can be hardcoded as a numeric string literal.
3. Every table figure must originate from a verified JSON report in evals/reports/.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from evals.generate_readme_tables import (
    generate_bakeoff_table,
    generate_cuad_table,
    generate_domain_calibration_table,
    generate_llm_verification_table,
)


def test_no_hardcoded_numbers_in_domain_table_generator() -> None:
    """Verify that generate_domain_calibration_table contains zero ungrounded numeric literals."""
    import evals.generate_readme_tables as gen_module

    source = inspect.getsource(gen_module.generate_domain_calibration_table)

    # Disallow previously hallucinated/ungrounded literal numbers
    forbidden_literals = ["0.9818", "0.9940", "0.9730"]
    for lit in forbidden_literals:
        assert lit not in source, (
            f"Forbidden hardcoded numeric literal '{lit}' found in table generator source."
        )

    # Assert that all metric variables are read from loaded JSON report dictionaries
    assert "synth_data[" in source, (
        "Must read synthetic metrics directly from synth_data dictionary."
    )
    assert "cuad_data[" in source, "Must read CUAD metrics directly from cuad_data dictionary."


def test_readme_domain_calibration_table_in_sync() -> None:
    """Verify that README.md domain calibration table matches generate_domain_calibration_table() exactly."""
    readme_path = Path("README.md")
    assert readme_path.exists(), "README.md not found in repository root."

    readme_content = readme_path.read_text(encoding="utf-8")
    generated_table = generate_domain_calibration_table().strip()

    assert generated_table in readme_content, (
        "README.md domain calibration table does not match generated table from JSON reports. Run 'make sync-readme-tables' to update."
    )


def test_readme_cuad_table_in_sync() -> None:
    """Verify that README.md CUAD benchmark table matches generate_cuad_table() exactly."""
    readme_path = Path("README.md")
    readme_content = readme_path.read_text(encoding="utf-8")
    generated_table = generate_cuad_table().strip()

    assert generated_table in readme_content, (
        "README.md CUAD benchmark table is out of sync with evals/reports/cuad_benchmark_latest.json."
    )


def test_readme_bakeoff_table_in_sync() -> None:
    """Verify that README.md Bakeoff table matches generate_bakeoff_table() exactly."""
    readme_path = Path("README.md")
    readme_content = readme_path.read_text(encoding="utf-8")
    generated_table = generate_bakeoff_table().strip()

    assert generated_table in readme_content, (
        "README.md Bake-Off table is out of sync with evals/reports/bakeoff_latest.json."
    )


def test_readme_llm_verification_table_in_sync() -> None:
    """Verify that README.md LLM verification table matches generate_llm_verification_table() exactly."""
    readme_path = Path("README.md")
    readme_content = readme_path.read_text(encoding="utf-8")
    generated_table = generate_llm_verification_table().strip()

    assert generated_table in readme_content, (
        "README.md LLM verification table is out of sync with evals/reports/bartleby_llm_verification_report.json."
    )


def test_table_numbers_grounded_in_json_reports() -> None:
    """Verify that every numeric metric rendered in the domain table matches source report fields."""
    cal_cuad = Path("evals/reports/calibration_cuad_report.json")
    cal_synth = Path("evals/reports/calibration_bartleby_report.json")

    assert cal_cuad.exists(), "calibration_cuad_report.json must exist."
    assert cal_synth.exists(), "calibration_bartleby_report.json must exist."

    with cal_cuad.open("r", encoding="utf-8") as f:
        cuad_data = json.load(f)
    with cal_synth.open("r", encoding="utf-8") as f:
        synth_data = json.load(f)

    table = generate_domain_calibration_table()

    # Extract all bolded metrics like **0.50**, **1.0000**, etc.
    bold_metrics = re.findall(r"\*\*([0-9]+\.[0-9]+)\*\*", table)
    assert len(bold_metrics) >= 6, (
        "Expected at least 6 bold numeric metrics in domain calibration table."
    )

    # Grounding check: verify values exist in JSON sources
    assert f"{synth_data['optimal_threshold']:.2f}" in bold_metrics
    assert f"{synth_data['validation_macro_f1']:.4f}" in bold_metrics
    assert f"{synth_data['auc_roc_val']:.4f}" in bold_metrics
    assert f"{cuad_data['cuad_optimal_threshold']:.2f}" in bold_metrics
    assert f"{cuad_data['validation_f1_at_optimal']:.4f}" in bold_metrics
    assert f"{cuad_data['auc_roc_val']:.4f}" in bold_metrics


def test_committed_reports_are_not_git_dirty() -> None:
    """Verify that committed benchmark reports in evals/reports/ do not have git_dirty: true."""
    report_files = [
        Path("evals/reports/bartleby_llm_verification_report.json"),
        Path("evals/reports/cuad_benchmark_latest.json"),
        Path("evals/reports/bakeoff_latest.json"),
    ]
    for rep_path in report_files:
        if rep_path.exists():
            data = json.loads(rep_path.read_text(encoding="utf-8"))
            if "git_dirty" in data:
                assert data["git_dirty"] is False, (
                    f"Report {rep_path} was generated from a dirty git tree (git_dirty: True). "
                    "Reports must be generated with a clean git working directory."
                )


def test_readme_prose_verification_metrics_in_sync() -> None:
    """Verify that prose claims in README.md match the verified figures in reports."""
    readme_path = Path("README.md")
    content = readme_path.read_text(encoding="utf-8")
    rep_path = Path("evals/reports/bartleby_llm_verification_report.json")
    assert rep_path.exists(), "bartleby_llm_verification_report.json must exist."
    rep_data = json.loads(rep_path.read_text(encoding="utf-8"))
    m3 = rep_data["modes"][2]

    # Check that prose cites the exact count of accepted hits (12 de 16) and rejected misses (8 de los 10)
    expected_hit = f"{m3['llm_accepted_on_hit']} de 16"
    expected_miss = f"{m3['llm_rejected_on_miss']} de los 10"
    assert expected_hit in content, f"Expected '{expected_hit}' in README prose."
    assert expected_miss in content, f"Expected '{expected_miss}' in README prose."

    # Check that prose cites the exact rates
    assert f"{m3['correct_abstention_rate'] * 100:.1f}%" in content
    assert f"{m3['false_abstention_rate'] * 100:.1f}%" in content
    assert f"{m3['macro_f1']:.4f}" in content


def test_readme_all_gen_blocks_in_sync() -> None:
    """Verify that every GEN block in README.md matches its generator byte-for-byte."""
    from evals.generate_readme_tables import GENERATORS

    readme_path = Path("README.md")
    content = readme_path.read_text(encoding="utf-8")

    for name, gen_fn in GENERATORS.items():
        pattern = re.compile(
            rf"<!--\s*GEN:{name}\s*-->([\s\S]*?)<!--\s*/GEN\s*-->"
        )
        match = pattern.search(content)
        assert match is not None, f"Block <!-- GEN:{name} --> missing from README.md"

        actual_block = match.group(1).strip()
        expected_block = gen_fn().strip()
        assert actual_block == expected_block, (
            f"Block <!-- GEN:{name} --> in README.md is out of sync with generator. Run 'python evals/generate_readme_tables.py --sync'."
        )


def test_readme_prose_claim_audit() -> None:
    """Verify that no ungrounded orphan numbers exist in README.md prose outside GEN blocks."""
    from tools.claim_audit import audit_markdown_file, load_allowlist

    allowlist_path = Path("tools/claim_allowlist.yaml")
    allowed_numbers, allowed_regexes = load_allowlist(allowlist_path)
    violations = audit_markdown_file(Path("README.md"), allowed_numbers, allowed_regexes)

    assert len(violations) == 0, (
        f"Found {len(violations)} ungrounded numbers in README.md prose. Violations: {violations[:5]}"
    )

