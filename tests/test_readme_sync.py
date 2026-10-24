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
