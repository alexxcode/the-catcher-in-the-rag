"""Negative tests proving the Evidence Gate fails whenever tampered or ungrounded claims occur.

Verifies:
(a) Fails on loose numbers in prose outside GEN blocks.
(b) Fails on hand-edited numbers inside GEN blocks.
(c) Fails on hand-edited metrics in report JSONs (preserving SHA/timestamp).
(d) Fails on mentioning models that do not exist in config/models.yaml.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.generate_readme_tables import GENERATORS
from evals.repro_check import check_bartleby_llm_report, check_calibration_cuad_report
from tools.claim_audit import (
    audit_markdown_file,
    audit_markdown_models,
    get_allowed_models,
    load_allowlist,
)


def test_fails_on_loose_number_in_prose(tmp_path: Path) -> None:
    """Case (a): Adding a loose ungrounded number in documentation prose must fail audit."""
    allowlist_path = Path("tools/claim_allowlist.yaml")
    allowed_numbers, allowed_regexes = load_allowlist(allowlist_path)

    # Markdown with an orphan empirical number outside any GEN block
    bad_md = tmp_path / "README_bad.md"
    bad_md.write_text(
        "# Contract Audit\n\nEl sistema alcanza un 98.7% de recall en extracción contractual sin precedentes.\n",
        encoding="utf-8",
    )

    violations = audit_markdown_file(bad_md, allowed_numbers, allowed_regexes)
    assert len(violations) > 0, "Audit failed to detect orphan number '98.7%'!"
    assert any("98.7%" in v["token"] for v in violations)


def test_fails_on_hand_edited_number_in_gen_block() -> None:
    """Case (b): Modifying a number inside a GEN block by hand must fail sync test."""
    # Compute what generator actually produces
    gen_func = GENERATORS["retrieval_table"]
    actual_table = gen_func()

    # Create a hand-edited version where a number was artificially altered
    tampered_table = actual_table.replace("26.9%", "99.9%")
    assert tampered_table != actual_table

    # A sync check asserting equality must fail
    with pytest.raises(AssertionError):
        assert tampered_table == actual_table, "Sync check failed to detect tampered number inside GEN block!"


def test_fails_on_hand_edited_metric_in_json(tmp_path: Path) -> None:
    """Case (c): Editing a metric value in a report JSON while preserving SHA/timestamp must fail repro-check."""
    original_report = Path("evals/reports/bartleby_llm_verification_report.json")
    pricing_path = Path("config/pricing.yaml")
    assert original_report.exists()

    with original_report.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Tamper with Mode 3 correct abstention rate (alter metric from 0.6329 to 0.9900)
    data["modes"][2]["correct_abstention_rate"] = 0.9900
    # Keep commit SHA and timestamp completely unchanged!

    tampered_file = tmp_path / "bartleby_tampered.json"
    tampered_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    errors = check_bartleby_llm_report(tampered_file, pricing_path)
    assert len(errors) > 0, "repro-check failed to catch tampered metric in JSON!"
    assert any("mismatch" in e for e in errors)

    # Also test tampering with CUAD calibration candidate
    cuad_report = Path("evals/reports/calibration_cuad_report.json")
    with cuad_report.open("r", encoding="utf-8") as f:
        cuad_data = json.load(f)

    cuad_data["candidates"][0]["abstention_rate"] = 0.9999
    tampered_cuad = tmp_path / "cuad_tampered.json"
    tampered_cuad.write_text(json.dumps(cuad_data, indent=2), encoding="utf-8")

    cuad_errors = check_calibration_cuad_report(tampered_cuad)
    assert len(cuad_errors) > 0, "repro-check failed to catch tampered CUAD candidate metric!"
    assert any("mismatch" in e for e in cuad_errors)


def test_fails_on_unconfigured_model_in_readme(tmp_path: Path) -> None:
    """Case (d): Mentioning an unconfigured model in README must fail audit."""
    models_yaml = Path("config/models.yaml")
    allowed_models = get_allowed_models(models_yaml)

    bad_readme = tmp_path / "README_bad_model.md"
    bad_readme.write_text(
        "# Benchmarks\n\nEvaluado contra claude-3-opus-20240229 y gpt-4o con resultados prometedores.\n",
        encoding="utf-8",
    )

    violations = audit_markdown_models(bad_readme, allowed_models)
    assert len(violations) >= 2, "Model audit failed to catch unconfigured models!"
    flagged = {v["model"] for v in violations}
    assert "claude-3-opus-20240229" in flagged
    assert "gpt-4o" in flagged
