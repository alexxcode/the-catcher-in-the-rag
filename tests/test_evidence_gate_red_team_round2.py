"""Red-team round 2: attacks the evidence gate still does NOT catch.

Same convention as round 1: each known gap was xfail(strict=True). The gate was fixed,
turning tests red, and the markers were removed. Do NOT delete these tests or weaken their assertions.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_IGNORE = shutil.ignore_patterns(
    ".venv", ".git", "data", ".cache", "__pycache__", ".pytest_cache", ".ruff_cache", ".terraform"
)
LLM_REPORT = "evals/reports/bartleby_llm_verification_report.json"
OOS_REPORT = "evals/reports/out_of_sample_cuad_report.json"


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    dst = tmp_path / "repo"
    shutil.copytree(ROOT, dst, ignore=_IGNORE)
    return dst


def _run(repo: Path, *args: str) -> int:
    return subprocess.run(
        [sys.executable, *args], cwd=repo, capture_output=True, text=True, timeout=300
    ).returncode


def gate_passes(repo: Path) -> bool:
    if _run(repo, "tools/claim_audit.py") != 0:
        return False
    if _run(repo, "evals/repro_check.py") != 0:
        return False
    return (
        _run(repo, "-m", "pytest", "tests/test_readme_sync.py", "-q", "--no-cov", "-p", "no:cacheprovider") == 0
    )


def _append(path: Path, text: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


def test_control_untouched_tree_passes(repo: Path) -> None:
    """Guards the red-team suite itself: if this fails, every xfail below is meaningless."""
    assert gate_passes(repo)


def test_forged_out_of_sample_report_is_caught(repo: Path) -> None:
    p = repo / OOS_REPORT
    d = json.loads(p.read_text(encoding="utf-8"))
    pos = d["metrics"]["false_abstention_on_positives"]
    pos["count"], pos["rate"], pos["wilson_ci95"] = 30, round(30 / pos["total"], 4), [0.14, 0.26]
    d["metrics"]["macro_f1"] = 0.86
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    _run(repo, "evals/generate_readme_tables.py", "--sync")
    assert not gate_passes(repo)


@pytest.mark.parametrize("phrase", ["tres cuartos de los casos", "la mitad de los contratos", "x3 mas rapido", "el doble de precision"])
def test_fraction_words_and_multipliers_are_caught(repo: Path, phrase: str) -> None:
    _append(repo / "README.md", f"\nLa mejora es de {phrase} en produccion.\n")
    assert not gate_passes(repo)


def test_number_in_exempt_progress_doc_is_caught(repo: Path) -> None:
    (repo / "StateOfCatcher.md").write_text("Precision 98.7% en produccion.\n", encoding="utf-8")
    assert not gate_passes(repo)
