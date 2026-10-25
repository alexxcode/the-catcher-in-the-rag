"""Red-team tests for the evidence gate.

Each test tampers with a COPY of the repo and asserts the gate catches it.
Attacks the gate does NOT yet catch are marked xfail(strict=True): they keep CI
green today and turn RED the moment the gate is fixed, forcing removal of the
xfail marker. Do NOT delete these tests and do NOT weaken the assertions.
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
REPORT = "evals/reports/bartleby_llm_verification_report.json"


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
    """True if EVERY evidence-gate component accepts the (possibly tampered) tree."""
    if _run(repo, "tools/claim_audit.py") != 0:
        return False
    if _run(repo, "evals/repro_check.py") != 0:
        return False
    return _run(repo, "-m", "pytest", "tests/test_readme_sync.py", "-q", "--no-cov", "-p", "no:cacheprovider") == 0


def _append(path: Path, text: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


def test_control_untouched_tree_passes(repo: Path) -> None:
    assert gate_passes(repo)


# ---- attacks the gate already catches (must stay caught) -------------------------------------
def test_loose_digit_in_readme_prose_is_caught(repo: Path) -> None:
    _append(repo / "README.md", "\nEl sistema alcanza 98.7% de precision en produccion.\n")
    assert not gate_passes(repo)


def test_single_metric_edit_in_json_is_caught(repo: Path) -> None:
    p = repo / REPORT
    d = json.loads(p.read_text(encoding="utf-8"))
    d["modes"][2]["correct_abstention_rate"] = 0.99
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    assert not gate_passes(repo)


# ---- previously known gaps (now fixed and actively enforced) -------------------------
def test_number_in_words_is_caught(repo: Path) -> None:
    _append(repo / "README.md", "\nLa precision es noventa y ocho por ciento en produccion.\n")
    assert not gate_passes(repo)


def test_loose_number_in_root_markdown_is_caught(repo: Path) -> None:
    (repo / "SPEC.md").write_text("Precision 98.7% en produccion.\n", encoding="utf-8")
    assert not gate_passes(repo)


def test_loose_number_in_docs_folder_is_caught(repo: Path) -> None:
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs" / "claims.md").write_text("Precision 98.7% en produccion.\n", encoding="utf-8")
    assert not gate_passes(repo)


def test_numeric_default_hidden_in_generator_is_caught(repo: Path) -> None:
    _append(
        repo / "evals" / "generate_readme_tables.py",
        "\n\ndef _hidden_default():\n    return {}.get('missing_metric', 0.987)\n",
    )
    assert not gate_passes(repo)


def test_consistent_forgery_of_report_counts_is_caught(repo: Path) -> None:
    p = repo / REPORT
    d = json.loads(p.read_text(encoding="utf-8"))
    m = d["modes"][2]
    ab, pr, fa = m["total_absences"], m["total_presences"], m["false_abstentions"]
    ca = 70
    m["correct_abstentions"], m["correct_abstention_rate"] = ca, round(ca / ab, 4)
    pa, ra = ca / (ca + fa), ca / ab
    f1a = 2 * pa * ra / (pa + ra)
    tp, fp, fn = pr - fa, ab - ca, fa
    pp, rp = tp / (tp + fp), tp / (tp + fn)
    f1p = 2 * pp * rp / (pp + rp)
    m.update(
        precision_abstain=round(pa, 4), recall_abstain=round(ra, 4),
        f1_abstain=round(f1a, 4), f1_presence=round(f1p, 4), macro_f1=round((f1a + f1p) / 2, 4),
    )
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    _run(repo, "evals/generate_readme_tables.py", "--sync")  # attacker regenerates README tables too
    assert not gate_passes(repo)
