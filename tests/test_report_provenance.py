"""Regression tests for evaluation report provenance (Fase 0 - Evidence Gate).

Enforces:
(a) Timestamps have non-zero microseconds (no manual simulation).
(b) git_commit_sha exists in git commit history.
(c) git_dirty is False on committed reports.
(d) command appears in evals/COMMANDS.md registry.
(e) A timestamped copy exists alongside *_latest.json.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

CANONICAL_REPORTS = [
    "evals/reports/bakeoff_latest.json",
    "evals/reports/cuad_benchmark_latest.json",
    "evals/reports/bartleby_llm_verification_report.json",
    "evals/reports/out_of_sample_cuad_report.json",
]


def test_canonical_reports_have_valid_microseconds() -> None:
    for rep in CANONICAL_REPORTS:
        p = Path(rep)
        assert p.exists(), f"Report {rep} not found."
        data = json.loads(p.read_text(encoding="utf-8"))
        ts = data.get("timestamp") or data.get("report", {}).get("timestamp") or data.get("provenance", {}).get("timestamp", "")
        assert ts, f"No timestamp found in {rep}"

        # Match ISO timestamps with microsecond precision: e.g. .123456
        match = re.search(r"\.(\d{6})", ts)
        assert match is not None, f"Report {rep} timestamp '{ts}' lacks 6-digit microsecond precision."
        assert match.group(1) != "000000", f"Report {rep} has simulated zero microseconds '.000000'."


def test_canonical_reports_git_commit_exists() -> None:
    for rep in CANONICAL_REPORTS:
        p = Path(rep)
        assert p.exists(), f"Report {rep} not found."
        data = json.loads(p.read_text(encoding="utf-8"))
        sha = data.get("git_commit_sha") or data.get("report", {}).get("git_commit_sha") or data.get("provenance", {}).get("git_commit_sha", "")
        assert sha, f"No git_commit_sha found in {rep}"
        assert re.match(r"^[0-9a-f]{7,40}$", sha), f"Invalid git_commit_sha format '{sha}' in {rep}"

        # Verify commit exists in git tree
        res = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            capture_output=True,
            check=False,
        )
        assert res.returncode == 0, f"git_commit_sha '{sha}' in {rep} does not exist in git history."


def test_canonical_reports_are_not_dirty() -> None:
    for rep in CANONICAL_REPORTS:
        p = Path(rep)
        assert p.exists(), f"Report {rep} not found."
        data = json.loads(p.read_text(encoding="utf-8"))
        dirty = data.get("git_dirty")
        if dirty is None:
            dirty = data.get("report", {}).get("git_dirty")
        if dirty is None:
            dirty = data.get("provenance", {}).get("git_dirty")

        assert dirty is False, f"Report {rep} has git_dirty != False (found: {dirty}). Reports must be run on clean tree."


def test_canonical_commands_registered_in_commands_md() -> None:
    commands_md = Path("evals/COMMANDS.md")
    assert commands_md.exists(), "evals/COMMANDS.md must exist."
    registry_text = commands_md.read_text(encoding="utf-8")

    for rep in CANONICAL_REPORTS:
        p = Path(rep)
        assert p.exists(), f"Report {rep} not found."
        data = json.loads(p.read_text(encoding="utf-8"))
        cmd = data.get("command") or data.get("report", {}).get("command") or data.get("provenance", {}).get("command", "")
        assert cmd, f"No command found in {rep}"

        # The base command script name or key flags must be present in COMMANDS.md
        cmd_base = cmd.split("/")[-1].strip()
        assert (cmd_base in registry_text) or (p.name in registry_text), (
            f"Command '{cmd}' from {rep} is not registered in evals/COMMANDS.md."
        )


def test_latest_reports_have_timestamped_copies() -> None:
    latest_reports = [
        Path("evals/reports/bakeoff_latest.json"),
        Path("evals/reports/cuad_benchmark_latest.json"),
    ]
    for p in latest_reports:
        assert p.exists(), f"Latest report {p} must exist."
        prefix = p.name.replace("_latest.json", "")
        # Find matching timestamped sibling
        siblings = list(p.parent.glob(f"{prefix}*.json"))
        timestamped_siblings = [s for s in siblings if s.name != p.name and re.search(r"\d{8}T\d{6}Z", s.name)]
        assert len(timestamped_siblings) >= 1, (
            f"No timestamped copy found alongside {p.name}. Found: {[s.name for s in siblings]}"
        )
