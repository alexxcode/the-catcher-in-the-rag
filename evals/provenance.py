"""Evaluation Provenance Utility.

Captures execution provenance metadata:
- Git commit SHA and branch
- Exact CLI command invocation
- Python environment version
- Model and embedding provider identifiers
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def get_git_commit_sha() -> str:
    """Retrieve the current Git commit SHA."""
    env_sha = os.getenv("GIT_COMMIT_SHA") or os.getenv("GITHUB_SHA")
    if env_sha:
        return env_sha.strip()

    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return "unversioned"


def get_git_branch() -> str:
    """Retrieve the current Git branch name."""
    env_branch = os.getenv("GIT_BRANCH") or os.getenv("GITHUB_REF_NAME")
    if env_branch:
        return env_branch.strip()

    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_git_dirty() -> bool:
    """Check if the Git working tree has uncommitted changes."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return bool(res.stdout.strip())
    except Exception:
        return False


def get_cli_command() -> str:
    """Retrieve sanitized command-line invocation without host-specific absolute paths."""
    args = list(sys.argv)
    if args:
        # Sanitize absolute script path to standard invocation format
        base = Path(args[0]).name
        if base == "run.py" and "evals" in args[0].replace("\\", "/"):
            args[0] = "python -m evals.run"
        elif base == "__main__.py":
            args[0] = "python -m evals"
        else:
            args[0] = base
    return " ".join(args)


def get_provenance_metadata(
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile comprehensive provenance metadata for benchmark reports."""
    meta: dict[str, Any] = {
        "git_commit_sha": get_git_commit_sha(),
        "git_branch": get_git_branch(),
        "git_dirty": get_git_dirty(),
        "command": get_cli_command(),
        "python_version": sys.version.split()[0],
    }
    if extra:
        meta.update(extra)
    return meta
