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


def get_cli_command() -> str:
    """Retrieve the exact command-line invocation."""
    return " ".join(sys.argv)


def get_provenance_metadata(
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile comprehensive provenance metadata for benchmark reports."""
    meta: dict[str, Any] = {
        "git_commit_sha": get_git_commit_sha(),
        "git_branch": get_git_branch(),
        "command": get_cli_command(),
        "python_version": sys.version.split()[0],
    }
    if extra:
        meta.update(extra)
    return meta
