"""Lineage metadata attached to every run and every registered model version."""

from __future__ import annotations

import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version

from credit_default.config import DATASET_DOI, DATASET_LICENSE, DATASET_NAME, DATASET_URL


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def git_info() -> dict[str, str]:
    """Commit the code was trained from. CI and containers pass GIT_COMMIT explicitly."""
    commit = os.environ.get("GIT_COMMIT") or _git("rev-parse", "HEAD") or "unknown"
    branch = os.environ.get("GIT_BRANCH") or _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    status = _git("status", "--porcelain", "--untracked-files=no")
    dirty = "unknown" if commit == "unknown" else str(bool(status)).lower()
    return {"git_commit": commit, "git_branch": branch, "git_dirty": dirty}


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def lineage_tags(dataset_hash: str) -> dict[str, str]:
    return {
        "dataset_name": DATASET_NAME,
        "dataset_source": DATASET_URL,
        "dataset_doi": DATASET_DOI,
        "dataset_license": DATASET_LICENSE,
        "dataset_sha256": dataset_hash,
        **git_info(),
        "python_version": platform.python_version(),
        "sklearn_version": _pkg_version("scikit-learn"),
        "lightgbm_version": _pkg_version("lightgbm"),
        "package_version": _pkg_version("credit-default"),
    }
