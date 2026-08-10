"""
BugsInPy interface layer — metadata parsing.

Parses bug.info and project.info files from the BugsInPy repository.
All fields used purely as ground-truth labels are clearly marked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BugInfo:
    """
    Contents of a BugsInPy bug.info file.

    IMPORTANT: `test_cases` contains the known triggering test IDs.
    These are GROUND-TRUTH LABELS only and must NEVER be exposed to
    any policy during an episode.
    """
    project: str
    bug_id: str
    python_version: str
    buggy_commit_id: str
    fixed_commit_id: str
    # These are the triggering test files — ground truth, not for policy use
    test_file: str = ""
    test_cases: list[str] = field(default_factory=list)


@dataclass
class ProjectInfo:
    """Contents of a BugsInPy project.info file."""
    project: str
    github_url: str
    status: str = "OK"
    cause: str = "N.A."


# ─── Parsers ───────────────────────────────────────────────────────────────────

def _parse_kv(text: str) -> dict[str, str]:
    """Parse simple key=\"value\" shell-style files."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r'^(\w+)="([^"]*)"$', line)
        if m:
            result[m.group(1)] = m.group(2)
    return result


def parse_bug_info(path: Path, project: str, bug_id: str) -> BugInfo:
    """
    Parse a bug.info file.

    Args:
        path: Path to the bug.info file.
        project: Project name (for identification).
        bug_id: Bug identifier string.

    Returns:
        BugInfo dataclass.
    """
    text = path.read_text(encoding="utf-8")
    kv = _parse_kv(text)

    # test_cases may be a single string with spaces/newlines separating items
    raw_cases = kv.get("test_cases", "")
    test_cases = [tc.strip() for tc in raw_cases.split() if tc.strip()]

    return BugInfo(
        project=project,
        bug_id=bug_id,
        python_version=kv.get("python_version", ""),
        buggy_commit_id=kv.get("buggy_commit_id", ""),
        fixed_commit_id=kv.get("fixed_commit_id", ""),
        test_file=kv.get("test_file", ""),
        test_cases=test_cases,
    )


def parse_project_info(path: Path, project: str) -> ProjectInfo:
    """Parse a project.info file."""
    text = path.read_text(encoding="utf-8")
    kv = _parse_kv(text)
    return ProjectInfo(
        project=project,
        github_url=kv.get("github_url", ""),
        status=kv.get("status", "OK"),
        cause=kv.get("cause", "N.A."),
    )


def list_bugs(bugsinpy_root: Path, project: str) -> list[str]:
    """
    List all bug IDs for a project.

    Args:
        bugsinpy_root: Root of the BugsInPy repository clone.
        project: Project name.

    Returns:
        Sorted list of bug ID strings (e.g. ["1", "2", "3"]).
    """
    bugs_dir = bugsinpy_root / "projects" / project / "bugs"
    if not bugs_dir.exists():
        return []
    ids = [d.name for d in bugs_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    return sorted(ids, key=int)


def list_projects(bugsinpy_root: Path) -> list[str]:
    """List all project names in a BugsInPy clone."""
    projects_dir = bugsinpy_root / "projects"
    return sorted(d.name for d in projects_dir.iterdir() if d.is_dir())
