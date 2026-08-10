"""
Git diff analysis and Python AST-based change extraction.

Identifies changed files, line ranges, functions, and classes between
the buggy and fixed commits of a BugsInPy bug.

All information derived here is STATIC — it is computed from the
git diff and does not depend on test execution outcomes.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from bugsinpy.checkout import wsl, LINUX_BUGSINPY


# ─── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class LineRange:
    """A contiguous range of changed lines in one file."""
    start: int
    end: int

    def contains(self, line: int) -> bool:
        return self.start <= line <= self.end

    def overlaps(self, other: "LineRange") -> bool:
        return self.start <= other.end and other.start <= self.end


@dataclass
class ChangedEntity:
    """A source code entity (function or class) that intersects a change."""
    qualified_name: str   # e.g. "pandas.core.arrays.categorical.CategoricalDtype.__init__"
    file_path: str
    start_line: int
    end_line: int
    entity_type: str      # "function" or "class"


@dataclass
class DiffResult:
    """Structured output of analysing the diff between buggy and fixed commits."""
    buggy_commit: str
    fixed_commit: str
    changed_files: list[str] = field(default_factory=list)
    added_lines: dict[str, list[int]] = field(default_factory=dict)
    deleted_lines: dict[str, list[int]] = field(default_factory=dict)
    modified_ranges: dict[str, list[LineRange]] = field(default_factory=dict)
    changed_entities: list[ChangedEntity] = field(default_factory=list)
    error: Optional[str] = None


# ─── Git diff parsing ──────────────────────────────────────────────────────────

_DIFF_FILE_RE = re.compile(r"^diff --git a/(.+) b/(.+)$")
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def extract_diff(
    project_workspace: str,
    buggy_commit: str,
    fixed_commit: str,
) -> DiffResult:
    """
    Extract structured diff information between the buggy and fixed commits.

    Runs `git diff` inside WSL on the checked-out project.

    Args:
        project_workspace: Linux path to the checked-out project (either version).
        buggy_commit: Short or full buggy commit hash.
        fixed_commit: Short or full fixed commit hash.

    Returns:
        DiffResult containing changed files, line ranges, and identified entities.
    """
    cmd = f"git diff {buggy_commit} {fixed_commit} -- '*.py' 2>&1"
    result = wsl(cmd, cwd=project_workspace, timeout=60)

    if not result.ok and result.returncode not in (0, 1):
        return DiffResult(
            buggy_commit=buggy_commit,
            fixed_commit=fixed_commit,
            error=f"git diff failed: {result.stderr[:300]}",
        )

    return _parse_unified_diff(result.stdout, buggy_commit, fixed_commit)


def _parse_unified_diff(
    diff_text: str,
    buggy_commit: str,
    fixed_commit: str,
) -> DiffResult:
    """Parse a unified diff string into a DiffResult."""
    dr = DiffResult(buggy_commit=buggy_commit, fixed_commit=fixed_commit)

    current_file: Optional[str] = None
    current_line_new = 0

    for line in diff_text.splitlines():
        # New file in diff
        file_match = _DIFF_FILE_RE.match(line)
        if file_match:
            current_file = file_match.group(2)  # b/... side (fixed version)
            if current_file.startswith("b/"):
                current_file = current_file[2:]
            if current_file not in dr.changed_files:
                dr.changed_files.append(current_file)
                dr.added_lines[current_file] = []
                dr.deleted_lines[current_file] = []
                dr.modified_ranges[current_file] = []
            continue

        # Hunk header
        hunk_match = _HUNK_RE.match(line)
        if hunk_match and current_file:
            current_line_new = int(hunk_match.group(3))
            hunk_new_count = int(hunk_match.group(4) or 1)
            dr.modified_ranges[current_file].append(
                LineRange(current_line_new, current_line_new + hunk_new_count - 1)
            )
            continue

        if current_file is None:
            continue

        if line.startswith("+") and not line.startswith("+++"):
            dr.added_lines[current_file].append(current_line_new)
            current_line_new += 1
        elif line.startswith("-") and not line.startswith("---"):
            dr.deleted_lines[current_file].append(current_line_new)
            # Deleted lines don't increment new-file line number
        elif not line.startswith("\\"):
            current_line_new += 1

    return dr


# ─── AST analysis ──────────────────────────────────────────────────────────────

def identify_changed_entities(
    diff_result: DiffResult,
    project_workspace_linux: str,
) -> list[ChangedEntity]:
    """
    Use Python AST to identify functions/classes that contain changed lines.

    Reads source files from the FIXED version (available in project_workspace).
    This is safe because:
    - We are reading the AST structure (function/class boundaries), not logic
    - The policy never sees this information labelled as "what changed"
    - Changed entity names are used as coverage features only

    Args:
        diff_result: Parsed diff result.
        project_workspace_linux: Linux path to the fixed version of the project.

    Returns:
        List of ChangedEntity objects.
    """
    entities: list[ChangedEntity] = []

    for filepath in diff_result.changed_files:
        if not filepath.endswith(".py"):
            continue

        ranges = diff_result.modified_ranges.get(filepath, [])
        if not ranges:
            continue

        # Read the source file from WSL
        full_linux_path = f"{project_workspace_linux}/{filepath}"
        read_result = wsl(f"cat '{full_linux_path}' 2>/dev/null || echo '__MISSING__'")
        if "__MISSING__" in read_result.stdout or not read_result.ok:
            continue

        source = read_result.stdout
        file_entities = _extract_entities_from_source(
            source=source,
            filepath=filepath,
            changed_ranges=ranges,
        )
        entities.extend(file_entities)

    return entities


def _extract_entities_from_source(
    source: str,
    filepath: str,
    changed_ranges: list[LineRange],
) -> list[ChangedEntity]:
    """Parse AST and find functions/classes that overlap changed line ranges."""
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return []

    # Build module prefix from file path
    module_prefix = filepath.replace("/", ".").removesuffix(".py")

    entities: list[ChangedEntity] = []

    class EntityVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope_stack: list[str] = []

        def _check_and_add(
            self,
            node: ast.AST,
            entity_type: str,
            name: str,
        ) -> None:
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            node_range = LineRange(start, end)
            for cr in changed_ranges:
                if node_range.overlaps(cr):
                    scope = ".".join(self.scope_stack)
                    qualified = f"{module_prefix}.{scope}.{name}".replace("..", ".")
                    qualified = qualified.strip(".")
                    entities.append(ChangedEntity(
                        qualified_name=qualified,
                        file_path=filepath,
                        start_line=start,
                        end_line=end,
                        entity_type=entity_type,
                    ))
                    break

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._check_and_add(node, "function", node.name)
            self.scope_stack.append(node.name)
            self.generic_visit(node)
            self.scope_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._check_and_add(node, "class", node.name)
            self.scope_stack.append(node.name)
            self.generic_visit(node)
            self.scope_stack.pop()

    visitor = EntityVisitor()
    visitor.visit(tree)
    return entities
