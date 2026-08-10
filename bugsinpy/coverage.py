"""
Coverage collection using coverage.py inside WSL.

Builds a sparse test × code-entity (function/file) coverage matrix.
Function-level coverage is preferred; file-level is the fallback.

All coverage data is PRE-EPISODE information: it is computed once
during data collection and stored in the experiment record. It does NOT
change during policy evaluation.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from bugsinpy.checkout import wsl, LINUX_BUGSINPY, LINUX_PROJECT_ROOT


# ─── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CoverageResult:
    """Coverage collected for one test."""
    test_id: str
    covered_files: list[str] = field(default_factory=list)
    covered_functions: list[str] = field(default_factory=list)
    coverage_level: str = "none"   # "function", "file", or "none"
    error: Optional[str] = None


# ─── Coverage collection ───────────────────────────────────────────────────────

def install_coverage(project_workspace: str, python_exe: str) -> bool:
    """Ensure coverage.py is installed in the project venv."""
    result = wsl(
        f"{python_exe} -m pip install coverage pytest-cov --quiet",
        cwd=project_workspace,
        timeout=60,
    )
    return result.ok


def collect_suite_coverage(
    test_ids: list[str],
    project_workspace: str,
    python_exe: str,
    source_package: Optional[str] = None,
    timeout: int = 1800,
) -> dict[str, list[str]]:
    """
    Run the entire suite under coverage.py with dynamic contexts to trace
    exactly which tests execute which lines of code in a single invocation.
    
    Returns:
        Dict mapping test_id -> list of covered entities.
    """
    args_file = f"/tmp/pytest_args_{abs(hash(project_workspace))}.txt"
    json_out = f"/tmp/coverage_{abs(hash(project_workspace))}.json"
    
    # Write test IDs to an args file to avoid shell length limits
    test_list_content = "\n".join(test_ids)
    with open(args_file, "w", encoding="utf-8") as f:
        f.write(test_list_content)
    
    # Write a small python runner to bypass pytest's lack of @file support and shell length limits
    runner_file = f"{project_workspace}/_pytest_runner.py"
    runner_script = f"""import sys
import pytest
with open("{args_file}", "r", encoding="utf-8") as f:
    test_ids = [line.strip() for line in f if line.strip()]
args = ["--cov={source_package or '.'}", "--cov-context=test", "--cov-branch", "--no-header", "-q"] + test_ids
sys.exit(pytest.main(args))
"""
    with open(runner_file, "w", encoding="utf-8") as f:
        f.write(runner_script)

    cov_run_cmd = f"{python_exe} {runner_file} 2>&1"
    cov_json_cmd = f"{python_exe} -m coverage json --show-contexts -o {json_out} --quiet 2>&1"
    
    # Run coverage
    run_result = wsl(cov_run_cmd, cwd=project_workspace, timeout=timeout)
    # Generate JSON
    json_result = wsl(cov_json_cmd, cwd=project_workspace, timeout=120)
    
    if not json_result.ok:
        print(f"WARNING: coverage json failed. Stdout: {json_result.stdout}, Stderr: {json_result.stderr}")
        return {}
        
    parse_result = wsl(f"cat {json_out} 2>/dev/null || echo 'MISSING'")
    if "MISSING" in parse_result.stdout or not parse_result.ok:
        return {}
        
    try:
        data = json.loads(parse_result.stdout)
    except json.JSONDecodeError:
        return {}
        
    # Build coverage matrix directly from JSON contexts
    matrix: dict[str, set[str]] = {tid: set() for tid in test_ids}
    
    files_data = data.get("files", {})
    for filepath, fdata in files_data.items():
        clean_path = filepath.lstrip("./")
        if not clean_path.endswith(".py"):
            continue
            
        contexts_dict = fdata.get("contexts", {})
        if not contexts_dict:
            continue
            
        # Contexts map line_number -> list of test contexts (e.g. "tests/test_x.py::test_y")
        # We want to map each context to the file it touched.
        # Ideally we map to functions, but file-level `module.*` is fully compatible 
        # with our ast_analysis pipeline which refines them.
        module_name = clean_path.replace("/", ".").removesuffix(".py")
        fallback_func = f"{module_name}.*"
        
        for line_str, context_list in contexts_dict.items():
            for ctx in context_list:
                # coverage.py contexts look like: "test_file.py::test_func|run" or just "test_file.py::test_func"
                # Strip any trailing coverage flags separated by |
                clean_ctx = ctx.split("|")[0] if "|" in ctx else ctx
                if not clean_ctx:
                    continue
                    
                # We need to map clean_ctx back to our original test_ids
                # If they match exactly, great. If not, fallback to suffix match.
                matched_tid = None
                if clean_ctx in matrix:
                    matched_tid = clean_ctx
                else:
                    for tid in test_ids:
                        if clean_ctx.endswith(tid) or tid.endswith(clean_ctx):
                            matched_tid = tid
                            break
                            
                if matched_tid:
                    matrix[matched_tid].add(fallback_func)
                    
    return {k: sorted(list(v)) for k, v in matrix.items()}


def _extract_functions_from_coverage(
    filepath: str,
    fdata: dict,
) -> list[str]:
    """
    Approximate function-level coverage using executed lines.

    Uses the 'executed_lines' from coverage.py JSON to determine
    which line ranges in the source file were hit, then maps those
    lines to AST-identified function definitions.

    Note: This is an approximation. True function-level coverage
    requires coverage.py context features or a more sophisticated
    analysis pipeline.
    """
    executed_lines: set[int] = set(fdata.get("executed_lines", []))
    if not executed_lines:
        return []

    # We use a pre-parsed function map if available (populated by ast_analysis.py)
    # Here we return the file-level entity as a fallback
    # The analysis/ast_analysis.py module performs the precise mapping
    module_name = filepath.replace("/", ".").removesuffix(".py")
    return [f"{module_name}.*"]  # Placeholder — will be refined by ast_analysis


def _fallback_file_coverage(
    test_id: str,
    project_workspace: str,
    python_exe: str,
    json_out: str,
) -> CoverageResult:
    """Collect file-level coverage as fallback using coverage report."""
    report_cmd = f"{python_exe} -m coverage report 2>/dev/null"
    result = wsl(report_cmd, cwd=project_workspace, timeout=30)

    covered_files: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].endswith(".py"):
            # Format: filename statements missing coverage%
            try:
                stmts = int(parts[1])
                miss = int(parts[2])
                if stmts - miss > 0:  # At least one line executed
                    covered_files.append(parts[0].lstrip("./"))
            except ValueError:
                pass

    if covered_files:
        return CoverageResult(
            test_id=test_id,
            covered_files=covered_files,
            coverage_level="file",
        )
    return CoverageResult(test_id=test_id, error="Coverage fallback also failed")
