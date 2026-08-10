"""
Test runner — executes individual tests in WSL and collects timing and outcomes.

Uses coverage.py (installed in the project's venv) to optionally collect
function-level coverage during the same run.

Critical leakage rule: This module may only write to GROUND_TRUTH_LABELS fields.
The policy receives only pre-episode static features.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from bugsinpy.checkout import wsl, CommandResult, LINUX_BUGSINPY


# ─── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TestOutcome:
    """
    Result of executing a single test.

    This is a GROUND_TRUTH_LABEL container. Its fields must not be
    exposed to any policy before the test is actually selected and executed.
    """
    test_id: str
    outcome: str          # "PASS", "FAIL", "ERROR", "TIMEOUT", "SKIP"
    runtime_seconds: float
    stdout: str = ""
    stderr: str = ""


@dataclass
class SuiteRun:
    """Results from running an entire test suite on one bug version."""
    project: str
    bug_id: str
    version: int          # 0 = buggy, 1 = fixed
    outcomes: list[TestOutcome] = field(default_factory=list)
    infrastructure_error: Optional[str] = None
    total_elapsed: float = 0.0


# ─── Test discovery ────────────────────────────────────────────────────────────

def discover_tests(project_workspace: str, python_exe: str) -> list[str]:
    """
    Discover all test IDs in the project using pytest's collection mode.

    Always force-installs a modern pytest (>=6.0) into the project venv so that
    --rootdir is available. This overrides any old pinned pytest in the project's
    requirements.txt; we only need pytest for *collection*, not for running.

    Args:
        project_workspace: Linux path to the checked-out project.
        python_exe: Linux path to the Python executable in the project venv.

    Returns:
        List of test IDs (e.g. "tests/test_foo.py::TestFoo::test_bar").
    """
    # Force-install a modern pytest + pytest-cov — upgrades any stale pinned version.
    # pytest-cov needed because many projects have `addopts = --cov=...` in setup.cfg.
    wsl(
        f"~/.local/bin/uv pip install --python {python_exe} 'pytest>=6.0' pytest-cov --upgrade -q "
        f"2>/dev/null",
        cwd=project_workspace, timeout=120,
    )

    # --rootdir=. anchors pytest to the project dir.
    # --override-ini=addopts= strips any project-level addopts (e.g. --cov flags)
    # that would require additional plugins not installed in our collection env.
    # -p no:cacheprovider avoids writing .pytest_cache to the workspace.
    cmd = (
        f"{python_exe} -m pytest --collect-only -q "
        f"--rootdir=. "
        f"--override-ini=addopts= "
        f"-p no:cacheprovider "
        f"2>/dev/null"
    )
    result = wsl(cmd, cwd=project_workspace, timeout=120)

    test_ids: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        # pytest --collect-only -q format: "tests/foo.py::TestClass::test_method"
        if "::" in line and not line.startswith(("ERRORS", "warnings", "no tests")):
            test_ids.append(line)
    return test_ids



# ─── Single test execution ─────────────────────────────────────────────────────

_FAIL_PATTERNS = re.compile(
    r"(FAILED|ERROR|AssertionError|Traceback|Error:|ERRORS)", re.IGNORECASE
)
_PASS_PATTERNS = re.compile(r"(passed|PASSED)", re.IGNORECASE)


def run_single_test(
    test_id: str,
    project_workspace: str,
    python_exe: str,
    timeout: int = 120,
) -> TestOutcome:
    """
    Execute one test and return its outcome and runtime.

    Uses `pytest -x` for clean exit on first failure within the test.

    Args:
        test_id: Full pytest test ID.
        project_workspace: Linux path to project.
        python_exe: Linux path to Python executable.
        timeout: Per-test timeout in seconds.

    Returns:
        TestOutcome.
    """
    # Escape special characters that could break shell quoting
    safe_id = test_id.replace("'", "\\'")
    cmd = (
        f"{python_exe} -m pytest '{safe_id}' "
        f"-rN --tb=short -q 2>&1"
    )

    t0 = time.perf_counter()
    result = wsl(cmd, cwd=project_workspace, timeout=timeout + 10)
    elapsed = time.perf_counter() - t0

    combined = result.stdout + result.stderr

    if result.returncode == -1:
        outcome = "TIMEOUT"
    elif result.returncode == 0:
        outcome = "PASS"
    elif "no tests ran" in combined.lower() or "collected 0 items" in combined.lower():
        outcome = "ERROR"
    else:
        outcome = "FAIL"

    return TestOutcome(
        test_id=test_id,
        outcome=outcome,
        runtime_seconds=round(elapsed, 4),
        stdout=result.stdout[:4000],   # Truncate to avoid huge records
        stderr=result.stderr[:2000],
    )


# ─── Full suite run ────────────────────────────────────────────────────────────

def run_full_suite(
    project: str,
    bug_id: str,
    version: int,
    project_workspace: str,
    python_exe: str,
    test_ids: list[str],
    timeout_per_test: int = 120,
    parallel: int = 1,
) -> SuiteRun:
    """
    Run every test in test_ids and collect outcomes.

    Args:
        project: Project name.
        bug_id: Bug identifier.
        version: 0 for buggy, 1 for fixed.
        project_workspace: Linux path to checked-out project.
        python_exe: Linux path to Python executable.
        test_ids: Ordered list of test IDs to execute.
        timeout_per_test: Per-test timeout in seconds.
        parallel: Number of concurrent worker processes (default 1 = sequential).
                  NOTE: parallel > 1 may cause interference for tests that share
                  global state. Use parallel=1 for reproducibility-sensitive runs.

    Returns:
        SuiteRun containing all individual outcomes (ordered to match test_ids).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    suite = SuiteRun(project=project, bug_id=bug_id, version=version)
    t0 = time.perf_counter()

    if parallel <= 1:
        # Sequential path (guaranteed ordering, safe for all projects)
        for test_id in test_ids:
            outcome = run_single_test(
                test_id=test_id,
                project_workspace=project_workspace,
                python_exe=python_exe,
                timeout=timeout_per_test,
            )
            suite.outcomes.append(outcome)
    else:
        # Parallel path — results are re-ordered to match test_ids
        outcomes_map: dict[str, TestOutcome] = {}
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            future_to_id = {
                executor.submit(
                    run_single_test,
                    test_id=test_id,
                    project_workspace=project_workspace,
                    python_exe=python_exe,
                    timeout=timeout_per_test,
                ): test_id
                for test_id in test_ids
            }
            for future in as_completed(future_to_id):
                test_id = future_to_id[future]
                try:
                    outcomes_map[test_id] = future.result()
                except Exception as exc:  # noqa: BLE001
                    outcomes_map[test_id] = TestOutcome(
                        test_id=test_id,
                        outcome="ERROR",
                        runtime_seconds=0.0,
                        stdout="",
                        stderr=str(exc),
                    )
        # Re-order to original test_ids ordering
        for test_id in test_ids:
            if test_id in outcomes_map:
                suite.outcomes.append(outcomes_map[test_id])

    suite.total_elapsed = round(time.perf_counter() - t0, 3)
    return suite


# ─── Reproducibility check ─────────────────────────────────────────────────────

def check_reproducibility(
    triggering_tests: list[str],
    buggy_outcomes: SuiteRun,
    fixed_outcomes: SuiteRun,
) -> tuple[bool, list[str], list[str]]:
    """
    Verify that the known triggering tests FAIL on buggy and PASS on fixed.
    If triggering_tests is empty, dynamically discover tests that FAIL->PASS.

    Args:
        triggering_tests: Test IDs from bug.info (ground truth labels).
        buggy_outcomes: SuiteRun from buggy version.
        fixed_outcomes: SuiteRun from fixed version.

    Returns:
        (is_reproducible, list of failure reasons, list of actual triggering test IDs)
    """
    buggy_map = {o.test_id: o.outcome for o in buggy_outcomes.outcomes}
    fixed_map = {o.test_id: o.outcome for o in fixed_outcomes.outcomes}

    reasons: list[str] = []
    actual_triggering = []

    # If bug.info provides explicit test cases, verify them
    if triggering_tests:
        for tt in triggering_tests:
            # Match by suffix since full paths may differ from stored IDs
            buggy_hit = _find_test(tt, buggy_map)
            fixed_hit = _find_test(tt, fixed_map)

            if buggy_hit is None:
                reasons.append(f"Triggering test not found in buggy suite: {tt}")
                continue
            if fixed_hit is None:
                reasons.append(f"Triggering test not found in fixed suite: {tt}")
                continue

            if buggy_hit not in ("FAIL", "ERROR"):
                reasons.append(f"Expected FAIL on buggy but got {buggy_hit}: {tt}")
            if fixed_hit not in ("PASS", "SKIP"):
                reasons.append(f"Expected PASS on fixed but got {fixed_hit}: {tt}")
            
            if buggy_hit in ("FAIL", "ERROR") and fixed_hit in ("PASS", "SKIP"):
                actual_triggering.append(tt)
    else:
        # Dynamically discover FAIL -> PASS transitions
        for test_id, buggy_outcome in buggy_map.items():
            if buggy_outcome in ("FAIL", "ERROR"):
                fixed_outcome = fixed_map.get(test_id)
                if fixed_outcome in ("PASS", "SKIP"):
                    actual_triggering.append(test_id)
        
        if not actual_triggering:
            reasons.append("No dynamic FAIL->PASS triggering tests found")

    return len(reasons) == 0, reasons, actual_triggering


def _find_test(trigger: str, outcome_map: dict[str, str]) -> Optional[str]:
    """Find outcome for a trigger test ID, allowing partial path matching."""
    if trigger in outcome_map:
        return outcome_map[trigger]
    # Try suffix match
    for test_id, outcome in outcome_map.items():
        if test_id.endswith(trigger) or trigger.endswith(test_id):
            return outcome
    return None
