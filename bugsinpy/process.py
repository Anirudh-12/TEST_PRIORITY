"""
Main end-to-end pipeline for processing a single BugsInPy bug.

Usage:
    python -m bugsinpy.process --project thefuck --bug 1

Produces:
    data/processed/<project>/<bug_id>.json
    data/dataset_status.jsonl  (append-only log)

Processing steps:
    1. Parse bug.info / project.info
    2. Clone BugsInPy (if needed) and checkout buggy version
    3. Install environment (uv / pip)
    4. Run full test suite on buggy version
    5. Run full test suite on fixed version
    6. Verify reproducibility (triggering tests FAIL on buggy, PASS on fixed)
    7. Collect coverage per test on the buggy version
    8. Run git diff + AST analysis
    9. Compute pre-episode features
    10. Validate against JSON Schema
    11. Write experiment record
    12. Append status to dataset_status.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Optional

import jsonschema

from analysis.diff import DiffResult, extract_diff, identify_changed_entities
from analysis.features import (
    PreEpisodeFeatures,
    build_pre_episode_features,
)
from bugsinpy.checkout import (
    LINUX_BUGSINPY,
    LINUX_PROJECT_ROOT,
    LINUX_WORKSPACE,
    WSL_DISTRO,
    BugStatus,
    checkout_bug,
    clone_bugsinpy,
    ensure_workspace,
    get_installed_python,
    install_bug,
    wsl,
)
from bugsinpy.coverage import (
    collect_suite_coverage,
    install_coverage,
)
from bugsinpy.metadata import BugInfo, ProjectInfo, parse_bug_info, parse_project_info
from bugsinpy.runner import (
    SuiteRun,
    TestOutcome,
    check_reproducibility,
    discover_tests,
    run_full_suite,
)

# ─── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
STATUS_LOG = DATA_DIR / "dataset_status.jsonl"
SCHEMA_PATH = DATA_DIR / "schemas" / "experiment_record.json"

# BugsInPy clone on Windows (for reading bug.info files)
# We read these from the Windows clone since we can access them directly
BUGSINPY_WIN_PATH = PROJECT_ROOT / "data" / "raw" / "BugsInPy"


# ─── Logging helpers ───────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class ProcessingLog:
    """Append-only log for pipeline steps."""

    def __init__(self) -> None:
        self.entries: list[dict] = []
        self._step_start: float = 0.0

    def start(self, step: str) -> None:
        self._step_start = time.perf_counter()
        self.entries.append(
            {
                "step": step,
                "status": "STARTED",
                "timestamp": _now(),
            }
        )
        print(f"  [{step}] ...", flush=True)

    def done(self, step: str, message: str = "") -> None:
        elapsed = time.perf_counter() - self._step_start
        self.entries.append(
            {
                "step": step,
                "status": "OK",
                "timestamp": _now(),
                "message": message,
                "elapsed_seconds": round(elapsed, 3),
            }
        )
        print(f"  [{step}] OK  ({elapsed:.1f}s)  {message}", flush=True)

    def fail(self, step: str, message: str) -> None:
        elapsed = time.perf_counter() - self._step_start
        self.entries.append(
            {
                "step": step,
                "status": "FAILED",
                "timestamp": _now(),
                "message": message,
                "elapsed_seconds": round(elapsed, 3),
            }
        )
        print(f"  [{step}] FAILED: {message}", file=sys.stderr, flush=True)


# ─── Main pipeline ─────────────────────────────────────────────────────────────


def process_bug(
    project: str,
    bug_id: str,
    skip_coverage: bool = False,
    timeout_per_test: int = 120,
    force_reprocess: bool = False,
    parallel: int = 1,
) -> dict:
    """
    Full end-to-end pipeline for a single bug.

    Args:
        project: BugsInPy project name.
        bug_id: Bug identifier string.
        skip_coverage: Skip coverage collection (faster, less complete).
        timeout_per_test: Per-test timeout in seconds.
        force_reprocess: Re-process even if output file already exists.
        parallel: Number of parallel test workers (1 = sequential).

    Returns:
        Experiment record dict (also written to disk).
    """
    out_dir = PROCESSED_DIR / project
    out_file = out_dir / f"{bug_id}.json"

    if out_file.exists() and not force_reprocess:
        print(f"Already processed: {out_file}. Use --force to reprocess.")
        return json.loads(out_file.read_text())

    log = ProcessingLog()
    record: dict = {
        "schema_version": "1.0",
        "project": project,
        "bug_id": bug_id,
        "environment": {},
        "changed_entities": {"files": [], "functions": [], "line_ranges": {}},
        "tests": [],
        "coverage_matrix": {},
        "dataset_status": BugStatus.AVAILABLE.value,
        "processing_log": [],
    }

    print(f"\n{'=' * 60}")
    print(f"Processing {project} bug #{bug_id}")
    print(f"{'=' * 60}")

    # ── Step 0: Ensure workspace and BugsInPy clone ────────────────────────────
    log.start("workspace_setup")
    ensure_workspace()
    clone_result = clone_bugsinpy()
    if not clone_result.ok and "Already cloned" not in clone_result.stdout:
        log.fail("workspace_setup", f"BugsInPy clone failed: {clone_result.stderr[:200]}")
        return _finalize(record, BugStatus.CHECKOUT_FAILED, log, out_dir, out_file)
    log.done("workspace_setup", "BugsInPy ready")

    # ── Step 1: Parse metadata ─────────────────────────────────────────────────
    log.start("parse_metadata")
    # Read bug.info from WSL (BugsInPy clone on Linux filesystem)
    bug_info_linux = f"{LINUX_BUGSINPY}/projects/{project}/bugs/{bug_id}/bug.info"
    project_info_linux = f"{LINUX_BUGSINPY}/projects/{project}/project.info"

    bug_info_result = wsl(f"cat '{bug_info_linux}'")
    project_info_result = wsl(f"cat '{project_info_linux}'")

    if not bug_info_result.ok:
        log.fail("parse_metadata", f"Cannot read bug.info: {bug_info_result.stderr[:100]}")
        return _finalize(record, BugStatus.CHECKOUT_FAILED, log, out_dir, out_file)

    # Parse using temporary files
    import tempfile, os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".info", delete=False) as tf:
        tf.write(bug_info_result.stdout)
        tmp_path = Path(tf.name)
    try:
        bug_info = parse_bug_info(tmp_path, project, bug_id)
    finally:
        os.unlink(tmp_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".info", delete=False) as tf:
        tf.write(project_info_result.stdout)
        tmp_path = Path(tf.name)
    try:
        proj_info = parse_project_info(tmp_path, project)
    finally:
        os.unlink(tmp_path)

    record["environment"] = {
        "python_version": bug_info.python_version,
        "buggy_commit": bug_info.buggy_commit_id,
        "fixed_commit": bug_info.fixed_commit_id,
        "github_url": proj_info.github_url,
        "os": "Ubuntu 26.04 (WSL2)",
        "collection_timestamp": _now(),
    }
    log.done(
        "parse_metadata",
        f"Python {bug_info.python_version}, commits {bug_info.buggy_commit_id[:7]}→{bug_info.fixed_commit_id[:7]}",
    )

    # ── Step 2: Checkout buggy version ─────────────────────────────────────────
    log.start("checkout_buggy")
    buggy_workspace = f"{LINUX_WORKSPACE}/{project}_{bug_id}_buggy/{project}"
    checkout_result = checkout_bug(
        project, bug_id, version=0, workspace=f"{LINUX_WORKSPACE}/{project}_{bug_id}_buggy"
    )
    if not checkout_result.ok:
        log.fail("checkout_buggy", checkout_result.stderr[:200])
        return _finalize(record, BugStatus.CHECKOUT_FAILED, log, out_dir, out_file)
    log.done("checkout_buggy", f"Workspace: {buggy_workspace}")

    # ── Step 3: Install environment ────────────────────────────────────────────
    log.start("install_env")
    install_result = install_bug(buggy_workspace, python_version=bug_info.python_version)
    if not install_result.ok:
        log.fail("install_env", install_result.stderr[:300])
        return _finalize(record, BugStatus.INSTALL_FAILED, log, out_dir, out_file)

    python_exe = get_installed_python(buggy_workspace, python_version=bug_info.python_version)
    if python_exe is None:
        # Fallback: try system python3
        python_exe = "python3"
    log.done("install_env", f"Python: {python_exe}")

    # ── Step 4: Discover tests ─────────────────────────────────────────────────
    log.start("discover_tests")
    test_ids = discover_tests(buggy_workspace, python_exe)
    if not test_ids:
        log.fail("discover_tests", "No tests discovered")
        return _finalize(record, BugStatus.TEST_FAILED_INFRASTRUCTURE, log, out_dir, out_file)
    log.done("discover_tests", f"{len(test_ids)} tests found")

    # ── Step 5: Run suite on buggy version ────────────────────────────────────
    log.start("run_buggy_suite")
    buggy_run = run_full_suite(
        project=project,
        bug_id=bug_id,
        version=0,
        project_workspace=buggy_workspace,
        python_exe=python_exe,
        test_ids=test_ids,
        timeout_per_test=timeout_per_test,
        parallel=parallel,
    )
    fail_count = sum(1 for o in buggy_run.outcomes if o.outcome in ("FAIL", "ERROR"))
    log.done("run_buggy_suite", f"{fail_count}/{len(test_ids)} tests failing")

    # ── Step 6: Checkout and run fixed version ─────────────────────────────────
    log.start("checkout_fixed")
    fixed_workspace = f"{LINUX_WORKSPACE}/{project}_{bug_id}_fixed/{project}"
    fixed_checkout = checkout_bug(
        project, bug_id, version=1, workspace=f"{LINUX_WORKSPACE}/{project}_{bug_id}_fixed"
    )
    if not fixed_checkout.ok:
        log.fail("checkout_fixed", fixed_checkout.stderr[:200])
        return _finalize(record, BugStatus.CHECKOUT_FAILED, log, out_dir, out_file)
    install_bug(fixed_workspace)

    fixed_python = get_installed_python(fixed_workspace) or "python3"
    fixed_run = run_full_suite(
        project=project,
        bug_id=bug_id,
        version=1,
        project_workspace=fixed_workspace,
        python_exe=fixed_python,
        test_ids=test_ids,
        timeout_per_test=timeout_per_test,
        parallel=parallel,
    )
    log.done("checkout_fixed", "Fixed version run complete")

    # ── Step 7: Reproducibility check ─────────────────────────────────────────
    log.start("reproducibility_check")
    is_repro, repro_reasons, actual_triggering_tests = check_reproducibility(
        triggering_tests=bug_info.test_cases,
        buggy_outcomes=buggy_run,
        fixed_outcomes=fixed_run,
    )
    if not is_repro:
        log.fail("reproducibility_check", "; ".join(repro_reasons))
        # Still proceed but mark as NON_REPRODUCIBLE — do not discard data
        record["dataset_status"] = BugStatus.NON_REPRODUCIBLE.value
    else:
        log.done("reproducibility_check", "Triggering tests verified")

    # ── Step 8: Coverage collection ────────────────────────────────────────────
    if not skip_coverage:
        log.start("coverage_collection")
        install_coverage(buggy_workspace, python_exe)

        coverage_matrix = collect_suite_coverage(
            test_ids=test_ids,
            project_workspace=buggy_workspace,
            python_exe=python_exe,
            timeout=timeout_per_test * 10,
        )

        if not coverage_matrix:
            log.fail("coverage_collection", "Coverage JSON failed or empty")
            if record["dataset_status"] == BugStatus.AVAILABLE.value:
                record["dataset_status"] = BugStatus.COVERAGE_FAILED.value
        else:
            covered_tests_count = sum(1 for t, ent in coverage_matrix.items() if ent)
            log.done("coverage_collection", f"{covered_tests_count}/{len(test_ids)} tests covered")
    else:
        coverage_matrix = {}

    record["coverage_matrix"] = coverage_matrix

    # ── Step 9: Diff + AST analysis ───────────────────────────────────────────
    log.start("diff_analysis")
    diff_result = extract_diff(
        project_workspace=fixed_workspace,  # Use fixed workspace for git history
        buggy_commit=bug_info.buggy_commit_id,
        fixed_commit=bug_info.fixed_commit_id,
    )
    if diff_result.error:
        log.fail("diff_analysis", diff_result.error)
    else:
        diff_result.changed_entities = identify_changed_entities(
            diff_result=diff_result,
            project_workspace_linux=fixed_workspace,
        )
        log.done(
            "diff_analysis",
            f"{len(diff_result.changed_files)} files, {len(diff_result.changed_entities)} entities changed",
        )

    record["changed_entities"] = {
        "files": diff_result.changed_files,
        "functions": [e.qualified_name for e in diff_result.changed_entities],
        "line_ranges": {
            fp: [[r.start, r.end] for r in ranges]
            for fp, ranges in diff_result.modified_ranges.items()
        },
        "added_line_count": sum(len(v) for v in diff_result.added_lines.values()),
        "deleted_line_count": sum(len(v) for v in diff_result.deleted_lines.values()),
    }

    # ── Step 10: Build test records ────────────────────────────────────────────
    log.start("build_test_records")
    buggy_map = {o.test_id: o for o in buggy_run.outcomes}
    fixed_map = {o.test_id: o for o in fixed_run.outcomes}
    triggering_set = set(actual_triggering_tests)

    test_records = []
    for test_id in test_ids:
        buggy_o = buggy_map.get(test_id)
        fixed_o = fixed_map.get(test_id)
        covered = coverage_matrix.get(test_id, [])

        features = build_pre_episode_features(
            test_id=test_id,
            covered_entities=covered,
            diff_result=diff_result,
            estimated_runtime=buggy_o.runtime_seconds if buggy_o else 0.0,
        )

        # Determine if this is a triggering test (ground truth only)
        is_triggering = any(t in test_id or test_id in t for t in triggering_set)

        test_records.append(
            {
                "test_id": test_id,
                "test_file": test_id.split("::")[0] if "::" in test_id else test_id,
                "test_function": "::".join(test_id.split("::")[1:]) if "::" in test_id else "",
                "coverage_level": "function" if covered else "none",
                "PRE_EPISODE_FEATURES": {
                    "estimated_runtime_seconds": features.estimated_runtime_seconds,
                    "coverage_overlap_ratio": features.coverage_overlap_ratio,
                    "coverage_entity_count": features.coverage_entity_count,
                    "dependency_distance": features.dependency_distance,
                    "changed_file_in_coverage": features.changed_file_in_coverage,
                    "changed_function_in_coverage": features.changed_function_in_coverage,
                    "historical_failure_rate": features.historical_failure_rate,
                    "historical_execution_count": features.historical_execution_count,
                },
                # GROUND TRUTH LABELS — never expose to policy
                "GROUND_TRUTH_LABELS": {
                    "outcome_buggy": (buggy_o.outcome if buggy_o else "UNKNOWN"),
                    "outcome_fixed": (fixed_o.outcome if fixed_o else "UNKNOWN"),
                    "is_triggering_test": is_triggering,
                    "actual_runtime_seconds": (buggy_o.runtime_seconds if buggy_o else 0.0),
                },
            }
        )

    record["tests"] = test_records
    fail_counts = sum(
        1 for t in test_records if t["GROUND_TRUTH_LABELS"]["outcome_buggy"] in ("FAIL", "ERROR")
    )
    record["suite_summary"] = {
        "total_tests": len(test_records),
        "failing_tests_count": fail_counts,
        "passing_tests_count": len(test_records) - fail_counts,
        "total_runtime_seconds": buggy_run.total_elapsed,
        "median_runtime_seconds": _median(
            [t["GROUND_TRUTH_LABELS"]["actual_runtime_seconds"] for t in test_records]
        ),
    }
    log.done("build_test_records", f"{len(test_records)} test records built")

    # ── Step 11: Validate and write ────────────────────────────────────────────
    log.start("schema_validation")
    schema = json.loads(SCHEMA_PATH.read_text())
    try:
        jsonschema.validate(record, schema)
        log.done("schema_validation", "Record is schema-valid")
    except jsonschema.ValidationError as ve:
        log.fail("schema_validation", str(ve.message)[:200])
        # Still write — we want to see the record even if schema fails

    if record["dataset_status"] == BugStatus.AVAILABLE.value:
        record["dataset_status"] = BugStatus.SUCCESSFULLY_PROCESSED.value

    return _finalize(record, BugStatus(record["dataset_status"]), log, out_dir, out_file)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _finalize(
    record: dict,
    status: BugStatus,
    log: ProcessingLog,
    out_dir: Path,
    out_file: Path,
) -> dict:
    """Write the record to disk and append to status log."""
    record["dataset_status"] = status.value
    record["processing_log"] = log.entries

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\n✓ Record written: {out_file}")
    print(f"  Status: {status.value}")

    # Append to status log (never overwrite)
    STATUS_LOG.parent.mkdir(parents=True, exist_ok=True)
    status_entry = {
        "project": record["project"],
        "bug_id": record["bug_id"],
        "status": status.value,
        "timestamp": _now(),
        "total_tests": record.get("suite_summary", {}).get("total_tests", 0),
    }
    with open(STATUS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(status_entry) + "\n")

    return record


# ─── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process one BugsInPy bug into a machine-readable experiment record."
    )
    parser.add_argument("--project", required=True, help="BugsInPy project name")
    parser.add_argument("--bug", required=True, help="Bug ID")
    parser.add_argument("--skip-coverage", action="store_true", help="Skip coverage collection")
    parser.add_argument("--timeout", type=int, default=30, help="Per-test timeout (seconds)")
    parser.add_argument("--parallel", type=int, default=8, help="Parallel test workers (default 8; matches physical core count)")
    parser.add_argument("--force", action="store_true", help="Reprocess even if output exists")
    args = parser.parse_args()

    record = process_bug(
        project=args.project,
        bug_id=args.bug,
        skip_coverage=args.skip_coverage,
        timeout_per_test=args.timeout,
        force_reprocess=args.force,
        parallel=args.parallel,
    )
    print(f"\nFinal status: {record['dataset_status']}")


if __name__ == "__main__":
    main()
