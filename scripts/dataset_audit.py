"""
Dataset audit script — summarize all processed bug records.

Usage:
    uv run python scripts/dataset_audit.py
    uv run python scripts/dataset_audit.py --output report.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
STATUS_LOG = PROJECT_ROOT / "data" / "dataset_status.jsonl"


def load_records() -> list[dict]:
    records = []
    if not PROCESSED_DIR.exists():
        return records
    for proj_dir in sorted(PROCESSED_DIR.iterdir()):
        if not proj_dir.is_dir():
            continue
        for json_file in sorted(proj_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                records.append(data)
            except Exception as e:
                records.append({"project": proj_dir.name, "bug_id": json_file.stem,
                                 "dataset_status": f"PARSE_ERROR: {e}"})
    return records


def summarize(records: list[dict]) -> str:
    by_status: dict[str, list[dict]] = defaultdict(list)
    by_project: dict[str, dict] = defaultdict(lambda: defaultdict(int))

    for r in records:
        status = r.get("dataset_status", "UNKNOWN")
        proj = r.get("project", "?")
        by_status[status].append(r)
        by_project[proj][status] += 1
        by_project[proj]["total"] += 1

    lines = [
        "# Dataset Audit Report",
        "",
        f"**Total records scanned**: {len(records)}",
        "",
        "## Status Summary",
        "",
        "| Status | Count |",
        "|--------|-------|",
    ]
    for status, recs in sorted(by_status.items()):
        lines.append(f"| `{status}` | {len(recs)} |")

    successfully = by_status.get("SUCCESSFULLY_PROCESSED", [])
    lines += [
        "",
        f"## Successfully Processed ({len(successfully)} records)",
        "",
        "| Project | Bug ID | Tests | Failing | Triggering | Runtime (s) |",
        "|---------|--------|-------|---------|------------|-------------|",
    ]
    for r in successfully:
        ss = r.get("suite_summary", {})
        total = ss.get("total_tests", "?")
        failing = ss.get("failing_tests_count", "?")
        runtime = ss.get("total_runtime_seconds", "?")
        # Count triggering tests
        triggering = sum(
            1 for t in r.get("tests", [])
            if t.get("GROUND_TRUTH_LABELS", {}).get("is_triggering_test", False)
        )
        lines.append(
            f"| {r.get('project')} | {r.get('bug_id')} | {total} | {failing} | {triggering} | {runtime} |"
        )

    lines += [
        "",
        "## Per-Project Breakdown",
        "",
        "| Project | Total | Successful | Non-Reproducible | Failed |",
        "|---------|-------|------------|------------------|--------|",
    ]
    for proj, counts in sorted(by_project.items()):
        lines.append(
            f"| {proj} | {counts['total']} "
            f"| {counts.get('SUCCESSFULLY_PROCESSED', 0)} "
            f"| {counts.get('NON_REPRODUCIBLE', 0)} "
            f"| {counts.get('INSTALL_FAILED', 0) + counts.get('CHECKOUT_FAILED', 0)} |"
        )

    non_repro = by_status.get("NON_REPRODUCIBLE", [])
    if non_repro:
        lines += [
            "",
            f"## Non-Reproducible Bugs ({len(non_repro)})",
            "",
            "| Project | Bug ID | Log Reason |",
            "|---------|--------|------------|",
        ]
        for r in non_repro:
            log = r.get("processing_log", [])
            reason = next(
                (e.get("message", "") for e in reversed(log) if e.get("status") == "FAILED"),
                "unknown",
            )[:80]
            lines.append(f"| {r.get('project')} | {r.get('bug_id')} | {reason} |")

    failed = by_status.get("INSTALL_FAILED", []) + by_status.get("CHECKOUT_FAILED", []) + \
             by_status.get("TEST_FAILED_INFRASTRUCTURE", [])
    if failed:
        lines += [
            "",
            f"## Pipeline Failures ({len(failed)})",
            "",
            "| Project | Bug ID | Status | Reason |",
            "|---------|--------|--------|--------|",
        ]
        for r in failed:
            log = r.get("processing_log", [])
            reason = next(
                (e.get("message", "") for e in reversed(log) if e.get("status") == "FAILED"),
                "unknown",
            )[:80]
            lines.append(
                f"| {r.get('project')} | {r.get('bug_id')} | `{r.get('dataset_status')}` | {reason} |"
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the processed bug dataset.")
    parser.add_argument("--output", help="Write report to this Markdown file (optional)")
    args = parser.parse_args()

    records = load_records()
    report = summarize(records)
    print(report)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\nReport written to: {args.output}")


if __name__ == "__main__":
    main()
