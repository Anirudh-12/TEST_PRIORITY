"""
Windows-native dataset collection runner.

Replaces run_dataset_collection.sh for use from Windows PowerShell/uv.
This runs the BugsInPy pipeline for all target bugs sequentially,
calling `uv run python -m bugsinpy.process` for each.

Usage (from Windows PowerShell, in the project root):
    uv run python scripts/collect_dataset.py
    uv run python scripts/collect_dataset.py --start-from thefuck 2
    uv run python scripts/collect_dataset.py --only thefuck
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_PATH = PROJECT_ROOT / "data" / f"collection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# All target bugs — ordered small-to-large for fast feedback
TARGET_BUGS: list[tuple[str, int]] = [
    # httpie — Python 3.7 (5 bugs, ~300 tests each, fast)
    ("httpie",       1), ("httpie",       2), ("httpie",       3),
    ("httpie",       4), ("httpie",       5),
    # black — Python 3.8 (10 bugs, ~400 tests each, medium)
    ("black",        1), ("black",        2), ("black",        3),
    ("black",        4), ("black",        5), ("black",        6),
    ("black",        7), ("black",        8), ("black",        9),
    ("black",       10),
    # thefuck — Python 3.7 (15 bugs, ~1700 tests each, slow)
    ("thefuck",      2), ("thefuck",      3), ("thefuck",      4),
    ("thefuck",      5), ("thefuck",      6), ("thefuck",      7),
    ("thefuck",      8), ("thefuck",      9), ("thefuck",     10),
    ("thefuck",     11), ("thefuck",     12), ("thefuck",     13),
    ("thefuck",     14), ("thefuck",     15),
    # ── BATCH 1: Reliable Python 3.8 projects ───────────────────────────────
    # tornado — Python 3.8 (16 bugs, async I/O framework, well-organized tests)
    ("tornado",     1),  ("tornado",     2),  ("tornado",     3),
    ("tornado",     4),  ("tornado",     5),  ("tornado",     6),
    ("tornado",     7),  ("tornado",     8),  ("tornado",     9),
    ("tornado",    10),  ("tornado",    11),  ("tornado",    12),
    ("tornado",    13),  ("tornado",    14),  ("tornado",    15),
    ("tornado",    16),
    # fastapi — Python 3.8 (16 bugs, REST API framework, modern test suite)
    ("fastapi",     1),  ("fastapi",     2),  ("fastapi",     3),
    ("fastapi",     4),  ("fastapi",     5),  ("fastapi",     6),
    ("fastapi",     7),  ("fastapi",     8),  ("fastapi",     9),
    ("fastapi",    10),  ("fastapi",    11),  ("fastapi",    12),
    ("fastapi",    13),  ("fastapi",    14),  ("fastapi",    15),
    ("fastapi",    16),
    # sanic — Python 3.8 (5 bugs, async web framework)
    ("sanic",       1),  ("sanic",       2),  ("sanic",       3),
    ("sanic",       4),  ("sanic",       5),
]


def log(msg: str, fp=None) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if fp:
        fp.write(line + "\n")
        fp.flush()


def is_already_processed(project: str, bug_id: int) -> bool:
    out_file = PROCESSED_DIR / project / f"{bug_id}.json"
    if not out_file.exists():
        return False
    try:
        data = json.loads(out_file.read_text(encoding="utf-8"))
        # Skip both successful and confirmed non-reproducible bugs — no point retrying
        return data.get("dataset_status") in ("SUCCESSFULLY_PROCESSED", "NON_REPRODUCIBLE")
    except Exception:
        return False


# Linux venv — all pipeline execution runs here, completely separate from
# the Windows .venv. Lives in the Linux filesystem so no lib64 issues.
LINUX_PYTHON = "/home/akshay/tp_venv/bin/python"
LINUX_PROJECT = "/mnt/c/Users/aksha/OneDrive/Documents/TEST_PRIORITY"


def run_bug(project: str, bug_id: int, fp, with_coverage: bool = False, force: bool = False) -> bool:
    """Run process pipeline for one bug inside WSL. Returns True if successful."""
    args = (
        f"{LINUX_PYTHON} -m bugsinpy.process "
        f"--project {project} "
        f"--bug {bug_id} "
        f"--timeout 30 "
        f"--parallel 8"
    )
    if not with_coverage:
        args += " --skip-coverage"
    if force:
        args += " --force"
        
    import sys
    if sys.platform == "linux":
        cmd = ["bash", "-c", f"cd {LINUX_PROJECT} && {args}"]
    else:
        cmd = ["wsl", "-d", "Ubuntu", "--", "bash", "-c", f"cd {LINUX_PROJECT} && {args}"]
        
    result = subprocess.run(
        cmd,
        capture_output=False,  # Let stdout/stderr flow to the terminal
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode == 0



def main() -> None:
    parser = argparse.ArgumentParser(description="Run BugsInPy dataset collection from Windows.")
    parser.add_argument("--only", help="Only process bugs from this project.")
    parser.add_argument("--start-from", nargs=2, metavar=("PROJECT", "BUG_ID"),
                        help="Skip bugs before this one in the list.")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess even if already successfully processed.")
    parser.add_argument("--with-coverage", action="store_true",
                        help="Enable coverage collection (WARNING: takes ~10x longer).")
    parser.add_argument("--max-bugs", type=int, default=None,
                        help="Maximum number of bugs to process per project (useful for massive projects).")
    args = parser.parse_args()

    bugs = TARGET_BUGS
    if args.only:
        bugs = [(p, b) for p, b in bugs if p == args.only]
        # If the project is not in our hardcoded TARGET_BUGS, dynamically generate 1..max_bugs
        if not bugs:
            limit = args.max_bugs if args.max_bugs else 50
            bugs = [(args.only, i) for i in range(1, limit + 1)]
            
    if args.max_bugs:
        # Cap the bugs per project
        capped_bugs = []
        counts = {}
        for p, b in bugs:
            counts[p] = counts.get(p, 0) + 1
            if counts[p] <= args.max_bugs:
                capped_bugs.append((p, b))
        bugs = capped_bugs

    if args.start_from:
        proj, bug_id = args.start_from[0], int(args.start_from[1])
        start_idx = next((i for i, (p, b) in enumerate(bugs) if p == proj and b == bug_id), 0)
        bugs = bugs[start_idx:]

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    success = failed = skipped = 0

    with open(LOG_PATH, "w", encoding="utf-8") as fp:
        log(f"Dataset Collection — {len(bugs)} bugs to process", fp)
        log(f"Log: {LOG_PATH}", fp)
        log("=" * 60, fp)

        for idx, (project, bug_id) in enumerate(bugs, 1):
            log(f"[{idx}/{len(bugs)}] {project}/{bug_id}", fp)

            if not args.force and is_already_processed(project, bug_id):
                log(f"  SKIP — already SUCCESSFULLY_PROCESSED", fp)
                skipped += 1
                continue

            ok = run_bug(project, bug_id, fp, with_coverage=args.with_coverage, force=args.force)
            if ok:
                log(f"  OK  — {project}/{bug_id}", fp)
                success += 1
            else:
                log(f"  WARN — non-zero exit (check data/processed/{project}/{bug_id}.json)", fp)
                failed += 1

        log("=" * 60, fp)
        log(f"Complete. Success={success}  Failed/partial={failed}  Skipped={skipped}", fp)
        log(f"Run `uv run python scripts/dataset_audit.py` for a full summary.", fp)


if __name__ == "__main__":
    main()
