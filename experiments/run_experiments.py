"""
Experiment runner — execute all policies across all processed bug records.

Usage:
    uv run python -m experiments.run_experiments [options]

    uv run python -m experiments.run_experiments \\
        --budgets 0.05 0.10 0.25 0.50 \\
        --policies random coverage historical cost_weighted full_suite \\
        --random-seeds 30 \\
        --output data/results/

Output:
    data/results/episode_results.jsonl   — one line per episode
    data/results/summary_table.md        — Markdown results table
    data/results/results_db.sqlite       — SQLite for querying
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from environment.episode import Episode
from evaluation.metrics import EpisodeResult, compute_all_metrics, summarize_metric
from policies.base import Policy, RevealedOutcome
from policies.cost_weighted_policy import CostWeightedPolicy
from policies.coverage_policy import CoveragePolicy
from policies.full_suite import FullSuitePolicy
from policies.historical_policy import HistoricalPolicy
from policies.random_policy import RandomPolicy

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


# ─── Record loading ─────────────────────────────────────────────────────────────

def load_all_records(processed_dir: Path) -> list[dict]:
    """Load all SUCCESSFULLY_PROCESSED records."""
    records = []
    if not processed_dir.exists():
        return records
    for proj_dir in sorted(processed_dir.iterdir()):
        if not proj_dir.is_dir():
            continue
        for json_file in sorted(proj_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                if data.get("dataset_status") == "SUCCESSFULLY_PROCESSED":
                    records.append(data)
            except Exception as e:
                print(f"  [WARN] Could not load {json_file}: {e}", file=sys.stderr)
    return records


# ─── Policy registry ────────────────────────────────────────────────────────────

def build_policies(
    policy_names: list[str],
    random_seeds: int = 30,
) -> list[Policy]:
    """Instantiate all requested policies."""
    all_policies: list[Policy] = []
    for name in policy_names:
        if name == "full_suite":
            all_policies.append(FullSuitePolicy())
        elif name == "random":
            for seed in range(random_seeds):
                all_policies.append(RandomPolicy(seed=seed))
        elif name == "coverage":
            all_policies.append(CoveragePolicy())
        elif name == "historical":
            all_policies.append(HistoricalPolicy())
        elif name == "cost_weighted":
            all_policies.append(CostWeightedPolicy())
        else:
            print(f"  [WARN] Unknown policy: {name}", file=sys.stderr)
    return all_policies


# ─── Results database ────────────────────────────────────────────────────────────

def init_db(db_path: Path) -> sqlite3.Connection:
    """Create or open the SQLite results database."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            bug_id TEXT NOT NULL,
            policy_name TEXT NOT NULL,
            budget_fraction REAL NOT NULL,
            fault_detected INTEGER NOT NULL,
            tests_executed INTEGER NOT NULL,
            tests_to_detection INTEGER,
            time_to_detection REAL,
            total_time_used REAL NOT NULL,
            total_suite_time REAL NOT NULL,
            total_tests INTEGER NOT NULL,
            apfd REAL,
            apfdc REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def insert_episode(conn: sqlite3.Connection, ep: EpisodeResult, apfd: float, apfdc: float) -> None:
    conn.execute(
        """INSERT INTO episodes
           (project, bug_id, policy_name, budget_fraction, fault_detected,
            tests_executed, tests_to_detection, time_to_detection,
            total_time_used, total_suite_time, total_tests, apfd, apfdc)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ep.project, ep.bug_id, ep.policy_name, ep.budget_fraction,
            int(ep.fault_detected), ep.tests_executed, ep.tests_to_detection,
            ep.time_to_detection, ep.total_time_used, ep.total_suite_time,
            ep.total_tests, apfd, apfdc,
        ),
    )
    conn.commit()


# ─── Markdown table generation ──────────────────────────────────────────────────

def build_summary_table(
    all_results: dict[str, dict[float, list[EpisodeResult]]],
) -> str:
    """
    Build a Markdown summary table.

    all_results: { policy_name -> { budget -> [EpisodeResult] } }
    """
    from evaluation.metrics import apfd, apfdc, fault_detection_rate

    budget_cols = sorted({b for p in all_results.values() for b in p})
    policy_names = sorted(all_results.keys())

    lines = [
        "# Experiment Results: Baseline Policy Comparison",
        "",
        "## FDR (Fault Detection Rate) by Budget",
        "",
        "| Policy | " + " | ".join(f"B={b:.0%}" for b in budget_cols) + " |",
        "|--------|" + "--------|" * len(budget_cols),
    ]
    for pname in policy_names:
        row = f"| `{pname}` |"
        for budget in budget_cols:
            eps = all_results[pname].get(budget, [])
            if not eps:
                row += " — |"
                continue
            fdr = fault_detection_rate([{"fault_detected": e.fault_detected} for e in eps])
            row += f" {fdr:.3f} |"
        lines.append(row)

    lines += [
        "",
        "## APFDc (Cost-aware Fault Detection) by Budget",
        "",
        "| Policy | " + " | ".join(f"B={b:.0%}" for b in budget_cols) + " |",
        "|--------|" + "--------|" * len(budget_cols),
    ]
    for pname in policy_names:
        row = f"| `{pname}` |"
        for budget in budget_cols:
            eps = all_results[pname].get(budget, [])
            if not eps:
                row += " — |"
                continue
            scores = [
                apfdc([v for _, v in e.execution_sequence], e.execution_costs)
                for e in eps if e.execution_costs
            ]
            mean_apfdc = sum(scores) / len(scores) if scores else 0.0
            row += f" {mean_apfdc:.3f} |"
        lines.append(row)

    lines += [
        "",
        f"*N bugs = {max(len(eps) for p in all_results.values() for eps in p.values() if eps)}*",
        "",
        "> APFDc: higher = fault detected earlier relative to test cost.",
        "> FDR: fraction of bugs where the fault was detected within budget.",
    ]

    return "\n".join(lines)


# ─── Main runner ────────────────────────────────────────────────────────────────

def run_experiments(
    policy_names: list[str],
    budget_fractions: list[float],
    random_seeds: int = 30,
    processed_dir: Path = PROCESSED_DIR,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    """
    Run all experiments and return summary metrics.

    Returns:
        Nested dict: { policy_name -> { budget -> [EpisodeResult] } }
    """
    from evaluation.metrics import apfd as compute_apfd, apfdc as compute_apfdc

    records = load_all_records(processed_dir)
    if not records:
        print("ERROR: No SUCCESSFULLY_PROCESSED records found.", file=sys.stderr)
        print(f"  Run the data collection pipeline first (scripts/run_dataset_collection.sh)")
        sys.exit(1)

    print(f"Loaded {len(records)} bug records from {len({r['project'] for r in records})} projects.")

    policies = build_policies(policy_names, random_seeds)
    print(f"Policies: {[p.name for p in policies]}")
    print(f"Budgets: {budget_fractions}")
    print(f"Total experiments: {len(records)} × {len(policies)} × {len(budget_fractions)} = "
          f"{len(records) * len(policies) * len(budget_fractions)}")

    results_dir.mkdir(parents=True, exist_ok=True)
    db_path = results_dir / "results_db.sqlite"
    jsonl_path = results_dir / "episode_results.jsonl"
    db = init_db(db_path)

    # { policy_canonical_name -> { budget -> [EpisodeResult] } }
    all_results: dict[str, dict[float, list[EpisodeResult]]] = {}

    total = len(records) * len(policies) * len(budget_fractions)
    done = 0

    with open(jsonl_path, "a", encoding="utf-8") as jsonl_file:
        for record in records:
            proj = record["project"]
            bug = record["bug_id"]

            for budget in budget_fractions:
                for policy in policies:
                    done += 1
                    if done % 50 == 0:
                        print(f"  Progress: {done}/{total} ({100*done/total:.0f}%)")

                    # Run episode
                    episode = Episode.from_record(
                        record=record,
                        budget_fraction=budget,
                        policy_name=policy.name,
                    )
                    features = episode.get_features()
                    ordered_ids = policy.select_tests(
                        features=features,
                        budget_seconds=episode.get_budget_seconds(),
                    )
                    result = episode.simulate(ordered_ids)

                    # Compute APFD/APFDc for this episode
                    outcomes = [v for _, v in result.execution_sequence]
                    ep_apfd = compute_apfd(outcomes)
                    ep_apfdc = compute_apfdc(outcomes, result.execution_costs) if result.execution_costs else 0.0

                    # Aggregate (group random seeds under canonical name)
                    canonical = "random" if policy.name.startswith("random_seed") else policy.name
                    if canonical not in all_results:
                        all_results[canonical] = {}
                    if budget not in all_results[canonical]:
                        all_results[canonical][budget] = []
                    all_results[canonical][budget].append(result)

                    # Persist
                    insert_episode(db, result, ep_apfd, ep_apfdc)
                    ep_dict = asdict(result)
                    ep_dict["apfd"] = ep_apfd
                    ep_dict["apfdc"] = ep_apfdc
                    jsonl_file.write(json.dumps(ep_dict) + "\n")

    # Generate Markdown summary
    table = build_summary_table(all_results)
    table_path = results_dir / "summary_table.md"
    table_path.write_text(table, encoding="utf-8")

    print(f"\n{'='*60}")
    print(table)
    print(f"\nResults saved to: {results_dir}")
    print(f"  - Episode log: {jsonl_path}")
    print(f"  - Database: {db_path}")
    print(f"  - Summary table: {table_path}")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline policy experiments on the processed BugsInPy dataset."
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["full_suite", "random", "coverage", "historical", "cost_weighted"],
        choices=["full_suite", "random", "coverage", "historical", "cost_weighted"],
        help="Policies to evaluate (default: all).",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.25, 0.50, 1.00],
        help="Budget fractions to evaluate (default: 0.05 0.10 0.25 0.50 1.00).",
    )
    parser.add_argument(
        "--random-seeds",
        type=int,
        default=30,
        help="Number of seeds for random policy (default 30).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR,
        help=f"Results output directory (default: {RESULTS_DIR}).",
    )
    args = parser.parse_args()

    run_experiments(
        policy_names=args.policies,
        budget_fractions=args.budgets,
        random_seeds=args.random_seeds,
        results_dir=args.output,
    )


if __name__ == "__main__":
    main()
