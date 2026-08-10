"""
Leave-One-Project-Out (LOPO) evaluation for the RL Agent.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from environment.episode import Episode
from evaluation.metrics import EpisodeResult
from policies.rl_policy import NeuralBanditPolicy
from policies.base import RevealedOutcome
from experiments.run_experiments import (
    load_all_records, init_db, insert_episode, build_summary_table,
    PROCESSED_DIR, RESULTS_DIR
)


def run_lopo_evaluation(
    budget_fractions: list[float],
    epochs: int = 5,
    processed_dir: Path = PROCESSED_DIR,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    records = load_all_records(processed_dir)
    if not records:
        print("ERROR: No records found.", file=sys.stderr)
        sys.exit(1)

    projects = sorted(list({r["project"] for r in records}))
    print(f"Projects for LOPO: {projects}")

    results_dir.mkdir(parents=True, exist_ok=True)
    db_path = results_dir / "lopo_results_db.sqlite"
    jsonl_path = results_dir / "lopo_episode_results.jsonl"
    db = init_db(db_path)

    all_results: dict[str, dict[float, list[EpisodeResult]]] = defaultdict(lambda: defaultdict(list))
    
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for target_proj in projects:
            print(f"\n{'='*60}")
            print(f"LOPO Target Project: {target_proj}")
            print(f"{'='*60}")
            
            train_records = [r for r in records if r["project"] != target_proj]
            eval_records = [r for r in records if r["project"] == target_proj]
            
            # Initialize RL Policy
            policy = NeuralBanditPolicy(epsilon=0.1, lr=0.01, train=True)
            
            # Pre-training phase
            print(f"  Pre-training on {len(train_records)} bugs from other projects (Epochs: {epochs})...")
            for epoch in range(epochs):
                random.shuffle(train_records)
                for record in train_records:
                    policy.reset()
                    # Train using 100% budget so it explores all tests
                    episode = Episode.from_record(record, 1.0, policy.name)
                    features = episode.get_features()
                    
                    ordered_ids = policy.select_tests(features, episode.get_budget_seconds())
                    result = episode.simulate(ordered_ids)
                    
                    # Construct revealed outcomes for update
                    revealed = []
                    for i, (tid, outcome) in enumerate(result.execution_sequence):
                        rt = result.execution_costs[i] if result.execution_costs else 0.0
                        revealed.append(RevealedOutcome(tid, outcome, rt))
                        
                    policy.update(revealed)
                    
            # Evaluation phase
            print(f"  Evaluating on {len(eval_records)} bugs from {target_proj}...")
            policy.train = False  # Freeze weights, no exploration
            
            for budget in budget_fractions:
                for record in eval_records:
                    policy.reset()
                    episode = Episode.from_record(record, budget, policy.name)
                    features = episode.get_features()
                    
                    ordered_ids = policy.select_tests(features, episode.get_budget_seconds())
                    result = episode.simulate(ordered_ids)
                    
                    # Compute APFD/APFDc
                    from evaluation.metrics import apfd as compute_apfd, apfdc as compute_apfdc
                    outcomes = [v for _, v in result.execution_sequence]
                    ep_apfd = compute_apfd(outcomes)
                    ep_apfdc = compute_apfdc(outcomes, result.execution_costs) if result.execution_costs else 0.0
                    
                    # Canonical name
                    canonical = "neural_bandit_lopo"
                    all_results[canonical][budget].append(result)
                    
                    insert_episode(db, result, ep_apfd, ep_apfdc)
                    ep_dict = asdict(result)
                    ep_dict["apfd"] = ep_apfd
                    ep_dict["apfdc"] = ep_apfdc
                    jsonl_file.write(json.dumps(ep_dict) + "\n")

    # Generate Markdown summary of just the RL policy
    table = build_summary_table(all_results)
    print(f"\n{'='*60}")
    print("LOPO RL Results:")
    print(table)
    
    return all_results

def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOPO evaluation for RL policy.")
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Training epochs on the out-of-bag data (default: 20)."
    )
    args = parser.parse_args()
    
    budgets = [0.05, 0.10, 0.25, 0.50, 1.00]
    run_lopo_evaluation(budgets, epochs=args.epochs)

if __name__ == "__main__":
    main()
