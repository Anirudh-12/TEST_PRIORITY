"""
Statistical Validation script for RL Test Prioritization.
Computes Wilcoxon Signed-Rank tests and Vargha-Delaney A12 effect sizes
comparing the RL agent (neural_bandit_lopo) against heuristic baselines.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, rankdata
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "data" / "results"
BASELINE_JSONL = RESULTS_DIR / "episode_results.jsonl"
LOPO_JSONL = RESULTS_DIR / "lopo_episode_results.jsonl"
STATIC_JSONL = RESULTS_DIR / "static_episode_results.jsonl"

def load_results() -> dict[str, dict[float, dict[str, float]]]:
    """
    Returns nested structure:
    results[policy_name][budget_fraction][bug_id] = apfdc_score
    """
    results: dict[str, dict[float, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    
    def parse_file(path: Path):
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                pol = data.get("policy_name")
                # Group random seeds
                if pol.startswith("random_seed"):
                    pol = "random"
                
                budget = float(data.get("budget_fraction"))
                bug_key = f"{data['project']}_{data['bug_id']}"
                apfdc = float(data.get("apfdc", 0.0))
                
                if pol == "random":
                    # Average over random seeds if multiple
                    if bug_key not in results[pol][budget]:
                        results[pol][budget][bug_key] = []
                    results[pol][budget][bug_key].append(apfdc)
                else:
                    results[pol][budget][bug_key] = apfdc

    parse_file(BASELINE_JSONL)
    parse_file(LOPO_JSONL)
    parse_file(STATIC_JSONL)
    
    # Average random baseline across seeds
    for budget in results["random"]:
        for bug in results["random"][budget]:
            if isinstance(results["random"][budget][bug], list):
                vals = results["random"][budget][bug]
                results["random"][budget][bug] = sum(vals)/len(vals)
                
    return results


def vargha_delaney_A12(x: list[float], y: list[float]) -> float:
    """
    Computes Vargha-Delaney A12 effect size.
    A12 > 0.5 means x is generally higher than y.
    A12 < 0.5 means y is generally higher than x.
    """
    m = len(x)
    n = len(y)
    if m == 0 or n == 0:
        return 0.5
    r = rankdata(x + y)
    r_x = sum(r[:m])
    return (r_x / m - (m + 1) / 2) / n


def get_effect_size_label(a12: float) -> str:
    """Get standard magnitude label for A12."""
    mag = abs(a12 - 0.5)
    if mag >= 0.21:
        return "Large"
    elif mag >= 0.14:
        return "Medium"
    elif mag >= 0.06:
        return "Small"
    return "Negligible"


def run_statistical_validation():
    results = load_results()
    if not results:
        print("No results found. Run experiments first.")
        return

    budgets = sorted({b for pol in results.values() for b in pol.keys()})
    baselines = [p for p in results.keys() if p != "neural_bandit"]
    
    if "neural_bandit" not in results:
        print("Error: RL LOPO results not found.")
        return

    console = Console()
    console.print("\n[bold cyan]Statistical Validation: RL Agent vs Baselines[/bold cyan]\n")
    
    for budget in budgets:
        table = Table(title=f"Budget = {budget:.0%}", show_header=True, header_style="bold magenta")
        table.add_column("Baseline Policy", style="dim", width=20)
        table.add_column("N Bugs", justify="right")
        table.add_column("Wilcoxon p-value", justify="right")
        table.add_column("A12 Effect Size", justify="right")
        table.add_column("Magnitude", justify="left")
        table.add_column("Winner", justify="center")
        
        rl_scores_dict = results["neural_bandit"].get(budget, {})
        
        for baseline in baselines:
            base_scores_dict = results[baseline].get(budget, {})
            
            # Align pairs exactly by bug_id
            common_bugs = set(rl_scores_dict.keys()).intersection(base_scores_dict.keys())
            if not common_bugs:
                continue
                
            rl_array = [rl_scores_dict[b] for b in sorted(common_bugs)]
            base_array = [base_scores_dict[b] for b in sorted(common_bugs)]
            
            n = len(common_bugs)
            
            # Check if arrays are identical (p-value undefined/irrelevant)
            if np.allclose(rl_array, base_array):
                p_val = 1.0
                a12 = 0.5
            else:
                try:
                    res = wilcoxon(rl_array, base_array, alternative="two-sided")
                    p_val = res.pvalue
                except ValueError:
                    p_val = 1.0 # E.g., all differences are zero
                
                a12 = vargha_delaney_A12(rl_array, base_array)
                
            label = get_effect_size_label(a12)
            
            winner = "-"
            if p_val < 0.05:
                if a12 > 0.5:
                    winner = "[green]RL Agent[/green]"
                elif a12 < 0.5:
                    winner = f"[red]{baseline}[/red]"
                    
            table.add_row(
                baseline,
                str(n),
                f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
                f"{a12:.3f}",
                label,
                winner
            )
            
        console.print(table)
        console.print()


if __name__ == "__main__":
    run_statistical_validation()
