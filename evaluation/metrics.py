"""
Evaluation metrics for regression test selection.

All metrics are computed from episode results. An episode is a single run
of a policy on one bug (one test-selection sequence).

References:
    - APFD: Rothermel et al. (2001) TSE
    - APFDc: Elbaum et al. (2001) ICSE  
    - Budget Success Rate: custom metric for this work
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional


# ─── Episode result ────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """Result of one policy episode on one bug."""
    project: str
    bug_id: str
    policy_name: str
    budget_fraction: float

    fault_detected: bool
    tests_executed: int           # Number of tests executed before stop/budget
    tests_to_detection: Optional[int]   # None if not detected
    time_to_detection: Optional[float]  # Wall-clock seconds, None if not detected
    total_time_used: float        # Total time consumed by executed tests
    total_suite_time: float       # Total time if all tests were run
    total_tests: int              # Size of full test suite

    # Ordered list of (test_id, was_triggering_test) for APFD computation
    execution_sequence: list[tuple[str, bool]] = field(default_factory=list)
    execution_costs: list[float] = field(default_factory=list)


# ─── Primary metrics ───────────────────────────────────────────────────────────

def fault_detection_rate(episodes: list[dict]) -> float:
    """
    Fraction of episodes where the fault was detected.

    Args:
        episodes: List of dicts with 'fault_detected' key.

    Returns:
        Float in [0, 1].
    """
    if not episodes:
        return 0.0
    return sum(1 for e in episodes if e.get("fault_detected")) / len(episodes)


def suite_reduction(selected: int, total: int) -> float:
    """
    Fraction of the test suite NOT executed.

    TSR = 1 - selected / total

    Args:
        selected: Number of tests executed.
        total: Total number of candidate tests.

    Returns:
        Float in [0, 1]. Higher is better (fewer tests run).
    """
    if total == 0:
        return 0.0
    return 1.0 - (selected / total)


def cost_reduction(time_used: float, total_suite_time: float) -> float:
    """
    Fraction of total test suite execution cost saved.

    CR = 1 - time_used / total_suite_time

    Returns:
        Float in [0, 1].
    """
    if total_suite_time <= 0:
        return 0.0
    return max(0.0, 1.0 - time_used / total_suite_time)


def apfd(outcomes: list[bool]) -> float:
    """
    Average Percentage of Faults Detected (APFD).

    APFD = 1 - (T_F / (n * m)) + (1 / (2 * n))

    Where:
        T_F = position of first fault-revealing test (1-indexed)
        n   = total number of tests in the sequence
        m   = number of faults (1 for single-fault setting)

    Args:
        outcomes: List of booleans where True means this test detected the fault.

    Returns:
        Float in [0, 1]. Higher is better.
    """
    if not outcomes:
        return 0.0

    n = len(outcomes)
    fault_positions = [i + 1 for i, v in enumerate(outcomes) if v]

    if not fault_positions:
        return 0.0

    # Single-fault setting: use first detection position
    # For multi-fault: sum over all faults
    m = len(fault_positions)
    sum_pos = sum(fault_positions)

    return 1.0 - (sum_pos / (n * m)) + (1.0 / (2.0 * n))


def apfdc(outcomes: list[bool], costs: list[float]) -> float:
    """
    Cost-aware APFD (APFDc).

    Formula (Elbaum et al. 2001):
        APFDc = (sum over each fault f: sum of costs of tests executed before f,
                 plus half the cost of the test that detected f) / (total_cost * m)

    Note: Higher APFDc means the fault was detected earlier relative to cost consumed.
    Cheaper tests detecting a fault yield higher APFDc.

    Args:
        outcomes: List of booleans (True = fault detected at this position).
        costs: Execution cost of each test (same length as outcomes).

    Returns:
        Float in [0, 1]. Higher is better.
    """
    if not outcomes or not costs or len(outcomes) != len(costs):
        return 0.0

    total_cost = sum(costs)
    if total_cost <= 0:
        return apfd(outcomes)

    fault_positions = [i for i, v in enumerate(outcomes) if v]
    if not fault_positions:
        return 0.0

    m = len(fault_positions)
    apfdc_sum = 0.0

    for fault_idx in fault_positions:
        # Cost of all tests executed BEFORE the detecting test
        cost_before = sum(costs[:fault_idx])
        # Half the cost of the detecting test itself
        cost_detecting = costs[fault_idx] / 2.0
        # Total cost budget NOT consumed before detection (inverted for APFDc)
        # APFDc measures how much of the suite we avoided running
        apfdc_sum += total_cost - cost_before - cost_detecting

    return apfdc_sum / (total_cost * m)


def budget_success_rate(
    episode_results: list[EpisodeResult],
    budget_fractions: list[float] = None,
) -> dict[float, float]:
    """
    Percentage of bugs detected within each budget level.

    Args:
        episode_results: List of episode results.
        budget_fractions: Budget levels to evaluate (e.g. [0.05, 0.10, 0.25, 0.50]).

    Returns:
        Dict mapping budget fraction → detection rate.
    """
    if budget_fractions is None:
        budget_fractions = [0.05, 0.10, 0.25, 0.50, 1.00]

    results: dict[float, float] = {}
    for bf in budget_fractions:
        within_budget = []
        for ep in episode_results:
            if ep.budget_fraction <= bf and ep.fault_detected:
                within_budget.append(True)
            else:
                within_budget.append(False)
        results[bf] = sum(within_budget) / len(within_budget) if within_budget else 0.0
    return results


# ─── Aggregate statistics ──────────────────────────────────────────────────────

def summarize_metric(values: list[float]) -> dict:
    """
    Compute descriptive statistics for a list of metric values.

    Returns:
        Dict with mean, median, std, min, max, and 95% CI.
    """
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}

    n = len(values)
    mean = statistics.mean(values)
    median = statistics.median(values)
    std = statistics.stdev(values) if n > 1 else 0.0

    # 95% bootstrap CI placeholder (full bootstrap in statistical.py)
    se = std / math.sqrt(n) if n > 0 else 0.0
    ci_lower = mean - 1.96 * se
    ci_upper = mean + 1.96 * se

    return {
        "mean": round(mean, 4),
        "median": round(median, 4),
        "std": round(std, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": n,
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
    }


def compute_all_metrics(episode_results: list[EpisodeResult]) -> dict:
    """
    Compute all primary metrics from a list of episode results.

    Returns:
        Nested dict of metric name → summary stats.
    """
    if not episode_results:
        return {}

    fdr = fault_detection_rate([{"fault_detected": e.fault_detected} for e in episode_results])
    ttds = [e.tests_to_detection for e in episode_results if e.tests_to_detection is not None]
    t2ds = [e.time_to_detection for e in episode_results if e.time_to_detection is not None]
    tsrs = [test_suite_reduction(e.tests_executed, e.total_tests) for e in episode_results]
    crs = [cost_reduction(e.total_time_used, e.total_suite_time) for e in episode_results]

    apfd_scores = []
    apfdc_scores = []
    for ep in episode_results:
        outcomes = [v for _, v in ep.execution_sequence]
        apfd_scores.append(apfd(outcomes))
        if ep.execution_costs:
            apfdc_scores.append(apfdc(outcomes, ep.execution_costs))

    return {
        "fault_detection_rate": fdr,
        "tests_to_detection": summarize_metric(ttds),
        "time_to_detection": summarize_metric(t2ds),
        "test_suite_reduction": summarize_metric(tsrs),
        "cost_reduction": summarize_metric(crs),
        "apfd": summarize_metric(apfd_scores),
        "apfdc": summarize_metric(apfdc_scores),
        "budget_success_rate": budget_success_rate(episode_results),
        "n_episodes": len(episode_results),
        "n_detected": sum(1 for e in episode_results if e.fault_detected),
    }
