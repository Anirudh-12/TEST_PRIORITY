"""
Baseline 4: Cost-Weighted Coverage Policy.

Uses a composite score that balances coverage relevance against execution cost:

    score = coverage_overlap_ratio / estimated_runtime_seconds

Tests that cover the most changed code in the least time are prioritised first.
This is the strongest hand-engineered baseline and represents the "upper bound"
that we expect a learned ML policy to outperform.

Special handling for zero-coverage tests and zero-runtime estimates.
"""
from __future__ import annotations

from typing import Optional

from policies.base import Policy, RevealedOutcome


class CostWeightedPolicy(Policy):
    """
    Ranks tests by coverage_overlap_ratio divided by estimated_runtime_seconds.

    This embodies the classic "bang-per-buck" heuristic from cost-aware TCP
    literature (Elbaum et al. 2001, Yoo & Harman survey 2012).

    Tests with zero coverage_overlap_ratio are placed last, ordered by ascending
    runtime (cheap unknown tests before expensive unknown tests).
    """

    @property
    def name(self) -> str:
        return "cost_weighted_coverage"

    def select_tests(
        self,
        features: list[dict],
        budget_seconds: float,
        history: Optional[list[RevealedOutcome]] = None,
    ) -> list[str]:
        covered = []
        uncovered = []

        for f in features:
            feat = f.get("PRE_EPISODE_FEATURES", {})
            overlap = feat.get("coverage_overlap_ratio", 0.0)
            runtime = max(feat.get("estimated_runtime_seconds", 0.001), 1e-6)

            if overlap > 0.0:
                score = overlap / runtime
                covered.append((score, runtime, f["test_id"]))
            else:
                uncovered.append((runtime, f["test_id"]))

        # Sort covered tests by descending score (best bang-per-buck first)
        covered.sort(key=lambda x: (-x[0], x[1]))
        # Sort uncovered tests by ascending runtime (cheapest first)
        uncovered.sort(key=lambda x: x[0])

        return [t[2] for t in covered] + [t[1] for t in uncovered]
