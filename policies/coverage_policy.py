"""
Baseline 2: Coverage-Greedy Policy.

Sorts tests by descending coverage_overlap_ratio (static coverage overlap
with the changed code entities). Ties are broken by ascending estimated runtime.

This mimics common industrial practice of prioritizing tests that cover the
changed files/functions without any historical information.
"""
from __future__ import annotations

from typing import Optional

from policies.base import Policy, RevealedOutcome


class CoveragePolicy(Policy):
    """
    Ranks tests by static coverage overlap with the code change.

    Score = coverage_overlap_ratio  (higher = more changed code covered)
    Tie-break = ascending estimated_runtime_seconds (cheaper tests first)

    This is a pure pre-episode, zero-shot baseline — requires no history.
    """

    @property
    def name(self) -> str:
        return "coverage_greedy"

    def select_tests(
        self,
        features: list[dict],
        budget_seconds: float,
        history: Optional[list[RevealedOutcome]] = None,
    ) -> list[str]:
        def score(f: dict) -> tuple:
            feat = f.get("PRE_EPISODE_FEATURES", {})
            overlap = feat.get("coverage_overlap_ratio", 0.0)
            runtime = feat.get("estimated_runtime_seconds", float("inf"))
            # Primary: descending overlap; secondary: ascending runtime
            return (-overlap, runtime)

        return [f["test_id"] for f in sorted(features, key=score)]
