"""
Baseline 3: Historical Failure Rate Policy.

Sorts tests by descending historical_failure_rate, which is computed from
prior bugs in the dataset (exponentially weighted). Represents the best
"pure history" approach — close to what RETECS (Spieker et al. 2017) uses.

This requires at least some historical data. For the first bug in a project
(cold start), it degrades to random ordering of tests.
"""
from __future__ import annotations

from typing import Optional

from policies.base import Policy, RevealedOutcome


class HistoricalPolicy(Policy):
    """
    Ranks tests by their historical failure rate across prior bugs.

    The historical_failure_rate in PRE_EPISODE_FEATURES is computed during
    dataset construction from earlier bugs only (no leakage).

    Tie-break: ascending estimated_runtime_seconds (cheaper first).
    """

    @property
    def name(self) -> str:
        return "historical_failure_rate"

    def select_tests(
        self,
        features: list[dict],
        budget_seconds: float,
        history: Optional[list[RevealedOutcome]] = None,
    ) -> list[str]:
        def score(f: dict) -> tuple:
            feat = f.get("PRE_EPISODE_FEATURES", {})
            rate = feat.get("historical_failure_rate", 0.0)
            runtime = feat.get("estimated_runtime_seconds", float("inf"))
            return (-rate, runtime)

        return [f["test_id"] for f in sorted(features, key=score)]
