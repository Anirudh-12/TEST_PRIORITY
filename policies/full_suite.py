"""
Baseline 0: Full Suite Policy.

Runs all candidate tests in arbitrary (discovery) order up to budget.
This is the performance ceiling — FDR should approach 1.0 when budget=1.0.
Used as a normalization reference.
"""
from __future__ import annotations
from typing import Optional
from policies.base import Policy, RevealedOutcome


class FullSuitePolicy(Policy):
    """
    Runs tests in their original discovery order (no reordering).
    Represents the default behaviour of most CI systems with no prioritization.
    """

    @property
    def name(self) -> str:
        return "full_suite"

    def select_tests(
        self,
        features: list[dict],
        budget_seconds: float,
        history: Optional[list[RevealedOutcome]] = None,
    ) -> list[str]:
        return [f["test_id"] for f in features]
