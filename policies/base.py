"""
Abstract base class for all test-selection policies.

A policy receives only PRE_EPISODE_FEATURES and must return an ordered
list of test IDs. It must NEVER access GROUND_TRUTH_LABELS.

Leakage contract:
  - `select_tests()` receives `features` which contains only PRE_EPISODE_FEATURES.
  - `update()` receives revealed outcomes (PASS/FAIL strings) from already-executed
    tests. This is NOT leakage — the policy observed these outcomes by running the test.
  - The policy must NEVER inspect the episode record directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RevealedOutcome:
    """A test outcome revealed to the policy after execution (not leakage)."""
    test_id: str
    outcome: str        # "PASS", "FAIL", "ERROR", "TIMEOUT", "SKIP"
    runtime_seconds: float


class Policy(ABC):
    """
    Abstract base for all test-selection policies.

    Subclasses implement one required method:
        select_tests(features, budget_seconds, history) -> list[str]

    And one optional method:
        update(revealed) -> None   (for online/adaptive policies)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable policy name (used in results tables)."""

    @abstractmethod
    def select_tests(
        self,
        features: list[dict],
        budget_seconds: float,
        history: Optional[list[RevealedOutcome]] = None,
    ) -> list[str]:
        """
        Return an ordered list of test IDs to execute.

        The simulator will run tests in this order, stopping when the budget
        is exhausted. Tests not included in the list are never run.

        Args:
            features: List of feature dicts, each with:
                - "test_id": str
                - "test_file": str
                - "test_function": str
                - "PRE_EPISODE_FEATURES": dict
            budget_seconds: Remaining execution budget in seconds.
            history: Outcomes from previous bugs in the sequence (for online learning).
                     None for the first bug. Never contains GROUND_TRUTH_LABELS —
                     only outcomes the policy observed after executing tests.

        Returns:
            Ordered list of test IDs (first = highest priority).
        """

    def update(self, revealed: list[RevealedOutcome]) -> None:
        """
        Called after each bug's episode with the revealed test outcomes.

        Offline (static) policies ignore this. Online policies use this to
        update their model before processing the next bug.

        Args:
            revealed: List of RevealedOutcome for all executed tests.
                      IMPORTANT: This only includes tests that were ACTUALLY
                      executed within the budget. Tests not executed are NOT
                      revealed (the policy cannot know their outcomes).
        """

    def reset(self) -> None:
        """Reset any per-episode state. Called before each new episode."""
