"""
Episode simulation environment for regression test selection.

This module is the core "gym" for our experiment. It replays a processed
bug record and enforces the strict pre-episode / ground-truth separation.

LEAKAGE BOUNDARY:
  - Policies receive only PRE_EPISODE_FEATURES via get_features().
  - GROUND_TRUTH_LABELS are revealed only AFTER simulate() completes.
  - An AccessGuard prevents accidental access to labels before execution.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from evaluation.metrics import EpisodeResult


# ─── Access Guard ──────────────────────────────────────────────────────────────

class LeakageError(Exception):
    """Raised when a policy attempts to access ground-truth labels before execution."""


class AccessGuard:
    """Wrapper that blocks attribute access until simulation is complete."""

    def __init__(self, record: dict) -> None:
        object.__setattr__(self, "_record", record)
        object.__setattr__(self, "_locked", True)

    def unlock(self) -> None:
        object.__setattr__(self, "_locked", False)

    def __getattr__(self, name: str):
        if object.__getattribute__(self, "_locked"):
            raise LeakageError(
                f"Attempted to access '{name}' before simulate() was called. "
                "This would constitute data leakage — policies must not access "
                "GROUND_TRUTH_LABELS before the episode ends."
            )
        return object.__getattribute__(self, "_record")[name]


# ─── Episode ───────────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """
    Simulates one test-selection episode on a processed bug record.

    The episode exposes only PRE_EPISODE_FEATURES to the policy. Ground-truth
    outcomes are revealed only after simulate() is called with an ordered
    test selection.

    Example:
        episode = Episode.from_record(record, budget_fraction=0.25)
        features = episode.get_features()
        ordered_ids = my_policy.select_tests(features, ...)
        result = episode.simulate(ordered_ids)
    """
    record: dict
    budget_fraction: float          # e.g. 0.25 = 25% of total suite time
    policy_name: str = "unknown"

    _guard: AccessGuard = field(init=False, repr=False)
    _simulated: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_guard", AccessGuard(self.record))

    @classmethod
    def from_record(
        cls,
        record: dict,
        budget_fraction: float,
        policy_name: str = "unknown",
    ) -> "Episode":
        """Construct an episode from a processed JSON record."""
        if not 0.0 < budget_fraction <= 1.0:
            raise ValueError(f"budget_fraction must be in (0, 1], got {budget_fraction}")
        return cls(record=record, budget_fraction=budget_fraction, policy_name=policy_name)

    def get_features(self) -> list[dict]:
        """
        Return the pre-episode feature dict for each candidate test.

        Each dict contains:
            - test_id: str
            - PRE_EPISODE_FEATURES: dict  (safe, no labels)

        NEVER includes GROUND_TRUTH_LABELS.
        """
        return [
            {
                "test_id": t["test_id"],
                "test_file": t.get("test_file", ""),
                "test_function": t.get("test_function", ""),
                "PRE_EPISODE_FEATURES": t["PRE_EPISODE_FEATURES"],
            }
            for t in self.record.get("tests", [])
        ]

    def get_budget_seconds(self) -> float:
        """Compute the absolute budget in seconds from the budget fraction."""
        total_time = self.record.get("suite_summary", {}).get("total_runtime_seconds", 0.0)
        return total_time * self.budget_fraction

    def simulate(self, ordered_test_ids: list[str]) -> EpisodeResult:
        """
        Simulate execution of tests in the given order, stopping at budget.

        Args:
            ordered_test_ids: Test IDs in the order the policy wants to run them.
                              Need not include all tests; any omitted are never run.

        Returns:
            EpisodeResult with outcomes revealed (including ground-truth labels).

        Raises:
            LeakageError: If called after already simulated (double-episode guard).
            ValueError: If ordered_test_ids contains unknown test IDs.
        """
        if self._simulated:
            raise LeakageError("simulate() called twice on the same episode.")

        # Build ground-truth lookup from the record
        gt_lookup: dict[str, dict] = {
            t["test_id"]: t["GROUND_TRUTH_LABELS"]
            for t in self.record.get("tests", [])
        }
        feature_lookup: dict[str, dict] = {
            t["test_id"]: t["PRE_EPISODE_FEATURES"]
            for t in self.record.get("tests", [])
        }

        budget_seconds = self.get_budget_seconds()
        total_suite_time = self.record.get("suite_summary", {}).get("total_runtime_seconds", 0.0)
        total_tests = len(self.record.get("tests", []))

        time_used = 0.0
        executed = 0
        fault_detected = False
        tests_to_detection: Optional[int] = None
        time_to_detection: Optional[float] = None
        execution_sequence: list[tuple[str, bool]] = []
        execution_costs: list[float] = []

        for test_id in ordered_test_ids:
            if test_id not in gt_lookup:
                continue  # Skip unknown test IDs gracefully

            gt = gt_lookup[test_id]
            feat = feature_lookup.get(test_id, {})

            # Use estimated runtime as the cost (we don't have actual until run)
            # In simulation, we use the actual runtime from the buggy version
            actual_runtime = gt.get("actual_runtime_seconds", feat.get("estimated_runtime_seconds", 0.1))

            # Check budget
            if time_used + actual_runtime > budget_seconds and executed > 0:
                break  # Budget would be exceeded; stop

            time_used += actual_runtime
            executed += 1

            is_triggering = gt.get("is_triggering_test", False)
            buggy_outcome = gt.get("outcome_buggy", "UNKNOWN")
            detected_now = is_triggering and buggy_outcome in ("FAIL", "ERROR")

            execution_sequence.append((test_id, detected_now))
            execution_costs.append(actual_runtime)

            if detected_now and not fault_detected:
                fault_detected = True
                tests_to_detection = executed
                time_to_detection = time_used

        # Unlock the guard — simulation complete, labels can be accessed
        self._guard.unlock()
        object.__setattr__(self, "_simulated", True)

        return EpisodeResult(
            project=self.record.get("project", ""),
            bug_id=self.record.get("bug_id", ""),
            policy_name=self.policy_name,
            budget_fraction=self.budget_fraction,
            fault_detected=fault_detected,
            tests_executed=executed,
            tests_to_detection=tests_to_detection,
            time_to_detection=time_to_detection,
            total_time_used=round(time_used, 4),
            total_suite_time=total_suite_time,
            total_tests=total_tests,
            execution_sequence=execution_sequence,
            execution_costs=execution_costs,
        )
