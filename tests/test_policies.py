"""
Tests for all baseline policies.

Verifies correctness and the leakage contract:
  - Policies receive only PRE_EPISODE_FEATURES.
  - Policies return valid ordered test IDs.
  - Specific ordering properties for each policy.
"""
from __future__ import annotations

import json
import random
import pytest

from policies.base import RevealedOutcome
from policies.coverage_policy import CoveragePolicy
from policies.cost_weighted_policy import CostWeightedPolicy
from policies.full_suite import FullSuitePolicy
from policies.historical_policy import HistoricalPolicy
from policies.random_policy import RandomPolicy


# ─── Fixtures ──────────────────────────────────────────────────────────────────

def make_features(n: int, seed: int = 42) -> list[dict]:
    """Create n fake feature dicts."""
    rng = random.Random(seed)
    return [
        {
            "test_id": f"tests/test_foo.py::test_{i}",
            "test_file": "tests/test_foo.py",
            "test_function": f"test_{i}",
            "PRE_EPISODE_FEATURES": {
                "estimated_runtime_seconds": round(rng.uniform(0.05, 5.0), 4),
                "coverage_overlap_ratio": round(rng.uniform(0.0, 1.0) if i < n // 2 else 0.0, 4),
                "coverage_entity_count": rng.randint(0, 20),
                "dependency_distance": rng.choice([-1, 1, 2, 3]),
                "changed_file_in_coverage": i < n // 4,
                "changed_function_in_coverage": i < n // 8,
                "historical_failure_rate": round(rng.uniform(0.0, 0.5), 4),
                "historical_execution_count": rng.randint(0, 50),
            },
        }
        for i in range(n)
    ]


FEATURES_20 = make_features(20)
BUDGET = 30.0


# ─── FullSuitePolicy ───────────────────────────────────────────────────────────

class TestFullSuitePolicy:
    def test_returns_all_tests(self):
        policy = FullSuitePolicy()
        result = policy.select_tests(FEATURES_20, BUDGET)
        assert len(result) == 20

    def test_preserves_order(self):
        policy = FullSuitePolicy()
        expected_order = [f["test_id"] for f in FEATURES_20]
        assert policy.select_tests(FEATURES_20, BUDGET) == expected_order

    def test_returns_test_ids(self):
        policy = FullSuitePolicy()
        ids = policy.select_tests(FEATURES_20, BUDGET)
        assert all(isinstance(t, str) for t in ids)


# ─── RandomPolicy ──────────────────────────────────────────────────────────────

class TestRandomPolicy:
    def test_returns_all_tests(self):
        policy = RandomPolicy(seed=0)
        assert len(policy.select_tests(FEATURES_20, BUDGET)) == 20

    def test_deterministic_with_seed(self):
        p1 = RandomPolicy(seed=42)
        p2 = RandomPolicy(seed=42)
        assert p1.select_tests(FEATURES_20, BUDGET) == p2.select_tests(FEATURES_20, BUDGET)

    def test_different_seeds_produce_different_orders(self):
        orders = [RandomPolicy(seed=s).select_tests(FEATURES_20, BUDGET) for s in range(10)]
        # At least some should differ
        assert len(set(tuple(o) for o in orders)) > 1

    def test_reset_reproduces_same_order(self):
        policy = RandomPolicy(seed=7)
        first = policy.select_tests(FEATURES_20, BUDGET)
        policy.reset()
        second = policy.select_tests(FEATURES_20, BUDGET)
        assert first == second

    def test_does_not_access_ground_truth(self):
        """Policy must only use PRE_EPISODE_FEATURES."""
        policy = RandomPolicy(seed=0)
        # If it tries to access GROUND_TRUTH_LABELS, it will get a KeyError
        result = policy.select_tests(FEATURES_20, BUDGET)
        assert len(result) == 20


# ─── CoveragePolicy ────────────────────────────────────────────────────────────

class TestCoveragePolicy:
    def test_returns_all_tests(self):
        policy = CoveragePolicy()
        assert len(policy.select_tests(FEATURES_20, BUDGET)) == 20

    def test_high_coverage_first(self):
        """Tests with higher coverage_overlap_ratio must come first."""
        policy = CoveragePolicy()
        ids = policy.select_tests(FEATURES_20, BUDGET)
        feat_map = {f["test_id"]: f["PRE_EPISODE_FEATURES"] for f in FEATURES_20}

        # Find the split point between covered and uncovered
        covered_ids = [t for t in ids if feat_map[t]["coverage_overlap_ratio"] > 0]
        uncovered_ids = [t for t in ids if feat_map[t]["coverage_overlap_ratio"] == 0]

        if covered_ids and uncovered_ids:
            # All covered tests should appear before all uncovered tests
            last_covered_idx = ids.index(covered_ids[-1])
            first_uncovered_idx = ids.index(uncovered_ids[0])
            assert last_covered_idx < first_uncovered_idx

    def test_stable_on_empty_features(self):
        policy = CoveragePolicy()
        result = policy.select_tests([], BUDGET)
        assert result == []


# ─── HistoricalPolicy ──────────────────────────────────────────────────────────

class TestHistoricalPolicy:
    def test_returns_all_tests(self):
        policy = HistoricalPolicy()
        assert len(policy.select_tests(FEATURES_20, BUDGET)) == 20

    def test_high_failure_rate_first(self):
        """Tests with higher historical failure rate should be ranked first."""
        features = [
            {"test_id": "A", "PRE_EPISODE_FEATURES": {"historical_failure_rate": 0.8,
                                                        "estimated_runtime_seconds": 1.0}},
            {"test_id": "B", "PRE_EPISODE_FEATURES": {"historical_failure_rate": 0.2,
                                                        "estimated_runtime_seconds": 1.0}},
            {"test_id": "C", "PRE_EPISODE_FEATURES": {"historical_failure_rate": 0.5,
                                                        "estimated_runtime_seconds": 1.0}},
        ]
        policy = HistoricalPolicy()
        result = policy.select_tests(features, BUDGET)
        assert result == ["A", "C", "B"]


# ─── CostWeightedPolicy ────────────────────────────────────────────────────────

class TestCostWeightedPolicy:
    def test_returns_all_tests(self):
        policy = CostWeightedPolicy()
        assert len(policy.select_tests(FEATURES_20, BUDGET)) == 20

    def test_covered_tests_before_uncovered(self):
        """Tests with non-zero coverage must come before zero-coverage tests."""
        features = [
            {"test_id": "A", "PRE_EPISODE_FEATURES": {"coverage_overlap_ratio": 0.0,
                                                        "estimated_runtime_seconds": 0.1}},
            {"test_id": "B", "PRE_EPISODE_FEATURES": {"coverage_overlap_ratio": 0.5,
                                                        "estimated_runtime_seconds": 1.0}},
            {"test_id": "C", "PRE_EPISODE_FEATURES": {"coverage_overlap_ratio": 0.2,
                                                        "estimated_runtime_seconds": 0.5}},
        ]
        policy = CostWeightedPolicy()
        result = policy.select_tests(features, BUDGET)
        # B has score 0.5/1.0=0.5, C has score 0.2/0.5=0.4 → B before C before A
        assert result[0] == "B"
        assert result[1] == "C"
        assert result[2] == "A"

    def test_bang_per_buck_ordering(self):
        """Equal coverage, cheaper test should rank higher."""
        features = [
            {"test_id": "CHEAP", "PRE_EPISODE_FEATURES": {"coverage_overlap_ratio": 0.5,
                                                            "estimated_runtime_seconds": 0.5}},
            {"test_id": "EXPENSIVE", "PRE_EPISODE_FEATURES": {"coverage_overlap_ratio": 0.5,
                                                                "estimated_runtime_seconds": 5.0}},
        ]
        policy = CostWeightedPolicy()
        result = policy.select_tests(features, BUDGET)
        assert result[0] == "CHEAP"
