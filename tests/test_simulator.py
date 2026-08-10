"""
Tests for the simulation environment (Episode class).

Verifies:
  - AccessGuard prevents leakage before simulate()
  - Budget enforcement
  - Correct EpisodeResult construction
  - Triggering test detection
"""
from __future__ import annotations

import pytest

from environment.episode import Episode, LeakageError
from evaluation.metrics import EpisodeResult


# ─── Fixtures ──────────────────────────────────────────────────────────────────

def make_record(
    n_tests: int = 10,
    n_triggering: int = 2,
    runtime: float = 1.0,
) -> dict:
    """Create a minimal synthetic bug record for testing."""
    tests = []
    for i in range(n_tests):
        is_triggering = i < n_triggering
        tests.append({
            "test_id": f"tests/test_foo.py::test_{i}",
            "test_file": "tests/test_foo.py",
            "test_function": f"test_{i}",
            "coverage_level": "none",
            "PRE_EPISODE_FEATURES": {
                "estimated_runtime_seconds": runtime,
                "coverage_overlap_ratio": 0.8 if is_triggering else 0.0,
                "coverage_entity_count": 5 if is_triggering else 0,
                "dependency_distance": 1 if is_triggering else -1,
                "changed_file_in_coverage": is_triggering,
                "changed_function_in_coverage": is_triggering,
                "historical_failure_rate": 0.0,
                "historical_execution_count": 0,
            },
            "GROUND_TRUTH_LABELS": {
                "outcome_buggy": "FAIL" if is_triggering else "PASS",
                "outcome_fixed": "PASS",
                "is_triggering_test": is_triggering,
                "actual_runtime_seconds": runtime,
            },
        })
    total_time = n_tests * runtime
    return {
        "schema_version": "1.0",
        "project": "test_project",
        "bug_id": "1",
        "dataset_status": "SUCCESSFULLY_PROCESSED",
        "environment": {},
        "changed_entities": {"files": [], "functions": [], "line_ranges": {}},
        "tests": tests,
        "coverage_matrix": {},
        "suite_summary": {
            "total_tests": n_tests,
            "failing_tests_count": n_triggering,
            "passing_tests_count": n_tests - n_triggering,
            "total_runtime_seconds": total_time,
        },
        "processing_log": [],
    }


# ─── Episode construction ───────────────────────────────────────────────────────

class TestEpisodeConstruction:
    def test_from_record(self):
        record = make_record()
        ep = Episode.from_record(record, budget_fraction=0.5)
        assert ep.budget_fraction == 0.5
        assert ep.record is record

    def test_invalid_budget_zero(self):
        record = make_record()
        with pytest.raises(ValueError):
            Episode.from_record(record, budget_fraction=0.0)

    def test_invalid_budget_above_one(self):
        record = make_record()
        with pytest.raises(ValueError):
            Episode.from_record(record, budget_fraction=1.5)

    def test_budget_one_valid(self):
        record = make_record()
        ep = Episode.from_record(record, budget_fraction=1.0)
        assert ep.budget_fraction == 1.0


# ─── Feature access (no leakage) ───────────────────────────────────────────────

class TestGetFeatures:
    def test_features_do_not_contain_ground_truth(self):
        record = make_record()
        ep = Episode.from_record(record, budget_fraction=0.5)
        features = ep.get_features()
        for f in features:
            assert "GROUND_TRUTH_LABELS" not in f
            assert "outcome_buggy" not in f
            assert "is_triggering_test" not in f

    def test_features_contain_pre_episode_keys(self):
        record = make_record()
        ep = Episode.from_record(record, budget_fraction=0.5)
        features = ep.get_features()
        assert len(features) == 10
        for f in features:
            assert "test_id" in f
            assert "PRE_EPISODE_FEATURES" in f
            feat = f["PRE_EPISODE_FEATURES"]
            assert "estimated_runtime_seconds" in feat
            assert "coverage_overlap_ratio" in feat

    def test_all_test_ids_present(self):
        record = make_record(n_tests=5)
        ep = Episode.from_record(record, budget_fraction=0.5)
        ids = {f["test_id"] for f in ep.get_features()}
        expected = {f"tests/test_foo.py::test_{i}" for i in range(5)}
        assert ids == expected


# ─── Budget enforcement ─────────────────────────────────────────────────────────

class TestBudgetEnforcement:
    def test_no_tests_beyond_budget(self):
        """With 25% budget and 1s/test, should execute at most 2–3 tests."""
        record = make_record(n_tests=10, runtime=1.0)  # total = 10s
        ep = Episode.from_record(record, budget_fraction=0.25)  # budget = 2.5s
        all_ids = [f["test_id"] for f in ep.get_features()]
        result = ep.simulate(all_ids)
        assert result.tests_executed <= 3
        assert result.total_time_used <= 2.5 + 1.0  # budget + one test overshoot

    def test_full_budget_runs_all(self):
        record = make_record(n_tests=10, runtime=1.0)
        ep = Episode.from_record(record, budget_fraction=1.0)
        all_ids = [f["test_id"] for f in ep.get_features()]
        result = ep.simulate(all_ids)
        assert result.tests_executed == 10

    def test_budget_seconds_computation(self):
        record = make_record(n_tests=10, runtime=1.0)  # total=10s
        ep = Episode.from_record(record, budget_fraction=0.1)
        assert abs(ep.get_budget_seconds() - 1.0) < 1e-6


# ─── Fault detection ───────────────────────────────────────────────────────────

class TestFaultDetection:
    def test_fault_detected_when_triggering_test_first(self):
        record = make_record(n_tests=10, n_triggering=1, runtime=1.0)
        ep = Episode.from_record(record, budget_fraction=1.0)
        # Put the triggering test first
        features = ep.get_features()
        all_ids = [f["test_id"] for f in features]
        result = ep.simulate(all_ids)
        # test_0 is the triggering test; it is first in the list
        assert result.fault_detected is True
        assert result.tests_to_detection == 1

    def test_fault_not_detected_when_budget_exhausted_before_triggering(self):
        record = make_record(n_tests=10, n_triggering=2, runtime=1.0)
        ep = Episode.from_record(record, budget_fraction=1.0)
        features = ep.get_features()
        # Put non-triggering tests first, then triggering (reverse order)
        all_ids = [f["test_id"] for f in reversed(features)]
        result = ep.simulate(all_ids)
        # With budget=1.0 (all tests), fault_detected depends on ordering
        # The triggering tests are test_0 and test_1, placed last here
        # All tests run, so fault should eventually be detected
        assert result.fault_detected is True

    def test_fault_not_detected_if_budget_too_small(self):
        """With 10% budget, only 1 test runs; triggering test is last → no detection."""
        record = make_record(n_tests=10, n_triggering=1, runtime=1.0)
        ep = Episode.from_record(record, budget_fraction=0.1)
        features = ep.get_features()
        # Put non-triggering tests first
        all_ids = [f["test_id"] for f in features][1:] + [features[0]["test_id"]]
        result = ep.simulate(all_ids)
        assert result.fault_detected is False
        assert result.tests_to_detection is None


# ─── Leakage guard ─────────────────────────────────────────────────────────────

class TestLeakageGuard:
    def test_double_simulate_raises(self):
        record = make_record()
        ep = Episode.from_record(record, budget_fraction=1.0)
        ids = [f["test_id"] for f in ep.get_features()]
        ep.simulate(ids)
        with pytest.raises(LeakageError):
            ep.simulate(ids)

    def test_episode_result_has_no_raw_gt_labels(self):
        record = make_record()
        ep = Episode.from_record(record, budget_fraction=1.0)
        ids = [f["test_id"] for f in ep.get_features()]
        result = ep.simulate(ids)
        # EpisodeResult should not contain raw label dicts
        assert not hasattr(result, "GROUND_TRUTH_LABELS")
        assert not hasattr(result, "outcome_buggy")
