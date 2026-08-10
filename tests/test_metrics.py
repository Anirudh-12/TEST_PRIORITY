"""
Unit tests for metric computation.

These use synthetic data to verify metric correctness.
"""
from __future__ import annotations
import pytest
from evaluation.metrics import apfd, apfdc, fault_detection_rate, suite_reduction


class TestAPFD:
    def test_perfect_ordering(self):
        """Triggering test ranked first → maximum APFD."""
        # 5 tests, triggering test is test 1
        outcomes = [True, False, False, False, False]
        score = apfd(outcomes)
        # APFD = 1 - (1/5) + (1/(2*5)) = 0.9
        assert abs(score - 0.9) < 0.01

    def test_worst_ordering(self):
        """Triggering test ranked last → minimum APFD."""
        outcomes = [False, False, False, False, True]
        score = apfd(outcomes)
        # APFD = 1 - (5/5) + (1/(2*5)) = 0.1
        assert abs(score - 0.1) < 0.01

    def test_random_ordering(self):
        """APFD should be ~0.5 for random ordering (expected value)."""
        outcomes = [False, True, False, False, False]
        score = apfd(outcomes)
        # Triggering test at position 2/5: APFD = 1 - 2/5 + 1/10 = 0.7
        assert abs(score - 0.7) < 0.01

    def test_no_detection(self):
        """No detection → APFD = 0."""
        outcomes = [False, False, False, False, False]
        assert apfd(outcomes) == 0.0

    def test_empty(self):
        """Empty list → APFD = 0."""
        assert apfd([]) == 0.0


class TestAPFDc:
    def test_cost_aware_ordering(self):
        """APFDc: fault detected at position 1, total cost 5, half cost = 0.5
        APFDc = (total - 0 - 0.5) / (5 * 1) = 4.5/5 = 0.9"""
        outcomes = [True, False, False, False, False]
        costs = [1.0, 1.0, 1.0, 1.0, 1.0]
        score_c = apfdc(outcomes, costs)
        assert abs(score_c - 0.9) < 0.01

    def test_worst_case_apfdc(self):
        """APFDc: fault detected last, most cost consumed before detection."""
        outcomes = [False, False, False, False, True]
        costs = [1.0, 1.0, 1.0, 1.0, 1.0]
        score = apfdc(outcomes, costs)
        # APFDc = (5 - 4 - 0.5) / (5*1) = 0.5/5 = 0.1
        assert abs(score - 0.1) < 0.01

    def test_cheaper_failing_test_better(self):
        """With 2 tests: cheap fault detector (cost 0.1) vs expensive (cost 1.0).
        Cheap: APFDc = (1.1 - 0 - 0.05) / 1.1 ≈ 0.955
        Expensive: APFDc = (1.1 - 0 - 0.5) / 1.1 ≈ 0.545
        Cheap should be higher APFDc."""
        # cheap fault test first, expensive second
        outcomes_cheap_first = [True, False]
        costs_cheap_first = [0.1, 1.0]
        # expensive fault test first, cheap second
        outcomes_exp_first = [True, False]
        costs_exp_first = [1.0, 0.1]
        score_cheap = apfdc(outcomes_cheap_first, costs_cheap_first)
        score_exp = apfdc(outcomes_exp_first, costs_exp_first)
        assert score_cheap > score_exp


class TestFaultDetectionRate:
    def test_all_detected(self):
        episodes = [
            {"fault_detected": True},
            {"fault_detected": True},
            {"fault_detected": True},
        ]
        assert fault_detection_rate(episodes) == 1.0

    def test_none_detected(self):
        episodes = [
            {"fault_detected": False},
            {"fault_detected": False},
        ]
        assert fault_detection_rate(episodes) == 0.0

    def test_partial(self):
        episodes = [
            {"fault_detected": True},
            {"fault_detected": False},
            {"fault_detected": True},
            {"fault_detected": False},
        ]
        assert abs(fault_detection_rate(episodes) - 0.5) < 0.01

    def test_empty(self):
        assert fault_detection_rate([]) == 0.0


class TestTestSuiteReduction:
    def test_full_reduction(self):
        assert suite_reduction(selected=0, total=10) == 1.0

    def test_no_reduction(self):
        assert suite_reduction(selected=10, total=10) == 0.0

    def test_half_reduction(self):
        assert abs(suite_reduction(selected=5, total=10) - 0.5) < 0.01

    def test_zero_total(self):
        assert suite_reduction(selected=0, total=0) == 0.0
