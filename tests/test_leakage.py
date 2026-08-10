"""
Automated leakage checks for experiment records.

These tests must pass before any experiment is run.
A leakage failure is a critical research integrity issue.

Run: pytest tests/test_leakage.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# ─── Fixtures ─────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
GROUND_TRUTH_KEYS = {"outcome_buggy", "outcome_fixed", "is_triggering_test", "actual_runtime_seconds"}
PRE_EPISODE_KEYS = {
    "estimated_runtime_seconds",
    "coverage_overlap_ratio",
    "coverage_entity_count",
    "dependency_distance",
    "changed_file_in_coverage",
    "changed_function_in_coverage",
    "historical_failure_rate",
    "historical_execution_count",
}


def _all_records() -> list[tuple[str, dict]]:
    """Load all processed experiment records."""
    records = []
    for json_file in PROCESSED_DIR.rglob("*.json"):
        try:
            record = json.loads(json_file.read_text(encoding="utf-8"))
            records.append((str(json_file), record))
        except json.JSONDecodeError:
            pass
    return records


def _collect_feature_keys(d: dict, prefix: str = "") -> set[str]:
    """Recursively collect all keys in a dict (flattened with dot notation)."""
    keys: set[str] = set()
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        keys.add(full_key)
        if isinstance(v, dict):
            keys |= _collect_feature_keys(v, full_key)
    return keys


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestLeakagePrevention:
    """
    Critical leakage checks. These represent research integrity requirements.
    Failure of any test here is a CRITICAL problem that must be fixed before
    running experiments.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_records(self):
        """Skip these tests if no processed records exist yet."""
        if not PROCESSED_DIR.exists() or not list(PROCESSED_DIR.rglob("*.json")):
            pytest.skip("No processed records found yet — run bugsinpy.process first")

    def test_ground_truth_not_in_pre_episode_features(self):
        """
        CRITICAL: Ground truth labels must not appear in PRE_EPISODE_FEATURES.
        
        This would allow the policy to see test outcomes before executing tests.
        """
        for path, record in _all_records():
            for test in record.get("tests", []):
                pre_keys = set(test.get("PRE_EPISODE_FEATURES", {}).keys())
                overlap = pre_keys & GROUND_TRUTH_KEYS
                assert not overlap, (
                    f"LEAKAGE DETECTED in {path}:\n"
                    f"  Ground truth keys found in PRE_EPISODE_FEATURES: {overlap}\n"
                    f"  Test: {test.get('test_id')}\n"
                    f"  This is a CRITICAL leakage — fix immediately."
                )

    def test_is_triggering_test_not_in_features(self):
        """
        CRITICAL: The 'is_triggering_test' flag must NEVER appear in
        PRE_EPISODE_FEATURES or anywhere a policy could access it.
        
        This is the most dangerous form of leakage — it tells the policy
        exactly which test to select.
        """
        for path, record in _all_records():
            for test in record.get("tests", []):
                pre = test.get("PRE_EPISODE_FEATURES", {})
                assert "is_triggering_test" not in pre, (
                    f"CRITICAL LEAKAGE: 'is_triggering_test' in PRE_EPISODE_FEATURES\n"
                    f"  File: {path}\n"
                    f"  Test: {test.get('test_id')}"
                )

    def test_outcome_not_in_pre_episode_features(self):
        """
        CRITICAL: Test outcomes (PASS/FAIL) from the current episode must not
        appear in PRE_EPISODE_FEATURES.
        """
        outcome_strings = {"PASS", "FAIL", "ERROR", "TIMEOUT"}
        for path, record in _all_records():
            for test in record.get("tests", []):
                pre = test.get("PRE_EPISODE_FEATURES", {})
                for key, value in pre.items():
                    if isinstance(value, str) and value.upper() in outcome_strings:
                        pytest.fail(
                            f"LEAKAGE: Outcome string '{value}' found in PRE_EPISODE_FEATURES.{key}\n"
                            f"  File: {path}\n"
                            f"  Test: {test.get('test_id')}"
                        )

    def test_all_pre_episode_keys_are_valid(self):
        """
        PRE_EPISODE_FEATURES must only contain expected keys.
        Any unexpected key is a potential leakage vector.
        """
        for path, record in _all_records():
            for test in record.get("tests", []):
                pre_keys = set(test.get("PRE_EPISODE_FEATURES", {}).keys())
                unexpected = pre_keys - PRE_EPISODE_KEYS
                assert not unexpected, (
                    f"Unexpected keys in PRE_EPISODE_FEATURES: {unexpected}\n"
                    f"  File: {path}\n"
                    f"  Test: {test.get('test_id')}\n"
                    f"  If these are new intentional features, add them to PRE_EPISODE_KEYS"
                )

    def test_ground_truth_section_is_present(self):
        """
        Every test record must have a GROUND_TRUTH_LABELS section.
        This ensures the evaluator can always access ground truth.
        """
        for path, record in _all_records():
            for test in record.get("tests", []):
                assert "GROUND_TRUTH_LABELS" in test, (
                    f"Missing GROUND_TRUTH_LABELS in {path}, test {test.get('test_id')}"
                )
                gt = test["GROUND_TRUTH_LABELS"]
                for required_key in ("outcome_buggy", "outcome_fixed", "is_triggering_test"):
                    assert required_key in gt, (
                        f"Missing ground truth key '{required_key}' in {path}"
                    )

    def test_pre_episode_features_section_is_present(self):
        """Every test record must have a PRE_EPISODE_FEATURES section."""
        for path, record in _all_records():
            for test in record.get("tests", []):
                assert "PRE_EPISODE_FEATURES" in test, (
                    f"Missing PRE_EPISODE_FEATURES in {path}, test {test.get('test_id')}"
                )

    def test_triggering_tests_are_marked(self):
        """
        At least one test in each SUCCESSFULLY_PROCESSED record must be
        marked as is_triggering_test=True (in GROUND_TRUTH_LABELS).
        """
        for path, record in _all_records():
            if record.get("dataset_status") != "SUCCESSFULLY_PROCESSED":
                continue
            triggering = [
                t for t in record.get("tests", [])
                if t.get("GROUND_TRUTH_LABELS", {}).get("is_triggering_test") is True
            ]
            assert triggering, (
                f"No triggering tests found in SUCCESSFULLY_PROCESSED record: {path}\n"
                f"  This may indicate a metadata parsing error."
            )

    def test_historical_failure_rate_uses_only_prior_bugs(self):
        """
        historical_failure_rate must be 0.0 for the first bug in a project
        (no prior history available).
        """
        for path, record in _all_records():
            if record.get("bug_id") == "1":
                for test in record.get("tests", []):
                    hr = test.get("PRE_EPISODE_FEATURES", {}).get("historical_failure_rate", 0.0)
                    # Bug 1 has no history — rate should be 0.0
                    assert hr == 0.0, (
                        f"historical_failure_rate is {hr} for bug_id=1 (should be 0.0):\n"
                        f"  File: {path}, Test: {test.get('test_id')}\n"
                        f"  This indicates history data from the current bug leaked into features."
                    )


class TestSchemaValidity:
    """Check that all processed records conform to the JSON schema."""

    def test_all_records_have_required_top_level_fields(self):
        required = {"schema_version", "project", "bug_id", "environment",
                    "changed_entities", "tests", "coverage_matrix", "dataset_status"}
        for path, record in _all_records():
            missing = required - set(record.keys())
            assert not missing, f"Missing required fields {missing} in {path}"

    def test_no_records_have_unknown_status(self):
        valid_statuses = {
            "AVAILABLE", "CHECKOUT_FAILED", "INSTALL_FAILED",
            "TEST_FAILED_INFRASTRUCTURE", "COVERAGE_FAILED",
            "NON_REPRODUCIBLE", "SUCCESSFULLY_PROCESSED"
        }
        for path, record in _all_records():
            status = record.get("dataset_status")
            assert status in valid_statuses, (
                f"Unknown dataset_status '{status}' in {path}"
            )
