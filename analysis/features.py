"""
Feature engineering — computes pre-episode and dynamic features for the agent.

CRITICAL LEAKAGE BOUNDARY:
  PRE_EPISODE_FEATURES: computed before any test is run. Safe for the policy.
  DYNAMIC_FEATURES: updated after each test execution. Safe only for the selected test.
  GROUND_TRUTH_LABELS: never exposed to the policy.

This module only produces PRE_EPISODE_FEATURES (static features computed
during dataset collection). Dynamic features are computed by environment/state.py
during simulation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from analysis.diff import DiffResult, ChangedEntity


# ─── Pre-episode feature set ───────────────────────────────────────────────────

@dataclass
class PreEpisodeFeatures:
    """
    Features that are available BEFORE the testing episode begins.

    These may be used by any policy, including static ML baselines.
    They must not contain any information from the test's own execution
    in the current episode.
    """
    test_id: str

    # Coverage-based features
    coverage_overlap_ratio: float = 0.0
    """Jaccard similarity between test's covered entities and changed entities."""

    coverage_entity_count: int = 0
    """Number of source entities covered by this test."""

    changed_file_in_coverage: bool = False
    """True if any changed file appears in this test's coverage."""

    changed_function_in_coverage: bool = False
    """True if any changed function appears in this test's coverage."""

    # Dependency features
    dependency_distance: int = -1
    """
    Minimum graph distance from a changed entity to any entity in this test's
    coverage. -1 means no path found (disconnected).
    """

    # Cost features
    estimated_runtime_seconds: float = 0.0
    """Estimated runtime from prior executions or median for this project."""

    # Historical features (from previous bugs in the same project)
    historical_failure_rate: float = 0.0
    """Fraction of previous bugs where this test detected the regression."""

    historical_execution_count: int = 0
    """Number of bugs in history where this test was included."""


# ─── Feature computation ───────────────────────────────────────────────────────

def compute_coverage_features(
    test_id: str,
    covered_entities: list[str],
    changed_entities: list[ChangedEntity],
    changed_files: list[str],
) -> tuple[float, int, bool, bool]:
    """
    Compute coverage-based features.

    Args:
        test_id: Test identifier.
        covered_entities: Entities covered by this test (files or functions).
        changed_entities: Entities changed between buggy and fixed versions.
        changed_files: Files changed between buggy and fixed versions.

    Returns:
        (coverage_overlap_ratio, coverage_entity_count,
         changed_file_in_coverage, changed_function_in_coverage)
    """
    covered_set = set(covered_entities)
    coverage_entity_count = len(covered_set)

    changed_function_names = {e.qualified_name for e in changed_entities}
    changed_file_set = set(changed_files)

    # File-level check: does any changed file appear in covered entities?
    changed_file_in_coverage = any(
        cf in covered_entity
        for cf in changed_file_set
        for covered_entity in covered_set
    )

    # Function-level check
    changed_function_in_coverage = bool(
        covered_set & changed_function_names
    )

    # Jaccard overlap between covered entities and all changed entities
    all_changed = changed_function_names | changed_file_set
    if all_changed and covered_set:
        intersection = len(covered_set & all_changed)
        union = len(covered_set | all_changed)
        coverage_overlap_ratio = intersection / union if union > 0 else 0.0
    else:
        coverage_overlap_ratio = 0.0

    return (
        round(coverage_overlap_ratio, 6),
        coverage_entity_count,
        changed_file_in_coverage,
        changed_function_in_coverage,
    )


def compute_dependency_distance(
    covered_entities: list[str],
    changed_entities: list[ChangedEntity],
    call_graph: Optional[dict[str, list[str]]] = None,
) -> int:
    """
    Compute the minimum graph distance from any changed entity to any
    entity in this test's coverage.

    Args:
        covered_entities: Entities covered by this test.
        changed_entities: Entities changed between versions.
        call_graph: Optional call graph {caller: [callees]}.
                    If None, a direct coverage overlap distance is used.

    Returns:
        Integer distance. 0 = direct overlap, 1 = one hop, -1 = disconnected.
    """
    changed_names = {e.qualified_name for e in changed_entities}
    covered_set = set(covered_entities)

    # Distance 0: direct overlap
    if covered_set & changed_names:
        return 0

    if call_graph is None:
        # No call graph — use file-level distance heuristic
        changed_files = {e.file_path for e in changed_entities}
        for entity in covered_set:
            for cf in changed_files:
                if cf in entity:
                    return 1
        return -1

    # BFS from changed entities through the call graph
    frontier = set(changed_names)
    visited = set(changed_names)
    distance = 0

    while frontier and distance < 5:  # Cap at depth 5
        distance += 1
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbour in call_graph.get(node, []):
                if neighbour in covered_set:
                    return distance
                if neighbour not in visited:
                    next_frontier.add(neighbour)
                    visited.add(neighbour)
        frontier = next_frontier

    return -1


def build_pre_episode_features(
    test_id: str,
    covered_entities: list[str],
    diff_result: DiffResult,
    estimated_runtime: float = 0.0,
    historical_failure_rate: float = 0.0,
    historical_execution_count: int = 0,
    call_graph: Optional[dict[str, list[str]]] = None,
) -> PreEpisodeFeatures:
    """
    Build the complete pre-episode feature set for one test.

    Args:
        test_id: Test identifier.
        covered_entities: Entities covered by this test.
        diff_result: Parsed diff between buggy and fixed commits.
        estimated_runtime: Estimated runtime in seconds.
        historical_failure_rate: Historical failure rate for this test.
        historical_execution_count: Number of times seen in history.
        call_graph: Optional call graph for dependency distance computation.

    Returns:
        PreEpisodeFeatures.
    """
    (
        overlap_ratio,
        entity_count,
        changed_file_hit,
        changed_fn_hit,
    ) = compute_coverage_features(
        test_id=test_id,
        covered_entities=covered_entities,
        changed_entities=diff_result.changed_entities,
        changed_files=diff_result.changed_files,
    )

    dep_distance = compute_dependency_distance(
        covered_entities=covered_entities,
        changed_entities=diff_result.changed_entities,
        call_graph=call_graph,
    )

    return PreEpisodeFeatures(
        test_id=test_id,
        coverage_overlap_ratio=overlap_ratio,
        coverage_entity_count=entity_count,
        changed_file_in_coverage=changed_file_hit,
        changed_function_in_coverage=changed_fn_hit,
        dependency_distance=dep_distance,
        estimated_runtime_seconds=round(estimated_runtime, 4),
        historical_failure_rate=round(historical_failure_rate, 6),
        historical_execution_count=historical_execution_count,
    )


def to_feature_vector(features: PreEpisodeFeatures) -> list[float]:
    """
    Convert PreEpisodeFeatures to a numeric vector for ML models.

    Returns:
        Fixed-length list of floats suitable for sklearn/numpy.
    """
    dep_dist_normalized = (
        features.dependency_distance / 5.0
        if features.dependency_distance >= 0
        else -1.0
    )
    return [
        features.coverage_overlap_ratio,
        min(features.coverage_entity_count / 100.0, 1.0),  # Normalize
        float(features.changed_file_in_coverage),
        float(features.changed_function_in_coverage),
        dep_dist_normalized,
        math.log1p(features.estimated_runtime_seconds),
        features.historical_failure_rate,
        min(features.historical_execution_count / 50.0, 1.0),  # Normalize
    ]


FEATURE_NAMES = [
    "coverage_overlap_ratio",
    "coverage_entity_count_norm",
    "changed_file_in_coverage",
    "changed_function_in_coverage",
    "dependency_distance_norm",
    "log_estimated_runtime",
    "historical_failure_rate",
    "historical_execution_count_norm",
]
