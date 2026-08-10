"""
Contextual Bandit (Neural Network) Policy for Test Selection.

Language-agnostic: all features are normalized continuous values in [0, 1],
so this policy works identically for Python (BugsInPy) and Java (Defects4J)
or any future language benchmark, as long as the feature extractor populates
the standard PRE_EPISODE_FEATURES schema.

Features used (all normalized to [0, 1]):
  - coverage_overlap_ratio     : fraction of changed lines touched by this test
  - estimated_runtime_norm     : log-normalized runtime (lang-agnostic cost proxy)
  - historical_failure_rate    : fraction of past runs where this test failed
  - files_changed_norm         : normalized number of changed files (commit scope)
"""
from __future__ import annotations

import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from policies.base import Policy, RevealedOutcome

# Max runtime cap for log-normalization (seconds). Tests exceeding this are
# clipped. 600s = 10 min, a safe ceiling for both Python and Java suites.
_MAX_RUNTIME_SECONDS = 600.0
# Max files changed for normalization. Commits with > this are clipped.
_MAX_FILES_CHANGED = 20.0


def _log_normalize_runtime(runtime_seconds: float) -> float:
    """Map runtime to [0, 1] using log scale — language-agnostic cost proxy."""
    clipped = min(max(runtime_seconds, 0.0), _MAX_RUNTIME_SECONDS)
    return math.log1p(clipped) / math.log1p(_MAX_RUNTIME_SECONDS)


def _normalize_files_changed(n: int) -> float:
    """Map number of changed files to [0, 1]."""
    return min(n, _MAX_FILES_CHANGED) / _MAX_FILES_CHANGED


class NeuralBanditNet(nn.Module):
    """
    Lightweight MLP Q-value estimator.

    Input:  4-dimensional normalized feature vector (language-agnostic).
    Output: Scalar Q-value (expected reward for executing this test next).
    """
    INPUT_DIM = 4

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralBanditPolicy(Policy):
    """
    Online Contextual Bandit policy using epsilon-greedy exploration.

    Reward function:
        - FAIL (bug found):  +10.0  (large, clear fault-seeking signal)
        - PASS (no bug):     -0.1 × budget_fraction_consumed  (soft cost penalty)

    This reward is intentionally language-agnostic — the magnitudes are
    fixed constants independent of language or framework.

    The policy is updated in batched mode at the end of each episode
    (after all executed tests have been revealed), which is stable and
    standard practice for Contextual Bandits.
    """

    def __init__(self, epsilon: float = 0.1, lr: float = 0.001, train: bool = True):
        super().__init__()
        self._name = "neural_bandit"
        self.epsilon = epsilon
        self.train = train

        self.model = NeuralBanditNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Per-episode state — cleared in reset()
        self._last_features: dict[str, torch.Tensor] = {}
        self._last_budget: float = 1.0

    @property
    def name(self) -> str:
        if self.train:
            return f"{self._name}_eps{self.epsilon}"
        return self._name

    def _extract_features(self, feat: dict) -> torch.Tensor:
        """
        Build a normalized, language-agnostic feature vector from a test record.

        All values are guaranteed to be in [0, 1] regardless of language.
        """
        pef = feat.get("PRE_EPISODE_FEATURES", {})

        coverage = float(pef.get("coverage_overlap_ratio", 0.0))
        runtime_norm = _log_normalize_runtime(
            float(pef.get("estimated_runtime_seconds", 0.0))
        )
        history = float(pef.get("historical_failure_rate", 0.0))
        files_norm = _normalize_files_changed(
            int(pef.get("files_changed_count", 1))
        )

        return torch.tensor(
            [coverage, runtime_norm, history, files_norm],
            dtype=torch.float32,
        )

    def select_tests(
        self,
        features: list[dict],
        budget_seconds: float,
        history: Optional[list[RevealedOutcome]] = None,
    ) -> list[str]:
        if not features:
            return []

        self._last_budget = max(budget_seconds, 1e-5)

        test_ids: list[str] = []
        X_list: list[torch.Tensor] = []

        for f in features:
            tid = f["test_id"]
            x = self._extract_features(f)
            self._last_features[tid] = x
            test_ids.append(tid)
            X_list.append(x)

        X = torch.stack(X_list)

        self.model.eval()
        with torch.no_grad():
            q_values = self.model(X).clone()

        # Epsilon-greedy exploration during training
        if self.train and self.epsilon > 0:
            for i in range(len(q_values)):
                if random.random() < self.epsilon:
                    q_values[i] = torch.rand(1).item() * 2.0 - 1.0

        sorted_indices = torch.argsort(q_values, descending=True)
        return [test_ids[i] for i in sorted_indices]

    def update(self, revealed: list[RevealedOutcome]) -> None:
        """Batched update at end of episode — called by the training loop."""
        if not self.train or not revealed:
            return

        X_batch: list[torch.Tensor] = []
        y_batch: list[float] = []

        cumulative_budget = 0.0

        for r in revealed:
            if r.test_id not in self._last_features:
                cumulative_budget += r.runtime_seconds
                continue

            x = self._last_features[r.test_id]
            
            # How much budget was consumed *before* this test ran?
            budget_consumed_before = cumulative_budget / self._last_budget
            budget_frac = min(budget_consumed_before, 1.0)

            if r.outcome == "FAIL":
                # Time-Discounted Reward: finding bug at 0% budget = 10.0, at 100% budget = 0.0
                reward = 10.0 * (1.0 - budget_frac)
            else:
                # Soft cost penalty for passing tests, based on how much budget this specific test wasted
                test_cost_frac = r.runtime_seconds / self._last_budget
                reward = -0.1 * min(test_cost_frac, 1.0)

            X_batch.append(x)
            y_batch.append(reward)
            
            cumulative_budget += r.runtime_seconds

        if not X_batch:
            return

        X = torch.stack(X_batch)
        y = torch.tensor(y_batch, dtype=torch.float32)

        self.model.train()
        self.optimizer.zero_grad()
        predictions = self.model(X)
        loss = self.criterion(predictions, y)
        loss.backward()
        # Gradient clipping for stability across different input scales
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

    def reset(self) -> None:
        self._last_features.clear()
        self._last_budget = 1.0

    def get_weights(self) -> dict:
        """Serialize model weights for checkpointing."""
        return self.model.state_dict()

    def load_weights(self, weights: dict) -> None:
        """Load serialized model weights."""
        self.model.load_state_dict(weights)
