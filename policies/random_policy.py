"""
Baseline 1: Random Policy.

Shuffles all candidate tests uniformly at random.

This is the absolute lower-bound baseline. Any sensible policy should achieve
higher FDR than this. Must be run with multiple seeds and results averaged.
"""
from __future__ import annotations

import random
from typing import Optional

from policies.base import Policy, RevealedOutcome


class RandomPolicy(Policy):
    """
    Selects tests by uniform random shuffling.

    Args:
        seed: Random seed for reproducibility. Use seed=None for a new random
              sequence each call. In experiments, use 30 different seeds (0–29)
              and report the mean ± std.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        if self.seed is not None:
            return f"random_seed{self.seed}"
        return "random"

    def select_tests(
        self,
        features: list[dict],
        budget_seconds: float,
        history: Optional[list[RevealedOutcome]] = None,
    ) -> list[str]:
        ids = [f["test_id"] for f in features]
        self._rng.shuffle(ids)
        return ids

    def reset(self) -> None:
        """Reset to initial seed for reproducibility across episodes."""
        self._rng = random.Random(self.seed)
