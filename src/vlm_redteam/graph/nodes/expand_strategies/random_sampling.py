"""Random sampling strategy (original weighted random sampling)."""

from __future__ import annotations

from random import Random
from typing import Any

from .base import ExpandContext, ExpandStrategy


class RandomSamplingStrategy(ExpandStrategy):
    """
    Weighted random sampling of attacks.

    This is the original expand strategy that randomly samples attacks
    based on their configured weights.
    """

    name = "random_sampling"

    def select_attacks(
        self,
        ctx: ExpandContext,
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Sample attacks randomly with weights.

        Args:
            ctx: Expand context.

        Returns:
            List of (attack_name, params) tuples.
        """
        rng = Random(ctx.seed)
        sampled_attacks = ctx.registry.sample_attacks(ctx.per_branch_candidates, rng)

        results: list[tuple[str, dict[str, Any]]] = []
        for attack_name in sampled_attacks:
            sampled_seed = rng.randint(0, 2**31 - 1)
            params = {"seed": sampled_seed}
            results.append((attack_name, params))

        return results
