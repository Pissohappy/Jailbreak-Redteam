"""In-memory registry for attack adapters."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .adapter import AttackAdapter


@dataclass
class _RegistryEntry:
    adapter: AttackAdapter
    weight: float


class AttackRegistry:
    """Registry with weighted sampling support for attack adapters."""

    def __init__(self) -> None:
        self._registry: dict[str, _RegistryEntry] = {}

    def register(self, name: str, adapter: AttackAdapter, weight: float = 1.0) -> None:
        if weight <= 0:
            raise ValueError("weight must be > 0")
        self._registry[name] = _RegistryEntry(adapter=adapter, weight=weight)

    def list_attacks(self) -> list[tuple[str, float]]:
        return [(name, entry.weight) for name, entry in self._registry.items()]

    def get(self, name: str) -> AttackAdapter:
        return self._registry[name].adapter

    def sample_attacks(self, k: int, rng: Random) -> list[str]:
        """Sample up to k unique attack names according to weights."""

        if k <= 0 or not self._registry:
            return []

        items = list(self._registry.items())
        k = min(k, len(items))
        selected: list[str] = []

        for _ in range(k):
            names = [name for name, _ in items]
            weights = [entry.weight for _, entry in items]
            picked = rng.choices(names, weights=weights, k=1)[0]
            selected.append(picked)
            items = [(name, entry) for name, entry in items if name != picked]

        return selected
