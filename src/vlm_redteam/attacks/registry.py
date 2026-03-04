"""In-memory registry for attack adapters."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .adapter import AttackAdapter


@dataclass
class _RegistryEntry:
    adapter: AttackAdapter
    weight: float


_DEFAULT_REGISTRY: "AttackRegistry | None" = None


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
        """Sample up to k attack names according to weights (with replacement)."""

        if k <= 0 or not self._registry:
            return []

        names = [name for name, _ in self._registry.items()]
        weights = [entry.weight for _, entry in self._registry.items()]
        return rng.choices(names, weights=weights, k=k)


def set_default_registry(registry: AttackRegistry) -> None:
    """Set process-wide default registry for graph nodes."""

    global _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = registry


def get_default_registry() -> AttackRegistry:
    """Get process-wide default registry (lazy-initialized)."""

    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = AttackRegistry()
    return _DEFAULT_REGISTRY
