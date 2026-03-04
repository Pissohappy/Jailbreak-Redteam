"""Registry placeholder for attack adapters."""

from .adapter import AttackAdapter


class AttackRegistry:
    """Simple in-memory adapter registry."""

    def __init__(self) -> None:
        self._registry: dict[str, AttackAdapter] = {}

    def register(self, name: str, adapter: AttackAdapter) -> None:
        self._registry[name] = adapter

    def get(self, name: str) -> AttackAdapter:
        return self._registry[name]
