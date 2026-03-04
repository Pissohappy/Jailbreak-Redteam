"""Attack adapter placeholder abstractions."""

from typing import Protocol


class AttackAdapter(Protocol):
    """Protocol for attack generation adapters."""

    def generate(self, prompt: str) -> list[str]:
        """Return candidate attacks for a prompt."""
        ...
