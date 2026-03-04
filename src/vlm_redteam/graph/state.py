"""State definitions for the LangGraph workflow."""

from typing import Any

from pydantic import BaseModel, Field


class BeamState(BaseModel):
    """Minimal placeholder state for graph compilation."""

    run_id: str
    round_idx: int = 0
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
