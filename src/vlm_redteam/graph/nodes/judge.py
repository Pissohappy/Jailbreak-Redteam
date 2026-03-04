"""Judge node placeholder."""

from ..state import BeamState


def judge_node(state: BeamState) -> BeamState:
    """Placeholder judge node."""
    state.notes.append("judge")
    return state
