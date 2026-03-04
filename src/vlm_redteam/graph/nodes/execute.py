"""Execute node placeholder."""

from ..state import BeamState


def execute_node(state: BeamState) -> BeamState:
    """Placeholder execute node."""
    state.notes.append("execute")
    return state
