"""Select node placeholder."""

from ..state import BeamState


def select_node(state: BeamState) -> BeamState:
    """Placeholder select node."""
    state.notes.append("select")
    return state
