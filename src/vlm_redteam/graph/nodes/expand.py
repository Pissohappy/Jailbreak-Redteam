"""Expand node placeholder."""

from ..state import BeamState


def expand_node(state: BeamState) -> BeamState:
    """Placeholder expand node."""
    state.notes.append("expand")
    return state
