"""History strategy helpers for multi-turn branch execution."""

from __future__ import annotations

from typing import Any, Callable

from .state import Branch, Candidate, GraphState

HistoryProvider = Callable[..., list[dict[str, Any]] | None]


VALID_HISTORY_STRATEGIES = {"inherit_parent", "none", "memory"}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def get_history_strategy(stats: dict[str, Any]) -> str:
    """Return normalized history strategy with safe fallback."""

    strategy = str(stats.get("history_strategy", "inherit_parent")).strip().lower()
    if strategy not in VALID_HISTORY_STRATEGIES:
        return "inherit_parent"
    return strategy


def resolve_execution_history(
    *,
    state: GraphState,
    candidate: Candidate | dict[str, Any],
    parent_branch: Branch | dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Resolve history sent to target model under configured strategy."""

    stats = state.get("stats", {})
    strategy = get_history_strategy(stats)
    parent_history = list(_get(parent_branch, "history", [])) if parent_branch else []

    if strategy == "none":
        return None
    if strategy == "inherit_parent":
        return parent_history or None

    provider = stats.get("history_memory_provider")
    if callable(provider):
        provided = provider(
            stage="execute",
            state=state,
            candidate=candidate,
            parent_branch=parent_branch,
            parent_history=parent_history,
        )
        if provided is not None:
            return list(provided)

    memory_store = state.get("memory") if isinstance(state, dict) else None
    if isinstance(memory_store, dict):
        branch_id = str(_get(candidate, "from_branch_id", ""))
        memory_key = str(stats.get("history_memory_key", "history"))
        record = memory_store.get(branch_id)
        if isinstance(record, dict) and isinstance(record.get(memory_key), list):
            return list(record[memory_key])

    return parent_history or None


def compose_branch_history(
    *,
    state: GraphState,
    candidate: Candidate | dict[str, Any],
    parent_branch: Branch | dict[str, Any] | None,
    current_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compose new branch history under configured strategy."""

    stats = state.get("stats", {})
    strategy = get_history_strategy(stats)
    parent_history = list(_get(parent_branch, "history", [])) if parent_branch else []

    if strategy == "none":
        return [current_entry]
    if strategy == "inherit_parent":
        return parent_history + [current_entry]

    provider = stats.get("history_memory_provider")
    if callable(provider):
        provided = provider(
            stage="select",
            state=state,
            candidate=candidate,
            parent_branch=parent_branch,
            parent_history=parent_history,
            current_entry=current_entry,
        )
        if provided is not None:
            return list(provided)

    return parent_history + [current_entry]
