"""Expand node implementation with strategy pattern support."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from vlm_redteam.attacks.registry import get_default_registry
from vlm_redteam.graph.state import BeamState, Branch, Candidate, GraphState
from vlm_redteam.storage.artifacts import ArtifactStore
from vlm_redteam.storage.event_log import EventLogger

from .expand_strategies.base import ExpandContext
from .expand_strategies.random_sampling import RandomSamplingStrategy
from .expand_strategies.llm_guided import LLMGuidedStrategy


# Default strategy instance
_default_strategy = RandomSamplingStrategy()

# Global strategy cache (not stored in state to avoid serialization issues)
_strategy_cache: dict[str, Any] = {}


def _build_strategy_from_config(stats: dict[str, Any]) -> Any:
    """Build expand strategy from configuration stored in stats."""
    strategy_name = stats.get("expand_strategy_name", "random_sampling")
    
    if strategy_name == "llm_guided":
        llm_config = stats.get("llm_guide_config") or {}
        return LLMGuidedStrategy(
            base_url=llm_config.get("base_url"),
            model=llm_config.get("model"),
            api_key=llm_config.get("api_key"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 1024),
            max_history_rounds=llm_config.get("max_history_rounds"),
        )
    else:
        return _default_strategy


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _branch_to_dict(branch: Branch | dict[str, Any]) -> dict[str, Any]:
    if is_dataclass(branch):
        return asdict(branch)
    return dict(branch)


def expand_node_with_strategy(
    state: GraphState,
    strategy: Any = None,
) -> GraphState:
    """
    Generate candidates using a configurable expand strategy.

    Args:
        state: Graph state.
        strategy: ExpandStrategy instance. If None, uses default random sampling.

    Returns:
        Updated graph state with candidates.
    """
    from random import Random

    if strategy is None:
        strategy = _default_strategy

    run_id = str(state.get("run_id", ""))
    round_idx = int(state.get("round_idx", 0))
    per_branch = int(state.get("per_branch_candidates", 0))
    beam = state.get("beam", [])
    if not beam or per_branch <= 0:
        state["candidates"] = []
        return state

    stats = state.get("stats", {})
    registry = stats.get("attack_registry") or get_default_registry()

    task = state.get("task", {})
    original_prompt = _get(task, "goal", "")
    original_image_path = _get(task, "image_path")

    candidates: list[Candidate] = []
    for branch in beam:
        seed = int(stats.get("seed", 0)) + round_idx + hash(str(_get(branch, "branch_id", "")))

        ctx = ExpandContext(
            run_id=run_id,
            round_idx=round_idx,
            branch=branch,
            original_prompt=original_prompt,
            original_image_path=original_image_path,
            registry=registry,
            per_branch_candidates=per_branch,
            seed=seed,
            stats=stats,
        )

        branch_candidates = strategy.generate_candidates(ctx)
        candidates.extend(branch_candidates)

    state["candidates"] = candidates
    stats["total_candidates"] = int(stats.get("total_candidates", 0)) + len(candidates)
    return state


def expand_node(state: GraphState | BeamState) -> GraphState | BeamState:
    """Generate candidates for each branch in beam with weighted attack sampling."""

    if is_dataclass(state) and isinstance(state, BeamState):
        state.notes.append("expand")
        return state

    assert isinstance(state, dict)

    # Build strategy from config (not stored directly to avoid serialization issues)
    stats = state.get("stats", {})
    strategy = _build_strategy_from_config(stats)

    return expand_node_with_strategy(state, strategy=strategy)
