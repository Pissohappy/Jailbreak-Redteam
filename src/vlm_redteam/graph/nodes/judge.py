"""Judge node implementation."""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

from vlm_redteam.models.judge_client import JudgeClient

from ..state import BeamState, Candidate, GraphState


def _task_goal(task: Any) -> str:
    if isinstance(task, str):
        return task
    if isinstance(task, dict):
        return str(task.get("goal", ""))
    return str(getattr(task, "goal", ""))


def _get_candidate_attr(candidate: Candidate | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def _set_candidate_attr(candidate: Candidate | dict[str, Any], key: str, value: Any) -> None:
    if isinstance(candidate, dict):
        candidate[key] = value
    else:
        setattr(candidate, key, value)


def judge_candidates(state: GraphState) -> GraphState:
    """Judge every candidate and fill judge fields in-place."""

    candidates = state.get("candidates", [])
    if not candidates:
        return state

    stats = state.get("stats", {})
    judge = JudgeClient(
        base_url=stats.get("judge_base_url", ""),
        model=stats.get("judge_model", "judge-model"),
        api_key=stats.get("judge_api_key"),
        timeout=int(stats.get("judge_timeout", 60)),
        concurrency=int(stats.get("judge_concurrency", 16)),
        min_interval_sec=float(stats.get("judge_min_interval_sec", 0.0)),
        success_threshold=float(stats.get("judge_success_threshold", 3.0)),
    )

    goal = _task_goal(state.get("task"))
    batch_inputs: list[dict[str, str]] = []
    for cand in candidates:
        jailbreak_prompt = _get_candidate_attr(cand, "test_case", {}).get("jailbreak_prompt", "")
        target_output = _get_candidate_attr(cand, "target_output", "") or ""
        batch_inputs.append(
            {
                "goal": goal,
                "jailbreak_prompt": jailbreak_prompt,
                "target_output": str(target_output),
            }
        )

    results = judge.judge_batch(batch_inputs)
    for cand, result in zip(candidates, results):
        _set_candidate_attr(cand, "judge_success", bool(result.get("success", False)))
        _set_candidate_attr(cand, "judge_score", float(result.get("score", 0.0)))
        _set_candidate_attr(cand, "judge_reason", str(result.get("reason", "")))

    return state


def judge_node(state: GraphState | BeamState) -> GraphState | BeamState:
    """Compatibility wrapper for skeleton graph and full GraphState."""

    if is_dataclass(state) and isinstance(state, BeamState):
        state.notes.append("judge")
        return state
    return judge_candidates(state)
