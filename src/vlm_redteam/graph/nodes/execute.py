"""Execute target model node implementation."""

from __future__ import annotations

import time
from dataclasses import is_dataclass
from typing import Any

from vlm_redteam.models.vllm_client import VLLMClient
from vlm_redteam.storage.event_log import EventLogger

from ..state import BeamState, Candidate, GraphState


def _get_candidate_attr(candidate: Candidate | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def _set_candidate_attr(candidate: Candidate | dict[str, Any], key: str, value: Any) -> None:
    if isinstance(candidate, dict):
        candidate[key] = value
    else:
        setattr(candidate, key, value)


def execute_target_node(state: GraphState) -> GraphState:
    """Call target model in batch for all candidates and fill target_output."""

    candidates = state.get("candidates", [])
    if not candidates:
        return state

    stats = state.get("stats", {})
    task = state.get("task", {})
    goal = task["goal"] if isinstance(task, dict) else getattr(task, "goal", "")

    client = VLLMClient(
        base_url=str(stats.get("target_vllm_base_url", "")),
        model=str(stats.get("target_vllm_model", "dummy-model")),
        api_key=stats.get("target_vllm_api_key"),
        timeout=int(stats.get("target_timeout", 60)),
        enable_vision=bool(stats.get("enable_vision", True)),
        concurrency=int(stats.get("concurrency", 16)),
    )
    event_logger: EventLogger = stats.get("event_logger") or EventLogger(
        run_id=str(state.get("run_id", "")),
        runs_root=stats.get("runs_root", "runs"),
    )

    batch_inputs: list[dict[str, Any]] = []
    cand_ids: list[str] = []
    for cand in candidates:
        test_case = _get_candidate_attr(cand, "test_case", {}) or {}
        user_text = f"{test_case.get('jailbreak_prompt', '')}\n\n{goal}"
        image_path = test_case.get("jailbreak_image_path")
        batch_inputs.append(
            {
                "user_text": user_text,
                "image_path": image_path,
                "temperature": float(stats.get("temperature", 0.2)),
                "max_tokens": int(stats.get("max_tokens", 512)),
            }
        )
        cand_ids.append(str(_get_candidate_attr(cand, "cand_id", "")))

    started = time.perf_counter()
    error_text = ""
    outputs: list[str] = []
    try:
        outputs = client.generate_batch(batch_inputs)
    except Exception as exc:  # noqa: BLE001
        error_text = str(exc)
        outputs = [f"[target_error] {error_text}" for _ in candidates]
    elapsed = time.perf_counter() - started

    has_error = bool(error_text)
    for cand, cand_id, output in zip(candidates, cand_ids, outputs):
        _set_candidate_attr(cand, "target_output", output)
        event_logger.log_event(
            "TargetCalled",
            {
                "cand_id": cand_id,
                "elapsed_sec": elapsed,
                "error": has_error,
                "error_message": error_text,
            },
        )

    return state


def execute_node(state: GraphState | BeamState) -> GraphState | BeamState:
    """Compatibility wrapper for graph skeleton and graph state."""

    if is_dataclass(state) and isinstance(state, BeamState):
        state.notes.append("execute")
        return state
    return execute_target_node(state)
