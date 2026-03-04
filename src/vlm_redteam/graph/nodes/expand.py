"""Expand node implementation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from random import Random
from typing import Any

from vlm_redteam.attacks.registry import get_default_registry
from vlm_redteam.graph.state import BeamState, Branch, Candidate, GraphState, stable_hash
from vlm_redteam.storage.artifacts import ArtifactStore
from vlm_redteam.storage.event_log import EventLogger


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


def expand_node(state: GraphState | BeamState) -> GraphState | BeamState:
    """Generate candidates for each branch in beam with weighted attack sampling."""

    if is_dataclass(state) and isinstance(state, BeamState):
        state.notes.append("expand")
        return state

    assert isinstance(state, dict)
    run_id = str(state.get("run_id", ""))
    round_idx = int(state.get("round_idx", 0))
    per_branch = int(state.get("per_branch_candidates", 0))
    beam = state.get("beam", [])
    if not beam or per_branch <= 0:
        state["candidates"] = []
        return state

    stats = state.get("stats", {})
    registry = stats.get("attack_registry") or get_default_registry()
    artifact_store: ArtifactStore = stats.get("artifact_store") or ArtifactStore(stats.get("runs_root", "runs"))
    event_logger: EventLogger = stats.get("event_logger") or EventLogger(
        run_id=run_id,
        runs_root=stats.get("runs_root", "runs"),
    )
    seed = int(stats.get("seed", 0)) + round_idx
    rng = Random(seed)

    task = state.get("task", {})
    original_prompt = _get(task, "goal", "")
    original_image_path = _get(task, "image_path")

    candidates: list[Candidate] = []
    for branch in beam:
        branch_id = str(_get(branch, "branch_id", ""))
        sampled_attacks = registry.sample_attacks(per_branch, rng)
        for attack_name in sampled_attacks:
            sampled_seed = rng.randint(0, 2**31 - 1)
            params = {"seed": sampled_seed}
            case_sig = stable_hash(
                {
                    "run_id": run_id,
                    "round_idx": round_idx,
                    "branch_id": branch_id,
                    "attack_name": attack_name,
                    "params": params,
                }
            )
            case_id = f"c-{case_sig[:16]}"

            event_logger.log_event(
                "ActionProposed",
                {
                    "branch_id": branch_id,
                    "attack_name": attack_name,
                    "params": params,
                    "case_id": case_id,
                },
            )

            adapter = registry.get(attack_name)
            test_case = adapter.generate(
                original_prompt=original_prompt,
                image_path=original_image_path,
                case_id=case_id,
                params=params,
            )

            jb_image_path = test_case.get("jailbreak_image_path")
            if jb_image_path:
                normalized_path, image_hash = artifact_store.save_image(run_id, jb_image_path)
                test_case["jailbreak_image_path"] = normalized_path
                test_case["jailbreak_image_hash"] = image_hash

            event_logger.log_event(
                "TestCaseGenerated",
                {
                    "branch_id": branch_id,
                    "attack_name": attack_name,
                    "case_id": case_id,
                    "test_case": test_case,
                },
            )

            candidates.append(
                Candidate(
                    cand_id=case_id,
                    from_branch_id=branch_id,
                    attack_name=attack_name,
                    params=params,
                    test_case=test_case,
                    target_output=None,
                    judge_success=None,
                    judge_score=None,
                    judge_reason=None,
                )
            )

    state["candidates"] = candidates
    stats["total_candidates"] = int(stats.get("total_candidates", 0)) + len(candidates)
    return state
