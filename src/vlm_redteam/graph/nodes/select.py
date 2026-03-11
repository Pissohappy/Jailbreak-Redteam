"""Select/beam-update node implementation."""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

from vlm_redteam.storage.run_outputs import append_state_snapshot

from ..history_policy import compose_branch_history
from ..state import BeamState, Branch, Candidate, GraphState, stable_hash


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _candidate_score(candidate: Candidate | dict[str, Any]) -> float:
    score = _get(candidate, "judge_score", 0.0)
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def _candidate_success(candidate: Candidate | dict[str, Any]) -> bool:
    return bool(_get(candidate, "judge_success", False))


def select_beam_node(state: GraphState) -> GraphState:
    """Select top-k deduplicated branches and decide loop termination."""

    candidates = state.get("candidates", [])
    if not candidates:
        state["beam"] = []
        state["round_idx"] = int(state.get("round_idx", 0)) + 1
        state["done"] = True
        return state

    task = state.get("task", {})
    goal = str(task.get("goal", "")) if isinstance(task, dict) else str(getattr(task, "goal", ""))
    next_round_idx = int(state.get("round_idx", 0)) + 1
    parent_map = {str(_get(branch, "branch_id", "")): branch for branch in state.get("beam", [])}

    deduped: dict[str, Branch] = {}
    best_success_branch: Branch | None = None

    for candidate in candidates:
        test_case = _get(candidate, "test_case", {}) or {}
        jailbreak_prompt = str(test_case.get("jailbreak_prompt", ""))
        image_path = test_case.get("jailbreak_image_path")
        image_hash = test_case.get("jailbreak_image_hash") or ""
        target_output = str(_get(candidate, "target_output", "") or "")
        score = _candidate_score(candidate)

        signature = stable_hash(
            {
                "jailbreak_prompt": jailbreak_prompt,
                "image_hash": str(image_hash),
                "target_output": target_output,
            }
        )
        parent_id = str(_get(candidate, "from_branch_id", "")) or None
        parent_branch = parent_map.get(parent_id or "")
        branch_id = f"b-{str(_get(candidate, 'cand_id', signature))}"
        current_entry = {
            "round_idx": next_round_idx,
            "branch_id": branch_id,
            "user_text": f"{jailbreak_prompt}".strip(),
            "image_path": image_path,
            "target_output": target_output,
            "judge_reason": _get(candidate, "judge_reason", ""),
        }
        branch = Branch(
            branch_id=branch_id,
            parent_id=parent_id,
            round_idx=next_round_idx,
            history=compose_branch_history(
                state=state,
                candidate=candidate,
                parent_branch=parent_branch,
                current_entry=current_entry,
            ),
            user_text=f"{jailbreak_prompt}".strip(),
            user_image=image_path,
            target_output=target_output,
            judge_success=_candidate_success(candidate),
            judge_score=score,
            judge_reason=_get(candidate, "judge_reason"),
            aggregate_score=score,
            signature=signature,
        )

        prev = deduped.get(signature)
        if prev is None or branch.aggregate_score > prev.aggregate_score:
            deduped[signature] = branch

    ranked = sorted(deduped.values(), key=lambda b: b.aggregate_score, reverse=True)
    beam_width = int(state.get("beam_width", 1))
    new_beam = ranked[:beam_width]

    for branch in ranked:
        if branch.judge_success and (
            best_success_branch is None or branch.aggregate_score > best_success_branch.aggregate_score
        ):
            best_success_branch = branch

    done = False
    best: Branch | None = state.get("best")
    if best_success_branch is not None:
        done = True
        best = best_success_branch
    elif next_round_idx >= int(state.get("max_rounds", 1)):
        done = True
        best = ranked[0] if ranked else best

    state["beam"] = new_beam
    state["round_idx"] = next_round_idx
    state["done"] = done
    state["best"] = best

    stats = state.get("stats", {})
    topk = [float(branch.aggregate_score) for branch in new_beam]
    stats.setdefault("round_topk_scores", []).append({"round_idx": next_round_idx, "scores": topk})
    append_state_snapshot(
        run_id=str(state.get("run_id", "")),
        runs_root=stats.get("runs_root", "runs"),
        round_idx=next_round_idx,
        beam=new_beam,
        best=best,
    )
    return state


def select_node(state: GraphState | BeamState) -> GraphState | BeamState:
    """Compatibility wrapper for both skeleton BeamState and GraphState."""

    if is_dataclass(state) and isinstance(state, BeamState):
        state.notes.append("select")
        return state
    return select_beam_node(state)
