"""Select beam node implementation."""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

from ..state import BeamState, Branch, Candidate, GraphState, stable_hash


def _get_candidate_attr(candidate: Candidate | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(key, default)
    return getattr(candidate, key, default)




def _get_branch_attr(branch: Branch | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(branch, dict):
        return branch.get(key, default)
    return getattr(branch, key, default)

def _candidate_to_branch(
    candidate: Candidate | dict[str, Any],
    next_round: int,
    parent_history: list[dict[str, Any]],
) -> Branch:
    test_case = _get_candidate_attr(candidate, "test_case", {}) or {}
    jailbreak_prompt = str(test_case.get("jailbreak_prompt", ""))
    image_hash = str(test_case.get("jailbreak_image_hash") or test_case.get("jailbreak_image_path") or "")
    target_output = str(_get_candidate_attr(candidate, "target_output", "") or "")
    signature = stable_hash(f"{jailbreak_prompt}|{image_hash}|{target_output}")

    return Branch(
        branch_id=str(_get_candidate_attr(candidate, "cand_id", "")),
        parent_id=str(_get_candidate_attr(candidate, "from_branch_id", "")) or None,
        round_idx=next_round,
        history=parent_history + [
            {
                "user_text": jailbreak_prompt,
                "image_path": test_case.get("jailbreak_image_path"),
                "target_output": target_output,
            }
        ],
        user_text=jailbreak_prompt,
        user_image=test_case.get("jailbreak_image_path"),
        target_output=target_output,
        judge_success=bool(_get_candidate_attr(candidate, "judge_success", False)),
        judge_score=float(_get_candidate_attr(candidate, "judge_score", 0.0) or 0.0),
        judge_reason=str(_get_candidate_attr(candidate, "judge_reason", "") or ""),
        aggregate_score=float(_get_candidate_attr(candidate, "judge_score", 0.0) or 0.0),
        signature=signature,
    )


def select_beam_node(state: GraphState) -> GraphState:
    """Select top-k branches by judge score with signature dedup and termination."""

    candidates = state.get("candidates", [])
    next_round = int(state.get("round_idx", 0)) + 1
    max_rounds = int(state.get("max_rounds", 1))
    beam_width = int(state.get("beam_width", 1))

    prev_beam = state.get("beam", [])
    parent_histories = {
        str(_get_branch_attr(b, "branch_id", "")): list(_get_branch_attr(b, "history", []))
        for b in prev_beam
    }

    branches = []
    for cand in candidates:
        parent_id = str(_get_candidate_attr(cand, "from_branch_id", ""))
        branches.append(
            _candidate_to_branch(
                cand,
                next_round=next_round,
                parent_history=parent_histories.get(parent_id, []),
            )
        )

    # Deduplicate by signature while keeping highest-scoring branch.
    dedup: dict[str, Branch] = {}
    for branch in branches:
        sig = branch.signature or ""
        if sig not in dedup or branch.aggregate_score > dedup[sig].aggregate_score:
            dedup[sig] = branch

    ranked = sorted(dedup.values(), key=lambda b: b.aggregate_score, reverse=True)
    new_beam = ranked[:beam_width]

    state["beam"] = new_beam
    state["round_idx"] = next_round

    success_branch = next((b for b in ranked if b.judge_success), None)
    if success_branch is not None:
        state["done"] = True
        state["best"] = success_branch
        return state

    if next_round >= max_rounds:
        state["done"] = True
        state["best"] = ranked[0] if ranked else state.get("best")
        return state

    state["done"] = False
    state["best"] = ranked[0] if ranked else state.get("best")
    return state


def select_node(state: GraphState | BeamState) -> GraphState | BeamState:
    """Compatibility wrapper for select node."""

    if is_dataclass(state) and isinstance(state, BeamState):
        state.notes.append("select")
        return state
    return select_beam_node(state)
