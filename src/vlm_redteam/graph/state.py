"""State definitions for the LangGraph workflow."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict


@dataclass
class Branch:
    """A beam-search branch carrying interaction and judging metadata."""

    branch_id: str
    parent_id: Optional[str]
    round_idx: int
    history: list[dict[str, Any]] = field(default_factory=list)
    user_text: str = ""
    user_image: Optional[str] = None
    target_output: Optional[str] = None
    judge_success: Optional[bool] = None
    judge_score: Optional[float] = None
    judge_reason: Optional[str] = None
    aggregate_score: float = 0.0
    signature: Optional[str] = None


@dataclass
class Candidate:
    """Candidate attack generated from a branch."""

    cand_id: str
    from_branch_id: str
    attack_name: str
    params: dict[str, Any] = field(default_factory=dict)
    test_case: dict[str, Any] = field(default_factory=dict)
    target_output: Optional[str] = None
    judge_success: Optional[bool] = None
    judge_score: Optional[float] = None
    judge_reason: Optional[str] = None


class GraphState(TypedDict):
    """Global workflow state passed across graph nodes."""

    run_id: str
    task: str
    round_idx: int
    max_rounds: int
    beam_width: int
    per_branch_candidates: int
    beam: list[Branch]
    candidates: list[Candidate]
    done: bool
    best: Optional[Branch]
    stats: dict[str, Any]


def stable_hash(obj: Any) -> str:
    """Compute deterministic SHA256 for nested JSON-compatible objects."""

    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_signature(jailbreak_prompt: str, image_path: Optional[str]) -> str:
    """Build a stable signature from prompt and image path."""

    payload = {
        "jailbreak_prompt": jailbreak_prompt,
        "image_path": image_path,
    }
    return stable_hash(payload)
