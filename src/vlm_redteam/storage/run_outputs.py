"""Run-level checkpoint export, snapshot logging, and report generation."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [_to_jsonable(v) for v in sorted(obj)]
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(obj).items()}
    return obj


def append_state_snapshot(
    run_id: str,
    runs_root: str | Path,
    round_idx: int,
    beam: list[Any],
    best: Any,
) -> None:
    run_dir = Path(runs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = run_dir / "state_snapshots.jsonl"

    beam_scores: list[float] = []
    for branch in beam:
        if isinstance(branch, dict):
            beam_scores.append(float(branch.get("aggregate_score", 0.0)))
        else:
            beam_scores.append(float(getattr(branch, "aggregate_score", 0.0)))

    best_branch_id = None
    best_score = None
    if best is not None:
        if isinstance(best, dict):
            best_branch_id = best.get("branch_id")
            best_score = float(best.get("aggregate_score", 0.0))
        else:
            best_branch_id = getattr(best, "branch_id", None)
            best_score = float(getattr(best, "aggregate_score", 0.0))

    payload = {
        "round_idx": round_idx,
        "beam_scores": beam_scores,
        "best": {
            "branch_id": best_branch_id,
            "aggregate_score": best_score,
        },
    }
    with snapshot_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def export_checkpoints(checkpointer: Any, config: dict[str, Any], run_id: str, runs_root: str | Path) -> None:
    run_dir = Path(runs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "checkpoints.json"

    items: list[dict[str, Any]] = []
    for item in checkpointer.list(config):
        items.append(
            {
                "config": _to_jsonable(item.config),
                "checkpoint": _to_jsonable(item.checkpoint),
                "metadata": _to_jsonable(item.metadata),
                "parent_config": _to_jsonable(item.parent_config),
                "pending_writes": _to_jsonable(item.pending_writes),
            }
        )

    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def write_run_reports(final_state: dict[str, Any], runs_root: str | Path = "runs") -> None:
    run_id = str(final_state.get("run_id", ""))
    run_dir = Path(runs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    stats = final_state.get("stats", {}) or {}
    topk_scores = stats.get("round_topk_scores", [])
    total_candidates = int(stats.get("total_candidates", 0))

    best = final_state.get("best")
    if is_dataclass(best):
        best = asdict(best)

    best_path_history = []
    best_branch_path = []
    success = False
    if isinstance(best, dict):
        best_path_history = list(best.get("history", []))
        best_branch_path = [entry.get("branch_id") for entry in best_path_history if isinstance(entry, dict)]
        success = bool(best.get("judge_success", False))

    summary_payload = {
        "total_candidates": total_candidates,
        "success": success,
        "best_branch_path": best_branch_path,
        "round_topk_scores": topk_scores,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# Best Path Review", ""]
    for idx, step in enumerate(best_path_history, start=1):
        user_text = str(step.get("user_text", ""))
        user_text_short = user_text.replace("\n", " ").strip()
        if len(user_text_short) > 280:
            user_text_short = f"{user_text_short[:277]}..."

        lines.extend(
            [
                f"## Round {step.get('round_idx', idx)}",
                f"- Branch: `{step.get('branch_id', '')}`",
                f"- User 摘要: {user_text_short}",
                f"- 图片路径: `{step.get('image_path')}`",
                f"- Target 输出: {step.get('target_output', '')}",
                f"- Judge Reason: {step.get('judge_reason', '')}",
                "",
            ]
        )

    (run_dir / "best_path.md").write_text("\n".join(lines), encoding="utf-8")
