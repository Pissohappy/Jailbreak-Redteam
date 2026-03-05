"""Abstract base class for expand strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from vlm_redteam.attacks.registry import AttackRegistry
from vlm_redteam.graph.state import Branch, Candidate


@dataclass
class ExpandContext:
    """Context passed to expand strategies."""

    run_id: str
    round_idx: int
    branch: Branch | dict[str, Any]
    original_prompt: str
    original_image_path: str | None
    registry: AttackRegistry
    per_branch_candidates: int
    seed: int
    stats: dict[str, Any]


class ExpandStrategy(ABC):
    """Abstract base class for attack selection strategies."""

    name: str = "base"

    @abstractmethod
    def select_attacks(
        self,
        ctx: ExpandContext,
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Select attacks for a single branch.

        Args:
            ctx: Expand context containing all necessary information.

        Returns:
            List of (attack_name, params) tuples.
        """
        pass

    def generate_candidates(
        self,
        ctx: ExpandContext,
    ) -> list[Candidate]:
        """
        Generate candidates for a branch using the selected attacks.

        This is a template method that calls select_attacks and then
        generates candidates from the selected attacks.

        Args:
            ctx: Expand context.

        Returns:
            List of generated candidates.
        """
        from random import Random

        from vlm_redteam.graph.state import stable_hash
        from vlm_redteam.storage.artifacts import ArtifactStore
        from vlm_redteam.storage.event_log import EventLogger

        rng = Random(ctx.seed)
        artifact_store: ArtifactStore = ctx.stats.get("artifact_store") or ArtifactStore(
            ctx.stats.get("runs_root", "runs")
        )
        event_logger: EventLogger = ctx.stats.get("event_logger") or EventLogger(
            run_id=ctx.run_id,
            runs_root=ctx.stats.get("runs_root", "runs"),
        )

        def _get(obj: Any, key: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        branch_id = str(_get(ctx.branch, "branch_id", ""))
        selected = self.select_attacks(ctx)
        candidates: list[Candidate] = []

        for attack_name, params in selected:
            if "seed" not in params:
                params["seed"] = rng.randint(0, 2**31 - 1)

            case_sig = stable_hash(
                {
                    "run_id": ctx.run_id,
                    "round_idx": ctx.round_idx,
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
                    "strategy": self.name,
                },
            )

            adapter = ctx.registry.get(attack_name)
            test_case = adapter.generate(
                original_prompt=ctx.original_prompt,
                image_path=ctx.original_image_path,
                case_id=case_id,
                params=params,
            )

            jb_image_path = test_case.get("jailbreak_image_path")
            if jb_image_path:
                normalized_path, image_hash = artifact_store.save_image(
                    ctx.run_id, jb_image_path
                )
                test_case["jailbreak_image_path"] = normalized_path
                test_case["jailbreak_image_hash"] = image_hash

            event_logger.log_event(
                "TestCaseGenerated",
                {
                    "branch_id": branch_id,
                    "attack_name": attack_name,
                    "case_id": case_id,
                    "test_case": test_case,
                    "strategy": self.name,
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

        return candidates
