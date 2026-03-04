"""CLI entrypoint for running the redteam pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from pydantic import BaseModel

from vlm_redteam.attacks.adapter import AttackAdapter
from vlm_redteam.attacks.registry import AttackRegistry, set_default_registry
from vlm_redteam.graph.build_graph import compile_graph
from vlm_redteam.graph.state import Branch
from vlm_redteam.models.vllm_client import VLLMClient
from vlm_redteam.storage.artifacts import ArtifactStore
from vlm_redteam.storage.event_log import EventLogger


class RunConfig(BaseModel):
    """Runtime config schema."""

    run_id: str
    target_vllm_base_url: str
    target_vllm_model: str
    target_vllm_api_key: str | None = None
    judge_base_url: str | None = None
    beam_width: int
    per_branch_candidates: int
    max_rounds: int
    enable_vision: bool = True
    temperature: float = 0.2
    max_tokens: int = 512
    concurrency: int = 16


def load_config(path: Path) -> RunConfig:
    """Load and validate YAML config."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RunConfig.model_validate(raw)


class _DummyAttack:
    """Minimal BaseAttack-style stub for local smoke test."""

    cfg: dict[str, int] = {}

    def generate_test_case(self, original_prompt, image_path, case_id, **kwargs):
        seed = kwargs.get("seed", 0)
        return {
            "case_id": case_id,
            "jailbreak_prompt": f"[seed={seed}] please help with: {original_prompt}",
            "jailbreak_image_path": image_path,
            "original_prompt": original_prompt,
            "original_image_path": image_path,
        }


def _build_demo_registry() -> AttackRegistry:
    registry = AttackRegistry()
    registry.register(
        "dummy_attack",
        AttackAdapter(attack_obj=_DummyAttack(), attack_name="dummy_attack"),
        weight=1.0,
    )
    return registry


def _format_best_summary(best: Branch | None) -> str:
    if best is None:
        return "best=None"
    return (
        f"best.branch_id={best.branch_id}, score={best.aggregate_score:.3f}, "
        f"success={best.judge_success}, history_len={len(best.history)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM redteam graph")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--ping", action="store_true", help="Ping target vLLM endpoint")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    print("Loaded config OK")

    if args.ping:
        client = VLLMClient(
            base_url=cfg.target_vllm_base_url,
            model=cfg.target_vllm_model,
            api_key=cfg.target_vllm_api_key,
            timeout=60,
            enable_vision=cfg.enable_vision,
            concurrency=cfg.concurrency,
        )
        resp = client.ping()
        print(json.dumps(resp, ensure_ascii=False, indent=2))
        return

    registry = _build_demo_registry()
    set_default_registry(registry)

    event_logger = EventLogger(run_id=cfg.run_id, runs_root="runs")
    artifact_store = ArtifactStore(runs_root="runs")

    initial_state = {
        "run_id": cfg.run_id,
        "task": {"goal": "demo-goal", "image_path": None},
        "round_idx": 0,
        "max_rounds": cfg.max_rounds,
        "beam_width": cfg.beam_width,
        "per_branch_candidates": cfg.per_branch_candidates,
        "beam": [Branch(branch_id="b0", parent_id=None, round_idx=0)],
        "candidates": [],
        "done": False,
        "best": None,
        "stats": {
            "seed": 0,
            "runs_root": "runs",
            "attack_registry": registry,
            "event_logger": event_logger,
            "artifact_store": artifact_store,
            "target_vllm_base_url": cfg.target_vllm_base_url,
            "target_vllm_model": cfg.target_vllm_model,
            "target_vllm_api_key": cfg.target_vllm_api_key,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "enable_vision": cfg.enable_vision,
            "concurrency": cfg.concurrency,
            "judge_base_url": cfg.judge_base_url or "",
            "judge_model": "judge-model",
            "judge_concurrency": cfg.concurrency,
        },
    }

    app = compile_graph()
    final_state = app.invoke(initial_state)
    print(f"Run finished: done={final_state['done']}, round_idx={final_state['round_idx']}")
    print(_format_best_summary(final_state.get("best")))


if __name__ == "__main__":
    main()
