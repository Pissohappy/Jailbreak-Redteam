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


class RunConfig(BaseModel):
    """Runtime config schema."""

    run_id: str
    target_vllm_base_url: str
    target_vllm_model: str
    target_vllm_api_key: str | None = None
    judge_base_url: str | None = None
    judge_model: str = "judge-model"
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
    """Minimal BaseAttack-style stub for local EXPAND smoke test."""

    cfg: dict[str, int] = {}

    def generate_test_case(self, original_prompt, image_path, case_id, **kwargs):
        seed = kwargs.get("seed", 0)
        return {
            "case_id": case_id,
            "jailbreak_prompt": f"[seed={seed}] {original_prompt}",
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


def _print_best_summary(final_state: dict) -> None:
    best = final_state.get("best")
    if not best:
        print("No best branch found.")
        return

    history = best.history if hasattr(best, "history") else best.get("history", [])
    score = best.aggregate_score if hasattr(best, "aggregate_score") else best.get("aggregate_score")
    success = best.judge_success if hasattr(best, "judge_success") else best.get("judge_success")
    branch_id = best.branch_id if hasattr(best, "branch_id") else best.get("branch_id")

    print("=== Best Branch Summary ===")
    print(f"branch_id: {branch_id}")
    print(f"success: {success}")
    print(f"aggregate_score: {score}")
    print(f"history_len: {len(history)}")
    if history:
        last = history[-1]
        print(f"last_user_text: {last.get('user_text', '')[:120]}")
        print(f"last_image_path: {last.get('image_path')}")
        print(f"last_target_output: {str(last.get('target_output', ''))[:200]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM redteam graph")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--ping", action="store_true", help="Ping target vLLM endpoint")
    parser.add_argument("--goal", default="demo-goal", help="Task goal text")
    parser.add_argument("--image-path", default=None, help="Task image path")
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

    initial_state = {
        "run_id": cfg.run_id,
        "task": {"goal": args.goal, "image_path": args.image_path},
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
            "target_vllm_base_url": cfg.target_vllm_base_url,
            "target_vllm_model": cfg.target_vllm_model,
            "target_vllm_api_key": cfg.target_vllm_api_key,
            "judge_base_url": cfg.judge_base_url or "",
            "judge_model": cfg.judge_model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "enable_vision": cfg.enable_vision,
            "concurrency": cfg.concurrency,
        },
    }

    app = compile_graph()
    final_state = app.invoke(initial_state)
    print("Graph run finished")
    _print_best_summary(final_state)


if __name__ == "__main__":
    main()
