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
from vlm_redteam.graph.nodes.execute import execute_target_node
from vlm_redteam.graph.nodes.expand import expand_node
from vlm_redteam.graph.state import Branch
from vlm_redteam.models.vllm_client import VLLMClient


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

    demo_state = {
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
            "target_vllm_base_url": cfg.target_vllm_base_url,
            "target_vllm_model": cfg.target_vllm_model,
            "target_vllm_api_key": cfg.target_vllm_api_key,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "enable_vision": cfg.enable_vision,
            "concurrency": cfg.concurrency,
        },
    }
    demo_out = expand_node(demo_state)
    print(f"Expand generated candidates: {len(demo_out['candidates'])}")

    demo_out = execute_target_node(demo_out)
    print(f"Execute generated outputs: {sum(1 for c in demo_out['candidates'] if getattr(c, 'target_output', None))}")

    _ = compile_graph()
    print("Graph compiled OK")


if __name__ == "__main__":
    main()
