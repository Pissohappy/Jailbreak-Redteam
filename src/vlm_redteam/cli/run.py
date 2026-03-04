"""CLI entrypoint for running the redteam pipeline."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import yaml
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from vlm_redteam.attacks.adapter import AttackAdapter
from vlm_redteam.attacks.registry import AttackRegistry, set_default_registry
from vlm_redteam.graph.build_graph import compile_graph
from vlm_redteam.graph.state import Branch
from vlm_redteam.models.vllm_client import VLLMClient
from vlm_redteam.storage.run_outputs import export_checkpoints, write_run_reports


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
    target_concurrency: int | None = None
    judge_concurrency: int | None = None
    global_rate_limit_qps: float = 0.0
    enabled_attacks: list[str] = ["demo_figstep"]
    attack_weights: dict[str, float] = {}
    attack_init_kwargs: dict[str, dict[str, Any]] = {}


def load_config(path: Path) -> RunConfig:
    """Load and validate YAML config."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RunConfig.model_validate(raw)


def _load_attack_from_entrypoint(entrypoint: str, init_kwargs: dict[str, Any]) -> Any:
    """Load and instantiate attack class from `<module>:<ClassName>` entrypoint."""

    if ":" not in entrypoint:
        raise ValueError(f"invalid entrypoint: {entrypoint}")
    module_name, class_name = entrypoint.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    attack_cls = getattr(module, class_name)

    if callable(attack_cls) and inspect.isclass(attack_cls):
        return attack_cls(**init_kwargs)
    raise TypeError(f"entrypoint does not reference a class: {entrypoint}")


def _build_registry(cfg: RunConfig) -> AttackRegistry:
    from vlm_redteam.attacks.demo_figstep import DemoFigStepAttack

    registry = AttackRegistry()

    enabled_attacks = cfg.enabled_attacks or ["demo_figstep"]
    for attack_spec in enabled_attacks:
        if attack_spec == "demo_figstep":
            attack_name = "demo_figstep"
            output_image_dir = Path("runs") / cfg.run_id / "generated_images"
            attack_obj = DemoFigStepAttack(output_image_dir=output_image_dir)
        elif ":" in attack_spec:
            attack_name = attack_spec
            init_kwargs = cfg.attack_init_kwargs.get(attack_spec, {})
            attack_obj = _load_attack_from_entrypoint(attack_spec, init_kwargs)
        else:
            raise ValueError(
                f"Unsupported attack spec '{attack_spec}'. Use 'demo_figstep' or '<module>:<ClassName>'."
            )

        weight = float(cfg.attack_weights.get(attack_name, 1.0))
        registry.register(
            attack_name,
            AttackAdapter(attack_obj=attack_obj, attack_name=attack_name),
            weight=weight,
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

    registry = _build_registry(cfg)
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
            "target_vllm_base_url": cfg.target_vllm_base_url,
            "target_vllm_model": cfg.target_vllm_model,
            "target_vllm_api_key": cfg.target_vllm_api_key,
            "judge_base_url": cfg.judge_base_url or "",
            "judge_model": cfg.judge_model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "enable_vision": cfg.enable_vision,
            "concurrency": cfg.concurrency,
            "target_concurrency": cfg.target_concurrency or cfg.concurrency,
            "judge_concurrency": cfg.judge_concurrency or cfg.concurrency,
            "target_min_interval_sec": (1.0 / cfg.global_rate_limit_qps) if cfg.global_rate_limit_qps > 0 else 0.0,
            "judge_min_interval_sec": (1.0 / cfg.global_rate_limit_qps) if cfg.global_rate_limit_qps > 0 else 0.0,
            "total_candidates": 0,
            "round_topk_scores": [],
        },
    }

    checkpointer = InMemorySaver()
    app = compile_graph(checkpointer=checkpointer)

    # from langchain_core.runnables.graph import MermaidDrawMethod

    # # 获取图片的二进制数据 (bytes)
    # png_data = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)

    # # 写入文件
    # with open("graph.png", "wb") as f:
    #     f.write(png_data)

    # print("流程图已保存为 graph.png")

    thread_config = {"configurable": {"thread_id": cfg.run_id}}
    final_state = app.invoke(initial_state, config=thread_config)
    export_checkpoints(
        checkpointer=checkpointer,
        config=thread_config,
        run_id=cfg.run_id,
        runs_root="runs",
    )
    write_run_reports(final_state, runs_root="runs")
    print("Graph run finished")
    _print_best_summary(final_state)


if __name__ == "__main__":
    main()
