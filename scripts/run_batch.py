#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch run script for processing dataset with multiple attack strategies.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_redteam.cli.run import load_config, _build_registry, set_default_registry
from vlm_redteam.graph.build_graph import compile_graph
from vlm_redteam.graph.state import Branch
from vlm_redteam.storage.run_outputs import export_checkpoints, write_run_reports
from langgraph.checkpoint.memory import InMemorySaver


def main():
    parser = argparse.ArgumentParser(description="Batch run VLM redteam pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON file")
    parser.add_argument("--output-dir", default="runs", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()

    # Load config
    cfg = load_config(Path(args.config))
    print(f"Loaded config: {cfg.run_id}")

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    if args.limit:
        dataset = dataset[:args.limit]

    print(f"Processing {len(dataset)} samples")

    # Build registry once
    registry = _build_registry(cfg)
    set_default_registry(registry)

    results = []

    for idx, sample in enumerate(dataset):
        sample_id = sample.get("id", idx)
        goal = sample.get("original_prompt", "")
        image_path = sample.get("image_path", "")

        # Adjust image path if needed
        if image_path and not Path(image_path).is_absolute():
            # Try relative to dataset directory
            dataset_dir = Path(args.dataset).parent
            full_image_path = dataset_dir / image_path
            if full_image_path.exists():
                image_path = str(full_image_path)

        print(f"\n[{idx+1}/{len(dataset)}] Processing sample {sample_id}")
        print(f"  Goal: {goal[:80]}...")
        print(f"  Image: {image_path}")

        # Create run ID for this sample
        run_id = f"{cfg.run_id}_sample_{sample_id}"

        initial_state = {
            "run_id": run_id,
            "task": {"goal": goal, "image_path": image_path},
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
                "runs_root": args.output_dir,
                "target_vllm_base_url": cfg.target_vllm_base_url,
                "target_vllm_model": cfg.target_vllm_model,
                "target_vllm_api_key": cfg.target_vllm_api_key,
                "judge_base_url": cfg.judge_base_url or "",
                "judge_model": cfg.judge_model,
                "judge_mode": cfg.judge_mode,
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
                "enable_vision": cfg.enable_vision,
                "concurrency": cfg.concurrency,
                "target_concurrency": cfg.target_concurrency or cfg.concurrency,
                "judge_concurrency": cfg.judge_concurrency or cfg.concurrency,
                "target_min_interval_sec": 0.0,
                "judge_min_interval_sec": 0.0,
                "total_candidates": 0,
                "round_topk_scores": [],
                "expand_strategy_name": cfg.expand_strategy,
                "llm_guide_config": dict(cfg.llm_guide) if cfg.llm_guide else None,
            },
        }

        try:
            checkpointer = InMemorySaver()
            app = compile_graph(checkpointer=checkpointer)

            thread_config = {"configurable": {"thread_id": run_id}}
            final_state = app.invoke(initial_state, config=thread_config)

            # Export results
            export_checkpoints(
                checkpointer=checkpointer,
                config=thread_config,
                run_id=run_id,
                runs_root=args.output_dir,
            )

            # Extract result
            best = final_state.get("best")
            result = {
                "sample_id": sample_id,
                "run_id": run_id,
                "goal": goal,
                "image_path": str(image_path),
                "success": best.judge_success if best else False,
                "aggregate_score": best.aggregate_score if best else None,
            }
            results.append(result)

            print(f"  Success: {result['success']}, Score: {result['aggregate_score']}")

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "sample_id": sample_id,
                "run_id": run_id,
                "goal": goal,
                "image_path": str(image_path),
                "success": False,
                "error": str(e),
            })

    # Save summary
    output_path = Path(args.output_dir) / cfg.run_id / "batch_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nBatch completed. Results saved to {output_path}")

    # Print summary
    success_count = sum(1 for r in results if r.get("success"))
    print(f"Success rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")


if __name__ == "__main__":
    main()
