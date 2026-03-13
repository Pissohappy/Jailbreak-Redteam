#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch run script for processing dataset with multiple attack strategies.

Directory structure:
runs/
├── {experiment_name}/                    # Experiment name (e.g., visual-perturb-sample50)
│   ├── {timestamp}/                      # Timestamp directory (e.g., 20260311_143052)
│   │   ├── config.yaml                   # Saved config copy
│   │   ├── batch_summary.json            # Batch run summary
│   │   ├── samples/                      # All sample results
│   │   │   ├── sample_10/
│   │   │   │   ├── checkpoints.json
│   │   │   │   ├── summary.json
│   │   │   │   ├── best_path.md
│   │   │   │   └── events.jsonl
│   │   │   ├── sample_36/
│   │   │   └── ...
│   │   └── generated_images/             # Generated images
│   └── ...（other timestamp directories）
"""

import argparse
import json
import shutil
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
    parser.add_argument("--experiment-name", default=None, help="Experiment name (defaults to config run_id)")
    args = parser.parse_args()

    # Load config
    cfg = load_config(Path(args.config))
    print(f"Loaded config: {cfg.run_id}")

    # Generate experiment name and timestamp
    experiment_name = args.experiment_name or cfg.run_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create experiment directory structure
    output_dir = Path(args.output_dir)
    experiment_dir = output_dir / experiment_name / timestamp
    samples_dir = experiment_dir / "samples"
    generated_images_dir = experiment_dir / "generated_images"

    # Create directories
    samples_dir.mkdir(parents=True, exist_ok=True)
    generated_images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment directory: {experiment_dir}")
    print(f"Timestamp: {timestamp}")

    # Save config copy
    config_copy_path = experiment_dir / "config.yaml"
    shutil.copy(args.config, config_copy_path)
    print(f"Config saved to: {config_copy_path}")

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
        main_category = sample.get("main_category", "")
        subcategory = sample.get("subcategory", "")

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

        # Create sample directory
        sample_dir = samples_dir / f"sample_{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Create run ID for this sample (relative path within experiment)
        run_id = f"{experiment_name}/{timestamp}/samples/sample_{sample_id}"

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
                "seed": int(sample_id),  # Use sample_id to ensure different random seeds per sample
                "runs_root": args.output_dir,
                "target_vllm_base_url": cfg.target_vllm_base_url,
                "target_vllm_model": cfg.target_vllm_model,
                "target_vllm_api_key": cfg.target_vllm_api_key,
                "judge_base_url": cfg.judge_base_url or "",
                "judge_model": cfg.judge_model,
                "judge_mode": cfg.judge_mode,
                "judge_api_key": cfg.judge_api_key,
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

            # Export results to sample directory
            export_checkpoints(
                checkpointer=checkpointer,
                config=thread_config,
                run_id=run_id,
                runs_root=args.output_dir,
            )
            write_run_reports(final_state, runs_root=args.output_dir)

            # Extract result
            best = final_state.get("best")
            result = {
                "sample_id": sample_id,
                "run_id": run_id,
                "goal": goal,
                "image_path": str(image_path),
                "main_category": main_category,
                "subcategory": subcategory,
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
                "main_category": main_category,
                "subcategory": subcategory,
                "success": False,
                "error": str(e),
            })

    # Save batch summary
    batch_summary = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "config_path": str(config_copy_path),
        "dataset_path": str(args.dataset),
        "total_samples": len(results),
        "successful_samples": sum(1 for r in results if r.get("success")),
        "success_rate": sum(1 for r in results if r.get("success")) / len(results) if results else 0,
        "results": results,
    }

    batch_summary_path = experiment_dir / "batch_summary.json"
    with open(batch_summary_path, "w") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)

    print(f"\nBatch completed. Summary saved to {batch_summary_path}")

    # Print summary
    success_count = sum(1 for r in results if r.get("success"))
    print(f"Success rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")


if __name__ == "__main__":
    main()
