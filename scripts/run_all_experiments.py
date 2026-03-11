#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行所有实验配置，并发送邮件通知进度。

Usage:
    python scripts/run_all_experiments.py --dataset data_sample50.json

    # 只运行特定实验
    python scripts/run_all_experiments.py --dataset data_sample50.json --experiments exp1 exp2

    # 测试模式（不实际运行，只打印命令）
    python scripts/run_all_experiments.py --dataset data_sample50.json --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import monitor module
monitor_path = Path("/mnt/disk1/szchen/monitor.py")
if monitor_path.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("monitor", monitor_path)
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    send_notification = monitor.send_notification
    HAS_MONITOR = True
else:
    print("Warning: monitor.py not found, email notifications disabled")
    HAS_MONITOR = False


# 实验配置
EXPERIMENTS = [
    # 实验1: Expand策略对比
    {
        "name": "exp1-llm-guided",
        "config": "configs/experiments/exp1_llm_guided.yaml",
        "experiment_name": "exp1-expand-comparison",
        "description": "实验1a: LLM-guided expand策略",
    },
    {
        "name": "exp1-random-sampling",
        "config": "configs/experiments/exp1_random_sampling.yaml",
        "experiment_name": "exp1-expand-comparison",
        "description": "实验1b: Random-sampling expand策略",
    },
    # 实验2: Judge配置对比
    {
        "name": "exp2-judge-multidim",
        "config": "configs/experiments/exp2_judge_multidim.yaml",
        "experiment_name": "exp2-judge-comparison",
        "description": "实验2a: Judge multidim模式",
    },
    {
        "name": "exp2-judge-strongreject",
        "config": "configs/experiments/exp2_judge_strongreject.yaml",
        "experiment_name": "exp2-judge-comparison",
        "description": "实验2b: Judge strongreject模式",
    },
    {
        "name": "exp2-judge-qwen",
        "config": "configs/experiments/exp2_judge_qwen.yaml",
        "experiment_name": "exp2-judge-comparison",
        "description": "实验2c: Judge使用Qwen模型",
    },
    # 实验3: 轮数分析
    {
        "name": "exp3-max-rounds-10",
        "config": "configs/experiments/exp3_max_rounds_10.yaml",
        "experiment_name": "exp3-rounds-analysis",
        "description": "实验3: 10轮攻击分析",
    },
]


def send_email(subject: str, content: str):
    """发送邮件通知"""
    if HAS_MONITOR:
        return send_notification(subject, content)
    else:
        print(f"[Email] {subject}")
        print(content)
        return False


def run_experiment(exp: dict, dataset: str, dry_run: bool = False) -> dict:
    """运行单个实验"""
    result = {
        "name": exp["name"],
        "description": exp["description"],
        "config": exp["config"],
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "success": False,
        "success_rate": None,
        "total_samples": None,
        "error": None,
    }

    cmd = [
        "python", "scripts/run_batch.py",
        "--config", exp["config"],
        "--dataset", dataset,
        "--experiment-name", exp["experiment_name"],
    ]

    print(f"\n{'='*60}")
    print(f"Starting: {exp['description']}")
    print(f"Config: {exp['config']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    if dry_run:
        print("[DRY-RUN] Skipping actual execution")
        result["success"] = True
        result["end_time"] = datetime.now().isoformat()
        return result

    start = time.time()

    try:
        # Run the experiment
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        elapsed = time.time() - start
        result["elapsed_seconds"] = elapsed
        result["end_time"] = datetime.now().isoformat()

        if process.returncode == 0:
            result["success"] = True
            print(f"✓ Experiment completed successfully in {elapsed:.1f}s")

            # Try to parse results from output
            output = process.stdout
            if "Success rate:" in output:
                for line in output.split("\n"):
                    if "Success rate:" in line:
                        result["raw_output"] = line.strip()
        else:
            result["error"] = process.stderr or process.stdout
            print(f"✗ Experiment failed: {result['error'][:200]}")

    except Exception as e:
        result["error"] = str(e)
        result["end_time"] = datetime.now().isoformat()
        print(f"✗ Exception: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run all experiments sequentially")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset JSON file",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Specific experiments to run (by name). Default: all experiments",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running",
    )
    parser.add_argument(
        "--no-email",
        action="store_true",
        help="Disable email notifications",
    )
    args = parser.parse_args()

    # Filter experiments
    if args.experiments:
        exp_names = set(args.experiments)
        experiments_to_run = [e for e in EXPERIMENTS if e["name"] in exp_names]
        if not experiments_to_run:
            print(f"Error: No matching experiments found for: {args.experiments}")
            print(f"Available: {[e['name'] for e in EXPERIMENTS]}")
            sys.exit(1)
    else:
        experiments_to_run = EXPERIMENTS

    total = len(experiments_to_run)

    # Send start notification
    if not args.no_email:
        start_msg = f"""VLM Redteam 实验批次开始

开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
数据集: {args.dataset}
实验数量: {total}

实验列表:
"""
        for i, exp in enumerate(experiments_to_run, 1):
            start_msg += f"  {i}. {exp['name']}: {exp['description']}\n"

        send_email("[VLM实验] 批次开始", start_msg)

    # Run experiments
    results = []
    for i, exp in enumerate(experiments_to_run, 1):
        print(f"\n\n{'#'*60}")
        print(f"# 实验 {i}/{total}: {exp['name']}")
        print(f"{'#'*60}\n")

        result = run_experiment(exp, args.dataset, args.dry_run)
        results.append(result)

        # Send progress notification after each experiment
        if not args.no_email:
            elapsed = result.get("elapsed_seconds", 0)
            status = "成功" if result["success"] else "失败"

            progress_msg = f"""实验进度更新 ({i}/{total})

实验: {exp['name']}
描述: {exp['description']}
状态: {status}
耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)
"""
            if result.get("error"):
                progress_msg += f"\n错误信息:\n{result['error'][:500]}"

            send_email(f"[VLM实验] {i}/{total} {exp['name']} - {status}", progress_msg)

    # Send final summary
    if not args.no_email:
        successful = sum(1 for r in results if r["success"])
        failed = total - successful

        summary_msg = f"""VLM Redteam 实验批次完成

完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
总实验数: {total}
成功: {successful}
失败: {failed}

详细结果:
"""
        for r in results:
            status = "✓" if r["success"] else "✗"
            elapsed = r.get("elapsed_seconds", 0)
            summary_msg += f"  {status} {r['name']}: {r['description']} ({elapsed:.1f}s)\n"
            if r.get("error"):
                summary_msg += f"      错误: {r['error'][:100]}\n"

        send_email(f"[VLM实验] 批次完成 ({successful}/{total}成功)", summary_msg)

    # Save results to file
    results_file = Path("runs") / "experiment_batch_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    batch_summary = {
        "start_time": results[0]["start_time"] if results else None,
        "end_time": results[-1]["end_time"] if results else None,
        "total_experiments": total,
        "successful": sum(1 for r in results if r["success"]),
        "results": results,
    }

    with open(results_file, "w") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'='*60}")
    print("批次运行完成!")
    print(f"成功: {successful}/{total}")
    print(f"结果保存至: {results_file}")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
