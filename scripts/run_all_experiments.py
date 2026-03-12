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

    # 实验5: 测试不同的 target models
    python scripts/run_all_experiments.py --dataset data_sample50.json \
        --experiment-type target-models \
        --target-models "qwen3-vl-8b:http://localhost:8008" "gpt-4o:http://xxx"

    # 实验6: 测试不同的 LLM-guided models
    python scripts/run_all_experiments.py --dataset data_sample50.json \
        --experiment-type llm-guide-models \
        --guide-models "gpt-4o-mini:https://api.whatai.cc:sk-xxx" "gpt-4o:https://api.whatai.cc:sk-xxx"

    # 实验7: 稳定性测试 (默认3次)
    python scripts/run_all_experiments.py --dataset data_sample50.json \
        --experiment-type stability \
        --stability-runs 3
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import YAML for config override
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

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


# ============================================================================
# Configuration Override Functions
# ============================================================================

def apply_overrides(config_path: Path, overrides: dict[str, Any], run_id_suffix: str = "") -> Path:
    """
    创建临时配置文件，应用覆盖参数。

    Args:
        config_path: 原始配置文件路径
        overrides: 覆盖参数字典，支持嵌套如 {"llm_guide.model": "gpt-4o"}
        run_id_suffix: run_id 后缀，用于区分不同配置

    Returns:
        临时配置文件路径
    """
    if not HAS_YAML:
        raise RuntimeError("PyYAML is required for config override. Install with: pip install pyyaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 支持嵌套覆盖，如 "llm_guide.model": "gpt-4o"
    for key, value in overrides.items():
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    # 修改 run_id 以区分不同配置
    if run_id_suffix:
        original_run_id = cfg.get("run_id", config_path.stem)
        cfg["run_id"] = f"{original_run_id}-{run_id_suffix}"

    # 更新 attack_init_kwargs 中的 output_image_dir
    if "attack_init_kwargs" in cfg:
        new_run_id = cfg.get("run_id", config_path.stem)
        for attack_key, attack_config in cfg["attack_init_kwargs"].items():
            if isinstance(attack_config, dict) and "output_image_dir" in attack_config:
                attack_config["output_image_dir"] = f"runs/{new_run_id}/generated_images"

    # 保存到临时文件
    temp_dir = Path("configs/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"temp_{uuid.uuid4().hex[:8]}.yaml"
    with open(temp_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    return temp_path


def parse_model_spec(model_spec: str) -> dict[str, str]:
    """
    解析模型规格字符串。

    格式: "model_name:url" 或 "model_name:url:api_key"
    注意: URL 中可能包含冒号（如 http://localhost:8008）

    解析规则:
    1. 第一个冒号分隔 model_name 和剩余部分
    2. 如果剩余部分包含 "://"，则识别为完整 URL
    3. 检查最后一个冒号后的部分：
       - 如果是纯数字（端口号），视为 URL 的一部分
       - 如果以 "sk-" 开头或包含字母，视为 api_key

    Args:
        model_spec: 模型规格字符串

    Returns:
        包含 name, url, api_key (可选) 的字典
    """
    # 第一个冒号分隔 name 和 url+api_key
    first_colon = model_spec.find(":")
    if first_colon == -1:
        raise ValueError(f"Invalid model spec format: {model_spec}. Expected 'name:url' or 'name:url:api_key'")

    name = model_spec[:first_colon]
    remainder = model_spec[first_colon + 1:]

    # 检查是否有 http:// 或 https://
    if remainder.startswith("http://") or remainder.startswith("https://"):
        # URL 格式: http(s)://host:port 或 http(s)://host:port:api_key
        # 找所有冒号的位置
        colon_positions = [i for i, c in enumerate(remainder) if c == ":"]

        if len(colon_positions) <= 1:
            # 只有 http:// 或 https://，没有端口，没有 api_key
            url = remainder
            api_key = None
        else:
            # 有多个冒号，检查最后一部分是否像 api_key 或端口号
            last_colon_pos = colon_positions[-1]
            last_segment = remainder[last_colon_pos + 1:]

            # 判断是否为端口号（纯数字，通常 1-65535）
            is_port = last_segment.isdigit() and 1 <= int(last_segment) <= 65535

            # 判断是否为 api_key
            is_api_key = (
                last_segment.startswith("sk-") or  # OpenAI style
                (not last_segment.isdigit() and len(last_segment) > 0)  # Contains non-digits
            )

            if is_api_key and not is_port:
                url = remainder[:last_colon_pos]
                api_key = last_segment
            else:
                url = remainder
                api_key = None
    else:
        # 非 http URL，简单分割
        if ":" in remainder:
            parts = remainder.rsplit(":", 1)
            url = parts[0]
            api_key = parts[1]
        else:
            url = remainder
            api_key = None

    result = {"name": name, "url": url}
    if api_key:
        result["api_key"] = api_key

    return result


def generate_target_model_experiments(
    base_config: str,
    target_models: list[str],
    experiment_name: str = "exp5-target-comparison",
) -> list[dict]:
    """
    生成不同 target model 的实验配置。

    Args:
        base_config: 基础配置文件路径
        target_models: target model 规格列表，格式 "name:url" 或 "name:url:api_key"
        experiment_name: 实验名称

    Returns:
        实验配置列表
    """
    experiments = []
    for model_spec in target_models:
        spec = parse_model_spec(model_spec)
        overrides = {
            "target_vllm_model": spec["name"],
            "target_vllm_base_url": spec["url"],
        }
        if "api_key" in spec:
            overrides["target_vllm_api_key"] = spec["api_key"]

        experiments.append({
            "name": f"exp5-target-{spec['name']}",
            "config": base_config,
            "experiment_name": experiment_name,
            "description": f"实验5: Target Model {spec['name']}",
            "overrides": overrides,
            "type": "target_model",
        })

    return experiments


def generate_llm_guide_experiments(
    base_config: str,
    guide_models: list[str],
    experiment_name: str = "exp6-guide-comparison",
) -> list[dict]:
    """
    生成不同 LLM-guided model 的实验配置。

    Args:
        base_config: 基础配置文件路径
        guide_models: guide model 规格列表，格式 "name:url" 或 "name:url:api_key"
        experiment_name: 实验名称

    Returns:
        实验配置列表
    """
    experiments = []
    for model_spec in guide_models:
        spec = parse_model_spec(model_spec)
        overrides = {
            "llm_guide.model": spec["name"],
            "llm_guide.base_url": spec["url"],
        }
        if "api_key" in spec:
            overrides["llm_guide.api_key"] = spec["api_key"]

        experiments.append({
            "name": f"exp6-guide-{spec['name']}",
            "config": base_config,
            "experiment_name": experiment_name,
            "description": f"实验6: LLM-guide Model {spec['name']}",
            "overrides": overrides,
            "type": "llm_guide",
        })

    return experiments


def generate_stability_experiments(
    base_config: str,
    num_runs: int,
    experiment_name: str = "exp7-stability",
) -> list[dict]:
    """
    生成稳定性测试的实验配置。

    Args:
        base_config: 基础配置文件路径
        num_runs: 重复运行次数
        experiment_name: 实验名称

    Returns:
        实验配置列表（每个运行一个配置）
    """
    experiments = []
    for run_idx in range(num_runs):
        experiments.append({
            "name": f"exp7-stability-run{run_idx + 1}",
            "config": base_config,
            "experiment_name": experiment_name,
            "description": f"实验7: 稳定性测试 Run {run_idx + 1}/{num_runs}",
            "run_index": run_idx + 1,
            "type": "stability",
        })

    return experiments


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
    # 实验4: 多轮对话历史策略对比
    {
        "name": "exp4-history-inherit",
        "config": "configs/experiments/exp_history_inherit.yaml",
        "experiment_name": "exp4-history-comparison",
        "description": "实验4a: 完整多轮历史策略 (inherit_parent)",
    },
    {
        "name": "exp4-history-none",
        "config": "configs/experiments/exp_history_none.yaml",
        "experiment_name": "exp4-history-comparison",
        "description": "实验4b: 无历史策略 (每轮独立执行)",
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
        "run_dir": None,
    }

    # 处理配置覆盖
    config_path = exp["config"]
    temp_config_path = None
    if "overrides" in exp and exp["overrides"]:
        run_id_suffix = exp["name"].split("-", 1)[-1] if "-" in exp["name"] else exp["name"]
        temp_config_path = apply_overrides(
            Path(config_path),
            exp["overrides"],
            run_id_suffix=run_id_suffix,
        )
        config_path = str(temp_config_path)
        result["temp_config"] = str(temp_config_path)
        print(f"Created temp config with overrides: {temp_config_path}")

    cmd = [
        "python", "scripts/run_batch.py",
        "--config", config_path,
        "--dataset", dataset,
        "--experiment-name", exp["experiment_name"],
    ]

    print(f"\n{'='*60}")
    print(f"Starting: {exp['description']}")
    print(f"Config: {exp['config']}")
    if "overrides" in exp and exp["overrides"]:
        print(f"Overrides: {exp['overrides']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    if dry_run:
        print("[DRY-RUN] Skipping actual execution")
        result["success"] = True
        result["end_time"] = datetime.now().isoformat()
        # Clean up temp config if created
        if temp_config_path and temp_config_path.exists():
            temp_config_path.unlink()
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

            # Try to find run directory
            if "Experiment directory:" in output:
                for line in output.split("\n"):
                    if "Experiment directory:" in line:
                        result["run_dir"] = line.split(":", 1)[1].strip()
        else:
            result["error"] = process.stderr or process.stdout
            print(f"✗ Experiment failed: {result['error'][:200]}")

    except Exception as e:
        result["error"] = str(e)
        result["end_time"] = datetime.now().isoformat()
        print(f"✗ Exception: {e}")

    finally:
        # Clean up temp config if created
        if temp_config_path and temp_config_path.exists():
            temp_config_path.unlink()

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
    # 新增参数：实验类型
    parser.add_argument(
        "--experiment-type",
        choices=["default", "target-models", "llm-guide-models", "stability"],
        default="default",
        help="Type of experiments to run",
    )
    # 新增参数：target models
    parser.add_argument(
        "--target-models",
        nargs="+",
        default=None,
        help="Target models to test. Format: 'model_name:url' or 'model_name:url:api_key'",
    )
    # 新增参数：LLM-guided models
    parser.add_argument(
        "--guide-models",
        nargs="+",
        default=None,
        help="LLM-guided models to test. Format: 'model_name:url' or 'model_name:url:api_key'",
    )
    # 新增参数：稳定性测试运行次数
    parser.add_argument(
        "--stability-runs",
        type=int,
        default=3,
        help="Number of runs for stability testing (default: 3)",
    )
    # 新增参数：基础配置文件
    parser.add_argument(
        "--base-config",
        default="configs/experiments/exp1_llm_guided.yaml",
        help="Base config file for dynamic experiments (default: exp1_llm_guided.yaml)",
    )
    # 新增参数：限制样本数
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process",
    )
    args = parser.parse_args()

    # 根据实验类型生成实验配置
    if args.experiment_type == "target-models":
        if not args.target_models:
            print("Error: --target-models is required for target-models experiment type")
            sys.exit(1)
        experiments_to_run = generate_target_model_experiments(
            base_config=args.base_config,
            target_models=args.target_models,
        )
        print(f"Generated {len(experiments_to_run)} target model experiments")
    elif args.experiment_type == "llm-guide-models":
        if not args.guide_models:
            print("Error: --guide-models is required for llm-guide-models experiment type")
            sys.exit(1)
        experiments_to_run = generate_llm_guide_experiments(
            base_config=args.base_config,
            guide_models=args.guide_models,
        )
        print(f"Generated {len(experiments_to_run)} LLM-guide model experiments")
    elif args.experiment_type == "stability":
        experiments_to_run = generate_stability_experiments(
            base_config=args.base_config,
            num_runs=args.stability_runs,
        )
        print(f"Generated {len(experiments_to_run)} stability test runs")
    else:
        # 默认行为：使用 EXPERIMENTS 列表
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

    # 构建数据集参数（可能包含 --limit）
    dataset_arg = args.dataset
    if args.limit:
        # 对于 stability 测试，我们需要在 run_batch.py 层面传递 limit
        pass  # 暂时忽略，run_batch.py 不支持 --limit 参数

    # Send start notification
    if not args.no_email:
        start_msg = f"""VLM Redteam 实验批次开始

开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
数据集: {args.dataset}
实验类型: {args.experiment_type}
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_type_suffix = f"_{args.experiment_type}" if args.experiment_type != "default" else ""
    results_file = Path("runs") / f"experiment_batch_results{exp_type_suffix}_{timestamp}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    batch_summary = {
        "start_time": results[0]["start_time"] if results else None,
        "end_time": results[-1]["end_time"] if results else None,
        "experiment_type": args.experiment_type,
        "total_experiments": total,
        "successful": sum(1 for r in results if r["success"]),
        "results": results,
    }

    with open(results_file, "w") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)

    successful = sum(1 for r in results if r["success"])
    failed = total - successful

    print(f"\n\n{'='*60}")
    print("批次运行完成!")
    print(f"成功: {successful}/{total}")
    print(f"结果保存至: {results_file}")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
