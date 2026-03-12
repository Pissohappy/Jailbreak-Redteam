#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 LLM-guided 方法的稳定性。

读取多次运行的实验结果，分析：
1. ASR (Attack Success Rate) 的稳定性
2. LLM 策略选择的一致性

Usage:
    # 分析单次运行的结果
    python scripts/analyze_stability.py --run-dir runs/exp1-expand-comparison/20260311_171035

    # 分析稳定性测试的多运行结果
    python scripts/analyze_stability.py --experiment-name exp7-stability --num-runs 3

    # 生成可视化报告
    python scripts/analyze_stability.py --run-dir runs/exp1-expand-comparison/20260311_171035 --output report.html
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_events_from_run(run_dir: Path) -> list[dict]:
    """加载单个运行目录中的所有 events.jsonl 文件"""
    events = []
    samples_dir = run_dir / "samples"
    if not samples_dir.exists():
        print(f"Warning: No samples directory found in {run_dir}")
        return events

    for sample_dir in samples_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        events_file = sample_dir / "events.jsonl"
        if events_file.exists():
            with open(events_file) as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))

    return events


def load_batch_summary(run_dir: Path) -> dict | None:
    """加载 batch_summary.json"""
    summary_file = run_dir / "batch_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)
    return None


def extract_llm_selections(events: list[dict]) -> list[dict]:
    """从事件列表中提取 LLMGuidedSelection 事件"""
    selections = []
    for event in events:
        if event.get("event_type") == "LLMGuidedSelection":
            payload = event.get("payload", {})
            selections.append({
                "timestamp": event.get("timestamp"),
                "run_id": event.get("run_id"),
                "round_idx": payload.get("round_idx"),
                "branch_id": payload.get("branch_id"),
                "used_fallback": payload.get("used_fallback", False),
                "fallback_reason": payload.get("fallback_reason"),
                "selected_attacks": payload.get("selected_attacks", []),
                "llm_response": payload.get("llm_response"),
            })
    return selections


def calculate_asr_stability(run_dirs: list[Path]) -> dict[str, Any]:
    """计算 ASR 稳定性指标"""
    asr_values = []
    run_summaries = []

    for run_dir in run_dirs:
        summary = load_batch_summary(run_dir)
        if summary:
            asr = summary.get("success_rate", 0)
            asr_values.append(asr)
            run_summaries.append({
                "run_dir": str(run_dir),
                "asr": asr,
                "total_samples": summary.get("total_samples", 0),
                "successful_samples": summary.get("successful_samples", 0),
            })

    if not asr_values:
        return {"error": "No valid run summaries found"}

    import statistics

    mean_asr = statistics.mean(asr_values)
    std_asr = statistics.stdev(asr_values) if len(asr_values) > 1 else 0

    # 计算置信区间 (95%)
    n = len(asr_values)
    if n > 1:
        import math
        ci_95 = 1.96 * std_asr / math.sqrt(n)
    else:
        ci_95 = 0

    return {
        "num_runs": len(asr_values),
        "asr_values": asr_values,
        "mean_asr": mean_asr,
        "std_asr": std_asr,
        "ci_95": ci_95,
        "min_asr": min(asr_values),
        "max_asr": max(asr_values),
        "range_asr": max(asr_values) - min(asr_values),
        "run_summaries": run_summaries,
    }


def analyze_strategy_consistency(run_dirs: list[Path]) -> dict[str, Any]:
    """分析 LLM 策略选择的一致性"""
    # 按 sample_id 分组收集策略选择
    sample_selections: dict[str, list[dict]] = defaultdict(list)

    for run_idx, run_dir in enumerate(run_dirs):
        events = load_events_from_run(run_dir)
        selections = extract_llm_selections(events)

        for sel in selections:
            # 从 run_id 提取 sample_id
            run_id = sel.get("run_id", "")
            # run_id 格式: exp_name/timestamp/samples/sample_XXX
            parts = run_id.split("/")
            sample_id = parts[-1] if parts else "unknown"

            sample_selections[sample_id].append({
                "run_idx": run_idx,
                "round_idx": sel.get("round_idx"),
                "selected_attacks": sel.get("selected_attacks", []),
                "used_fallback": sel.get("used_fallback", False),
                "fallback_reason": sel.get("fallback_reason"),
            })

    # 分析一致性
    consistency_stats = {
        "total_samples": len(sample_selections),
        "samples_with_multiple_runs": 0,
        "samples_with_consistent_selection": 0,
        "fallback_rate_per_run": defaultdict(int),
        "fallback_reasons": Counter(),  # 统计各环节 fallback 原因
        "fallback_reasons_per_run": defaultdict(lambda: Counter()),  # 按 run 统计 fallback 原因
        "attack_frequency": Counter(),
        "attack_selection_matrix": defaultdict(lambda: defaultdict(int)),
    }

    for sample_id, selections in sample_selections.items():
        if len(selections) > 1:
            consistency_stats["samples_with_multiple_runs"] += 1

            # 检查策略选择是否一致
            first_attacks = set()
            for sel in selections:
                attacks = tuple(
                    sorted([a.get("name", "") for a in sel.get("selected_attacks", [])])
                )
                first_attacks.add(attacks)

                # 统计 fallback 使用
                if sel.get("used_fallback"):
                    run_idx = sel.get("run_idx", 0)
                    consistency_stats["fallback_rate_per_run"][run_idx] += 1

                    # 统计 fallback 原因
                    reason = sel.get("fallback_reason", "unknown")
                    consistency_stats["fallback_reasons"][reason] += 1
                    consistency_stats["fallback_reasons_per_run"][run_idx][reason] += 1

                # 统计攻击方法使用频率
                for attack in sel.get("selected_attacks", []):
                    attack_name = attack.get("name", "unknown")
                    consistency_stats["attack_frequency"][attack_name] += 1
                    consistency_stats["attack_selection_matrix"][sample_id][attack_name] += 1

            if len(first_attacks) == 1:
                consistency_stats["samples_with_consistent_selection"] += 1

    # 计算一致性比率
    if consistency_stats["samples_with_multiple_runs"] > 0:
        consistency_stats["consistency_ratio"] = (
            consistency_stats["samples_with_consistent_selection"]
            / consistency_stats["samples_with_multiple_runs"]
        )
    else:
        consistency_stats["consistency_ratio"] = None

    # 转换 Counter 为普通字典以便 JSON 序列化
    consistency_stats["attack_frequency"] = dict(consistency_stats["attack_frequency"])
    consistency_stats["fallback_rate_per_run"] = dict(consistency_stats["fallback_rate_per_run"])
    consistency_stats["fallback_reasons"] = dict(consistency_stats["fallback_reasons"])
    consistency_stats["fallback_reasons_per_run"] = {
        k: dict(v) for k, v in consistency_stats["fallback_reasons_per_run"].items()
    }

    # 攻击选择矩阵转换为普通字典
    consistency_stats["attack_selection_matrix"] = {
        k: dict(v) for k, v in consistency_stats["attack_selection_matrix"].items()
    }

    return consistency_stats


def generate_report(asr_stability: dict, strategy_consistency: dict, output_path: Path | None = None) -> str:
    """生成稳定性分析报告"""
    report_lines = []

    report_lines.append("# LLM-guided 稳定性分析报告")
    report_lines.append("")
    report_lines.append("## ASR 稳定性分析")
    report_lines.append("")

    if "error" in asr_stability:
        report_lines.append(f"错误: {asr_stability['error']}")
    else:
        report_lines.append(f"- 运行次数: {asr_stability['num_runs']}")
        report_lines.append(f"- 平均 ASR: {asr_stability['mean_asr']:.2%}")
        report_lines.append(f"- 标准差: {asr_stability['std_asr']:.4f}")
        report_lines.append(f"- 95% 置信区间: ±{asr_stability['ci_95']:.4f}")
        report_lines.append(f"- ASR 范围: [{asr_stability['min_asr']:.2%}, {asr_stability['max_asr']:.2%}]")
        report_lines.append(f"- ASR 差异: {asr_stability['range_asr']:.2%}")
        report_lines.append("")
        report_lines.append("### 各运行 ASR 详情")
        report_lines.append("")
        for summary in asr_stability.get("run_summaries", []):
            report_lines.append(
                f"- {Path(summary['run_dir']).name}: "
                f"{summary['asr']:.2%} ({summary['successful_samples']}/{summary['total_samples']})"
            )

    report_lines.append("")
    report_lines.append("## LLM 策略选择一致性分析")
    report_lines.append("")
    report_lines.append(f"- 总样本数: {strategy_consistency['total_samples']}")
    report_lines.append(
        f"- 多运行样本数: {strategy_consistency['samples_with_multiple_runs']}"
    )

    if strategy_consistency["consistency_ratio"] is not None:
        report_lines.append(
            f"- 一致性比率: {strategy_consistency['consistency_ratio']:.2%} "
            f"({strategy_consistency['samples_with_consistent_selection']}/"
            f"{strategy_consistency['samples_with_multiple_runs']})"
        )

    report_lines.append("")
    report_lines.append("### 攻击方法使用频率")
    report_lines.append("")
    for attack, count in sorted(
        strategy_consistency["attack_frequency"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        report_lines.append(f"- {attack}: {count} 次")

    if strategy_consistency["fallback_rate_per_run"]:
        report_lines.append("")
        report_lines.append("### Fallback 使用率")
        report_lines.append("")
        for run_idx, count in sorted(strategy_consistency["fallback_rate_per_run"].items()):
            report_lines.append(f"- Run {run_idx + 1}: {count} 次 fallback")

        # Fallback 原因分析
        if strategy_consistency.get("fallback_reasons"):
            report_lines.append("")
            report_lines.append("### Fallback 原因分析")
            report_lines.append("")
            report_lines.append("#### 总体统计")
            report_lines.append("")
            # 原因描述映射
            reason_descriptions = {
                "llm_call_failed": "LLM 调用失败（网络/API 错误）",
                "parse_failed": "响应解析失败（JSON 格式错误）",
                "invalid_attack_name": "攻击名称无效（不在可用列表中）",
                "unknown": "未知原因",
            }
            for reason, count in sorted(
                strategy_consistency["fallback_reasons"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                desc = reason_descriptions.get(reason, reason)
                report_lines.append(f"- {desc}: {count} 次")

            # 按运行统计 fallback 原因
            if strategy_consistency.get("fallback_reasons_per_run"):
                report_lines.append("")
                report_lines.append("#### 各运行详情")
                report_lines.append("")
                for run_idx in sorted(strategy_consistency["fallback_reasons_per_run"].keys()):
                    reasons = strategy_consistency["fallback_reasons_per_run"][run_idx]
                    report_lines.append(f"**Run {run_idx + 1}:**")
                    for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
                        desc = reason_descriptions.get(reason, reason)
                        report_lines.append(f"  - {desc}: {count} 次")
                    report_lines.append("")

    report_lines.append("")

    # 输出到文件或返回字符串
    report = "\n".join(report_lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"报告已保存到: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM-guided stability")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Single run directory to analyze",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for multi-run stability analysis",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs for stability testing (default: 3)",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Root directory for runs (default: runs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for report (default: stdout)",
    )
    args = parser.parse_args()

    # 收集运行目录
    run_dirs = []

    if args.run_dir:
        # 单个运行目录
        run_dir = Path(args.run_dir)
        if run_dir.exists():
            run_dirs.append(run_dir)
        else:
            print(f"Error: Run directory not found: {run_dir}")
            sys.exit(1)
    elif args.experiment_name:
        # 多个运行目录，按实验名称和时间戳分组
        runs_root = Path(args.runs_root)
        exp_dir = runs_root / args.experiment_name
        if not exp_dir.exists():
            print(f"Error: Experiment directory not found: {exp_dir}")
            sys.exit(1)

        # 获取最近的 N 个运行目录
        timestamp_dirs = sorted(
            [d for d in exp_dir.iterdir() if d.is_dir()],
            key=lambda x: x.name,
            reverse=True,
        )[: args.num_runs]

        run_dirs = timestamp_dirs
        print(f"Found {len(run_dirs)} run directories for experiment {args.experiment_name}")
    else:
        print("Error: Either --run-dir or --experiment-name is required")
        sys.exit(1)

    if not run_dirs:
        print("Error: No run directories found")
        sys.exit(1)

    # 执行分析
    print(f"Analyzing {len(run_dirs)} run(s)...")

    # ASR 稳定性分析
    asr_stability = calculate_asr_stability(run_dirs)

    # 策略一致性分析
    strategy_consistency = analyze_strategy_consistency(run_dirs)

    # 生成报告
    output_path = Path(args.output) if args.output else None
    report = generate_report(asr_stability, strategy_consistency, output_path)

    if not args.output:
        print("\n" + "=" * 60)
        print(report)

    # 保存详细分析结果到 JSON
    if args.experiment_name:
        results_file = Path(args.runs_root) / f"{args.experiment_name}_stability_analysis.json"
    else:
        results_file = run_dirs[0] / "stability_analysis.json"

    results = {
        "asr_stability": asr_stability,
        "strategy_consistency": strategy_consistency,
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n详细分析结果已保存到: {results_file}")


if __name__ == "__main__":
    main()
