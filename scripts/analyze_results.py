#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI tool for analyzing VLM redteam experiment results.

Usage:
    # List all experiments
    python scripts/analyze_results.py --list

    # List runs for an experiment
    python scripts/analyze_results.py --experiment visual-perturb-sample50 --list-runs

    # Analyze a specific run
    python scripts/analyze_results.py --experiment visual-perturb-sample50 --timestamp 20260311_143052

    # Analyze the latest run for an experiment
    python scripts/analyze_results.py --experiment visual-perturb-sample50 --latest

    # Generate plots
    python scripts/analyze_results.py --experiment visual-perturb-sample50 --latest --plots

    # Analyze attack details (calls, success rate by attack type)
    python scripts/analyze_results.py --experiment visual-perturb-sample50 --latest --attack-analysis

    # Round analysis (success rate by round)
    python scripts/analyze_results.py --experiment visual-perturb-sample50 --latest --round-analysis --plots

    # Case analysis
    python scripts/analyze_results.py --experiment visual-perturb-sample50 --latest --case-analysis

    # Compare multiple runs
    python scripts/analyze_results.py --compare visual-perturb-sample50:20260311_143052 visual-perturb-sample50:20260312_100000
"""

import argparse
import json
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_redteam.analysis.loader import ExperimentLoader
from vlm_redteam.analysis.metrics import (
    calculate_success_rate,
    calculate_category_metrics,
    calculate_score_distribution,
    calculate_cumulative_asr_by_round,
    generate_summary_report,
    calculate_attack_metrics,
    calculate_attack_success_by_round,
    calculate_round_success_distribution,
    get_case_analysis,
)
from vlm_redteam.analysis.visualization import (
    generate_all_plots,
    plot_strategy_comparison,
    generate_attack_analysis_plots,
    plot_attack_call_distribution,
    plot_attack_calls_by_round,
    plot_success_rate_by_round,
    plot_round_success_distribution,
    plot_round_success_line,
    plot_round_success_comparison,
    generate_round_analysis_plots,
)
from vlm_redteam.analysis.metrics import calculate_round_success_rate


def print_separator(char: str = "-", length: int = 60):
    print(char * length)


def cmd_list_experiments(loader: ExperimentLoader):
    """List all experiments."""
    experiments = loader.list_experiments()

    if not experiments:
        print("No experiments found in runs directory.")
        return

    print_separator("=")
    print("Available Experiments")
    print_separator("=")

    for exp in experiments:
        runs = loader.list_runs(exp)
        print(f"\n📁 {exp}")
        print(f"   Runs: {len(runs)}")
        if runs:
            print(f"   Latest: {runs[-1]}")


def cmd_list_runs(loader: ExperimentLoader, experiment_name: str):
    """List all runs for an experiment."""
    runs = loader.list_runs(experiment_name)

    if not runs:
        print(f"No runs found for experiment: {experiment_name}")
        return

    print_separator("=")
    print(f"Runs for: {experiment_name}")
    print_separator("=")

    for run_ts in runs:
        run = loader.load_run(experiment_name, run_ts)
        if run:
            print(f"\n📅 {run_ts}")
            print(f"   Samples: {run.total_samples}")
            print(f"   Success Rate: {run.success_rate * 100:.1f}%")


def cmd_analyze_run(
    loader: ExperimentLoader,
    experiment_name: str,
    timestamp: str | None = None,
    generate_plots: bool = False,
    output_dir: str | None = None,
):
    """Analyze a specific run."""
    if timestamp:
        run = loader.load_with_sample_details(experiment_name, timestamp)
    else:
        run = loader.load_latest_run(experiment_name)

    if not run:
        print(f"Run not found: {experiment_name}" + (f"/{timestamp}" if timestamp else ""))
        return

    print_separator("=")
    print(f"Experiment: {run.experiment_name}")
    if run.timestamp:
        print(f"Timestamp: {run.timestamp}")
    print_separator("=")

    # Overall metrics
    overall = calculate_success_rate(run)
    print("\n📊 Overall Metrics")
    print(f"   Total Samples: {overall['total_samples']}")
    print(f"   Successful: {overall['successful_samples']}")
    print(f"   Success Rate: {overall['success_rate'] * 100:.1f}%")
    if overall['avg_score'] is not None:
        print(f"   Average Score: {overall['avg_score']:.3f}")
    if overall['errors'] > 0:
        print(f"   Errors: {overall['errors']}")

    # Main category breakdown
    print("\n📈 Success Rate by Main Category")
    main_cat_metrics = calculate_category_metrics(run, "main_category")
    main_cat_metrics = sorted(main_cat_metrics, key=lambda m: m.success_rate, reverse=True)

    print(f"   {'Category':<40} {'Total':>6} {'Success':>8} {'Rate':>8}")
    print(f"   {'-'*40} {'-'*6} {'-'*8} {'-'*8}")
    for m in main_cat_metrics:
        print(f"   {m.category:<40} {m.total:>6} {m.successful:>8} {m.success_rate*100:>7.1f}%")

    # Subcategory breakdown (top 10 by success rate)
    print("\n📈 Top 10 Subcategories by Success Rate")
    subcat_metrics = calculate_category_metrics(run, "subcategory")
    subcat_metrics = sorted(subcat_metrics, key=lambda m: m.success_rate, reverse=True)[:10]

    print(f"   {'Subcategory':<50} {'Total':>6} {'Rate':>8}")
    print(f"   {'-'*50} {'-'*6} {'-'*8}")
    for m in subcat_metrics:
        print(f"   {m.category:<50} {m.total:>6} {m.success_rate*100:>7.1f}%")

    # Score distribution
    score_dist = calculate_score_distribution(run)
    if score_dist["min"] is not None:
        print("\n📉 Score Distribution")
        print(f"   Min: {score_dist['min']:.3f}")
        print(f"   Max: {score_dist['max']:.3f}")
        print(f"   Mean: {score_dist['mean']:.3f}")
        print(f"   Median: {score_dist['median']:.3f}")
        print(f"   Std: {score_dist['std']:.3f}")

    # Cumulative ASR by round
    cumulative_asr = calculate_cumulative_asr_by_round(run)
    if cumulative_asr and any(c.new_success_at_round > 0 for c in cumulative_asr):
        print("\n🔄 Cumulative ASR by Round")
        print(f"   {'Round':<6} {'New Success':>12} {'Cumulative':>12} {'ASR':>8}")
        print(f"   {'-'*6} {'-'*12} {'-'*12} {'-'*8}")
        for c in cumulative_asr:
            if c.new_success_at_round > 0 or c.round_idx <= 3:  # Show first 3 rounds and any with success
                print(f"   {c.round_idx:<6} {c.new_success_at_round:>12} {c.cumulative_success:>12} {c.cumulative_asr*100:>7.1f}%")

    # Generate plots if requested
    if generate_plots:
        plots_dir = output_dir or f"runs/{experiment_name}/{run.timestamp}/analysis"
        print(f"\n📊 Generating plots to: {plots_dir}")

        plots = generate_all_plots(run, plots_dir)
        for name, path in plots.items():
            print(f"   {name}: {path}")


def cmd_attack_analysis(
    loader: ExperimentLoader,
    experiment_name: str,
    timestamp: str | None = None,
    generate_plots: bool = False,
    output_dir: str | None = None,
    top_n: int = 15,
):
    """Analyze attack calls and success rates."""
    # Load with attack events
    if timestamp:
        run = loader.load_with_attack_events(experiment_name, timestamp)
    else:
        # Need to get the latest timestamp first
        runs = loader.list_runs(experiment_name)
        if not runs:
            print(f"No runs found for experiment: {experiment_name}")
            return
        run = loader.load_with_attack_events(experiment_name, runs[-1])

    if not run:
        print(f"Run not found: {experiment_name}" + (f"/{timestamp}" if timestamp else ""))
        return

    print_separator("=")
    print(f"Attack Analysis: {run.experiment_name}")
    if run.timestamp:
        print(f"Timestamp: {run.timestamp}")
    print_separator("=")

    # Check if attack events are loaded
    total_events = sum(len(r.attack_events) for r in run.results)
    if total_events == 0:
        print("\n⚠️ No attack events found. Make sure events.jsonl files exist in sample directories.")
        return

    # Attack metrics
    attack_metrics = calculate_attack_metrics(run)

    if not attack_metrics:
        print("\nNo attack metrics to display.")
        return

    print(f"\n🎯 Attack Call Analysis (Top {top_n} by calls)")
    print(f"   {'Attack Type':<30} {'Total Calls':>12} {'Successful':>10} {'Rate':>8}")
    print(f"   {'-'*30} {'-'*12} {'-'*10} {'-'*8}")

    for m in attack_metrics[:top_n]:
        print(f"   {m.short_name:<30} {m.total_calls:>12} {m.successful_calls:>10} {m.success_rate*100:>7.1f}%")

    # Total calls summary
    total_calls = sum(m.total_calls for m in attack_metrics)
    total_successes = sum(m.successful_calls for m in attack_metrics)
    print(f"\n   Total Attack Calls: {total_calls}")
    print(f"   Total Successful: {total_successes}")
    print(f"   Overall Attack Success Rate: {total_successes/total_calls*100:.1f}%" if total_calls > 0 else "   N/A")

    # Attack success by round
    print(f"\n📈 Attack Success Rate by Round (Top 10 attacks)")
    attack_round_data = calculate_attack_success_by_round(run)

    # Get top attacks
    attack_totals = {name: sum(r["calls"] for r in rounds)
                     for name, rounds in attack_round_data.items()}
    top_attacks = sorted(attack_totals.keys(), key=lambda x: attack_totals[x], reverse=True)[:10]

    for attack in top_attacks:
        rounds_data = attack_round_data[attack]
        print(f"\n   {attack}:")
        for r in rounds_data[:5]:  # Show first 5 rounds
            rate_str = f"{r['rate']*100:.1f}%" if r['calls'] > 0 else "N/A"
            print(f"      Round {r['round']}: {r['calls']} calls, {r['successes']} successes ({rate_str})")

    # Round success distribution
    print("\n🔄 Success Distribution by Round")
    round_dist = calculate_round_success_distribution(run)

    print(f"   {'Round':<6} {'Successes':>10} {'% of Total':>12} {'Cumulative':>12}")
    print(f"   {'-'*6} {'-'*10} {'-'*12} {'-'*12}")

    for d in round_dist:
        if d.total_successes > 0 or d.round_idx <= 3:
            print(f"   {d.round_idx:<6} {d.total_successes:>10} {d.percentage:>11.1f}% {d.cumulative_percentage:>11.1f}%")

    # Generate plots if requested
    if generate_plots:
        plots_dir = output_dir or f"runs/{experiment_name}/{run.timestamp}/analysis"
        print(f"\n📊 Generating attack analysis plots to: {plots_dir}")

        plots = generate_attack_analysis_plots(run, plots_dir)
        for name, path in plots.items():
            print(f"   {name}: {path}")


def cmd_case_analysis(
    loader: ExperimentLoader,
    experiment_name: str,
    timestamp: str | None = None,
    sample_id: int | str | None = None,
    limit: int = 10,
):
    """Analyze specific cases."""
    if timestamp:
        run = loader.load_with_attack_events(experiment_name, timestamp)
    else:
        runs = loader.list_runs(experiment_name)
        if not runs:
            print(f"No runs found for experiment: {experiment_name}")
            return
        run = loader.load_with_attack_events(experiment_name, runs[-1])

    if not run:
        print(f"Run not found: {experiment_name}")
        return

    print_separator("=")
    print(f"Case Analysis: {run.experiment_name}")
    if run.timestamp:
        print(f"Timestamp: {run.timestamp}")
    print_separator("=")

    # Get case analysis
    cases = get_case_analysis(run, sample_id=sample_id, limit=limit)

    if not cases:
        print("\nNo cases found.")
        return

    for case in cases:
        print(f"\n📝 Sample ID: {case['sample_id']}")
        print(f"   Goal: {case['goal']}")
        print(f"   Category: {case['main_category']} > {case['subcategory']}")
        print(f"   Success: {'✅ Yes' if case['success'] else '❌ No'}")
        print(f"   Score: {case['aggregate_score']}")
        print(f"   Rounds: {case['rounds']}")
        print(f"   Total Candidates: {case['total_candidates']}")

        if case['attack_sequence']:
            print(f"   Attack Sequence:")
            for i, attack in enumerate(case['attack_sequence']):
                success_marker = "✅" if attack['success'] else "❌"
                print(f"      {i+1}. Round {attack['round']}: {attack['attack']} {success_marker}")
        else:
            print(f"   Attack Sequence: (no detailed events)")


def cmd_round_analysis(
    loader: ExperimentLoader,
    experiment_name: str,
    timestamp: str | None = None,
    generate_plots: bool = False,
    output_dir: str | None = None,
    max_rounds: int = 10,
):
    """Analyze success rate evolution across rounds."""
    if timestamp:
        run = loader.load_with_sample_details(experiment_name, timestamp)
    else:
        run = loader.load_latest_run(experiment_name)

    if not run:
        print(f"Run not found: {experiment_name}" + (f"/{timestamp}" if timestamp else ""))
        return

    print_separator("=")
    print(f"Round Analysis: {run.experiment_name}")
    if run.timestamp:
        print(f"Timestamp: {run.timestamp}")
    print_separator("=")

    # Calculate round success rate metrics
    round_metrics = calculate_round_success_rate(run, max_rounds)

    if not round_metrics:
        print("\nNo round data to analyze.")
        return

    # Print round analysis table
    print(f"\n🔄 Success Rate by Round")
    print(f"   {'Round':<6} {'Samples':>8} {'New Success':>12} {'Round Rate':>12} {'Cumulative':>12}")
    print(f"   {'-'*6} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")

    for m in round_metrics:
        if m.new_successes > 0 or m.round_idx <= 3:
            print(f"   {m.round_idx:<6} {m.total_samples_at_round:>8} {m.new_successes:>12} "
                  f"{m.round_success_rate*100:>11.1f}% {m.cumulative_asr*100:>11.1f}%")

    # Print summary
    total_successes = sum(m.new_successes for m in round_metrics)
    final_asr = round_metrics[-1].cumulative_asr if round_metrics else 0

    print(f"\n📊 Summary")
    print(f"   Total Samples: {len(run.results)}")
    print(f"   Total Successes: {total_successes}")
    print(f"   Final ASR: {final_asr*100:.1f}%")

    # Find most effective rounds
    effective_rounds = sorted(
        [m for m in round_metrics if m.new_successes > 0],
        key=lambda x: x.new_successes,
        reverse=True
    )[:3]

    if effective_rounds:
        print(f"\n🎯 Most Effective Rounds")
        for i, m in enumerate(effective_rounds, 1):
            print(f"   {i}. Round {m.round_idx}: {m.new_successes} successes ({m.round_success_rate*100:.1f}% rate)")

    # Generate plots if requested
    if generate_plots:
        plots_dir = output_dir or f"runs/{experiment_name}/{run.timestamp}/analysis"
        print(f"\n📊 Generating round analysis plots to: {plots_dir}")

        plots = generate_round_analysis_plots(run, plots_dir, max_rounds=max_rounds)
        for name, path in plots.items():
            print(f"   {name}: {path}")


def cmd_compare_runs(
    loader: ExperimentLoader,
    run_specs: list[str],
    output_dir: str | None = None,
):
    """Compare multiple runs.

    run_specs format: experiment_name:timestamp or experiment_name (for latest)
    """
    runs = []

    for spec in run_specs:
        if ":" in spec:
            exp_name, timestamp = spec.split(":", 1)
            run = loader.load_run(exp_name, timestamp)
        else:
            run = loader.load_latest_run(spec)

        if run:
            runs.append(run)
        else:
            print(f"Warning: Could not load run: {spec}")

    if not runs:
        print("No valid runs to compare.")
        return

    print_separator("=")
    print("Run Comparison")
    print_separator("=")

    # Print comparison table
    print(f"\n{'Experiment':<30} {'Timestamp':<18} {'Samples':>8} {'Success Rate':>12}")
    print(f"{'-'*30} {'-'*18} {'-'*8} {'-'*12}")

    for run in runs:
        ts = run.timestamp or "N/A"
        print(f"{run.experiment_name:<30} {ts:<18} {run.total_samples:>8} {run.success_rate*100:>11.1f}%")

    # Generate comparison plot
    if len(runs) > 1:
        plots_dir = Path(output_dir) if output_dir else Path("runs")
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / "comparison.png"

        plot_strategy_comparison(runs, plot_path)
        print(f"\n📊 Comparison plot saved to: {plot_path}")

        # Also generate round comparison plot
        round_plot_path = plots_dir / "round_comparison.png"
        plot_round_success_comparison(runs, round_plot_path)
        print(f"📊 Round comparison plot saved to: {round_plot_path}")


def cmd_export_report(
    loader: ExperimentLoader,
    experiment_name: str,
    timestamp: str | None = None,
    output_path: str | None = None,
):
    """Export analysis report as JSON."""
    if timestamp:
        run = loader.load_with_sample_details(experiment_name, timestamp)
    else:
        run = loader.load_latest_run(experiment_name)

    if not run:
        print(f"Run not found: {experiment_name}")
        return

    report = generate_summary_report(run)

    # Determine output path
    if not output_path:
        ts = run.timestamp or "latest"
        output_path = f"runs/{experiment_name}/{ts}/analysis_report.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Report exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze VLM redteam experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory containing experiment runs (default: runs)",
    )

    # Action flags
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--list",
        action="store_true",
        help="List all experiments",
    )
    action_group.add_argument(
        "--list-runs",
        action="store_true",
        help="List runs for an experiment (requires --experiment)",
    )
    action_group.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze a run (requires --experiment)",
    )
    action_group.add_argument(
        "--attack-analysis",
        action="store_true",
        help="Analyze attack calls and success rates (requires --experiment)",
    )
    action_group.add_argument(
        "--round-analysis",
        action="store_true",
        help="Analyze success rate by round (requires --experiment)",
    )
    action_group.add_argument(
        "--case-analysis",
        action="store_true",
        help="Analyze specific cases (requires --experiment)",
    )
    action_group.add_argument(
        "--compare",
        nargs="+",
        metavar="RUN_SPEC",
        help="Compare multiple runs (format: experiment:timestamp or experiment)",
    )
    action_group.add_argument(
        "--export",
        action="store_true",
        help="Export analysis report as JSON",
    )

    # Parameters
    parser.add_argument(
        "--experiment",
        help="Experiment name to analyze",
    )
    parser.add_argument(
        "--timestamp",
        help="Specific run timestamp (default: latest)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest run for the experiment",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--output",
        help="Output directory for plots/reports",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset JSON file for category enrichment (useful for old format results)",
    )
    parser.add_argument(
        "--sample-id",
        type=int,
        help="Sample ID for case analysis",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of cases to show in case analysis (default: 10)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top attacks to show in attack analysis (default: 15)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum number of rounds to analyze in round analysis (default: 10)",
    )

    args = parser.parse_args()

    loader = ExperimentLoader(args.runs_dir)

    # Load dataset for category enrichment if provided
    if args.dataset:
        try:
            loader.load_dataset(args.dataset)
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    # Handle actions
    if args.list:
        cmd_list_experiments(loader)
    elif args.list_runs:
        if not args.experiment:
            print("Error: --experiment is required with --list-runs")
            sys.exit(1)
        cmd_list_runs(loader, args.experiment)
    elif args.compare:
        cmd_compare_runs(loader, args.compare, args.output)
    elif args.export:
        if not args.experiment:
            print("Error: --experiment is required with --export")
            sys.exit(1)
        cmd_export_report(loader, args.experiment, args.timestamp, args.output)
    elif args.attack_analysis:
        if not args.experiment:
            print("Error: --experiment is required with --attack-analysis")
            sys.exit(1)
        cmd_attack_analysis(
            loader,
            args.experiment,
            args.timestamp,
            args.plots,
            args.output,
            args.top_n,
        )
    elif args.round_analysis:
        if not args.experiment:
            print("Error: --experiment is required with --round-analysis")
            sys.exit(1)
        cmd_round_analysis(
            loader,
            args.experiment,
            args.timestamp,
            args.plots,
            args.output,
            args.max_rounds,
        )
    elif args.case_analysis:
        if not args.experiment:
            print("Error: --experiment is required with --case-analysis")
            sys.exit(1)
        cmd_case_analysis(
            loader,
            args.experiment,
            args.timestamp,
            args.sample_id,
            args.limit,
        )
    else:
        # Default: analyze a run
        if not args.experiment:
            print("Error: --experiment is required for analysis")
            print("\nUsage examples:")
            print("  # List all experiments")
            print("  python scripts/analyze_results.py --list")
            print()
            print("  # Analyze latest run")
            print("  python scripts/analyze_results.py --experiment my-experiment --latest")
            print()
            print("  # Analyze specific run with plots")
            print("  python scripts/analyze_results.py --experiment my-experiment --timestamp 20260311_143052 --plots")
            print()
            print("  # Round analysis with plots")
            print("  python scripts/analyze_results.py --experiment my-experiment --round-analysis --plots")
            print()
            print("  # Attack analysis")
            print("  python scripts/analyze_results.py --experiment my-experiment --attack-analysis --plots")
            print()
            print("  # Case analysis")
            print("  python scripts/analyze_results.py --experiment my-experiment --case-analysis")
            print()
            print("  # Case analysis for specific sample")
            print("  python scripts/analyze_results.py --experiment my-experiment --case-analysis --sample-id 10")
            sys.exit(1)

        cmd_analyze_run(
            loader,
            args.experiment,
            args.timestamp,
            args.plots,
            args.output,
        )


if __name__ == "__main__":
    main()
