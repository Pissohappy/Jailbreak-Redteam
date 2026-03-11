"""Metrics calculation for experiment results."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from vlm_redteam.analysis.loader import ExperimentRun, SampleResult, AttackEvent


@dataclass
class CategoryMetrics:
    """Metrics for a category."""

    category: str
    total: int
    successful: int
    success_rate: float
    avg_score: float | None


@dataclass
class StrategyMetrics:
    """Metrics for an attack strategy."""

    strategy: str
    total: int
    successful: int
    success_rate: float
    avg_score: float | None


@dataclass
class RoundDistribution:
    """Distribution of rounds."""

    rounds: int
    count: int
    successful: int


@dataclass
class CumulativeASR:
    """Cumulative ASR at each round."""

    round_idx: int
    cumulative_success: int
    cumulative_asr: float
    new_success_at_round: int  # Number of new successes at this round


def calculate_cumulative_asr_by_round(run: ExperimentRun, max_rounds: int = 10) -> list[CumulativeASR]:
    """Calculate cumulative ASR at each round.

    This is useful for understanding how many samples succeed at each round,
    and the cumulative success rate as rounds increase.

    Args:
        run: Experiment run to analyze
        max_rounds: Maximum number of rounds to consider

    Returns:
        List of CumulativeASR for each round
    """
    # Track which samples succeeded at which round
    # We need to determine the first round where each sample succeeded
    success_by_round: dict[int, set] = defaultdict(set)

    for result in run.results:
        if not result.success:
            continue

        # Determine the round of success from round_topk_scores or best_branch_path
        success_round = None
        if result.round_topk_scores:
            # The success happened at the last round
            success_round = len(result.round_topk_scores)
        elif result.best_branch_path:
            success_round = len(result.best_branch_path)

        if success_round is not None:
            success_by_round[success_round].add(result.sample_id)

    # Calculate cumulative ASR
    total_samples = len(run.results)
    cumulative_success = 0
    results = []

    for round_idx in range(1, max_rounds + 1):
        new_success = len(success_by_round.get(round_idx, set()))
        cumulative_success += new_success

        results.append(CumulativeASR(
            round_idx=round_idx,
            cumulative_success=cumulative_success,
            cumulative_asr=cumulative_success / total_samples if total_samples > 0 else 0.0,
            new_success_at_round=new_success,
        ))

    return results


def calculate_success_rate(run: ExperimentRun) -> dict[str, Any]:
    """Calculate overall success rate for an experiment run."""
    if not run.results:
        return {
            "total_samples": 0,
            "successful_samples": 0,
            "success_rate": 0.0,
        }

    total = len(run.results)
    successful = sum(1 for r in run.results if r.success)
    scores = [r.aggregate_score for r in run.results if r.aggregate_score is not None]

    return {
        "total_samples": total,
        "successful_samples": successful,
        "success_rate": successful / total if total > 0 else 0.0,
        "avg_score": sum(scores) / len(scores) if scores else None,
        "errors": sum(1 for r in run.results if r.error),
    }


def calculate_category_metrics(
    run: ExperimentRun,
    group_by: str = "main_category"
) -> list[CategoryMetrics]:
    """Calculate success rates grouped by category.

    Args:
        run: Experiment run to analyze
        group_by: Either "main_category" or "subcategory"

    Returns:
        List of CategoryMetrics for each category
    """
    category_data: dict[str, list[SampleResult]] = defaultdict(list)

    for result in run.results:
        key = getattr(result, group_by, "") or "Unknown"
        category_data[key].append(result)

    metrics = []
    for category, results in sorted(category_data.items()):
        total = len(results)
        successful = sum(1 for r in results if r.success)
        scores = [r.aggregate_score for r in results if r.aggregate_score is not None]

        metrics.append(CategoryMetrics(
            category=category,
            total=total,
            successful=successful,
            success_rate=successful / total if total > 0 else 0.0,
            avg_score=sum(scores) / len(scores) if scores else None,
        ))

    return metrics


def calculate_strategy_metrics(run: ExperimentRun) -> list[StrategyMetrics]:
    """Calculate success rates by attack strategy.

    Note: Strategy information should be extracted from run_id or stored in results.
    For now, this returns a placeholder that can be extended.
    """
    # Group by strategy if available in results
    # This is a placeholder - actual implementation depends on how strategy info is stored
    strategy_data: dict[str, list[SampleResult]] = defaultdict(list)

    for result in run.results:
        # Try to extract strategy from run_id or use "default"
        # In future, this could be stored in the result data
        strategy = "combined"  # Default strategy name
        strategy_data[strategy].append(result)

    metrics = []
    for strategy, results in sorted(strategy_data.items()):
        total = len(results)
        successful = sum(1 for r in results if r.success)
        scores = [r.aggregate_score for r in results if r.aggregate_score is not None]

        metrics.append(StrategyMetrics(
            strategy=strategy,
            total=total,
            successful=successful,
            success_rate=successful / total if total > 0 else 0.0,
            avg_score=sum(scores) / len(scores) if scores else None,
        ))

    return metrics


def calculate_round_distribution(run: ExperimentRun) -> list[RoundDistribution]:
    """Calculate distribution of rounds for successful vs failed samples."""
    round_data: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "successful": 0})

    for result in run.results:
        # Extract round count from round_topk_scores or best_branch_path
        rounds = 0
        if result.round_topk_scores:
            rounds = len(result.round_topk_scores)
        elif result.best_branch_path:
            rounds = len(result.best_branch_path)

        round_data[rounds]["total"] += 1
        if result.success:
            round_data[rounds]["successful"] += 1

    distribution = []
    for rounds in sorted(round_data.keys()):
        data = round_data[rounds]
        distribution.append(RoundDistribution(
            rounds=rounds,
            count=data["total"],
            successful=data["successful"],
        ))

    return distribution


def calculate_score_distribution(
    run: ExperimentRun,
    bins: int = 10
) -> dict[str, Any]:
    """Calculate distribution of aggregate scores.

    Args:
        run: Experiment run to analyze
        bins: Number of bins for histogram

    Returns:
        Dictionary with bin edges, counts, and statistics
    """
    scores = [r.aggregate_score for r in run.results if r.aggregate_score is not None]

    if not scores:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "histogram": None,
        }

    import statistics

    # Basic statistics
    min_score = min(scores)
    max_score = max(scores)
    mean_score = statistics.mean(scores)
    median_score = statistics.median(scores)
    std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0

    # Build histogram
    if max_score > min_score:
        bin_width = (max_score - min_score) / bins
        bin_edges = [min_score + i * bin_width for i in range(bins + 1)]
        bin_counts = [0] * bins

        for score in scores:
            # Find the appropriate bin
            bin_idx = min(int((score - min_score) / bin_width), bins - 1)
            bin_counts[bin_idx] += 1
    else:
        # All scores are the same
        bin_edges = [min_score, max_score + 0.001]
        bin_counts = [len(scores)]

    return {
        "min": min_score,
        "max": max_score,
        "mean": mean_score,
        "median": median_score,
        "std": std_score,
        "histogram": {
            "bin_edges": bin_edges,
            "counts": bin_counts,
        },
    }


def generate_summary_report(run: ExperimentRun, include_round_analysis: bool = True) -> dict[str, Any]:
    """Generate a comprehensive summary report for an experiment run."""
    report = {
        "experiment_name": run.experiment_name,
        "timestamp": run.timestamp,
        "overall_metrics": calculate_success_rate(run),
        "main_category_metrics": [
            {"category": m.category, "total": m.total, "successful": m.successful, "success_rate": m.success_rate}
            for m in calculate_category_metrics(run, "main_category")
        ],
        "subcategory_metrics": [
            {"category": m.category, "total": m.total, "successful": m.successful, "success_rate": m.success_rate}
            for m in calculate_category_metrics(run, "subcategory")
        ],
        "score_distribution": calculate_score_distribution(run),
    }

    if include_round_analysis:
        report["cumulative_asr_by_round"] = [
            {
                "round": c.round_idx,
                "cumulative_success": c.cumulative_success,
                "cumulative_asr": c.cumulative_asr,
                "new_success_at_round": c.new_success_at_round,
            }
            for c in calculate_cumulative_asr_by_round(run)
        ]

    return report


# ============ Attack Analysis Metrics ============

@dataclass
class AttackMetrics:
    """Metrics for a specific attack type."""

    attack_name: str
    short_name: str  # Simplified attack name
    total_calls: int  # Total number of times this attack was called
    successful_calls: int  # Number of calls that led to success
    success_rate: float  # successful_calls / total_calls
    calls_by_round: dict[int, int]  # Round -> number of calls
    success_by_round: dict[int, int]  # Round -> number of successful calls


@dataclass
class RoundSuccessDistribution:
    """Distribution of successful attacks by round."""

    round_idx: int
    total_successes: int  # Total successful attacks at this round
    percentage: float  # Percentage of all successes at this round
    cumulative_percentage: float  # Cumulative percentage up to this round


def get_short_attack_name(attack_name: str) -> str:
    """Extract a short name from the full attack name."""
    # e.g., "attacks_strategy.email.attack:EmailThreadAttack" -> "EmailThread"
    if ":" in attack_name:
        name = attack_name.split(":")[-1]
    else:
        name = attack_name.split(".")[-1]

    # Remove "Attack" suffix if present
    if name.endswith("Attack"):
        name = name[:-6]

    return name


def calculate_attack_metrics(run: ExperimentRun) -> list[AttackMetrics]:
    """Calculate metrics for each attack type used in the experiment.

    Args:
        run: Experiment run with attack events loaded

    Returns:
        List of AttackMetrics for each attack type
    """
    # Aggregate data by attack name
    attack_data: dict[str, dict] = defaultdict(lambda: {
        "total_calls": 0,
        "successful_calls": 0,
        "calls_by_round": defaultdict(int),
        "success_by_round": defaultdict(int),
    })

    for result in run.results:
        if not result.attack_events:
            continue

        for event in result.attack_events:
            attack_name = event.attack_name
            round_idx = event.round_idx if event.round_idx is not None else 0

            attack_data[attack_name]["total_calls"] += 1
            attack_data[attack_name]["calls_by_round"][round_idx] += 1

            if event.success:
                attack_data[attack_name]["successful_calls"] += 1
                attack_data[attack_name]["success_by_round"][round_idx] += 1

    metrics = []
    for attack_name, data in sorted(attack_data.items()):
        total = data["total_calls"]
        successful = data["successful_calls"]

        metrics.append(AttackMetrics(
            attack_name=attack_name,
            short_name=get_short_attack_name(attack_name),
            total_calls=total,
            successful_calls=successful,
            success_rate=successful / total if total > 0 else 0.0,
            calls_by_round=dict(data["calls_by_round"]),
            success_by_round=dict(data["success_by_round"]),
        ))

    # Sort by total calls descending
    metrics.sort(key=lambda m: m.total_calls, reverse=True)

    return metrics


def calculate_attack_success_by_round(run: ExperimentRun, max_rounds: int = 10) -> dict[str, list[dict]]:
    """Calculate attack success rates by round for each attack type.

    Args:
        run: Experiment run with attack events loaded
        max_rounds: Maximum number of rounds to analyze

    Returns:
        Dictionary mapping attack name to list of {round, calls, successes, rate}
    """
    # Aggregate data by attack name and round
    attack_round_data: dict[str, dict[int, dict]] = defaultdict(lambda: defaultdict(lambda: {"calls": 0, "successes": 0}))

    for result in run.results:
        if not result.attack_events:
            continue

        for event in result.attack_events:
            attack_name = event.attack_name
            round_idx = event.round_idx if event.round_idx is not None else 0

            if round_idx > max_rounds:
                continue

            attack_round_data[attack_name][round_idx]["calls"] += 1
            if event.success:
                attack_round_data[attack_name][round_idx]["successes"] += 1

    # Convert to output format
    result = {}
    for attack_name, round_data in attack_round_data.items():
        short_name = get_short_attack_name(attack_name)
        rounds_list = []
        for round_idx in sorted(round_data.keys()):
            data = round_data[round_idx]
            rounds_list.append({
                "round": round_idx,
                "calls": data["calls"],
                "successes": data["successes"],
                "rate": data["successes"] / data["calls"] if data["calls"] > 0 else 0.0,
            })
        result[short_name] = rounds_list

    return result


def calculate_round_success_distribution(run: ExperimentRun, max_rounds: int = 10) -> list[RoundSuccessDistribution]:
    """Calculate distribution of successful attacks by round.

    This shows what percentage of all successful attacks happened at each round.

    Args:
        run: Experiment run to analyze
        max_rounds: Maximum number of rounds to consider

    Returns:
        List of RoundSuccessDistribution for each round
    """
    success_by_round: dict[int, int] = defaultdict(int)
    total_successes = 0

    for result in run.results:
        if not result.success:
            continue

        # Determine the round of success
        success_round = None
        if result.round_topk_scores:
            success_round = len(result.round_topk_scores)
        elif result.best_branch_path:
            success_round = len(result.best_branch_path)

        if success_round is not None and success_round <= max_rounds:
            success_by_round[success_round] += 1
            total_successes += 1

    distribution = []
    cumulative = 0
    for round_idx in range(1, max_rounds + 1):
        count = success_by_round.get(round_idx, 0)
        percentage = (count / total_successes * 100) if total_successes > 0 else 0.0
        cumulative += percentage

        distribution.append(RoundSuccessDistribution(
            round_idx=round_idx,
            total_successes=count,
            percentage=percentage,
            cumulative_percentage=cumulative,
        ))

    return distribution


def get_case_analysis(run: ExperimentRun, sample_id: int | str | None = None, limit: int = 10) -> list[dict]:
    """Get detailed analysis for specific cases or top/bottom cases.

    Args:
        run: Experiment run to analyze
        sample_id: Specific sample ID to analyze (optional)
        limit: Number of cases to return if no specific sample_id

    Returns:
        List of case analysis dictionaries
    """
    cases = []

    # Filter results
    if sample_id is not None:
        results = [r for r in run.results if r.sample_id == sample_id]
    else:
        # Get top successful and top failed cases
        successful = sorted(
            [r for r in run.results if r.success],
            key=lambda r: r.aggregate_score or 0,
            reverse=True
        )[:limit // 2]
        failed = sorted(
            [r for r in run.results if not r.success],
            key=lambda r: r.aggregate_score or 0
        )[:limit // 2]
        results = successful + failed

    for result in results:
        case_info = {
            "sample_id": result.sample_id,
            "goal": result.goal[:100] + "..." if len(result.goal) > 100 else result.goal,
            "main_category": result.main_category,
            "subcategory": result.subcategory,
            "success": result.success,
            "aggregate_score": result.aggregate_score,
            "total_candidates": result.total_candidates,
            "rounds": len(result.round_topk_scores) if result.round_topk_scores else (
                len(result.best_branch_path) if result.best_branch_path else 0
            ),
            "attack_sequence": [],
        }

        # Add attack sequence if available
        if result.attack_events:
            for i, event in enumerate(result.attack_events):
                case_info["attack_sequence"].append({
                    "round": event.round_idx,
                    "attack": get_short_attack_name(event.attack_name),
                    "success": event.success,
                })

        cases.append(case_info)

    return cases
