"""Visualization tools for experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_redteam.analysis.loader import ExperimentRun
from vlm_redteam.analysis.metrics import (
    CategoryMetrics,
    RoundDistribution,
    calculate_category_metrics,
    calculate_score_distribution,
    calculate_round_distribution,
    calculate_attack_metrics,
    calculate_attack_success_by_round,
    calculate_round_success_distribution,
    AttackMetrics,
)


def _ensure_matplotlib():
    """Ensure matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def plot_category_success_rates(
    run: ExperimentRun,
    group_by: str = "main_category",
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
    title: str | None = None,
) -> Any:
    """Plot success rates by category.

    Args:
        run: Experiment run to visualize
        group_by: Either "main_category" or "subcategory"
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        title: Custom title for the plot

    Returns:
        matplotlib Figure object
    """
    plt = _ensure_matplotlib()

    metrics = calculate_category_metrics(run, group_by)
    if not metrics:
        print("No category data to plot")
        return None

    # Sort by success rate
    metrics = sorted(metrics, key=lambda m: m.success_rate, reverse=True)

    categories = [m.category for m in metrics]
    success_rates = [m.success_rate * 100 for m in metrics]
    totals = [m.total for m in metrics]

    fig, ax = plt.subplots(figsize=figsize)

    # Create bars with color based on success rate
    colors = []
    for rate in success_rates:
        if rate >= 50:
            colors.append("#e74c3c")  # Red for high attack success
        elif rate >= 25:
            colors.append("#f39c12")  # Orange for medium
        else:
            colors.append("#27ae60")  # Green for low

    bars = ax.bar(range(len(categories)), success_rates, color=colors)

    # Add count labels on bars
    for i, (bar, total) in enumerate(zip(bars, totals)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{int(height)}% ({total})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Category" if group_by == "main_category" else "Subcategory")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title(
        title or f"Attack Success Rate by {group_by.replace('_', ' ').title()}"
    )
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

    return fig


def plot_strategy_comparison(
    runs: list[ExperimentRun],
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Strategy Comparison",
) -> Any:
    """Compare success rates across multiple experiment runs.

    Args:
        runs: List of experiment runs to compare
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        title: Custom title for the plot

    Returns:
        matplotlib Figure object
    """
    plt = _ensure_matplotlib()

    if not runs:
        print("No runs to compare")
        return None

    labels = []
    success_rates = []
    totals = []

    for run in runs:
        label = f"{run.experiment_name}"
        if run.timestamp:
            label += f"\n({run.timestamp})"

        labels.append(label)
        success_rates.append(run.success_rate * 100)
        totals.append(run.total_samples)

    fig, ax = plt.subplots(figsize=figsize)

    x = range(len(labels))
    bars = ax.bar(x, success_rates, color="#3498db")

    # Add labels
    for i, (bar, total) in enumerate(zip(bars, totals)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{int(height)}% ({total})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Experiment Run")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

    return fig


def plot_score_distribution(
    run: ExperimentRun,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
    bins: int = 10,
    title: str | None = None,
) -> Any:
    """Plot distribution of aggregate scores.

    Args:
        run: Experiment run to visualize
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        bins: Number of bins for histogram
        title: Custom title for the plot

    Returns:
        matplotlib Figure object
    """
    plt = _ensure_matplotlib()

    dist = calculate_score_distribution(run, bins=bins)

    if dist["histogram"] is None:
        print("No score data to plot")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    bin_edges = dist["histogram"]["bin_edges"]
    counts = dist["histogram"]["counts"]

    # Create bar plot
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    bin_widths = [bin_edges[i + 1] - bin_edges[i] for i in range(len(bin_edges) - 1)]

    ax.bar(bin_centers, counts, width=bin_widths, color="#9b59b6", alpha=0.7, edgecolor="black")

    ax.set_xlabel("Aggregate Score")
    ax.set_ylabel("Count")
    ax.set_title(title or "Distribution of Aggregate Scores")

    # Add statistics text
    stats_text = f"Mean: {dist['mean']:.2f}\nMedian: {dist['median']:.2f}\nStd: {dist['std']:.2f}"
    ax.text(
        0.95, 0.95, stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

    return fig


def plot_round_distribution(
    run: ExperimentRun,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Round Distribution (Success vs Failure)",
) -> Any:
    """Plot distribution of rounds for successful vs failed attacks.

    Args:
        run: Experiment run to visualize
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        title: Custom title for the plot

    Returns:
        matplotlib Figure object
    """
    plt = _ensure_matplotlib()

    distribution = calculate_round_distribution(run)

    if not distribution:
        print("No round data to plot")
        return None

    rounds = [d.rounds for d in distribution]
    success_counts = [d.successful for d in distribution]
    fail_counts = [d.count - d.successful for d in distribution]

    fig, ax = plt.subplots(figsize=figsize)

    x = range(len(rounds))
    width = 0.35

    bars1 = ax.bar([i - width / 2 for i in x], success_counts, width, label="Successful", color="#e74c3c")
    bars2 = ax.bar([i + width / 2 for i in x], fail_counts, width, label="Failed", color="#27ae60")

    ax.set_xlabel("Number of Rounds")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(rounds)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

    return fig


def generate_all_plots(
    run: ExperimentRun,
    output_dir: str | Path,
    prefix: str = "",
) -> dict[str, Any]:
    """Generate all visualization plots for an experiment run.

    Args:
        run: Experiment run to visualize
        output_dir: Directory to save plots
        prefix: Prefix for output filenames

    Returns:
        Dictionary with paths to generated plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{prefix}_" if prefix else ""

    plots = {}

    # Main category success rates
    path = output_dir / f"{prefix}main_category_success.png"
    fig = plot_category_success_rates(run, "main_category", path)
    if fig:
        plots["main_category"] = str(path)

    # Subcategory success rates
    path = output_dir / f"{prefix}subcategory_success.png"
    fig = plot_category_success_rates(run, "subcategory", path)
    if fig:
        plots["subcategory"] = str(path)

    # Score distribution
    path = output_dir / f"{prefix}score_distribution.png"
    fig = plot_score_distribution(run, path)
    if fig:
        plots["score_distribution"] = str(path)

    # Round distribution
    path = output_dir / f"{prefix}round_distribution.png"
    fig = plot_round_distribution(run, path)
    if fig:
        plots["round_distribution"] = str(path)

    return plots


# ============ Attack Analysis Visualization ============

def plot_attack_call_distribution(
    run: ExperimentRun,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 8),
    top_n: int = 15,
    title: str | None = None,
) -> Any:
    """Plot the distribution of attack calls by attack type.

    Shows total calls and success rate for each attack type.

    Args:
        run: Experiment run with attack events loaded
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        top_n: Number of top attacks to show
        title: Custom title for the plot

    Returns:
        matplotlib Figure object
    """
    plt = _ensure_matplotlib()

    metrics = calculate_attack_metrics(run)
    if not metrics:
        print("No attack event data to plot. Make sure attack events are loaded.")
        return None

    # Take top N by total calls
    metrics = metrics[:top_n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Total calls by attack type
    names = [m.short_name for m in metrics]
    calls = [m.total_calls for m in metrics]
    successes = [m.successful_calls for m in metrics]

    x = range(len(names))
    bars1 = ax1.bar(x, calls, color="#3498db", label="Total Calls")
    bars2 = ax1.bar(x, successes, color="#e74c3c", label="Successful")

    ax1.set_xlabel("Attack Type")
    ax1.set_ylabel("Number of Calls")
    ax1.set_title("Attack Calls by Type")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, height,
                    f"{int(height)}", ha="center", va="bottom", fontsize=8)

    # Right plot: Success rate by attack type
    success_rates = [m.success_rate * 100 for m in metrics]

    colors = ["#e74c3c" if r >= 30 else "#f39c12" if r >= 10 else "#27ae60" for r in success_rates]
    bars = ax2.bar(x, success_rates, color=colors)

    ax2.set_xlabel("Attack Type")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title("Attack Success Rate by Type")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_ylim(0, max(success_rates) * 1.2 if success_rates else 100)
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, height,
                    f"{rate:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.suptitle(title or "Attack Type Analysis", fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

    return fig


def plot_attack_calls_by_round(
    run: ExperimentRun,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 6),
    top_n: int = 10,
    title: str | None = None,
) -> Any:
    """Plot attack calls by round for each attack type.

    Shows a heatmap or grouped bar chart of calls per round per attack.

    Args:
        run: Experiment run with attack events loaded
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        top_n: Number of top attacks to show
        title: Custom title for the plot

    Returns:
        matplotlib Figure object
    """
    plt = _ensure_matplotlib()

    attack_round_data = calculate_attack_success_by_round(run)
    if not attack_round_data:
        print("No attack round data to plot")
        return None

    # Get top attacks by total calls
    attack_totals = {name: sum(r["calls"] for r in rounds)
                     for name, rounds in attack_round_data.items()}
    top_attacks = sorted(attack_totals.keys(), key=lambda x: attack_totals[x], reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for stacked bar chart
    all_rounds = set()
    for attack in top_attacks:
        for r in attack_round_data[attack]:
            all_rounds.add(r["round"])
    all_rounds = sorted(all_rounds)

    # Create grouped bar chart
    x = range(len(top_attacks))
    width = 0.8 / len(all_rounds) if all_rounds else 0.8

    colors = plt.cm.viridis([i / max(len(all_rounds), 1) for i in range(len(all_rounds))])

    for i, round_idx in enumerate(all_rounds):
        values = []
        for attack in top_attacks:
            round_data = next((r for r in attack_round_data[attack] if r["round"] == round_idx), None)
            values.append(round_data["calls"] if round_data else 0)

        bars = ax.bar([j + i * width for j in x], values, width,
                     label=f"Round {round_idx}", color=colors[i])

    ax.set_xlabel("Attack Type")
    ax.set_ylabel("Number of Calls")
    ax.set_title(title or "Attack Calls by Round")
    ax.set_xticks([j + width * len(all_rounds) / 2 for j in x])
    ax.set_xticklabels(top_attacks, rotation=45, ha="right")
    ax.legend(title="Round", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

    return fig


def plot_success_rate_by_round(
    run: ExperimentRun,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
    top_n: int = 10,
    title: str | None = None,
) -> Any:
    """Plot success rate by round for top attack types.

    Args:
        run: Experiment run with attack events loaded
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        top_n: Number of top attacks to show
        title: Custom title for the plot

    Returns:
        matplotlib Figure object
    """
    plt = _ensure_matplotlib()

    attack_round_data = calculate_attack_success_by_round(run)
    if not attack_round_data:
        print("No attack round data to plot")
        return None

    # Get top attacks by total calls
    attack_totals = {name: sum(r["calls"] for r in rounds)
                     for name, rounds in attack_round_data.items()}
    top_attacks = sorted(attack_totals.keys(), key=lambda x: attack_totals[x], reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot line for each attack
    colors = plt.cm.tab10(range(len(top_attacks)))

    for i, attack in enumerate(top_attacks):
        rounds_data = attack_round_data[attack]
        rounds = [r["round"] for r in rounds_data]
        rates = [r["rate"] * 100 for r in rounds_data]

        ax.plot(rounds, rates, marker="o", label=attack, color=colors[i], linewidth=2)

    ax.set_xlabel("Round")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title(title or "Attack Success Rate by Round")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

    return fig


def plot_round_success_distribution(
    run: ExperimentRun,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
    title: str | None = None,
) -> Any:
    """Plot distribution of successful attacks by round.

    Shows what percentage of all successes happened at each round.

    Args:
        run: Experiment run to analyze
        output_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        title: Custom title for the plot

    Returns:
        matplotlib Figure object
    """
    plt = _ensure_matplotlib()

    distribution = calculate_round_success_distribution(run)
    if not distribution or all(d.total_successes == 0 for d in distribution):
        print("No success distribution data to plot")
        return None

    rounds = [d.round_idx for d in distribution]
    percentages = [d.percentage for d in distribution]
    cumulative = [d.cumulative_percentage for d in distribution]

    fig, ax1 = plt.subplots(figsize=figsize)

    # Bar chart for percentage at each round
    bars = ax1.bar(rounds, percentages, color="#3498db", alpha=0.7, label="% at Round")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Percentage of Successes (%)", color="#3498db")
    ax1.tick_params(axis="y", labelcolor="#3498db")

    # Add value labels on bars
    for bar, pct in zip(bars, percentages):
        if pct > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    # Line chart for cumulative percentage
    ax2 = ax1.twinx()
    ax2.plot(rounds, cumulative, color="#e74c3c", marker="s", linewidth=2,
            label="Cumulative %")
    ax2.set_ylabel("Cumulative Percentage (%)", color="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")
    ax2.set_ylim(0, 105)

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title(title or "Distribution of Successful Attacks by Round")
    ax1.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

    return fig


def generate_attack_analysis_plots(
    run: ExperimentRun,
    output_dir: str | Path,
    prefix: str = "",
) -> dict[str, Any]:
    """Generate all attack analysis plots for an experiment run.

    Args:
        run: Experiment run with attack events loaded
        output_dir: Directory to save plots
        prefix: Prefix for output filenames

    Returns:
        Dictionary with paths to generated plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{prefix}_" if prefix else ""

    plots = {}

    # Attack call distribution
    path = output_dir / f"{prefix}attack_call_distribution.png"
    fig = plot_attack_call_distribution(run, path)
    if fig:
        plots["attack_call_distribution"] = str(path)

    # Attack calls by round
    path = output_dir / f"{prefix}attack_calls_by_round.png"
    fig = plot_attack_calls_by_round(run, path)
    if fig:
        plots["attack_calls_by_round"] = str(path)

    # Success rate by round
    path = output_dir / f"{prefix}success_rate_by_round.png"
    fig = plot_success_rate_by_round(run, path)
    if fig:
        plots["success_rate_by_round"] = str(path)

    # Round success distribution
    path = output_dir / f"{prefix}round_success_distribution.png"
    fig = plot_round_success_distribution(run, path)
    if fig:
        plots["round_success_distribution"] = str(path)

    return plots
