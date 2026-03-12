"""Data analysis tools for VLM redteam experiment results."""

from vlm_redteam.analysis.loader import ExperimentLoader
from vlm_redteam.analysis.metrics import (
    calculate_success_rate,
    calculate_category_metrics,
    calculate_strategy_metrics,
    calculate_round_distribution,
    calculate_score_distribution,
    calculate_cumulative_asr_by_round,
    calculate_round_success_rate,
    RoundSuccessRate,
)
from vlm_redteam.analysis.visualization import (
    plot_category_success_rates,
    plot_strategy_comparison,
    plot_score_distribution,
    plot_round_distribution,
    plot_round_success_line,
    plot_round_success_comparison,
    generate_round_analysis_plots,
)

__all__ = [
    "ExperimentLoader",
    "calculate_success_rate",
    "calculate_category_metrics",
    "calculate_strategy_metrics",
    "calculate_round_distribution",
    "calculate_score_distribution",
    "calculate_cumulative_asr_by_round",
    "calculate_round_success_rate",
    "RoundSuccessRate",
    "plot_category_success_rates",
    "plot_strategy_comparison",
    "plot_score_distribution",
    "plot_round_distribution",
    "plot_round_success_line",
    "plot_round_success_comparison",
    "generate_round_analysis_plots",
]
