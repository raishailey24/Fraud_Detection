"""
Dashboard components package.
"""
from .kpi_cards import display_kpi_cards
from .visualizations import (
    plot_fraud_over_time,
    plot_amount_distribution,
    plot_fraud_by_category,
    plot_hourly_patterns,
    plot_risk_score_distribution,
    plot_merchant_analysis,
    plot_confusion_matrix
)
from .filters import apply_filters
from .ai_panel import display_ai_copilot, display_metrics_summary

__all__ = [
    "display_kpi_cards",
    "plot_fraud_over_time",
    "plot_amount_distribution",
    "plot_fraud_by_category",
    "plot_hourly_patterns",
    "plot_risk_score_distribution",
    "plot_merchant_analysis",
    "plot_confusion_matrix",
    "apply_filters",
    "display_ai_copilot",
    "display_metrics_summary"
]
