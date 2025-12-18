# src/evaluation/__init__.py
"""
Evaluation module
"""

from .metrics import (
    compute_metrics,
    compute_confusion_matrix,
    compute_eer,
    evaluate_model_comprehensive
)

from .visualizzation import (
    plot_training_history,
    plot_multiple_models_comparison,
    plot_confusion_matrix,
    plot_results_summary,
    create_training_report
)

__all__ = [
    'compute_metrics',
    'compute_confusion_matrix',
    'compute_eer',
    'evaluate_model_comprehensive',
    'plot_training_history',
    'plot_multiple_models_comparison',
    'plot_confusion_matrix',
    'plot_results_summary',
    'create_training_report',
]
