"""
Evaluation module
Unified metrics and visualization for model evaluation.
"""

from .metrics import (
    # Verification & classification metrics
    compute_metrics,
    print_results,
)

__all__ = [
    # Metrics
    'compute_metrics',
    'print_results',
]