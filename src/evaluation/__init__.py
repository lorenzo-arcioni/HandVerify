"""
Evaluation module
Unified metrics and visualization for model evaluation.
"""

from .metrics import (
    # Classification metrics
    compute_basic_metrics,
    compute_confusion_matrix,
    
    # Biometric metrics
    compute_eer,
    compute_biometric_metrics,
    compute_rank1_identification,
    
    # Comprehensive evaluation
    evaluate_comprehensive,
)

from .visualization import (
    # Training visualizations
    plot_training_history,
    plot_triplet_training_history,
    plot_multiple_models_comparison,
    
    # Biometric visualizations
    plot_biometric_results,
    plot_confusion_matrix,
    
    # Reports
    create_training_report,
)

__all__ = [
    # Metrics
    'compute_basic_metrics',
    'compute_confusion_matrix',
    'compute_eer',
    'compute_biometric_metrics',
    'compute_rank1_identification',
    'evaluate_comprehensive',
    
    # Visualizations
    'plot_training_history',
    'plot_triplet_training_history',
    'plot_multiple_models_comparison',
    'plot_biometric_results',
    'plot_confusion_matrix',
    'create_training_report',
]
