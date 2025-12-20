"""
Evaluation module
Unified metrics and visualization for model evaluation.
"""

from .metrics import (
    # Verification & classification metrics
    compute_eer,
    compute_classification_metrics,
    compute_verification_metrics,
    print_verification_results,
)

#from .visualization import (
#    # Training visualizations
#    plot_training_history,
#    plot_triplet_training_history,
#    plot_multiple_models_comparison,
#    
#    # Biometric visualizations
#    plot_biometric_results,
#    plot_confusion_matrix,
#    
#    # Reports
#    create_training_report,
#)

__all__ = [
    # Metrics
    'compute_eer',
    'compute_classification_metrics',
    'compute_verification_metrics',
    'print_verification_results',
    
    # Visualizations
    'plot_training_history',
    'plot_triplet_training_history',
    'plot_multiple_models_comparison',
    'plot_biometric_results',
    'plot_confusion_matrix',
    'create_training_report',
]