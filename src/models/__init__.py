# src/models/__init__.py
"""
Models module
Unified access to all model architectures (BCE, Contrastive, Triplet)
"""

from .base import BaseSiameseNetwork, BaseTripletNetwork, BaseContrastiveNetwork

from .registry import (
    get_model,
    list_models,
    get_model_info,
    MODEL_REGISTRY,
    BCE_MODELS,
    CONTRASTIVE_MODELS,
    TRIPLET_MODELS,
)

from .bce_backbones import (
    SiameseResNet18,
    SiameseResNet34,
    SiameseResNet50,
    SiameseEfficientNetB0,
    SiameseEfficientNetB1,
    SiameseEfficientNetV2,
    SiameseMobileNetV3Small,
    SiameseMobileNetV3Large,
    SiameseDenseNet121,
    SiameseRegNetY400MF,
)

from .triplet_backbones import (
    TripletMobileNetV3Small,
    TripletMobileNetV3Large,
    TripletResNet18,
    TripletResNet34,
    TripletEfficientNetB0,
)

from .contrastive_backbones import (
    ContrastiveMobileNetV3Small,
    ContrastiveMobileNetV3Large,
    ContrastiveResNet18,
    ContrastiveResNet34,
    ContrastiveResNet50,
    ContrastiveEfficientNetB0,
    ContrastiveDenseNet121,
)

__all__ = [
    # ========================================================================
    # Base classes
    # ========================================================================
    'BaseSiameseNetwork',
    'BaseTripletNetwork',
    'BaseContrastiveNetwork',
    
    # ========================================================================
    # Registry API (main interface)
    # ========================================================================
    'get_model',           # Main function to instantiate models
    'list_models',         # List available models by type
    'get_model_info',      # Get model availability info
    
    # ========================================================================
    # Registry dictionaries (for advanced usage)
    # ========================================================================
    'MODEL_REGISTRY',      # Backward compatibility (BCE models)
    'BCE_MODELS',          # All BCE Siamese models
    'CONTRASTIVE_MODELS',  # All Contrastive models
    'TRIPLET_MODELS',      # All Triplet models
    
    # ========================================================================
    # BCE Siamese models (direct import)
    # ========================================================================
    'SiameseResNet18',
    'SiameseResNet34',
    'SiameseResNet50',
    'SiameseEfficientNetB0',
    'SiameseEfficientNetB1',
    'SiameseEfficientNetV2',
    'SiameseMobileNetV3Small',
    'SiameseMobileNetV3Large',
    'SiameseDenseNet121',
    'SiameseRegNetY400MF',
    
    # ========================================================================
    # Triplet models (direct import)
    # ========================================================================
    'TripletMobileNetV3Small',
    'TripletMobileNetV3Large',
    'TripletResNet18',
    'TripletResNet34',
    'TripletEfficientNetB0',
    
    # ========================================================================
    # Contrastive models (direct import)
    # ========================================================================
    'ContrastiveMobileNetV3Small',
    'ContrastiveMobileNetV3Large',
    'ContrastiveResNet18',
    'ContrastiveResNet34',
    'ContrastiveResNet50',
    'ContrastiveEfficientNetB0',
    'ContrastiveDenseNet121',
]