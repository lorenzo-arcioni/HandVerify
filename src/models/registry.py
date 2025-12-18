"""
Model Registry
Central registry for all available model architectures.
"""

from typing import Dict, Type
from .base import BaseSiameseNetwork
from .backbones import (
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


MODEL_REGISTRY: Dict[str, Type[BaseSiameseNetwork]] = {
    # ResNet family
    'resnet18': SiameseResNet18,
    'resnet34': SiameseResNet34,
    'resnet50': SiameseResNet50,
    
    # EfficientNet family
    'efficientnet_b0': SiameseEfficientNetB0,
    'efficientnet_b1': SiameseEfficientNetB1,
    'efficientnet_v2': SiameseEfficientNetV2,
    
    # MobileNet family
    'mobilenet_v3_small': SiameseMobileNetV3Small,
    'mobilenet_v3_large': SiameseMobileNetV3Large,
    
    # DenseNet family
    'densenet121': SiameseDenseNet121,
    
    # RegNet family
    'regnet_y_400mf': SiameseRegNetY400MF,
}


def get_model(model_name: str, **kwargs) -> BaseSiameseNetwork:
    """
    Get a model instance by name.
    
    Args:
        model_name: Name of the model (must be in MODEL_REGISTRY)
        **kwargs: Additional arguments to pass to model constructor
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_name not found in registry
    """
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)


def list_models() -> list:
    """Get list of all available model names"""
    return list(MODEL_REGISTRY.keys())