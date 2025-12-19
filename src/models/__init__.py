# src/models/__init__.py
"""
Models module
"""

from .base import BaseSiameseNetwork, BaseTripletNetwork, BaseContrastiveNetwork
from .registry import get_model, list_models, MODEL_REGISTRY
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
    'BaseSiameseNetwork',
    'BaseTripletNetwork',
    'BaseContrastiveNetwork',
    'get_model',
    'list_models',
    'MODEL_REGISTRY',
    'TripletMobileNetV3Small',
    'TripletMobileNetV3Large',
    'TripletResNet18',
    'TripletResNet34',
    'TripletEfficientNetB0',
    'ContrastiveMobileNetV3Small',
    'ContrastiveMobileNetV3Large',
    'ContrastiveResNet18',
    'ContrastiveResNet34',
    'ContrastiveResNet50',
    'ContrastiveEfficientNetB0',
    'ContrastiveDenseNet121',
]