# src/models/__init__.py
"""
Models module
"""

from .base import BaseSiameseNetwork, BaseTripletNetwork
from .registry import get_model, list_models, MODEL_REGISTRY
from .triplet_backbones import (
    TripletMobileNetV3Small,
    TripletMobileNetV3Large,
    TripletResNet18,
    TripletResNet34,
    TripletEfficientNetB0,
)

__all__ = [
    'BaseSiameseNetwork',
    'BaseTripletNetwork',
    'get_model',
    'list_models',
    'MODEL_REGISTRY',
    'TripletMobileNetV3Small',
    'TripletMobileNetV3Large',
    'TripletResNet18',
    'TripletResNet34',
    'TripletEfficientNetB0',
]