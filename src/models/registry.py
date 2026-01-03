"""
Model Registry
Central registry for all available model architectures (BCE, Contrastive, Triplet).
Updated to support freeze_backbone_layers and dropout parameters.
"""

from typing import Dict, Type, Union
from .base import BaseSiameseNetwork, BaseContrastiveNetwork, BaseTripletNetwork

# Import BCE Siamese backbones
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

# Import Contrastive backbones
from .contrastive_backbones import (
    ContrastiveMobileNetV3Small,
    ContrastiveMobileNetV3Large,
    ContrastiveResNet18,
    ContrastiveResNet34,
    ContrastiveResNet50,
    ContrastiveEfficientNetB0,
    ContrastiveDenseNet121,
)

# Import Triplet backbones
from .triplet_backbones import (
    TripletMobileNetV3Small,
    TripletMobileNetV3Large,
    TripletResNet18,
    TripletResNet34,
    TripletEfficientNetB0,
)

# ============================================================================
# Model Registry organized by architecture type
# ============================================================================

BCE_MODELS: Dict[str, Type[BaseSiameseNetwork]] = {
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

CONTRASTIVE_MODELS: Dict[str, Type[BaseContrastiveNetwork]] = {
    # MobileNet family
    'mobilenet_v3_small': ContrastiveMobileNetV3Small,
    'mobilenet_v3_large': ContrastiveMobileNetV3Large,
    
    # ResNet family
    'resnet18': ContrastiveResNet18,
    'resnet34': ContrastiveResNet34,
    'resnet50': ContrastiveResNet50,
    
    # EfficientNet family
    'efficientnet_b0': ContrastiveEfficientNetB0,
    
    # DenseNet family
    'densenet121': ContrastiveDenseNet121,
}

TRIPLET_MODELS: Dict[str, Type[BaseTripletNetwork]] = {
    # MobileNet family
    'mobilenet_v3_small': TripletMobileNetV3Small,
    'mobilenet_v3_large': TripletMobileNetV3Large,
    
    # ResNet family
    'resnet18': TripletResNet18,
    'resnet34': TripletResNet34,
    
    # EfficientNet family
    'efficientnet_b0': TripletEfficientNetB0,
}


# ============================================================================
# Main API Functions
# ============================================================================

def get_model(
    model_name: str,
    model_type: str = 'bce',
    freeze_backbone_layers: int = 2,
    dropout: float = 0.5,
    **kwargs
) -> Union[BaseSiameseNetwork, BaseContrastiveNetwork, BaseTripletNetwork]:
    """
    Get a model instance by name and type.
    
    Args:
        model_name: Name of the backbone architecture (e.g., 'resnet18')
        model_type: Type of model - 'bce', 'contrastive', or 'triplet'
        freeze_backbone_layers: Number of initial encoder layers to freeze (default: 2)
                               Set to 0 to disable freezing
        dropout: Dropout rate for regularization (default: 0.5)
        **kwargs: Additional arguments to pass to model constructor
                  - in_channels: Number of input channels (default: 1)
                  - projection_dim: For BCE models (default: 512)
                  - embedding_dim: For Contrastive/Triplet models (default: 128)
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_name or model_type not found in registry
        
    Examples:
        >>> # BCE Siamese with frozen layers
        >>> model = get_model('resnet18', model_type='bce', 
        ...                   freeze_backbone_layers=2, dropout=0.5)
        
        >>> # Contrastive without freezing
        >>> model = get_model('resnet18', model_type='contrastive', 
        ...                   freeze_backbone_layers=0, dropout=0.6)
        
        >>> # Triplet with 3 frozen layers
        >>> model = get_model('resnet18', model_type='triplet', 
        ...                   freeze_backbone_layers=3, dropout=0.5)
    """
    model_type = model_type.lower()
    
    # Select registry based on model type
    if model_type == 'bce':
        registry = BCE_MODELS
    elif model_type == 'contrastive':
        registry = CONTRASTIVE_MODELS
    elif model_type == 'triplet':
        registry = TRIPLET_MODELS
    else:
        raise ValueError(
            f"Invalid model_type '{model_type}'. "
            f"Must be one of: 'bce', 'contrastive', 'triplet'"
        )
    
    # Check if model exists in registry
    if model_name not in registry:
        available = ', '.join(registry.keys())
        raise ValueError(
            f"Model '{model_name}' not found in {model_type.upper()} registry. "
            f"Available models: {available}"
        )
    
    # Handle parameter naming differences
    if model_type == 'bce':
        # BCE uses 'projection_dim'
        if 'embedding_dim' in kwargs and 'projection_dim' not in kwargs:
            kwargs['projection_dim'] = kwargs.pop('embedding_dim')
    else:
        # Contrastive/Triplet use 'embedding_dim'
        if 'projection_dim' in kwargs and 'embedding_dim' not in kwargs:
            kwargs['embedding_dim'] = kwargs.pop('projection_dim')
    
    # Add freeze and dropout parameters
    kwargs['freeze_backbone_layers'] = freeze_backbone_layers
    kwargs['dropout'] = dropout
    
    # Instantiate model
    model_class = registry[model_name]
    return model_class(**kwargs)


def list_models(model_type: str = None) -> Dict[str, list]:
    """
    Get list of available model names.
    
    Args:
        model_type: If specified, return only models of this type.
                   If None, return all models organized by type.
        
    Returns:
        Dictionary mapping model types to lists of model names,
        or a single list if model_type is specified.
        
    Examples:
        >>> # Get all models
        >>> list_models()
        {'bce': [...], 'contrastive': [...], 'triplet': [...]}
        
        >>> # Get only BCE models
        >>> list_models('bce')
        ['resnet18', 'resnet34', ...]
    """
    all_models = {
        'bce': list(BCE_MODELS.keys()),
        'contrastive': list(CONTRASTIVE_MODELS.keys()),
        'triplet': list(TRIPLET_MODELS.keys()),
    }
    
    if model_type is None:
        return all_models
    
    model_type = model_type.lower()
    if model_type not in all_models:
        raise ValueError(
            f"Invalid model_type '{model_type}'. "
            f"Must be one of: 'bce', 'contrastive', 'triplet'"
        )
    
    return all_models[model_type]


def get_model_info(model_name: str = None) -> Dict:
    """
    Get information about available model architectures.
    
    Args:
        model_name: If specified, return info for this model only.
                   If None, return info for all models.
    
    Returns:
        Dictionary with model availability across different types
        
    Examples:
        >>> get_model_info('resnet18')
        {'resnet18': {'bce': True, 'contrastive': True, 'triplet': True}}
        
        >>> get_model_info()  # All models
    """
    all_model_names = set(
        list(BCE_MODELS.keys()) + 
        list(CONTRASTIVE_MODELS.keys()) + 
        list(TRIPLET_MODELS.keys())
    )
    
    if model_name is not None:
        if model_name not in all_model_names:
            raise ValueError(f"Model '{model_name}' not found in any registry")
        all_model_names = {model_name}
    
    info = {}
    for name in sorted(all_model_names):
        info[name] = {
            'bce': name in BCE_MODELS,
            'contrastive': name in CONTRASTIVE_MODELS,
            'triplet': name in TRIPLET_MODELS,
        }
    
    return info