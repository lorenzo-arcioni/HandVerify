"""
Model Utilities
Helper functions for model initialization and weight transfer.
"""

import torch
import torch.nn as nn
from typing import Union


def adapt_conv_layer_for_grayscale(
    conv_layer: nn.Conv2d,
    target_in_channels: int = 1
) -> nn.Conv2d:
    """
    Adapt a pretrained RGB convolutional layer for grayscale input.
    
    Transfers pretrained weights by averaging across RGB channels instead of
    random initialization. This preserves learned low-level features.
    
    Args:
        conv_layer: Pretrained Conv2d layer (typically with 3 input channels)
        target_in_channels: Target number of input channels (default: 1 for grayscale)
    
    Returns:
        New Conv2d layer with adapted weights
        
    Example:
        >>> resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        >>> resnet.conv1 = adapt_conv_layer_for_grayscale(resnet.conv1, target_in_channels=1)
    """
    
    # If already the correct number of channels, return as-is
    if conv_layer.in_channels == target_in_channels:
        return conv_layer
    
    # Extract pretrained weights: [out_channels, in_channels, H, W]
    pretrained_weights = conv_layer.weight.data
    
    # Average across input channels: [out_channels, 1, H, W]
    # This preserves the learned edge/texture filters
    adapted_weights = pretrained_weights.mean(dim=1, keepdim=True)
    
    # If target has more than 1 channel, repeat the averaged weights
    if target_in_channels > 1:
        adapted_weights = adapted_weights.repeat(1, target_in_channels, 1, 1)
        # Normalize to maintain similar activation magnitudes
        adapted_weights = adapted_weights / target_in_channels
    
    # Create new Conv2d layer with same architecture
    new_conv = nn.Conv2d(
        in_channels=target_in_channels,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        dilation=conv_layer.dilation,
        groups=conv_layer.groups,
        bias=conv_layer.bias is not None,
        padding_mode=conv_layer.padding_mode
    )
    
    # Transfer adapted weights
    new_conv.weight.data = adapted_weights
    
    # Transfer bias if present
    if conv_layer.bias is not None:
        new_conv.bias.data = conv_layer.bias.data.clone()
    
    return new_conv


def adapt_model_for_grayscale(
    model: nn.Module,
    first_conv_path: str,
    target_in_channels: int = 1
) -> nn.Module:
    """
    Adapt a pretrained model's first convolutional layer for grayscale input.
    
    Supports nested module paths (e.g., 'features.0.0' for nested Sequential modules).
    
    Args:
        model: Pretrained model
        first_conv_path: Dot-separated path to first conv layer
                        Examples: 'conv1', 'features.0.0', 'stem.0'
        target_in_channels: Target number of input channels (default: 1)
    
    Returns:
        Model with adapted first conv layer
        
    Examples:
        >>> # Simple case: ResNet
        >>> resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        >>> resnet = adapt_model_for_grayscale(resnet, 'conv1')
        
        >>> # Nested case: EfficientNet
        >>> effnet = models.efficientnet_b0(weights=...)
        >>> effnet = adapt_model_for_grayscale(effnet, 'features.0.0')
    """
    
    # Navigate to parent module and get attribute name
    path_parts = first_conv_path.split('.')
    parent_module = model
    
    # Navigate to parent
    for part in path_parts[:-1]:
        parent_module = getattr(parent_module, part)
    
    # Get the conv layer
    attr_name = path_parts[-1]
    conv_layer = getattr(parent_module, attr_name)
    
    # Adapt and replace
    adapted_conv = adapt_conv_layer_for_grayscale(conv_layer, target_in_channels)
    setattr(parent_module, attr_name, adapted_conv)
    
    return model


def print_weight_transfer_info(original_conv: nn.Conv2d, adapted_conv: nn.Conv2d):
    """
    Print information about weight transfer process.
    
    Args:
        original_conv: Original pretrained conv layer
        adapted_conv: Adapted conv layer
    """
    orig_shape = tuple(original_conv.weight.shape)
    new_shape = tuple(adapted_conv.weight.shape)
    
    print(f"  ✓ Weight transfer: {orig_shape} → {new_shape}")
    print(f"    Original channels: {original_conv.in_channels} (RGB)")
    print(f"    Adapted channels:  {adapted_conv.in_channels} (Grayscale)")
    print(f"    Method: Averaging across RGB channels")
