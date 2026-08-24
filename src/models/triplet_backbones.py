"""
Triplet Network Backbones
Implementation of various backbone architectures for triplet learning.
Updated with proper weight transfer from RGB to grayscale.
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .base import BaseTripletNetwork
from .utils import adapt_conv_layer_for_grayscale


class TripletMobileNetV3Small(BaseTripletNetwork):
    """Triplet network based on MobileNetV3-Small"""
    
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        
        # Adapt first conv with weight transfer
        if in_channels != 3:
            mobilenet.features[0][0] = adapt_conv_layer_for_grayscale(
                mobilenet.features[0][0], in_channels
            )
        
        encoder = nn.Sequential(
            mobilenet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        super().__init__(
            encoder=encoder, 
            feature_dim=576, 
            embedding_dim=embedding_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )


class TripletMobileNetV3Large(BaseTripletNetwork):
    """Triplet network based on MobileNetV3-Large"""
    
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        
        # Adapt first conv with weight transfer
        if in_channels != 3:
            mobilenet.features[0][0] = adapt_conv_layer_for_grayscale(
                mobilenet.features[0][0], in_channels
            )
        
        encoder = nn.Sequential(
            mobilenet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        super().__init__(
            encoder=encoder, 
            feature_dim=960, 
            embedding_dim=embedding_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )


class TripletResNet18(BaseTripletNetwork):
    """Triplet network based on ResNet18"""
    
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Adapt conv1 with weight transfer
        if in_channels != 3:
            resnet.conv1 = adapt_conv_layer_for_grayscale(resnet.conv1, in_channels)
        
        encoder = nn.Sequential(
            nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
                resnet.avgpool
            ),
            nn.Flatten()
        )
        
        super().__init__(
            encoder=encoder, 
            feature_dim=512, 
            embedding_dim=embedding_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )


class TripletResNet34(BaseTripletNetwork):
    """Triplet network based on ResNet34"""
    
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Adapt conv1 with weight transfer
        if in_channels != 3:
            resnet.conv1 = adapt_conv_layer_for_grayscale(resnet.conv1, in_channels)
        
        encoder = nn.Sequential(
            nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
                resnet.avgpool
            ),
            nn.Flatten()
        )
        
        super().__init__(
            encoder=encoder, 
            feature_dim=512, 
            embedding_dim=embedding_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )


class TripletResNet50(BaseTripletNetwork):
    """Triplet network based on ResNet50"""
    
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Adapt conv1 with weight transfer
        if in_channels != 3:
            resnet.conv1 = adapt_conv_layer_for_grayscale(resnet.conv1, in_channels)
        
        encoder = nn.Sequential(
            nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
                resnet.avgpool
            ),
            nn.Flatten()
        )
        
        super().__init__(
            encoder=encoder, 
            feature_dim=2048, 
            embedding_dim=embedding_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )


class TripletEfficientNetB0(BaseTripletNetwork):
    """Triplet network based on EfficientNet-B0"""
    
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Adapt first conv with weight transfer
        if in_channels != 3:
            effnet.features[0][0] = adapt_conv_layer_for_grayscale(
                effnet.features[0][0], in_channels
            )
        
        encoder = nn.Sequential(
            effnet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        super().__init__(
            encoder=encoder, 
            feature_dim=1280, 
            embedding_dim=embedding_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )


class TripletEfficientNetB1(BaseTripletNetwork):
    """Triplet network based on EfficientNet-B1"""
    
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        effnet = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        
        # Adapt first conv with weight transfer
        if in_channels != 3:
            effnet.features[0][0] = adapt_conv_layer_for_grayscale(
                effnet.features[0][0], in_channels
            )
        
        encoder = nn.Sequential(
            effnet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        super().__init__(
            encoder=encoder, 
            feature_dim=1280, 
            embedding_dim=embedding_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )
