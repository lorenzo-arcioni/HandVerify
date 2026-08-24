"""
Contrastive Network Backbones
Implementation of various backbone architectures for contrastive learning.
Updated with proper weight transfer from RGB to grayscale.
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .base import BaseContrastiveNetwork
from .utils import adapt_conv_layer_for_grayscale


class ContrastiveMobileNetV3Small(BaseContrastiveNetwork):
    """Contrastive network based on MobileNetV3-Small"""
    
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


class ContrastiveMobileNetV3Large(BaseContrastiveNetwork):
    """Contrastive network based on MobileNetV3-Large"""
    
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


class ContrastiveResNet18(BaseContrastiveNetwork):
    """Contrastive network based on ResNet18"""
    
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


class ContrastiveResNet34(BaseContrastiveNetwork):
    """Contrastive network based on ResNet34"""
    
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


class ContrastiveResNet50(BaseContrastiveNetwork):
    """Contrastive network based on ResNet50"""
    
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


class ContrastiveEfficientNetB0(BaseContrastiveNetwork):
    """Contrastive network based on EfficientNet-B0"""
    
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


class ContrastiveEfficientNetB1(BaseContrastiveNetwork):
    """Contrastive network based on EfficientNet-B1"""
    
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


class ContrastiveDenseNet121(BaseContrastiveNetwork):
    """Contrastive network based on DenseNet121"""
    
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        
        # Adapt first conv (conv0) with weight transfer
        if in_channels != 3:
            densenet.features.conv0 = adapt_conv_layer_for_grayscale(
                densenet.features.conv0, in_channels
            )
        
        encoder = nn.Sequential(
            densenet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        super().__init__(
            encoder=encoder, 
            feature_dim=1024, 
            embedding_dim=embedding_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )
