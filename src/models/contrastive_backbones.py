"""
Contrastive Network Backbones
Implementation of various backbone architectures for contrastive learning.
Updated with layer freezing support and increased regularization.
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .base import BaseContrastiveNetwork


class ContrastiveMobileNetV3Small(BaseContrastiveNetwork):
    """Contrastive network based on MobileNetV3-Small"""
    
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        mobilenet.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        
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
        mobilenet.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        
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
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        encoder = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        
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
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        encoder = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        
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
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        encoder = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        
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
        effnet.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
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
        densenet.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
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