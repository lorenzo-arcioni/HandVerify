"""
Triplet Network Backbones
Implementation of various backbone architectures for triplet learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .base import BaseTripletNetwork


class TripletMobileNetV3Small(BaseTripletNetwork):
    """
    Triplet network based on MobileNetV3-Small
    with frozen backbone and regularized embedding head.
    """

    def __init__(self, in_channels: int = 1, embedding_dim: int = 128):
        super().__init__()

        # ---------------------------------------------------------
        # Backbone: MobileNetV3-Small (PRETRAINED)
        # ---------------------------------------------------------
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )

        # Adatta input a grayscale
        mobilenet.features[0][0] = nn.Conv2d(
            in_channels,
            16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        # 🔒 FREEZE COMPLETO DEL BACKBONE
        for p in mobilenet.features.parameters():
            p.requires_grad = False

        # Encoder CNN
        self.encoder = nn.Sequential(
            mobilenet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        feature_dim = 576  # MobileNetV3-Small output

        # ---------------------------------------------------------
        # Embedding Head (REGOLARIZZATA)
        # ---------------------------------------------------------
        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing L2-normalized embeddings.
        """
        x = self.encoder(x)
        x = self.embedding_head(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class TripletMobileNetV3Large(BaseTripletNetwork):
    """Triplet network based on MobileNetV3-Large"""
    
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128):
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        mobilenet.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        
        encoder = nn.Sequential(
            mobilenet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        super().__init__(encoder=encoder, feature_dim=960, embedding_dim=embedding_dim)


class TripletResNet18(BaseTripletNetwork):
    """Triplet network based on ResNet18"""
    
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128):
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        encoder = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        
        super().__init__(encoder=encoder, feature_dim=512, embedding_dim=embedding_dim)


class TripletResNet34(BaseTripletNetwork):
    """Triplet network based on ResNet34"""
    
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128):
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        encoder = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        
        super().__init__(encoder=encoder, feature_dim=512, embedding_dim=embedding_dim)


class TripletEfficientNetB0(BaseTripletNetwork):
    """Triplet network based on EfficientNet-B0"""
    
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128):
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        effnet.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        encoder = nn.Sequential(
            effnet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        super().__init__(encoder=encoder, feature_dim=1280, embedding_dim=embedding_dim)