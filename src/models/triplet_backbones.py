"""
Triplet Network Backbones
Implementation of various backbone architectures for triplet learning.
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BaseTripletNetwork(nn.Module):
    """Base class for Triplet networks"""
    
    def __init__(self, encoder: nn.Module, feature_dim: int, embedding_dim: int = 128):
        """
        Args:
            encoder: Backbone network for feature extraction
            feature_dim: Dimension of features from encoder
            embedding_dim: Dimension of final embedding
        """
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        # Projection head
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, x):
        """
        Forward pass to get normalized embeddings.
        
        Args:
            x: Input image
            
        Returns:
            L2-normalized embedding
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)
    
    def get_embedding(self, x):
        """Alias for forward"""
        return self.forward(x)


class TripletMobileNetV3Small(BaseTripletNetwork):
    """Triplet network based on MobileNetV3-Small"""
    
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128):
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        mobilenet.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        
        encoder = nn.Sequential(
            mobilenet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        super().__init__(encoder=encoder, feature_dim=576, embedding_dim=embedding_dim)


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