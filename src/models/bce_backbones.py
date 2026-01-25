"""
Siamese Network Backbones
Implementation of various backbone architectures for handwriting verification.
Updated with layer freezing support and increased regularization.
"""

import torch.nn as nn
from torchvision import models
from .base import BaseSiameseNetwork


class SiameseResNet18(BaseSiameseNetwork):
    """Siamese network based on ResNet18"""
    
    def __init__(self, in_channels: int = 1, projection_dim: int = 512, 
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        super().__init__(
            encoder=encoder, 
            feature_dim=512, 
            projection_dim=projection_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )

class SiameseResNet34(BaseSiameseNetwork):
    """Siamese network based on ResNet34"""
    
    def __init__(self, in_channels: int = 1, projection_dim: int = 512,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        super().__init__(
            encoder=encoder, 
            feature_dim=512, 
            projection_dim=projection_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )

class SiameseResNet50(BaseSiameseNetwork):
    """Siamese network based on ResNet50"""
    
    def __init__(self, in_channels: int = 1, projection_dim: int = 512,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        super().__init__(
            encoder=encoder, 
            feature_dim=2048, 
            projection_dim=projection_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )

class SiameseEfficientNetB0(BaseSiameseNetwork):
    """Siamese network based on EfficientNet-B0"""
    
    def __init__(self, in_channels: int = 1, projection_dim: int = 512,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        effnet.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        encoder = nn.Sequential(effnet.features, nn.AdaptiveAvgPool2d(1))
        
        super().__init__(
            encoder=encoder, 
            feature_dim=1280, 
            projection_dim=projection_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )

class SiameseEfficientNetB1(BaseSiameseNetwork):
    """Siamese network based on EfficientNet-B1"""
    
    def __init__(self, in_channels: int = 1, projection_dim: int = 512,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        effnet = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        effnet.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        encoder = nn.Sequential(effnet.features, nn.AdaptiveAvgPool2d(1))
        
        super().__init__(
            encoder=encoder, 
            feature_dim=1280, 
            projection_dim=projection_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )

class SiameseEfficientNetV2(BaseSiameseNetwork):
    """Siamese network based on EfficientNetV2-S"""
    
    def __init__(self, in_channels: int = 1, projection_dim: int = 512,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        backbone.features[0][0] = nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        encoder = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d(1))
        
        super().__init__(
            encoder=encoder, 
            feature_dim=1280, 
            projection_dim=projection_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )

class SiameseMobileNetV3Small(BaseSiameseNetwork):
    """Siamese network based on MobileNetV3-Small"""
    
    def __init__(self, in_channels: int = 1, projection_dim: int = 512,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        mobilenet.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        encoder = nn.Sequential(mobilenet.features, nn.AdaptiveAvgPool2d(1))
        
        super().__init__(
            encoder=encoder, 
            feature_dim=576, 
            projection_dim=projection_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )

class SiameseMobileNetV3Large(BaseSiameseNetwork):
    """Siamese network based on MobileNetV3-Large"""
    
    def __init__(self, in_channels: int = 1, projection_dim: int = 512,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        mobilenet.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        encoder = nn.Sequential(mobilenet.features, nn.AdaptiveAvgPool2d(1))
        
        super().__init__(
            encoder=encoder, 
            feature_dim=960, 
            projection_dim=projection_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )

class SiameseDenseNet121(BaseSiameseNetwork):
    """Siamese network based on DenseNet121"""
    
    def __init__(self, in_channels: int = 1, projection_dim: int = 512,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        densenet.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        encoder = nn.Sequential(densenet.features, nn.AdaptiveAvgPool2d(1))
        
        super().__init__(
            encoder=encoder, 
            feature_dim=1024, 
            projection_dim=projection_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )

class SiameseRegNetY400MF(BaseSiameseNetwork):
    """Siamese network based on RegNet-Y-400MF"""
    
    def __init__(self, in_channels: int = 1, projection_dim: int = 512,
                 freeze_backbone_layers: int = 2, dropout: float = 0.5):
        regnet = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.DEFAULT)
        regnet.stem[0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        encoder = nn.Sequential(*list(regnet.children())[:-1], nn.AdaptiveAvgPool2d(1))
        
        super().__init__(
            encoder=encoder, 
            feature_dim=440, 
            projection_dim=projection_dim,
            freeze_backbone_layers=freeze_backbone_layers,
            dropout=dropout
        )