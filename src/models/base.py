"""
Base Siamese Network Module
Defines the base architecture for Siamese networks with shared encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseSiameseNetwork(nn.Module):
    """
    Base class for Siamese networks.
    All specific architectures should inherit from this class.
    """
    
    def __init__(self, encoder: nn.Module, feature_dim: int, projection_dim: int = 512):
        """
        Args:
            encoder: Backbone network for feature extraction
            feature_dim: Dimension of features from encoder
            projection_dim: Dimension of projection layer
        """
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        
        # Classifier for similarity prediction
        self.fc = self._build_fc_block(feature_dim * 2, projection_dim)
    
    def _build_fc_block(self, input_dim: int, projection_dim: int) -> nn.Sequential:
        """
        Builds the fully connected block for similarity classification.
        
        Args:
            input_dim: Input dimension (concatenated features)
            projection_dim: Intermediate projection dimension
            
        Returns:
            Sequential module for classification
        """
        half = projection_dim // 2
        quarter = half // 2
        
        return nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(projection_dim, half),
            nn.BatchNorm1d(half),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(half, quarter),
            nn.BatchNorm1d(quarter),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(quarter, 1),
            nn.Sigmoid()
        )
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single image through the encoder.
        
        Args:
            x: Input image tensor
            
        Returns:
            Feature vector
        """
        x = self.encoder(x)
        return x.view(x.size(0), -1)
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor, 
                return_embeddings: bool = False):
        """
        Forward pass for image pair.
        
        Args:
            img1: First image
            img2: Second image
            return_embeddings: If True, return normalized embeddings instead of similarity
            
        Returns:
            Similarity score or embeddings tuple
        """
        feat1 = self.forward_one(img1)
        feat2 = self.forward_one(img2)
        
        if return_embeddings:
            # Normalize for metric learning
            feat1 = F.normalize(feat1, p=2, dim=1)
            feat2 = F.normalize(feat2, p=2, dim=1)
            return feat1, feat2
        
        # Concatenate and classify
        combined = torch.cat([feat1, feat2], dim=1)
        return self.fc(combined)

class BaseContrastiveNetwork(nn.Module):
    """
    Base class for Contrastive Learning networks.
    Projects features to a lower-dimensional embedding space.
    Enhanced with deeper projection head and stronger regularization.
    """
    
    def __init__(
        self, 
        encoder: nn.Module, 
        feature_dim: int, 
        embedding_dim: int = 128,
        freeze_backbone_layers: int = 0,
        dropout: float = 0.5
    ):
        """
        Args:
            encoder: Backbone network for feature extraction
            feature_dim: Dimension of features from encoder
            embedding_dim: Dimension of projection head output
            freeze_backbone_layers: Number of initial encoder layers to freeze (0 = no freezing)
            dropout: Dropout rate for regularization (default: 0.5)
        """
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.freeze_backbone_layers = freeze_backbone_layers
        
        # Freeze initial layers if requested
        if freeze_backbone_layers > 0:
            self._freeze_encoder_layers(freeze_backbone_layers)
        
        # Projection head (deeper MLP with more regularization)
        self.projection = self._build_projection_head(feature_dim, embedding_dim, dropout)
    
    def _freeze_encoder_layers(self, num_layers: int):
        """
        Freeze the first N layers of the encoder backbone.
        
        Args:
            num_layers: Number of initial layers to freeze
        """
        frozen_count = 0
        
        for i, child in enumerate(self.encoder[0].children()):
            if frozen_count < num_layers:
                for param in child.parameters():
                    param.requires_grad = False
                frozen_count += 1
            else:
                break
        
        frozen_params = sum(p.numel() for p in self.encoder[0].parameters() if not p.requires_grad)
        total_params = sum(p.numel() for p in self.encoder[0].parameters())
        
        print("Total encoder layers:", len(self.encoder[0].children()))
        print(f"  ❄️  Frozen {frozen_count} encoder layers")
        print(f"  ❄️  Frozen parameters: {frozen_params:,} / {total_params:,} "
              f"({100*frozen_params/total_params:.1f}%)")
    
    def _build_projection_head(self, feature_dim: int, embedding_dim: int, dropout: float) -> nn.Sequential:
        """
        Build deeper projection head with stronger regularization.
        
        Args:
            feature_dim: Input feature dimension
            embedding_dim: Output embedding dimension
            dropout: Dropout rate
            
        Returns:
            Sequential projection head
        """
        # Intermediate dimensions
        hidden1 = 1024
        hidden2 = 512
        hidden3 = 256
        
        return nn.Sequential(
            # Layer 1
            nn.Linear(feature_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # 0.5
            
            # Layer 2
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # 0.5
            
            # Layer 3 (NEW)
            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),  # 0.4
            
            # Output
            nn.Linear(hidden3, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get normalized embeddings.
        
        Args:
            x: Input image
            
        Returns:
            L2-normalized projection
        """
        # Extract features
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        
        # Project to embedding space
        embeddings = self.projection(features)
        
        # Normalize
        return F.normalize(embeddings, p=2, dim=1)
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward"""
        return self.forward(x)


class BaseTripletNetwork(nn.Module):
    """
    Base class for Triplet networks.
    Enhanced with deeper projection head and stronger regularization.
    """
    
    def __init__(
        self, 
        encoder: nn.Module, 
        feature_dim: int, 
        embedding_dim: int = 128,
        freeze_backbone_layers: int = 0,
        dropout: float = 0.5
    ):
        """
        Args:
            encoder: Backbone network for feature extraction
            feature_dim: Dimension of features from encoder
            embedding_dim: Dimension of final embedding
            freeze_backbone_layers: Number of initial encoder layers to freeze (0 = no freezing)
            dropout: Dropout rate for regularization (default: 0.5)
        """
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.freeze_backbone_layers = freeze_backbone_layers
        
        # Freeze initial layers if requested
        if freeze_backbone_layers > 0:
            self._freeze_encoder_layers(freeze_backbone_layers)
        
        # Projection head (deeper with more regularization)
        self.fc = self._build_projection_head(feature_dim, embedding_dim, dropout)
    
    def _freeze_encoder_layers(self, num_layers: int):
        """
        Freeze the first N layers of the encoder backbone.
        
        Args:
            num_layers: Number of initial layers to freeze
        """
        frozen_count = 0
        
        for i, child in enumerate(self.encoder[0].children()):
            if frozen_count < num_layers:
                for param in child.parameters():
                    param.requires_grad = False
                frozen_count += 1
            else:
                break
        
        frozen_params = sum(p.numel() for p in self.encoder[0].parameters() if not p.requires_grad)
        total_params = sum(p.numel() for p in self.encoder[0].parameters())
        
        print("Total encoder layers:", len(self.encoder[0].children()))
        print(f"  ❄️  Frozen {frozen_count} encoder layers")
        print(f"  ❄️  Frozen parameters: {frozen_params:,} / {total_params:,} "
              f"({100*frozen_params/total_params:.1f}%)")
    
    def _build_projection_head(self, feature_dim: int, embedding_dim: int, dropout: float) -> nn.Sequential:
        """
        Build deeper projection head with stronger regularization.
        
        Args:
            feature_dim: Input feature dimension
            embedding_dim: Output embedding dimension
            dropout: Dropout rate
            
        Returns:
            Sequential projection head
        """
        # Intermediate dimensions
        hidden1 = 1024
        hidden2 = 512
        hidden3 = 256
        
        return nn.Sequential(
            # Layer 1
            nn.Linear(feature_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # 0.5
            
            # Layer 2
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # 0.5
            
            # Layer 3 (NEW)
            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),  # 0.4
            
            # Output
            nn.Linear(hidden3, embedding_dim)
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
