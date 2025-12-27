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
    """
    
    def __init__(self, encoder: nn.Module, feature_dim: int, projection_dim: int = 128):
        """
        Args:
            encoder: Backbone network for feature extraction
            feature_dim: Dimension of features from encoder
            projection_dim: Dimension of projection head output
        """
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        
        # Projection head (MLP)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, projection_dim)
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
