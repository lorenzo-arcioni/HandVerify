import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseSiameseNetwork(nn.Module):
    """Classe base per BCE e Triplet"""
    
    def __init__(self, encoder, feat_dim, mode='bce', projection_dim=1024):
        super().__init__()
        self.encoder = encoder
        self.mode = mode
        
        if mode == 'bce':
            # Classificatore per BCE
            self.fc = self._build_classifier(feat_dim * 2, projection_dim)
        # Per triplet: nessun classifier, solo embeddings
    
    def _build_classifier(self, input_dim, projection_dim):
        """Classificatore condiviso per BCE"""
        return nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(projection_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward_one(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
    
    def forward(self, img1, img2=None, img3=None):
        if self.mode == 'triplet':
            # Triplet: restituisce embeddings normalizzati
            if img3 is not None:  # anchor, pos, neg
                return tuple(F.normalize(self.forward_one(x), p=2, dim=1) 
                           for x in [img1, img2, img3])
            else:  # inference
                return F.normalize(self.forward_one(img1), p=2, dim=1)
        else:
            # BCE: restituisce probabilità
            feat1 = self.forward_one(img1)
            feat2 = self.forward_one(img2)
            return self.fc(torch.cat([feat1, feat2], dim=1))