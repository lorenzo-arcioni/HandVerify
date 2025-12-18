def get_model(name: str, mode='bce', **kwargs):
    """
    Factory unificato
    
    Args:
        name: 'resnet18', 'efficientnet_b0', etc.
        mode: 'bce' o 'triplet'
    """
    encoder, feat_dim = build_encoder(name)  # da backbones.py
    return BaseSiameseNetwork(encoder, feat_dim, mode=mode)