"""
Data Transforms
Custom transforms and augmentations for handwriting images.
"""

import torchvision.transforms as transforms


def get_train_transforms(target_size: int = 448):
    """
    Get training transforms with data augmentation.
    
    Args:
        target_size: Target image size
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(target_size, scale=(0.9, 1.1)),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])


def get_test_transforms(target_size: int = 448):
    """
    Get test/validation transforms (no augmentation).
    
    Args:
        target_size: Target image size
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor()
    ])
