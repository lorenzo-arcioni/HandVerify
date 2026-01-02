"""
Utility Helper Functions
General utility functions for the project.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the appropriate device (CUDA if available, else CPU).
    
    Returns:
        torch.device object
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    """
    Format large numbers with K/M/B suffixes.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string
    """
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


def print_model_info(model: torch.nn.Module, model_name: str):
    """
    Print information about a model.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
    """
    num_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Parameters: {format_number(num_params)} ({num_params:,})")
    print(f"{'='*60}\n")


def ensure_dir(directory: str):
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device
) -> torch.nn.Module:
    """
    Load model weights from checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Model with loaded weights
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    print(f"✓ Loaded checkpoint from {checkpoint_path}")
    
    return model


def save_checkpoint(
    model: torch.nn.Module,
    save_path: str
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        save_path: Path to save checkpoint
    """
    ensure_dir(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)
    print(f"✓ Saved checkpoint to {save_path}")