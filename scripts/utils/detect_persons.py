"""
Legacy interface for volleyball ball segmentation.
This file now imports from the modular structure in src/.
"""

# Import from the new modular structure
from src.models.unet import UNet
from src.data.dataset import SiameseDataset
from src.utils.data_processing import (
    create_one_shot_dataset,
    split_dataset_consistently,
    check_mask_values
)
from src.training.trainer import train_model
from src.visualization.visualization import plot_training_losses, visualize_masks
from src.evaluation.evaluator import evaluate_model

# Re-export for backward compatibility
__all__ = [
    'UNet',
    'SiameseDataset', 
    'create_one_shot_dataset',
    'split_dataset_consistently',
    'check_mask_values',
    'train_model',
    'plot_training_losses',
    'visualize_masks',
    'evaluate_model'
]

# Legacy main function for backward compatibility
def legacy_main():
    """Legacy main function - use main.py instead."""
    print("⚠️  This is the legacy interface. Please use main.py for the new modular structure.")
    print("📁 New structure available in src/ directory")
    print("🚀 Run: python main.py") 