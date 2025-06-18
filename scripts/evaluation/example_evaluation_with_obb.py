#!/usr/bin/env python3
"""
Example script demonstrating the modified evaluator with YOLOv8 OBB dataset creation.
This script evaluates a trained model and creates a dataset ready for labeling.
"""

import os
from src.evaluation.evaluator import evaluate_model

def main():
    # Configuration
    model_path = "path/to/your/trained/model.pth"
    test_query_dir = "data/train_val_test_prepared_for_training/test/query"
    test_mask_dir = "data/train_val_test_prepared_for_training/test/masks"
    support_path = "data/one_shot_segmentation_data/support/ball1.jpg"
    output_dir = "evaluation_results"
    
    # Optional: specify custom OBB dataset directory
    obb_dataset_dir = "obb_dataset_for_labeling"
    
    print("Starting evaluation with OBB dataset creation...")
    
    # Run evaluation with OBB dataset creation
    iou_scores = evaluate_model(
        model_path=model_path,
        test_query_dir=test_query_dir,
        test_mask_dir=test_mask_dir,
        support_path=support_path,
        output_dir=output_dir,
        save_obb_dataset=True,  # Enable OBB dataset creation
        obb_dataset_dir=obb_dataset_dir
    )
    
    print(f"\nEvaluation completed!")
    print(f"Average IoU: {sum(iou_scores) / len(iou_scores):.3f}")
    print(f"OBB dataset created in: {obb_dataset_dir}")
    print(f"Dataset structure:")
    print(f"  - {obb_dataset_dir}/images/ (original images)")
    print(f"  - {obb_dataset_dir}/labels/ (YOLOv8 OBB format labels)")
    print(f"  - {obb_dataset_dir}/data.yaml (dataset configuration)")
    print(f"  - {obb_dataset_dir}/evaluation_summary.txt (evaluation results)")

if __name__ == "__main__":
    main() 