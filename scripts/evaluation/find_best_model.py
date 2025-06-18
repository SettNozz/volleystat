#!/usr/bin/env python3
"""
Find Best Trained Model
Locates the best model from training checkpoints.
"""

import os
import glob
import json
from datetime import datetime


def find_best_model():
    """Find the best trained model."""
    print("🔍 Searching for trained models...")
    
    # Look for different types of model files
    model_patterns = [
        'best_few_shot_model.pt',
        'best_few_shot_v2_model.pt',
        'checkpoints/*.pt',
        'mlruns/*/artifacts/best_model/*.pt',
        'mlruns/*/artifacts/final_model/*.pt'
    ]
    
    found_models = []
    
    for pattern in model_patterns:
        matches = glob.glob(pattern, recursive=True)
        for match in matches:
            if os.path.isfile(match):
                file_size = os.path.getsize(match) / (1024 * 1024)  # MB
                mod_time = datetime.fromtimestamp(os.path.getmtime(match))
                found_models.append({
                    'path': match,
                    'size_mb': file_size,
                    'modified': mod_time
                })
    
    if not found_models:
        print("❌ No trained models found!")
        return None
    
    # Sort by modification time (newest first)
    found_models.sort(key=lambda x: x['modified'], reverse=True)
    
    print(f"\n📁 Found {len(found_models)} model files:")
    for i, model in enumerate(found_models):
        print(f"  {i+1}. {model['path']}")
        print(f"     Size: {model['size_mb']:.1f} MB")
        print(f"     Modified: {model['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return the most recent model
    best_model = found_models[0]
    print(f"\n🎯 Recommended model: {best_model['path']}")
    
    return best_model['path']


def find_test_dataset():
    """Find the test dataset directory."""
    print("\n🔍 Searching for test datasets...")
    
    # Look for test datasets
    test_patterns = [
        'combined_datasets/*/test',
        'few_shot_datasets_cleaned/*/test',
        'few_shot_datasets/*/test',
        'data/train_val_test_prepared_for_training/test'
    ]
    
    found_datasets = []
    
    for pattern in test_patterns:
        matches = glob.glob(pattern, recursive=True)
        for match in matches:
            if os.path.isdir(match):
                # Check if it has the required structure
                query_dir = os.path.join(match, 'query')
                support_dir = os.path.join(match, 'support')
                masks_dir = os.path.join(match, 'masks')
                
                if all(os.path.exists(d) for d in [query_dir, support_dir, masks_dir]):
                    num_samples = len([f for f in os.listdir(query_dir) if f.endswith('.jpg')])
                    found_datasets.append({
                        'path': match,
                        'num_samples': num_samples
                    })
    
    if not found_datasets:
        print("❌ No test datasets found!")
        return None
    
    # Sort by number of samples (most first)
    found_datasets.sort(key=lambda x: x['num_samples'], reverse=True)
    
    print(f"\n📁 Found {len(found_datasets)} test datasets:")
    for i, dataset in enumerate(found_datasets):
        print(f"  {i+1}. {dataset['path']}")
        print(f"     Samples: {dataset['num_samples']}")
    
    # Return the dataset with most samples
    best_dataset = found_datasets[0]
    print(f"\n🎯 Recommended test dataset: {best_dataset['path']}")
    
    return best_dataset['path']


def main():
    """Main function to find best model and test dataset."""
    print("🚀 Model and Dataset Finder")
    print("=" * 50)
    
    # Find best model
    best_model = find_best_model()
    
    # Find test dataset
    test_dataset = find_test_dataset()
    
    if best_model and test_dataset:
        print(f"\n🎉 Ready to test!")
        print(f"🤖 Model: {best_model}")
        print(f"📁 Test dataset: {test_dataset}")
        print(f"\n🚀 Run the test command:")
        print(f"python test_model.py \"{best_model}\" \"{test_dataset}\"")
    else:
        print(f"\n❌ Cannot proceed without model or test dataset")


if __name__ == "__main__":
    main() 