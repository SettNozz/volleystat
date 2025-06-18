#!/usr/bin/env python3
"""
Combine Two Few-Shot Datasets
Combines two datasets with the same train/val/test structure to increase the number of examples.
"""

import os
import shutil
import random
import json
from datetime import datetime


def get_dataset_info(dataset_dir):
    """Get information about a dataset."""
    info_file = os.path.join(dataset_dir, 'dataset_info.json')
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            return json.load(f)
    return None


def count_samples_in_split(split_dir):
    """Count samples in a split directory."""
    query_dir = os.path.join(split_dir, 'query')
    if os.path.exists(query_dir):
        return len([f for f in os.listdir(query_dir) if f.endswith('.jpg')])
    return 0


def combine_datasets(
    dataset1_dir,
    dataset2_dir,
    output_base_dir='combined_datasets',
    run_name=None
):
    """Combine two datasets with the same structure."""
    
    # Generate run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"combined_run_{timestamp}"
    
    print(f"🔄 Combining datasets...")
    print(f"📁 Dataset 1: {dataset1_dir}")
    print(f"📁 Dataset 2: {dataset2_dir}")
    
    # Check if both datasets exist
    if not os.path.exists(dataset1_dir):
        print(f"❌ Dataset 1 not found: {dataset1_dir}")
        return
    
    if not os.path.exists(dataset2_dir):
        print(f"❌ Dataset 2 not found: {dataset2_dir}")
        return
    
    # Get dataset info
    info1 = get_dataset_info(dataset1_dir)
    info2 = get_dataset_info(dataset2_dir)
    
    # Create output directory structure
    output_dir = os.path.join(output_base_dir, run_name)
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(os.path.join(split_dir, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'query'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'support'), exist_ok=True)
    
    # Combine each split
    combined_info = {
        'run_name': run_name,
        'created_at': datetime.now().isoformat(),
        'dataset1': dataset1_dir,
        'dataset2': dataset2_dir,
        'splits': {},
        'total_samples': 0
    }
    
    for split in ['train', 'val', 'test']:
        print(f"\n📁 Processing {split} split...")
        
        split1_dir = os.path.join(dataset1_dir, split)
        split2_dir = os.path.join(dataset2_dir, split)
        output_split_dir = os.path.join(output_dir, split)
        
        # Count samples in each dataset
        count1 = count_samples_in_split(split1_dir)
        count2 = count_samples_in_split(split2_dir)
        
        print(f"  📊 Dataset 1: {count1} samples")
        print(f"  📊 Dataset 2: {count2} samples")
        
        combined_count = 0
        
        # Copy files from dataset 1
        if os.path.exists(split1_dir):
            for folder in ['masks', 'query', 'support']:
                src_folder = os.path.join(split1_dir, folder)
                dst_folder = os.path.join(output_split_dir, folder)
                
                if os.path.exists(src_folder):
                    for file in os.listdir(src_folder):
                        if file.endswith(('.jpg', '.png')):
                            src_path = os.path.join(src_folder, file)
                            dst_path = os.path.join(dst_folder, f"d1_{file}")
                            shutil.copy2(src_path, dst_path)
                            combined_count += 1
        
        # Copy files from dataset 2
        if os.path.exists(split2_dir):
            for folder in ['masks', 'query', 'support']:
                src_folder = os.path.join(split2_dir, folder)
                dst_folder = os.path.join(output_split_dir, folder)
                
                if os.path.exists(src_folder):
                    for file in os.listdir(src_folder):
                        if file.endswith(('.jpg', '.png')):
                            src_path = os.path.join(src_folder, file)
                            dst_path = os.path.join(dst_folder, f"d2_{file}")
                            shutil.copy2(src_path, dst_path)
                            combined_count += 1
        
        combined_info['splits'][split] = combined_count
        combined_info['total_samples'] += combined_count
        
        print(f"  ✅ Combined: {combined_count} samples")
    
    # Save combined dataset info
    info_path = os.path.join(output_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(combined_info, f, indent=2)
    
    # Create detailed summary
    summary = {
        'combined_run_name': run_name,
        'combination_date': datetime.now().isoformat(),
        'dataset1_info': info1,
        'dataset2_info': info2,
        'combined_info': combined_info,
        'total_combined_samples': combined_info['total_samples']
    }
    
    summary_path = os.path.join(output_dir, 'combination_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 Dataset combination completed!")
    print(f"📁 Combined dataset: {output_dir}")
    print(f"📊 Total samples: {combined_info['total_samples']}")
    print(f"📄 Dataset info: {info_path}")
    print(f"📄 Combination summary: {summary_path}")
    
    return output_dir, combined_info


def main():
    """Main function to combine datasets."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python combine_datasets.py <dataset1_dir> <dataset2_dir>")
        print("\nExample:")
        print("  python combine_datasets.py few_shot_datasets_cleaned/few_shot_run_123 data/train_val_test_prepared_for_training")
        return
    
    # Configuration
    dataset1_dir = sys.argv[1]
    dataset2_dir = sys.argv[2]
    
    # Check if datasets exist
    if not os.path.exists(dataset1_dir):
        print(f"❌ Dataset 1 not found: {dataset1_dir}")
        return
    
    if not os.path.exists(dataset2_dir):
        print(f"❌ Dataset 2 not found: {dataset2_dir}")
        return
    
    # Combine datasets
    output_dir, combined_info = combine_datasets(
        dataset1_dir=dataset1_dir,
        dataset2_dir=dataset2_dir,
        output_base_dir='combined_datasets'
    )
    
    print(f"\n📊 Final dataset summary:")
    for split, count in combined_info['splits'].items():
        print(f"   - {split}: {count} samples")
    
    print(f"\n🚀 Ready for training!")
    print(f"Run: python train_few_shot_v2.py {output_dir} 100")


if __name__ == "__main__":
    main() 