#!/usr/bin/env python3
"""
Combine Two Few-Shot Datasets (Fixed Version)
Combines two datasets with proper file matching and validation.
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


def get_valid_samples(split_dir):
    """Get valid samples that have all required files (query, support, mask)."""
    query_dir = os.path.join(split_dir, 'query')
    support_dir = os.path.join(split_dir, 'support')
    masks_dir = os.path.join(split_dir, 'masks')
    
    if not all(os.path.exists(d) for d in [query_dir, support_dir, masks_dir]):
        return []
    
    valid_samples = []
    
    # Get all query images
    query_files = [f for f in os.listdir(query_dir) if f.endswith('.jpg')]
    
    for query_file in query_files:
        base_name = query_file.replace('.jpg', '')
        
        # Check if corresponding support and mask files exist
        support_file = f"{base_name}_support.jpg"
        mask_file = f"{base_name}_mask.png"
        
        support_path = os.path.join(support_dir, support_file)
        mask_path = os.path.join(masks_dir, mask_file)
        
        if os.path.exists(support_path) and os.path.exists(mask_path):
            valid_samples.append({
                'base_name': base_name,
                'query_file': query_file,
                'support_file': support_file,
                'mask_file': mask_file
            })
    
    return valid_samples


def combine_datasets_fixed(
    dataset1_dir,
    dataset2_dir,
    output_base_dir='combined_datasets_fixed',
    run_name=None
):
    """Combine two datasets with proper file validation."""
    
    # Generate run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"combined_run_{timestamp}"
    
    print(f"🔄 Combining datasets with validation...")
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
        
        # Get valid samples from each dataset
        samples1 = get_valid_samples(split1_dir)
        samples2 = get_valid_samples(split2_dir)
        
        print(f"  📊 Dataset 1: {len(samples1)} valid samples")
        print(f"  📊 Dataset 2: {len(samples2)} valid samples")
        
        combined_count = 0
        
        # Copy files from dataset 1
        for i, sample in enumerate(samples1):
            # Source paths
            query_src = os.path.join(split1_dir, 'query', sample['query_file'])
            support_src = os.path.join(split1_dir, 'support', sample['support_file'])
            mask_src = os.path.join(split1_dir, 'masks', sample['mask_file'])
            
            # Destination paths
            new_base_name = f"d1_{split}_{i+1:03d}"
            query_dst = os.path.join(output_split_dir, 'query', f"{new_base_name}.jpg")
            support_dst = os.path.join(output_split_dir, 'support', f"{new_base_name}_support.jpg")
            mask_dst = os.path.join(output_split_dir, 'masks', f"{new_base_name}_mask.png")
            
            # Copy files
            shutil.copy2(query_src, query_dst)
            shutil.copy2(support_src, support_dst)
            shutil.copy2(mask_src, mask_dst)
            combined_count += 1
        
        # Copy files from dataset 2
        for i, sample in enumerate(samples2):
            # Source paths
            query_src = os.path.join(split2_dir, 'query', sample['query_file'])
            support_src = os.path.join(split2_dir, 'support', sample['support_file'])
            mask_src = os.path.join(split2_dir, 'masks', sample['mask_file'])
            
            # Destination paths
            new_base_name = f"d2_{split}_{i+1:03d}"
            query_dst = os.path.join(output_split_dir, 'query', f"{new_base_name}.jpg")
            support_dst = os.path.join(output_split_dir, 'support', f"{new_base_name}_support.jpg")
            mask_dst = os.path.join(output_split_dir, 'masks', f"{new_base_name}_mask.png")
            
            # Copy files
            shutil.copy2(query_src, query_dst)
            shutil.copy2(support_src, support_dst)
            shutil.copy2(mask_src, mask_dst)
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


def validate_dataset(dataset_dir):
    """Validate that all files in the dataset exist and are properly matched."""
    print(f"🔍 Validating dataset: {dataset_dir}")
    
    total_valid = 0
    total_invalid = 0
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        valid_samples = get_valid_samples(split_dir)
        print(f"  📁 {split}: {len(valid_samples)} valid samples")
        total_valid += len(valid_samples)
        
        # Check for orphaned files
        query_dir = os.path.join(split_dir, 'query')
        support_dir = os.path.join(split_dir, 'support')
        masks_dir = os.path.join(split_dir, 'masks')
        
        if all(os.path.exists(d) for d in [query_dir, support_dir, masks_dir]):
            query_files = set(f.replace('.jpg', '') for f in os.listdir(query_dir) if f.endswith('.jpg'))
            support_files = set(f.replace('_support.jpg', '') for f in os.listdir(support_dir) if f.endswith('_support.jpg'))
            mask_files = set(f.replace('_mask.png', '') for f in os.listdir(masks_dir) if f.endswith('_mask.png'))
            
            orphaned = query_files - support_files - mask_files
            if orphaned:
                print(f"    ⚠️ {len(orphaned)} orphaned files in {split}")
                total_invalid += len(orphaned)
    
    print(f"📊 Validation complete: {total_valid} valid, {total_invalid} invalid")
    return total_valid, total_invalid


def main():
    """Main function to combine datasets."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python combine_datasets_fixed.py <dataset1_dir> <dataset2_dir>")
        print("\nExample:")
        print("  python combine_datasets_fixed.py few_shot_datasets_cleaned/few_shot_run_123 data/train_val_test_prepared_for_training")
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
    
    # Validate original datasets
    print("🔍 Validating original datasets...")
    validate_dataset(dataset1_dir)
    validate_dataset(dataset2_dir)
    
    # Combine datasets
    output_dir, combined_info = combine_datasets_fixed(
        dataset1_dir=dataset1_dir,
        dataset2_dir=dataset2_dir,
        output_base_dir='combined_datasets_fixed'
    )
    
    # Validate combined dataset
    print("\n🔍 Validating combined dataset...")
    validate_dataset(output_dir)
    
    print(f"\n📊 Final dataset summary:")
    for split, count in combined_info['splits'].items():
        print(f"   - {split}: {count} samples")
    
    print(f"\n🚀 Ready for training!")
    print(f"Run: python train_few_shot_v2.py {output_dir} 100")


if __name__ == "__main__":
    main() 