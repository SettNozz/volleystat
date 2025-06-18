#!/usr/bin/env python3
"""
Fix Combined Dataset
Removes orphaned files and ensures all samples have complete sets.
"""

import os
import shutil
import json
from datetime import datetime


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


def fix_dataset(dataset_dir, backup=True):
    """Fix dataset by removing orphaned files and ensuring complete sets."""
    print(f"🔧 Fixing dataset: {dataset_dir}")
    
    if backup:
        # Create backup
        backup_dir = f"{dataset_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"📦 Creating backup: {backup_dir}")
        shutil.copytree(dataset_dir, backup_dir)
    
    total_fixed = 0
    total_removed = 0
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        print(f"\n📁 Processing {split} split...")
        
        query_dir = os.path.join(split_dir, 'query')
        support_dir = os.path.join(split_dir, 'support')
        masks_dir = os.path.join(split_dir, 'masks')
        
        if not all(os.path.exists(d) for d in [query_dir, support_dir, masks_dir]):
            print(f"  ❌ Missing directories in {split}")
            continue
        
        # Get valid samples
        valid_samples = get_valid_samples(split_dir)
        print(f"  📊 Valid samples: {len(valid_samples)}")
        
        # Get all files
        query_files = set(f.replace('.jpg', '') for f in os.listdir(query_dir) if f.endswith('.jpg'))
        support_files = set(f.replace('_support.jpg', '') for f in os.listdir(support_dir) if f.endswith('_support.jpg'))
        mask_files = set(f.replace('_mask.png', '') for f in os.listdir(masks_dir) if f.endswith('_mask.png'))
        
        # Find orphaned files
        orphaned_query = query_files - support_files - mask_files
        orphaned_support = support_files - query_files - mask_files
        orphaned_masks = mask_files - query_files - support_files
        
        # Remove orphaned files
        removed_count = 0
        
        for orphaned in orphaned_query:
            file_path = os.path.join(query_dir, f"{orphaned}.jpg")
            if os.path.exists(file_path):
                os.remove(file_path)
                removed_count += 1
        
        for orphaned in orphaned_support:
            file_path = os.path.join(support_dir, f"{orphaned}_support.jpg")
            if os.path.exists(file_path):
                os.remove(file_path)
                removed_count += 1
        
        for orphaned in orphaned_masks:
            file_path = os.path.join(masks_dir, f"{orphaned}_mask.png")
            if os.path.exists(file_path):
                os.remove(file_path)
                removed_count += 1
        
        print(f"  🗑️ Removed {removed_count} orphaned files")
        total_removed += removed_count
        total_fixed += len(valid_samples)
    
    print(f"\n✅ Dataset fixed!")
    print(f"📊 Total valid samples: {total_fixed}")
    print(f"🗑️ Total removed files: {total_removed}")
    
    return total_fixed, total_removed


def main():
    """Main function to fix dataset."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fix_combined_dataset.py <dataset_dir>")
        print("\nExample:")
        print("  python fix_combined_dataset.py combined_datasets/combined_run_20250618_005020")
        return
    
    # Configuration
    dataset_dir = sys.argv[1]
    
    # Check if dataset exists
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset not found: {dataset_dir}")
        return
    
    # Fix dataset
    total_fixed, total_removed = fix_dataset(dataset_dir, backup=True)
    
    print(f"\n🚀 Dataset is now ready for training!")
    print(f"Run: python train_few_shot_v2.py {dataset_dir} 100")


if __name__ == "__main__":
    main() 