#!/usr/bin/env python3
"""
Check dataset statistics
"""

import json
from pathlib import Path

def check_dataset():
    """Check dataset statistics."""
    data_dir = Path("../data/result_detection_yolo")
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        print(f"   Try: ../data/result_detection_yolo_full")
        data_dir = Path("../data/result_detection_yolo_full")
        if not data_dir.exists():
            print(f"❌ Alternative directory also not found: {data_dir}")
            return
    
    print(f"📁 Checking dataset in: {data_dir}")
    print("=" * 60)
    
    total_images = 0
    total_annotations = 0
    
    # Find all video directories
    video_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("video_")]
    
    for video_dir in sorted(video_dirs):
        print(f"\n📹 {video_dir.name}")
        
        # Check annotations file
        annotations_file = video_dir / "annotations.json"
        if not annotations_file.exists():
            print(f"  ❌ No annotations.json found")
            continue
        
        # Count annotations
        try:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            video_annotations = len(annotations)
            total_annotations += video_annotations
            print(f"  📝 Annotations: {video_annotations}")
        except Exception as e:
            print(f"  ❌ Error reading annotations: {e}")
            continue
        
        # Check images directory
        images_dir = video_dir / "images"
        if not images_dir.exists():
            print(f"  ❌ No images directory found")
            continue
        
        # Count images
        image_files = list(images_dir.glob("*.jpg"))
        video_images = len(image_files)
        total_images += video_images
        print(f"  🖼️  Images: {video_images}")
        
        # Check if counts match
        if video_images == video_annotations:
            print(f"  ✅ Images and annotations match")
        else:
            print(f"  ⚠️  Mismatch: {video_images} images vs {video_annotations} annotations")
    
    print("\n" + "=" * 60)
    print(f"📊 TOTAL STATISTICS:")
    print(f"   • Video directories: {len(video_dirs)}")
    print(f"   • Total images: {total_images}")
    print(f"   • Total annotations: {total_annotations}")
    print(f"   • Average per video: {total_images / len(video_dirs) if video_dirs else 0:.1f}")
    
    if total_images > 0:
        print(f"\n🎯 Dataset ready for curation!")
        print(f"   • Target: 1000 images")
        print(f"   • Available: {total_images} images")
        print(f"   • Coverage: {min(100, total_images / 10):.1f}% of target")
        print(f"\n🚀 To start curation, run: python run_curator.py")
    else:
        print(f"\n❌ No images found for curation")

if __name__ == "__main__":
    check_dataset() 