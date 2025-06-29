#!/usr/bin/env python3
"""
Test script for Dataset Curator
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from curator import DatasetCurator

def test_curator_setup():
    """Test basic curator setup."""
    print("\n🔧 Testing curator setup...")
    
    # Test with default paths
    try:
        curator = DatasetCurator()
        print("✅ Curator initialized successfully")
        print(f"   - Input directory: {curator.input_data_dir}")
        print(f"   - Output directory: {curator.output_dataset_dir}")
        print(f"   - Available images: {len(curator.available_images)}")
        print(f"   - Already accepted: {len(curator.accepted_images)}")
        print(f"   - Target count: {curator.target_count}")
        return curator
    except Exception as e:
        print(f"❌ Error initializing curator: {e}")
        return None

def test_directory_structure():
    """Test directory structure."""
    print("\n📁 Testing directory structure...")
    
    # Check required directories
    data_dir = Path("../data")
    if not data_dir.exists():
        print(f"⚠️  Data directory missing: {data_dir}")
        print(f"   Copy some detection results to test the curator.")
        return False
    else:
        print(f"✅ Data directory exists: {data_dir}")
    
    # Check for detection results
    detection_dirs = [
        data_dir / "result_detection_yolo",
        data_dir / "result_detection_yolo_full"
    ]
    
    found_detection_data = False
    for dir_path in detection_dirs:
        if dir_path.exists():
            video_dirs = [d for d in dir_path.iterdir() if d.is_dir() and d.name.startswith("video_")]
            if video_dirs:
                print(f"✅ Found detection results: {dir_path} ({len(video_dirs)} video dirs)")
                found_detection_data = True
            else:
                print(f"⚠️  No video directories in: {dir_path}")
        else:
            print(f"⚠️  Detection directory missing: {dir_path}")
    
    if not found_detection_data:
        print("⚠️  No detection results found.")
        print("   Run ball detection scripts first to generate data.")
    
    return found_detection_data

def test_curator_basic():
    """Test basic curator functionality."""
    print("\n🔧 Testing curator functionality...")
    
    # Initialize curator
    curator = DatasetCurator(
        input_data_dir="../data/result_detection_yolo_full",
        output_dataset_dir="../data/test_curated_dataset",
        target_count=10  # Small target for testing
    )
    
    print(f"📊 Found {len(curator.available_images)} images")
    print(f"✅ Already accepted: {len(curator.accepted_images)} images")
    
    # Test getting stats
    stats = curator.get_stats()
    print(f"📈 Stats: {stats}")
    
    # Test getting current image
    current = curator.get_current_image()
    if current:
        print(f"🖼️  Current image: {current['filename']}")
        print(f"   Bboxes: {len(current['bboxes'])}")
    else:
        print("⚠️  No current image available")
    
    return curator

def main():
    """Main test function."""
    print("🧪 Dataset Curator Test")
    print("=" * 50)
    
    # Test 1: Directory structure
    print("\n1️⃣  Testing directory structure...")
    has_data = test_directory_structure()
    
    # Test 2: Curator setup
    print("\n2️⃣  Testing curator setup...")
    curator = test_curator_setup()
    
    if not curator:
        print("\n❌ Cannot continue tests - curator setup failed")
        return
    
    # Test 3: Basic functionality
    if has_data:
        print("\n3️⃣  Testing basic functionality...")
        test_curator_basic()
    else:
        print("\n⚠️  Skipping functionality tests - no data available")
    
    print("\n" + "=" * 50)
    print("🚀 Ready to run curator!")
    print("   Run: python run_curator.py")

if __name__ == "__main__":
    main() 