#!/usr/bin/env python3
"""
Complete Pipeline Runner for Volleyball Ball Detection Dataset
This script runs the complete workflow:
1. Process all videos for ball detection
2. Clean up dataset after manual review
3. Export to Label Studio format
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.process_all_videos_for_dataset import VideoDatasetProcessor
from scripts.cleanup_after_manual_review import DatasetCleanup
from scripts.export_to_label_studio import LabelStudioExporter
from configs.config import YOLO_MODEL_PATH, MODEL_LOAD_PATH


def run_video_processing():
    """Step 1: Process all videos for ball detection."""
    print("=" * 60)
    print("STEP 1: VIDEO PROCESSING")
    print("=" * 60)
    
    input_folder = r"C:/Users/illya/Videos/video_for_sharing"
    output_base = "data/pipeline_result"
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"❌ Input folder does not exist: {input_folder}")
        print("Please make sure the folder exists and contains video subfolders.")
        return False
    
    print(f"📁 Input folder: {input_folder}")
    print(f"📁 Output folder: {output_base}")
    
    # Create processor and run
    processor = VideoDatasetProcessor(
        input_folder=input_folder,
        output_base=output_base,
        yolo_model_path=YOLO_MODEL_PATH if os.path.exists(YOLO_MODEL_PATH) else None,
        unet_model_path=MODEL_LOAD_PATH if os.path.exists(MODEL_LOAD_PATH) else None,
        device='auto'
    )
    
    processor.process_all_videos()
    return True


def run_cleanup():
    """Step 2: Clean up dataset after manual review."""
    print("\n" + "=" * 60)
    print("STEP 2: DATASET CLEANUP")
    print("=" * 60)
    
    pipeline_result_path = "data/pipeline_result"
    cleaned_dataset_path = "data/cleaned_dataset"
    
    if not os.path.exists(pipeline_result_path):
        print(f"❌ Pipeline result not found: {pipeline_result_path}")
        print("Please run video processing first.")
        return False
    
    print(f"📁 Pipeline result: {pipeline_result_path}")
    print(f"📁 Cleaned dataset: {cleaned_dataset_path}")
    
    # Create cleanup instance
    cleanup = DatasetCleanup(pipeline_result_path, cleaned_dataset_path)
    
    # Perform cleanup
    success = cleanup.run_cleanup()
    
    if success:
        print(f"📊 Cleanup completed successfully!")
        print(f"   Remaining ball crops: {cleanup.stats['remaining_ball_crops']}")
        print(f"   Cleaned YOLO images: {cleanup.stats['cleaned_yolo_images']}")
        print(f"   Cleaned YOLO labels: {cleanup.stats['cleaned_yolo_labels']}")
        return True
    else:
        print("❌ Cleanup failed!")
        return False


def run_label_studio_export():
    """Step 3: Export to Label Studio format."""
    print("\n" + "=" * 60)
    print("STEP 3: LABEL STUDIO EXPORT")
    print("=" * 60)
    
    cleaned_dataset_path = "data/cleaned_dataset"
    export_path = "data/label_studio_export/volleyball_ball_detection.json"
    
    if not os.path.exists(cleaned_dataset_path):
        print(f"❌ Cleaned dataset not found: {cleaned_dataset_path}")
        print("Please run cleanup first.")
        return False
    
    print(f"📁 Cleaned dataset: {cleaned_dataset_path}")
    print(f"📁 Export path: {export_path}")
    
    # Create exporter
    exporter = LabelStudioExporter(
        dataset_path=cleaned_dataset_path,
        output_path=export_path,
        project_name="volleyball_ball_detection"
    )
    
    # Export dataset
    exporter.export_dataset()
    exporter.create_import_instructions()
    
    return True


def main():
    """Main function to run the complete pipeline."""
    print("🏐 Complete Volleyball Ball Detection Pipeline")
    print("=" * 60)
    print("This pipeline will:")
    print("1. Process all videos for ball detection")
    print("2. Clean up dataset after manual review")
    print("3. Export to Label Studio format")
    print("=" * 60)
    
    # Check if we should run all steps or just specific ones
    import argparse
    parser = argparse.ArgumentParser(description="Complete volleyball ball detection pipeline")
    parser.add_argument("--step", "-s", 
                       choices=["1", "2", "3", "all"],
                       default="all",
                       help="Which step to run (1=video processing, 2=cleanup, 3=export, all=complete pipeline)")
    
    args = parser.parse_args()
    
    success = True
    
    if args.step in ["1", "all"]:
        success = run_video_processing()
        if not success:
            print("❌ Video processing failed. Stopping pipeline.")
            return
    
    if args.step in ["2", "all"] and success:
        success = run_cleanup()
        if not success:
            print("❌ Cleanup failed. Stopping pipeline.")
            return
    
    if args.step in ["3", "all"] and success:
        success = run_label_studio_export()
        if not success:
            print("❌ Label Studio export failed.")
            return
    
    if success:
        print("\n" + "=" * 60)
        print("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        print("=" * 60)
        print("📊 Results:")
        print("   📁 Video processing results: data/pipeline_result/")
        print("   📁 Cleaned dataset: data/cleaned_dataset/")
        print("   📁 Label Studio export: data/label_studio_export/")
        print("\n📋 Next steps:")
        print("   1. Manually review and delete poor quality detections")
        print("   2. Run cleanup script again if needed")
        print("   3. Import Label Studio JSON to your Label Studio instance")
        print("   4. Review and refine annotations in Label Studio")


if __name__ == "__main__":
    main() 