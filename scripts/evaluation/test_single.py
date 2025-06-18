#!/usr/bin/env python3
"""
Simple Single Checkpoint Test
Tests one checkpoint to verify result saving functionality.
"""

import os
import sys
import subprocess

def test_single_checkpoint():
    """Test a single checkpoint and verify results are saved."""
    print("Testing Single Checkpoint")
    print("=" * 40)
    
    # Test parameters
    checkpoint_path = "models/checkpoints/checkpoint_epoch_50.pt"
    test_dataset_path = "data/datasets/combined_datasets/combined_run_20250618_005343/test"
    
    # Check if files exist
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return False
        
    if not os.path.exists(test_dataset_path):
        print(f"Test dataset not found: {test_dataset_path}")
        return False
    
    print(f"Using Python executable: {sys.executable}")
    print(f"Testing checkpoint: {checkpoint_path}")
    print(f"Test dataset: {test_dataset_path}")
    
    # Run the test
    try:
        cmd = [
            sys.executable, "scripts/evaluation/test_model_fixed.py",
            checkpoint_path, test_dataset_path
        ]
        
        print(f"\nRunning command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        print(f"\nReturn code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        if result.returncode == 0:
            print("\nTest completed successfully!")
            
            # Check if results were saved
            results_dir = "results/test_results"
            if os.path.exists(results_dir):
                print(f"Results directory created: {results_dir}")
                
                # Check for specific files
                files_to_check = [
                    "test_results.json",
                    "visualizations/test_samples.png",
                    "visualizations/metrics_summary.png"
                ]
                
                for file_path in files_to_check:
                    full_path = os.path.join(results_dir, file_path)
                    if os.path.exists(full_path):
                        size = os.path.getsize(full_path)
                        print(f"✓ {file_path} exists ({size} bytes)")
                    else:
                        print(f"✗ {file_path} missing")
            else:
                print(f"Results directory not created: {results_dir}")
            
            return True
        else:
            print(f"\nTest failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"Error running test: {e}")
        return False

if __name__ == "__main__":
    success = test_single_checkpoint()
    if success:
        print("\n✓ Single checkpoint test completed successfully!")
    else:
        print("\n✗ Single checkpoint test failed!") 