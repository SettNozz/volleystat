#!/usr/bin/env python3
"""
Test All Checkpoints Script
Tests all 25-epoch interval checkpoints and compares their performance.
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def test_checkpoint(checkpoint_path, test_dataset_path):
    """Test a single checkpoint and return results."""
    print(f"\nTesting checkpoint: {checkpoint_path}")
    
    try:
        # Run the test script using the current Python executable
        cmd = [
            sys.executable, "scripts/evaluation/test_model_fixed.py",
            checkpoint_path, test_dataset_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print(f"Test completed successfully")
            return True
        else:
            print(f"Test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error testing checkpoint: {e}")
        return False

def main():
    """Test all 25-epoch interval checkpoints."""
    print("Testing All 25-Epoch Interval Checkpoints")
    print("=" * 60)
    print(f"Using Python executable: {sys.executable}")
    
    # Define checkpoints to test (every 25 epochs)
    checkpoints = [
        "models/checkpoints/checkpoint_epoch_50.pt",
        "models/checkpoints/checkpoint_epoch_75.pt",  # if exists
        "models/checkpoints/checkpoint_epoch_100.pt",
        "models/checkpoints/checkpoint_epoch_125.pt",  # if exists
        "models/checkpoints/checkpoint_epoch_150.pt",
    ]
    
    # Test dataset path
    test_dataset_path = "data/datasets/combined_datasets/combined_run_20250618_005343/test"
    
    # Results tracking
    results = {
        'test_date': datetime.now().isoformat(),
        'checkpoints_tested': [],
        'successful_tests': [],
        'failed_tests': []
    }
    
    # Test each checkpoint
    for checkpoint in checkpoints:
        if os.path.exists(checkpoint):
            print(f"\nFound checkpoint: {checkpoint}")
            success = test_checkpoint(checkpoint, test_dataset_path)
            
            results['checkpoints_tested'].append(checkpoint)
            
            if success:
                results['successful_tests'].append(checkpoint)
            else:
                results['failed_tests'].append(checkpoint)
        else:
            print(f"Checkpoint not found: {checkpoint}")
    
    # Summary
    print(f"\nTest Summary:")
    print(f"  Total checkpoints tested: {len(results['checkpoints_tested'])}")
    print(f"  Successful tests: {len(results['successful_tests'])}")
    print(f"  Failed tests: {len(results['failed_tests'])}")
    
    # Save results
    results_path = "results/checkpoint_testing_summary.json"
    os.makedirs("results", exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    # Instructions for manual comparison
    print(f"\nNext Steps:")
    print(f"  1. Check the results in: results/test_results/")
    print(f"  2. Compare visualizations for each checkpoint")
    print(f"  3. Analyze metrics in the JSON files")
    print(f"  4. Look for the best performing checkpoint")

if __name__ == "__main__":
    main() 