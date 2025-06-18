#!/usr/bin/env python3
"""
Run Siamese UNet Training
Simple script to execute the comprehensive Siamese UNet training.
"""

import os
import sys

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts', 'training'))

from train_siamese_unet import main

if __name__ == "__main__":
    print("Starting Siamese UNet Training...")
    print("=" * 50)
    
    # Run the training
    main()
    
    print("=" * 50)
    print("Training completed!") 