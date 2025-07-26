#!/usr/bin/env python3
"""
Weight conversion script for yolov4
This script helps convert the exported weights back to TensorFlow format.
"""

import numpy as np
import json

def load_weights():
    """Load the exported weights"""
    weights = np.load('yolov4_weights.npy')
    
    with open('yolov4_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded weights:")
    print(f"  - Shape: {weights.shape}")
    print(f"  - Total parameters: {len(weights):,}")
    print(f"  - Original version: {metadata['header']['major']}.{metadata['header']['minor']}.{metadata['header']['revision']}")
    
    return weights, metadata

def get_weight_statistics():
    """Get statistics about the weights"""
    weights, metadata = load_weights()
    
    print(f"\nWeight statistics:")
    print(f"  - Min: {weights.min():.6f}")
    print(f"  - Max: {weights.max():.6f}")
    print(f"  - Mean: {weights.mean():.6f}")
    print(f"  - Std: {weights.std():.6f}")
    print(f"  - Non-zero: {np.count_nonzero(weights):,} / {len(weights):,} ({np.count_nonzero(weights)/len(weights)*100:.1f}%)")

if __name__ == '__main__':
    get_weight_statistics()
