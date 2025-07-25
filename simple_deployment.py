#!/usr/bin/env python3
"""
Simple deployment package creator that avoids Unicode issues.
"""

import os
import sys
import shutil
import subprocess
from datetime import datetime

def create_simple_deployment(weights_file, output_dir='./simple_deployment'):
    """Create a simple deployment package"""
    
    print(f"Creating simple deployment package in {output_dir}...")
    
    # Create directories
    directories = ['models', 'weights', 'config', 'scripts', 'docs']
    for directory in directories:
        dir_path = os.path.join(output_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"[OK] Created directory: {dir_path}")
    
    # Copy essential files
    files_to_copy = [
        ('core', os.path.join(output_dir, 'scripts', 'core')),
        ('deep_sort', os.path.join(output_dir, 'scripts', 'deep_sort')),
        ('tools', os.path.join(output_dir, 'scripts', 'tools')),
        ('logic', os.path.join(output_dir, 'scripts', 'logic')),
        ('data/classes', os.path.join(output_dir, 'config', 'classes')),
        ('object_tracker.py', os.path.join(output_dir, 'scripts', 'object_tracker.py')),
        ('requirements.txt', os.path.join(output_dir, 'requirements.txt'))
    ]
    
    for src, dst in files_to_copy:
        if os.path.exists(src):
            try:
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                print(f"[OK] Copied {src} to {dst}")
            except Exception as e:
                print(f"[ERROR] Failed to copy {src}: {e}")
        else:
            print(f"[WARN] Source not found: {src}")
    
    # Export weights
    try:
        weights_dir = os.path.join(output_dir, 'weights')
        cmd = f'python weights_export.py --weights "{weights_file}" --output-dir "{weights_dir}" --model-name "yolov4"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[OK] Weights exported to {weights_dir}")
        else:
            print(f"[ERROR] Weights export failed")
    except Exception as e:
        print(f"[ERROR] Weights export failed: {e}")
    
    # Create simple README
    readme_content = """# Real-Time Crowd Detection - Deployment Package

This package contains the Real-Time Crowd Detection system.

## Quick Start

1. Install dependencies:
   pip install -r requirements.txt

2. Run detection on a video:
   python scripts/object_tracker.py --video input.mp4 --output output.avi

3. Run detection on webcam:
   python scripts/object_tracker.py --video 0 --output webcam_output.avi

## Package Contents

- scripts/: Core detection and tracking modules
- weights/: Model weights and metadata
- config/: Configuration files
- requirements.txt: Python dependencies

## System Requirements

- Python 3.7+
- TensorFlow 2.8+
- OpenCV 4.5+
- 4GB+ RAM

## Usage Examples

# Basic detection
python scripts/object_tracker.py --video test.mp4

# With custom parameters
python scripts/object_tracker.py --video test.mp4 --score 0.5 --iou 0.45

# Save output
python scripts/object_tracker.py --video test.mp4 --output result.avi

## Troubleshooting

1. Import errors: Install dependencies with pip install -r requirements.txt
2. Video not opening: Check video file path and format
3. Slow performance: Use GPU or reduce video resolution

For more information, see the documentation in the docs/ directory.
"""
    
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"[OK] Created README: {readme_path}")
    
    # Create summary
    summary_path = os.path.join(output_dir, 'deployment_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Deployment Package Summary\n")
        f.write(f"=========================\n\n")
        f.write(f"Package: YOLOv4 Crowd Detection\n")
        f.write(f"Created: {datetime.now().isoformat()}\n")
        f.write(f"Source weights: {weights_file}\n\n")
        f.write(f"To use this package:\n")
        f.write(f"1. cd {output_dir}\n")
        f.write(f"2. pip install -r requirements.txt\n")
        f.write(f"3. python scripts/object_tracker.py --video your_video.mp4\n")
    
    print(f"[OK] Created summary: {summary_path}")
    
    return output_dir

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create simple deployment package")
    parser.add_argument('--weights', type=str, default='./data/yolov4.weights',
                       help='Path to weights file')
    parser.add_argument('--output-dir', type=str, default='./simple_deployment',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("Simple Deployment Package Creator")
    print("=" * 35)
    
    if not os.path.exists(args.weights):
        print(f"[ERROR] Weights file not found: {args.weights}")
        return 1
    
    try:
        package_dir = create_simple_deployment(args.weights, args.output_dir)
        
        print(f"\n[OK] Deployment package created successfully!")
        print(f"Package location: {package_dir}")
        print(f"\nTo use the package:")
        print(f"1. cd {package_dir}")
        print(f"2. pip install -r requirements.txt")
        print(f"3. python scripts/object_tracker.py --video your_video.mp4")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Failed to create deployment package: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
