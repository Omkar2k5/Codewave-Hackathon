#!/usr/bin/env python3
"""
Quick export script for Real-Time Crowd Detection models.
This script provides easy commands to export models with common configurations.
"""

import os
import sys
import subprocess
import argparse

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}")
    print("-" * len(description))
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def download_weights():
    """Download YOLOv4 weights if they don't exist"""
    weights_path = "./data/yolov4.weights"
    if os.path.exists(weights_path):
        print(f"✓ Weights already exist at {weights_path}")
        return True
    
    print("Downloading YOLOv4 weights...")
    return run_command("python download_weights.py --model yolov4", "Downloading YOLOv4 weights")

def export_yolov4_standard():
    """Export standard YOLOv4 model"""
    command = "python export_model.py --model yolov4 --input-size 416 --output-dir ./exported_models/yolov4_standard"
    return run_command(command, "Exporting YOLOv4 Standard Model")

def export_yolov4_tiny():
    """Export YOLOv4-tiny model"""
    # First download tiny weights
    if not os.path.exists("./data/yolov4-tiny.weights"):
        run_command("python download_weights.py --model yolov4-tiny", "Downloading YOLOv4-tiny weights")
    
    command = "python export_model.py --model yolov4 --tiny --input-size 416 --weights ./data/yolov4-tiny.weights --output-dir ./exported_models/yolov4_tiny"
    return run_command(command, "Exporting YOLOv4-tiny Model")

def export_for_mobile():
    """Export optimized models for mobile deployment"""
    commands = [
        ("python export_model.py --model yolov4 --tiny --input-size 320 --weights ./data/yolov4-tiny.weights --format tflite --output-dir ./exported_models/mobile", 
         "Exporting YOLOv4-tiny TensorFlow Lite (320x320) for mobile"),
        ("python export_model.py --model yolov4 --tiny --input-size 416 --weights ./data/yolov4-tiny.weights --format tflite --output-dir ./exported_models/mobile", 
         "Exporting YOLOv4-tiny TensorFlow Lite (416x416) for mobile")
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False
    
    return success

def export_for_production():
    """Export models optimized for production deployment"""
    commands = [
        ("python export_model.py --model yolov4 --input-size 416 --format savedmodel --output-dir ./exported_models/production", 
         "Exporting YOLOv4 SavedModel for production"),
        ("python export_model.py --model yolov4 --input-size 608 --format savedmodel --output-dir ./exported_models/production", 
         "Exporting YOLOv4 SavedModel (608x608) for high accuracy"),
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False
    
    return success

def export_all_variants():
    """Export all model variants"""
    print("Exporting all model variants...")
    
    # Download all weights first
    run_command("python download_weights.py --model all", "Downloading all model weights")
    
    # Export all variants
    variants = [
        ("python export_model.py --model yolov4 --input-size 416", "YOLOv4 Standard"),
        ("python export_model.py --model yolov4 --tiny --weights ./data/yolov4-tiny.weights", "YOLOv4-tiny"),
        ("python export_model.py --model yolov3 --weights ./data/yolov3.weights", "YOLOv3 Standard"),
        ("python export_model.py --model yolov3 --tiny --weights ./data/yolov3-tiny.weights", "YOLOv3-tiny"),
    ]
    
    success_count = 0
    for command, description in variants:
        full_command = f"{command} --output-dir ./exported_models/all_variants"
        if run_command(full_command, f"Exporting {description}"):
            success_count += 1
    
    print(f"\nExport Summary: {success_count}/{len(variants)} variants exported successfully")
    return success_count == len(variants)

def create_deployment_package():
    """Create a deployment package with all necessary files"""
    print("Creating deployment package...")
    
    # Create deployment directory structure
    deploy_dir = "./deployment_package"
    os.makedirs(f"{deploy_dir}/models", exist_ok=True)
    os.makedirs(f"{deploy_dir}/config", exist_ok=True)
    os.makedirs(f"{deploy_dir}/scripts", exist_ok=True)
    
    # Copy essential files
    files_to_copy = [
        ("./core", f"{deploy_dir}/core"),
        ("./data/classes", f"{deploy_dir}/config/classes"),
        ("./object_tracker.py", f"{deploy_dir}/scripts/"),
        ("./export_model.py", f"{deploy_dir}/scripts/"),
    ]
    
    for src, dst in files_to_copy:
        if os.path.exists(src):
            if os.path.isdir(src):
                run_command(f"xcopy /E /I /Y \"{src}\" \"{dst}\"", f"Copying {src} to {dst}")
            else:
                run_command(f"copy \"{src}\" \"{dst}\"", f"Copying {src} to {dst}")
    
    # Export a standard model to the deployment package
    run_command("python export_model.py --model yolov4 --tiny --weights ./data/yolov4-tiny.weights --format all --output-dir ./deployment_package/models", 
                "Exporting model for deployment package")
    
    # Create README for deployment
    readme_content = """# Real-Time Crowd Detection - Deployment Package

This package contains everything needed to deploy the crowd detection system.

## Contents:
- models/: Exported model files in various formats
- core/: Core detection and tracking modules  
- config/: Configuration files and class definitions
- scripts/: Utility scripts for running the system

## Quick Start:
1. Install dependencies: pip install tensorflow opencv-python numpy
2. Run detection: python scripts/object_tracker.py --weights models/yolov4_tiny_416_savedmodel --video your_video.mp4

## Model Files:
- SavedModel format: For TensorFlow serving and production deployment
- HDF5 format: For easy loading in Python applications
- TensorFlow Lite: For mobile and edge device deployment
- Weights only: For custom model architectures

For more information, see the main project documentation.
"""
    
    with open(f"{deploy_dir}/README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"✓ Deployment package created in {deploy_dir}")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Quick export for crowd detection models")
    parser.add_argument('action', choices=[
        'download', 'standard', 'tiny', 'mobile', 'production', 'all', 'deploy'
    ], help='Export action to perform')
    
    args = parser.parse_args()
    
    print("Real-Time Crowd Detection - Model Export Utility")
    print("=" * 50)
    
    success = False
    
    if args.action == 'download':
        success = download_weights()
    elif args.action == 'standard':
        download_weights()
        success = export_yolov4_standard()
    elif args.action == 'tiny':
        success = export_yolov4_tiny()
    elif args.action == 'mobile':
        success = export_for_mobile()
    elif args.action == 'production':
        download_weights()
        success = export_for_production()
    elif args.action == 'all':
        success = export_all_variants()
    elif args.action == 'deploy':
        download_weights()
        export_yolov4_tiny()
        success = create_deployment_package()
    
    if success:
        print("\n✓ Export completed successfully!")
        
        # Show next steps
        print("\nNext Steps:")
        if args.action in ['standard', 'tiny', 'production']:
            print("- Test the exported model with: python object_tracker.py --weights <exported_model_path>")
        elif args.action == 'mobile':
            print("- Deploy TensorFlow Lite models to mobile applications")
        elif args.action == 'deploy':
            print("- Use the deployment_package/ directory for production deployment")
        
        return 0
    else:
        print("\n✗ Export failed. Please check the error messages above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
