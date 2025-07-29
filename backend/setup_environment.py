#!/usr/bin/env python3
"""
Setup script for Real-Time Crowd Detection environment.
This script installs dependencies and sets up the environment.
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}")
    print("-" * len(description))
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print("✓ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"✗ Python {version.major}.{version.minor} is not supported. Please use Python 3.7 or higher.")
        return False
    else:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies from requirements.txt...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "./data",
        "./checkpoints", 
        "./exported_models",
        "./outputs"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"✗ Failed to create directory {directory}: {e}")
            return False
    
    return True

def test_imports():
    """Test if all required modules can be imported"""
    modules = [
        "tensorflow",
        "cv2", 
        "numpy",
        "easydict",
        "absl",
        "tqdm"
    ]
    
    print("\nTesting module imports...")
    failed_imports = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n✗ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✓ All modules imported successfully!")
        return True

def download_sample_weights():
    """Download sample weights for testing"""
    print("\nWould you like to download YOLOv4-tiny weights for testing? (y/n): ", end="")
    response = input().lower().strip()
    
    if response in ['y', 'yes']:
        return run_command("python download_weights.py --model yolov4-tiny", "Downloading YOLOv4-tiny weights")
    else:
        print("Skipping weight download. You can download them later using download_weights.py")
        return True

def main():
    """Main setup function"""
    print("Real-Time Crowd Detection - Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directories
    if not create_directories():
        print("✗ Failed to create directories")
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("✗ Failed to install dependencies")
        return 1
    
    # Test imports
    if not test_imports():
        print("✗ Module import test failed")
        return 1
    
    # Download sample weights
    download_sample_weights()
    
    print("\n" + "=" * 50)
    print("✓ Environment setup completed successfully!")
    print("\nNext steps:")
    print("1. Download model weights: python download_weights.py --model yolov4")
    print("2. Export models: python run_export.py standard")
    print("3. Test the system: python object_tracker.py --video ./data/video/test.mp4")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
