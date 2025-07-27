#!/usr/bin/env python3
"""
Working model export script that creates a deployable model package.
This script creates a complete deployment package without running into TensorFlow compatibility issues.
"""

import os
import sys
import json
import shutil
from datetime import datetime
import subprocess

def create_deployment_structure(output_dir):
    """Create deployment directory structure"""
    
    directories = [
        'models',
        'weights', 
        'config',
        'scripts',
        'docs'
    ]
    
    for directory in directories:
        dir_path = os.path.join(output_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"[OK] Created directory: {dir_path}")

    return directories

def copy_essential_files(output_dir):
    """Copy essential files for deployment"""

    files_to_copy = [
        # Core modules
        ('core', os.path.join(output_dir, 'scripts', 'core')),
        ('deep_sort', os.path.join(output_dir, 'scripts', 'deep_sort')),
        ('tools', os.path.join(output_dir, 'scripts', 'tools')),
        ('logic', os.path.join(output_dir, 'scripts', 'logic')),

        # Configuration
        ('data/classes', os.path.join(output_dir, 'config', 'classes')),

        # Main scripts
        ('object_tracker.py', os.path.join(output_dir, 'scripts', 'object_tracker.py')),
        ('weights_export.py', os.path.join(output_dir, 'scripts', 'weights_export.py')),

        # Documentation
        ('MODEL_EXPORT_README.md', os.path.join(output_dir, 'docs', 'MODEL_EXPORT_README.md')),
        ('requirements.txt', os.path.join(output_dir, 'requirements.txt'))
    ]

    copied_files = []

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

                copied_files.append(dst)
                print(f"[OK] Copied {src} to {dst}")
            except Exception as e:
                print(f"[ERROR] Failed to copy {src}: {e}")
        else:
            print(f"[WARN] Source not found: {src}")
    
    return copied_files

def export_weights_to_deployment(weights_file, output_dir, model_name):
    """Export weights to deployment package"""
    
    weights_dir = os.path.join(output_dir, 'weights')
    
    try:
        # Use our weights export script
        cmd = f'python weights_export.py --weights "{weights_file}" --output-dir "{weights_dir}" --model-name "{model_name}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[OK] Weights exported to {weights_dir}")
            return True
        else:
            print(f"[ERROR] Weights export failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"[ERROR] Weights export failed: {e}")
        return False

def create_deployment_scripts(output_dir):
    """Create deployment and usage scripts"""
    
    scripts = []
    
    # 1. Installation script
    install_script = f'''#!/usr/bin/env python3
"""
Installation script for Real-Time Crowd Detection deployment.
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[OK] Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install dependencies: {{e}}")
        return False

def verify_installation():
    """Verify installation"""
    modules = ["tensorflow", "cv2", "numpy", "easydict"]

    print("Verifying installation...")
    for module in modules:
        try:
            __import__(module)
            print(f"[OK] {{module}}")
        except ImportError:
            print(f"[ERROR] {{module}} not found")
            return False

    print("[OK] All modules verified")
    return True

if __name__ == "__main__":
    print("Real-Time Crowd Detection - Installation")
    print("=" * 40)

    if install_dependencies() and verify_installation():
        print("\\n[OK] Installation completed successfully!")
        print("\\nNext steps:")
        print("1. Run: python scripts/object_tracker.py --help")
        print("2. Test with: python run_detection.py")
    else:
        print("\\n[ERROR] Installation failed")
        sys.exit(1)
'''
    
    install_path = os.path.join(output_dir, 'install.py')
    with open(install_path, 'w') as f:
        f.write(install_script)
    scripts.append(install_path)
    print(f"[OK] Created installation script: {install_path}")

    # 2. Detection runner script
    detection_script = f'''#!/usr/bin/env python3
"""
Simple detection runner for the crowd detection system.
"""

import os
import sys
import subprocess

def run_detection(video_path=None, output_path=None):
    """Run crowd detection"""
    
    # Default paths
    if video_path is None:
        video_path = input("Enter video path (or press Enter for webcam): ").strip()
        if not video_path:
            video_path = "0"  # Webcam
    
    if output_path is None:
        output_path = "./output_detection.avi"
    
    # Check if we have a saved model or use weights
    model_path = None
    if os.path.exists("models"):
        # Look for saved models
        for file in os.listdir("models"):
            if file.endswith("_savedmodel") or file.endswith(".h5"):
                model_path = os.path.join("models", file)
                break
    
    if not model_path:
        # Use weights if available
        if os.path.exists("weights"):
            for file in os.listdir("weights"):
                if file.endswith(".weights"):
                    model_path = os.path.join("weights", file)
                    break
    
    if not model_path:
        print("[ERROR] No model or weights found. Please ensure you have:")
        print("  - A saved model in the 'models' directory, or")
        print("  - Weight files in the 'weights' directory")
        return False

    # Run detection
    cmd = [
        sys.executable,
        "scripts/object_tracker.py",
        "--video", video_path,
        "--output", output_path
    ]

    if model_path.endswith(".weights"):
        cmd.extend(["--weights", model_path])
    else:
        cmd.extend(["--weights", model_path])

    print(f"Running detection...")
    print(f"Command: {{' '.join(cmd)}}")

    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] Detection completed! Output saved to {{output_path}}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Detection failed: {{e}}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run crowd detection")
    parser.add_argument("--video", type=str, help="Input video path (default: webcam)")
    parser.add_argument("--output", type=str, default="./output_detection.avi", 
                       help="Output video path")
    
    args = parser.parse_args()
    
    print("Real-Time Crowd Detection - Runner")
    print("=" * 35)
    
    success = run_detection(args.video, args.output)
    sys.exit(0 if success else 1)
'''
    
    detection_path = os.path.join(output_dir, 'run_detection.py')
    with open(detection_path, 'w') as f:
        f.write(detection_script)
    scripts.append(detection_path)
    print(f"[OK] Created detection runner: {detection_path}")

    return scripts

def create_deployment_readme(output_dir, model_info):
    """Create deployment README"""
    
    readme_content = f'''# Real-Time Crowd Detection - Deployment Package

This package contains a complete deployment of the Real-Time Crowd Detection system.

## Package Contents

- `models/` - Exported model files
- `weights/` - Model weights and metadata  
- `config/` - Configuration files and class definitions
- `scripts/` - Core detection and tracking modules
- `docs/` - Documentation
- `requirements.txt` - Python dependencies

## Quick Start

### 1. Install Dependencies
```bash
python install.py
```

### 2. Run Detection
```bash
# On a video file
python run_detection.py --video path/to/your/video.mp4

# On webcam (default)
python run_detection.py
```

### 3. Advanced Usage
```bash
# Use specific model
python scripts/object_tracker.py --weights models/yolov4_model --video input.mp4 --output output.avi

# Adjust detection parameters
python scripts/object_tracker.py --weights weights/yolov4.weights --score 0.5 --iou 0.45 --video input.mp4
```

## Model Information

- **Model Type**: {model_info.get('model_type', 'YOLOv4')}
- **Input Size**: {model_info.get('input_size', '416x416')}
- **Classes**: {model_info.get('num_classes', '80')} (COCO dataset)
- **Export Date**: {model_info.get('export_date', datetime.now().isoformat())}

## System Requirements

- Python 3.7+
- TensorFlow 2.8+
- OpenCV 4.5+
- 4GB+ RAM
- GPU recommended for real-time performance

## Troubleshooting

### Common Issues

1. **Import errors**: Run `python install.py` to install dependencies
2. **Model not found**: Ensure weights are in the `weights/` directory
3. **Video not opening**: Check video file path and format
4. **Slow performance**: Consider using GPU or smaller input size

### Performance Tips

- Use GPU acceleration if available
- Reduce input video resolution for faster processing
- Use YOLOv4-tiny for real-time applications
- Close other applications to free up memory

## File Structure

```
deployment_package/
├── install.py                 # Installation script
├── run_detection.py          # Simple detection runner
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── models/                   # Exported models
├── weights/                  # Model weights
│   ├── yolov4_weights.npy   # Weights as numpy array
│   ├── yolov4_metadata.json # Weight metadata
│   └── yolov4_weight_loader.py # Weight loading utility
├── config/                   # Configuration
│   └── classes/             # Class definitions
├── scripts/                  # Core modules
│   ├── core/                # YOLO implementation
│   ├── deep_sort/           # Tracking algorithms
│   ├── tools/               # Utilities
│   ├── logic/               # Analysis logic
│   └── object_tracker.py    # Main detection script
└── docs/                     # Documentation
    └── MODEL_EXPORT_README.md
```

## Support

For issues and questions:
1. Check the documentation in `docs/`
2. Review the troubleshooting section above
3. Ensure all dependencies are installed correctly

## License

This deployment package contains the Real-Time Crowd Detection system.
Please refer to the original project for licensing information.
'''
    
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"[OK] Created deployment README: {readme_path}")
    return readme_path

def create_deployment_package(weights_file, output_dir='./deployment_package', model_name='yolov4'):
    """Create complete deployment package"""
    
    print(f"Creating deployment package in {output_dir}...")
    
    # Create directory structure
    directories = create_deployment_structure(output_dir)
    
    # Copy essential files
    copied_files = copy_essential_files(output_dir)
    
    # Export weights
    weights_success = export_weights_to_deployment(weights_file, output_dir, model_name)
    
    # Create deployment scripts
    scripts = create_deployment_scripts(output_dir)
    
    # Create README
    model_info = {
        'model_type': model_name,
        'input_size': '416x416',
        'num_classes': 80,
        'export_date': datetime.now().isoformat()
    }
    readme = create_deployment_readme(output_dir, model_info)
    
    # Create deployment summary
    summary_path = os.path.join(output_dir, 'deployment_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Deployment Package Summary\n")
        f.write(f"=========================\n\n")
        f.write(f"Package: {model_name} Crowd Detection\n")
        f.write(f"Created: {datetime.now().isoformat()}\n")
        f.write(f"Source weights: {weights_file}\n\n")
        f.write(f"Package contents:\n")
        f.write(f"- Directories: {len(directories)}\n")
        f.write(f"- Copied files: {len(copied_files)}\n")
        f.write(f"- Scripts: {len(scripts)}\n")
        f.write(f"- Weights exported: {'Yes' if weights_success else 'No'}\n\n")
        f.write(f"To use this package:\n")
        f.write(f"1. cd {output_dir}\n")
        f.write(f"2. python install.py\n")
        f.write(f"3. python run_detection.py\n")

    print(f"[OK] Created deployment summary: {summary_path}")

    return output_dir

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create deployment package")
    parser.add_argument('--weights', type=str, default='./data/yolov4.weights',
                       help='Path to weights file')
    parser.add_argument('--output-dir', type=str, default='./deployment_package',
                       help='Output directory')
    parser.add_argument('--model-name', type=str, default='yolov4',
                       help='Model name')
    
    args = parser.parse_args()
    
    print("Real-Time Crowd Detection - Deployment Package Creator")
    print("=" * 55)
    
    if not os.path.exists(args.weights):
        print(f"[ERROR] Weights file not found: {args.weights}")
        print("Please download weights first: python download_weights.py --model yolov4")
        return 1

    try:
        package_dir = create_deployment_package(args.weights, args.output_dir, args.model_name)

        print(f"\n[OK] Deployment package created successfully!")
        print(f"Package location: {package_dir}")
        print(f"\nTo use the package:")
        print(f"1. cd {package_dir}")
        print(f"2. python install.py")
        print(f"3. python run_detection.py")

        return 0

    except Exception as e:
        print(f"\n[ERROR] Failed to create deployment package: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
