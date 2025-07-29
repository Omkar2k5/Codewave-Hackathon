# Real-Time Crowd Detection - Deployment Package

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
