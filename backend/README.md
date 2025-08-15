# Real-Time Crowd Detection - Backend

This directory contains the core backend implementation for the real-time crowd detection system.

## 📁 Directory Structure

### Core Scripts
- **`advanced_crowd_detection.py`** - Main YOLOv8-based crowd detection system (29KB)
- **`object_tracker.py`** - Original YOLOv4 + DeepSORT tracking system (15KB)
- **`export_crowd_model.py`** - Model export script for web deployment (22KB)
- **`test_web_export.py`** - Test script for exported models (6KB)

### Core Modules
- **`core/`** - YOLOv4 model implementation and utilities
  - `yolov4.py` - YOLOv4 model architecture
  - `utils.py` - Utility functions
  - `config.py` - Configuration settings
  - `backbone.py`, `common.py`, `dataset.py` - Model components

- **`deep_sort/`** - DeepSORT tracking implementation
  - `tracker.py` - Main tracker class
  - `detection.py` - Detection handling
  - `kalman_filter.py` - Kalman filter for tracking
  - Other tracking utilities

- **`logic/`** - Analysis and clustering logic
  - `analyze.py` - Frame analysis and clustering
  - `model.py` - Model definitions
  - `util.py` - Utility functions

- **`tools/`** - Utility tools
  - `generate_detections.py` - Detection generation
  - `freeze_model.py` - Model freezing utilities

### Data & Models
- **`data/`** - Input data and configuration
  - `video/` - Test video files
  - `classes/` - Class definition files
  - `yolov4.cfg`, `yolov4.weights` - YOLOv4 configuration and weights

- **`model_data/`** - Model data files
  - `mars-small128.pb` - DeepSORT feature extraction model

- **`yolov8m.pt`** - YOLOv8 medium model weights

### Export & Output
- **`exported_models/`** - Previously exported models (legacy)
- **`web_export_models/`** - Web-ready exported models
  - `crowd_detection_model.onnx` - ONNX model for web deployment
  - `crowd_detection_model.torchscript` - TorchScript model
  - `web_integration_example.html` - Web integration example
  - `python_inference_example.py` - Python inference example
  - `README.md` - Web deployment documentation

- **`outputs/`** - Generated output videos
- **`checkpoints/`** - Model checkpoints (empty)

### Configuration
- **`requirements.txt`** - Python dependencies

### Documentation
- **`Report/`** - Project report and documentation
  - LaTeX source files and assets
  - Project documentation and figures

## 🚀 Usage

### Main Crowd Detection (YOLOv8)
```bash
python advanced_crowd_detection.py --video data/video/test_1.mp4 --output outputs/result.mp4
```

### Legacy Tracking (YOLOv4 + DeepSORT)
```bash
python object_tracker.py --video data/video/test_1.mp4 --output outputs/tracked.mp4
```

### Export Model for Web
```bash
python export_crowd_model.py --model yolov8m.pt --output-dir web_export_models
```

### Test Exported Model
```bash
python test_web_export.py
```

## 📊 Model Specifications

### YOLOv8 Model (Primary)
- **File**: `yolov8m.pt` (49.7 MB)
- **Architecture**: YOLOv8 Medium
- **Input Size**: 640x640
- **Classes**: 80 COCO classes (Person = class 0)
- **Performance**: High accuracy, real-time inference

### YOLOv4 Model (Legacy)
- **File**: `data/yolov4.weights`
- **Architecture**: YOLOv4
- **Input Size**: 416x416
- **Used with**: DeepSORT tracking

## 🔧 Dependencies

Key dependencies (see `requirements.txt` for complete list):
- `ultralytics` - YOLOv8 implementation
- `opencv-python` - Computer vision operations
- `tensorflow` - YOLOv4 model (legacy)
- `numpy` - Numerical operations
- `scikit-learn` - Clustering (DBSCAN)

## 📈 Performance

- **YOLOv8 System**: ~5-7 FPS on CPU, higher on GPU
- **YOLOv4 + DeepSORT**: ~3-5 FPS on CPU
- **Memory Usage**: ~2-4 GB RAM
- **Accuracy**: High precision person detection and tracking

## 🌐 Web Deployment

The `web_export_models/` directory contains everything needed for web deployment:
- ONNX model for browser inference
- JavaScript integration examples
- Python reference implementation
- Complete documentation

## 📝 Notes

- The system has been cleaned up to remove duplicate and experimental files
- Focus is on the YOLOv8-based implementation (`advanced_crowd_detection.py`)
- Legacy YOLOv4 system is kept for reference (`object_tracker.py`)
- All export functionality is consolidated in `export_crowd_model.py`

## 🔄 Recent Changes

- Removed 17 duplicate/experimental Python files
- Consolidated export functionality
- Cleaned up directory structure
- Added comprehensive web export capabilities
- Improved documentation and organization
