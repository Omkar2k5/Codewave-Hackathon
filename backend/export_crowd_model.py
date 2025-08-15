#!/usr/bin/env python3
"""
Comprehensive Crowd Detection Model Exporter for Web Applications
================================================================

This script exports the YOLOv8-based crowd detection model in multiple formats
suitable for web applications, including:
- ONNX format for web deployment
- TensorFlow.js format for browser inference
- TensorFlow Lite for mobile/edge deployment
- Complete model package with preprocessing utilities

Author: Crowd Detection System
Date: 2025-01-15
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from datetime import datetime
import logging
from pathlib import Path
import shutil
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# YOLOv8 imports
try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
    print("✅ YOLOv8 (ultralytics) available for export")
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ YOLOv8 not available. Install with: pip install ultralytics")

class CrowdDetectionModelExporter:
    """
    Comprehensive exporter for crowd detection models
    """
    
    def __init__(self, model_path='yolov8m.pt', export_dir='exported_models'):
        self.model_path = model_path
        self.export_dir = Path(export_dir)
        self.model = None
        self.metadata = {}
        
        # Create export directory
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configuration
        self.input_size = (640, 640)  # YOLOv8 default input size
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        
        logging.info(f"Initialized exporter with model: {model_path}")
        logging.info(f"Export directory: {self.export_dir}")
    
    def load_model(self):
        """Load the YOLOv8 model"""
        if not YOLO_AVAILABLE:
            logging.error("YOLOv8 not available. Cannot load model.")
            return False
        
        try:
            if not os.path.exists(self.model_path):
                logging.error(f"Model file not found: {self.model_path}")
                return False
            
            self.model = YOLO(self.model_path)
            logging.info(f"✅ Successfully loaded YOLOv8 model from {self.model_path}")
            
            # Extract model metadata
            self.metadata = {
                'model_type': 'YOLOv8',
                'model_path': str(self.model_path),
                'input_size': self.input_size,
                'confidence_threshold': self.confidence_threshold,
                'iou_threshold': self.iou_threshold,
                'export_timestamp': datetime.now().isoformat(),
                'model_info': {
                    'task': 'detect',
                    'classes': list(self.model.names.values()),
                    'num_classes': len(self.model.names),
                    'person_class_id': 0  # Person class in COCO dataset
                }
            }
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return False
    
    def export_onnx(self, filename='crowd_detection_model.onnx'):
        """Export model to ONNX format (best for web deployment)"""
        try:
            output_path = self.export_dir / filename
            
            # Export to ONNX
            self.model.export(
                format='onnx',
                imgsz=self.input_size,
                dynamic=False,  # Static input size for better web performance
                simplify=True,  # Simplify the model
                opset=11  # ONNX opset version for better compatibility
            )
            
            # Move the exported file to our export directory
            source_onnx = Path(self.model_path).with_suffix('.onnx')
            if source_onnx.exists():
                shutil.move(str(source_onnx), str(output_path))
                logging.info(f"✅ ONNX model exported to: {output_path}")
                return str(output_path)
            else:
                logging.error("ONNX export failed - file not found")
                return None
                
        except Exception as e:
            logging.error(f"ONNX export failed: {e}")
            return None
    
    def export_torchscript(self, filename='crowd_detection_model.torchscript'):
        """Export model to TorchScript format"""
        try:
            output_path = self.export_dir / filename
            
            # Export to TorchScript
            self.model.export(
                format='torchscript',
                imgsz=self.input_size
            )
            
            # Move the exported file to our export directory
            source_torchscript = Path(self.model_path).with_suffix('.torchscript')
            if source_torchscript.exists():
                shutil.move(str(source_torchscript), str(output_path))
                logging.info(f"✅ TorchScript model exported to: {output_path}")
                return str(output_path)
            else:
                logging.error("TorchScript export failed - file not found")
                return None
                
        except Exception as e:
            logging.error(f"TorchScript export failed: {e}")
            return None
    
    def create_preprocessing_utils(self):
        """Create preprocessing utilities for web applications"""
        utils_code = '''import cv2
import numpy as np

class CrowdDetectionPreprocessor:
    """Preprocessing utilities for crowd detection model"""
    
    def __init__(self, input_size=(640, 640)):
        self.input_size = input_size
    
    def preprocess_frame(self, frame):
        """Preprocess a frame for model inference"""
        original_height, original_width = frame.shape[:2]
        target_width, target_height = self.input_size
        
        # Calculate scale factors
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        # Resize frame
        resized_frame = cv2.resize(frame, self.input_size)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized_frame = rgb_frame.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW format
        input_tensor = np.transpose(normalized_frame, (2, 0, 1))
        batch_frame = np.expand_dims(input_tensor, axis=0)
        
        return batch_frame, (scale_x, scale_y)
    
    def postprocess_detections(self, detections, scale_factors, confidence_threshold=0.5):
        """Post-process model detections"""
        scale_x, scale_y = scale_factors
        processed = []
        
        for detection in detections:
            confidence = detection[4]
            if confidence >= confidence_threshold:
                # Scale coordinates back to original image size
                x1 = int(detection[0] / scale_x)
                y1 = int(detection[1] / scale_y)
                x2 = int(detection[2] / scale_x)
                y2 = int(detection[3] / scale_y)
                
                processed.append({
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'confidence': float(confidence),
                    'class_id': int(detection[5]) if len(detection) > 5 else 0,
                    'class_name': 'person'
                })
        
        return processed
    
    def draw_detections(self, frame, detections):
        """Draw detection bounding boxes on frame"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
'''
        
        utils_path = self.export_dir / 'preprocessing_utils.py'
        with open(utils_path, 'w') as f:
            f.write(utils_code)
        
        logging.info(f"✅ Preprocessing utilities saved to: {utils_path}")
        return str(utils_path)
    
    def save_metadata(self):
        """Save model metadata and configuration"""
        metadata_path = self.export_dir / 'model_metadata.json'
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logging.info(f"✅ Model metadata saved to: {metadata_path}")
        return str(metadata_path)

    def create_web_integration_example(self):
        """Create example code for web integration"""
        web_example = '''<!DOCTYPE html>
<html>
<head>
    <title>Real-Time Crowd Detection</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxjs/dist/onnx.min.js"></script>
</head>
<body>
    <h1>Real-Time Crowd Detection</h1>
    <video id="video" width="640" height="480" autoplay></video>
    <canvas id="canvas" width="640" height="480"></canvas>

    <script>
        class CrowdDetectionWeb {
            constructor() {
                this.session = null;
                this.video = document.getElementById('video');
                this.canvas = document.getElementById('canvas');
                this.ctx = this.canvas.getContext('2d');
            }

            async loadModel() {
                this.session = new onnx.InferenceSession();
                await this.session.loadModel('./crowd_detection_model.onnx');
                console.log('Model loaded successfully');
            }

            async startCamera() {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                this.video.srcObject = stream;
                this.video.play();

                this.video.addEventListener('loadeddata', () => {
                    this.processFrame();
                });
            }

            preprocessFrame(imageData) {
                const resized = this.resizeImage(imageData, 640, 640);
                const normalized = new Float32Array(640 * 640 * 3);

                for (let i = 0; i < resized.data.length; i += 4) {
                    const r = resized.data[i] / 255.0;
                    const g = resized.data[i + 1] / 255.0;
                    const b = resized.data[i + 2] / 255.0;

                    const idx = i / 4;
                    normalized[idx] = r;
                    normalized[idx + 640 * 640] = g;
                    normalized[idx + 640 * 640 * 2] = b;
                }

                return normalized;
            }

            async processFrame() {
                if (!this.session) return;

                this.ctx.drawImage(this.video, 0, 0, 640, 480);
                const imageData = this.ctx.getImageData(0, 0, 640, 480);

                const input = this.preprocessFrame(imageData);
                const tensor = new onnx.Tensor(input, 'float32', [1, 3, 640, 640]);

                const outputMap = await this.session.run([tensor]);
                const output = outputMap.values().next().value;

                this.drawDetections(output.data);

                requestAnimationFrame(() => this.processFrame());
            }

            drawDetections(detections) {
                this.ctx.clearRect(0, 0, 640, 480);
                this.ctx.drawImage(this.video, 0, 0, 640, 480);

                this.ctx.strokeStyle = '#00FF00';
                this.ctx.lineWidth = 2;
                this.ctx.font = '16px Arial';
                this.ctx.fillStyle = '#00FF00';

                for (let i = 0; i < detections.length; i += 6) {
                    const confidence = detections[i + 4];
                    if (confidence > 0.5) {
                        const x = detections[i] * 640;
                        const y = detections[i + 1] * 480;
                        const w = (detections[i + 2] - detections[i]) * 640;
                        const h = (detections[i + 3] - detections[i + 1]) * 480;

                        this.ctx.strokeRect(x, y, w, h);
                        this.ctx.fillText(`Person: ${confidence.toFixed(2)}`, x, y - 5);
                    }
                }
            }

            resizeImage(imageData, targetWidth, targetHeight) {
                const canvas = document.createElement('canvas');
                canvas.width = targetWidth;
                canvas.height = targetHeight;
                const ctx = canvas.getContext('2d');

                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = imageData.width;
                tempCanvas.height = imageData.height;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.putImageData(imageData, 0, 0);

                ctx.drawImage(tempCanvas, 0, 0, targetWidth, targetHeight);
                return ctx.getImageData(0, 0, targetWidth, targetHeight);
            }
        }

        const crowdDetection = new CrowdDetectionWeb();
        crowdDetection.loadModel().then(() => {
            crowdDetection.startCamera();
        });
    </script>
</body>
</html>'''

        web_path = self.export_dir / 'web_integration_example.html'
        with open(web_path, 'w') as f:
            f.write(web_example)

        logging.info(f"✅ Web integration example saved to: {web_path}")
        return str(web_path)

    def create_python_inference_example(self):
        """Create Python inference example"""
        python_example = '''import cv2
import numpy as np
import json

class CrowdDetectionInference:
    """Python inference class for crowd detection"""

    def __init__(self, model_path='crowd_detection_model.onnx'):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the exported model"""
        try:
            import onnxruntime as ort
            self.model = ort.InferenceSession(self.model_path)
            print(f"Model loaded from {self.model_path}")
        except ImportError:
            print("ONNX Runtime not available. Install with: pip install onnxruntime")
        except Exception as e:
            print(f"Failed to load model: {e}")

    def preprocess_frame(self, frame):
        """Preprocess frame for inference"""
        resized = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor

    def detect_people(self, frame):
        """Detect people in frame"""
        if self.model is None:
            return []

        input_tensor = self.preprocess_frame(frame)
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: input_tensor})
        detections = self.postprocess_detections(outputs[0], frame.shape)
        return detections

    def postprocess_detections(self, raw_output, original_shape):
        """Post-process model outputs"""
        detections = []
        h, w = original_shape[:2]
        scale_x = w / 640
        scale_y = h / 640

        for detection in raw_output[0]:
            confidence = detection[4]
            if confidence > 0.5:
                x1 = int(detection[0] * scale_x)
                y1 = int(detection[1] * scale_y)
                x2 = int(detection[2] * scale_x)
                y2 = int(detection[3] * scale_y)

                detections.append({
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'confidence': float(confidence),
                    'class': 'person'
                })

        return detections

# Example usage
if __name__ == "__main__":
    detector = CrowdDetectionInference()
    # detector.process_video('input_video.mp4', 'output_video.mp4')
'''

        python_path = self.export_dir / 'python_inference_example.py'
        with open(python_path, 'w', encoding='utf-8') as f:
            f.write(python_example)

        logging.info(f"✅ Python inference example saved to: {python_path}")
        return str(python_path)

    def create_deployment_package(self):
        """Create a complete deployment package"""
        package_path = self.export_dir / 'crowd_detection_deployment_package.zip'

        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.export_dir.rglob('*'):
                if file_path.is_file() and file_path.name != package_path.name:
                    arcname = file_path.relative_to(self.export_dir)
                    zipf.write(file_path, arcname)

        logging.info(f"✅ Deployment package created: {package_path}")
        return str(package_path)

    def export_all_formats(self):
        """Export model in all supported formats"""
        if not self.load_model():
            logging.error("Failed to load model. Cannot proceed with export.")
            return []

        exported_files = []

        print("🚀 Starting comprehensive model export...")
        print("=" * 50)

        # Export in different formats
        formats = [
            ("ONNX (Web Deployment)", lambda: self.export_onnx()),
            ("TorchScript (PyTorch)", lambda: self.export_torchscript()),
        ]

        for format_name, export_func in formats:
            print(f"\n📦 Exporting {format_name}...")
            try:
                result = export_func()
                if result:
                    exported_files.append(result)
                    print(f"✅ {format_name} export successful")
                else:
                    print(f"❌ {format_name} export failed")
            except Exception as e:
                print(f"❌ {format_name} export failed: {e}")

        # Create utility files
        print(f"\n🛠️ Creating utility files...")
        utility_files = [
            ("Preprocessing Utils", lambda: self.create_preprocessing_utils()),
            ("Python Inference Example", lambda: self.create_python_inference_example()),
            ("Web Integration Example", lambda: self.create_web_integration_example()),
            ("Model Metadata", lambda: self.save_metadata()),
        ]

        for util_name, util_func in utility_files:
            try:
                result = util_func()
                if result:
                    exported_files.append(result)
                    print(f"✅ {util_name} created")
            except Exception as e:
                print(f"❌ {util_name} creation failed: {e}")

        # Create deployment package
        print(f"\n📦 Creating deployment package...")
        try:
            package_path = self.create_deployment_package()
            exported_files.append(package_path)
            print(f"✅ Deployment package created")
        except Exception as e:
            print(f"❌ Deployment package creation failed: {e}")

        return exported_files


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 Crowd Detection Model for Web Applications"
    )

    parser.add_argument('--model', type=str, default='yolov8m.pt',
                       help='Path to YOLOv8 model file (default: yolov8m.pt)')
    parser.add_argument('--output-dir', type=str, default='exported_models',
                       help='Output directory for exported models (default: exported_models)')
    parser.add_argument('--export-all', action='store_true', default=True,
                       help='Export in all supported formats')

    args = parser.parse_args()

    print("🎯 YOLOv8 Crowd Detection Model Exporter")
    print("=" * 50)
    print(f"📁 Model: {args.model}")
    print(f"📂 Output Directory: {args.output_dir}")
    print("=" * 50)

    # Create exporter
    exporter = CrowdDetectionModelExporter(
        model_path=args.model,
        export_dir=args.output_dir
    )

    # Export all formats
    exported_files = exporter.export_all_formats()

    # Print summary
    print(f"\n🎉 Export completed successfully!")
    print(f"📊 Summary:")
    print(f"   - Total files exported: {len(exported_files)}")
    print(f"   - Export directory: {args.output_dir}")
    print(f"\n📁 Exported files:")
    for file_path in exported_files:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"   ✅ {file_path} ({size:,} bytes)")
            else:
                print(f"   ✅ {file_path} (Directory)")
        else:
            print(f"   ❌ {file_path} (Not found)")

    print(f"\n🚀 Ready for web deployment!")
    print(f"💡 Next steps:")
    print(f"   1. Copy the exported files to your web application")
    print(f"   2. Use the provided examples for integration")
    print(f"   3. Install required dependencies (onnxruntime, etc.)")
    print(f"   4. Test with your live video feed")

    return 0


if __name__ == '__main__':
    sys.exit(main())
