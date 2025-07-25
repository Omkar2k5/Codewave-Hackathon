#!/usr/bin/env python3
"""
Comprehensive model export utility for Real-Time Crowd Detection system.
This script provides various export options for the trained YOLO models.
"""

import os
import sys
import json
import argparse
import tensorflow as tf
from datetime import datetime
import logging

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.yolov4 import YOLO, decode, filter_boxes
import core.utils as utils
from core.config import cfg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelExporter:
    """Class to handle model export operations"""
    
    def __init__(self, model_type='yolov4', tiny=False, input_size=416, 
                 weights_path='./data/yolov4.weights', score_threshold=0.2):
        self.model_type = model_type
        self.tiny = tiny
        self.input_size = input_size
        self.weights_path = weights_path
        self.score_threshold = score_threshold
        self.model = None
        self.metadata = {}
        
    def create_model(self):
        """Create and load the YOLO model"""
        try:
            # Load configuration
            if self.tiny:
                if self.model_type == 'yolov3':
                    strides = [16, 32]
                    anchors = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
                else:  # yolov4-tiny
                    strides = [16, 32]
                    anchors = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
                xyscale = [1.05, 1.05]
            else:
                if self.model_type == 'yolov3':
                    strides = [8, 16, 32]
                    anchors = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
                else:  # yolov4
                    strides = [8, 16, 32]
                    anchors = [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
                xyscale = [1.2, 1.1, 1.05]
            
            # Read class names
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)
            num_classes = len(class_names)
            
            # Create model
            input_layer = tf.keras.layers.Input([self.input_size, self.input_size, 3])
            feature_maps = YOLO(input_layer, num_classes, self.model_type, self.tiny)
            
            bbox_tensors = []
            prob_tensors = []
            
            if self.tiny:
                for i, fm in enumerate(feature_maps):
                    if i == 0:
                        output_tensors = decode(fm, self.input_size // 16, num_classes, 
                                              strides, anchors, i, xyscale, 'tf')
                    else:
                        output_tensors = decode(fm, self.input_size // 32, num_classes, 
                                              strides, anchors, i, xyscale, 'tf')
                    bbox_tensors.append(output_tensors[0])
                    prob_tensors.append(output_tensors[1])
            else:
                for i, fm in enumerate(feature_maps):
                    if i == 0:
                        output_tensors = decode(fm, self.input_size // 8, num_classes, 
                                              strides, anchors, i, xyscale, 'tf')
                    elif i == 1:
                        output_tensors = decode(fm, self.input_size // 16, num_classes, 
                                              strides, anchors, i, xyscale, 'tf')
                    else:
                        output_tensors = decode(fm, self.input_size // 32, num_classes, 
                                              strides, anchors, i, xyscale, 'tf')
                    bbox_tensors.append(output_tensors[0])
                    prob_tensors.append(output_tensors[1])
            
            pred_bbox = tf.concat(bbox_tensors, axis=1)
            pred_prob = tf.concat(prob_tensors, axis=1)
            
            boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, 
                                          score_threshold=self.score_threshold, 
                                          input_shape=tf.constant([self.input_size, self.input_size]))
            pred = tf.concat([boxes, pred_conf], axis=-1)
            
            self.model = tf.keras.Model(input_layer, pred)
            
            # Load weights if they exist
            if os.path.exists(self.weights_path):
                utils.load_weights(self.model, self.weights_path, self.model_type, self.tiny)
                logging.info(f"Loaded weights from {self.weights_path}")
            else:
                logging.warning(f"Weights file {self.weights_path} not found. Model will be exported without pre-trained weights.")
            
            # Store metadata
            self.metadata = {
                'model_type': self.model_type,
                'is_tiny': self.tiny,
                'input_size': self.input_size,
                'score_threshold': self.score_threshold,
                'num_classes': num_classes,
                'class_names': list(class_names.values()),
                'strides': strides,
                'anchors': anchors,
                'xyscale': xyscale,
                'export_timestamp': datetime.now().isoformat(),
                'tensorflow_version': tf.__version__
            }
            
            return True
            
        except Exception as e:
            logging.error(f"Error creating model: {e}")
            return False
    
    def export_savedmodel(self, output_path):
        """Export model in TensorFlow SavedModel format"""
        try:
            self.model.save(output_path)
            logging.info(f"SavedModel exported to {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Failed to export SavedModel: {e}")
            return None
    
    def export_h5(self, output_path):
        """Export model in HDF5 format"""
        try:
            self.model.save(output_path, save_format='h5')
            logging.info(f"HDF5 model exported to {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Failed to export HDF5 model: {e}")
            return None
    
    def export_tflite(self, output_path, quantize=True):
        """Export model in TensorFlow Lite format"""
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            tflite_model = converter.convert()
            
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logging.info(f"TensorFlow Lite model exported to {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Failed to export TensorFlow Lite model: {e}")
            return None
    
    def export_weights(self, output_path):
        """Export model weights separately"""
        try:
            self.model.save_weights(output_path)
            logging.info(f"Model weights exported to {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Failed to export weights: {e}")
            return None
    
    def save_metadata(self, output_path):
        """Save model metadata"""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logging.info(f"Metadata saved to {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Failed to save metadata: {e}")
            return None
    
    def export_all(self, output_dir):
        """Export model in all supported formats"""
        if not self.model:
            if not self.create_model():
                return []
        
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = f"{self.model_type}{'_tiny' if self.tiny else ''}_{self.input_size}"
        exported_files = []
        
        # Export in different formats
        formats = [
            ('savedmodel', lambda: self.export_savedmodel(os.path.join(output_dir, f"{base_name}_savedmodel"))),
            ('h5', lambda: self.export_h5(os.path.join(output_dir, f"{base_name}.h5"))),
            ('tflite', lambda: self.export_tflite(os.path.join(output_dir, f"{base_name}.tflite"))),
            ('weights', lambda: self.export_weights(os.path.join(output_dir, f"{base_name}_weights.h5"))),
            ('metadata', lambda: self.save_metadata(os.path.join(output_dir, f"{base_name}_metadata.json")))
        ]
        
        for format_name, export_func in formats:
            try:
                result = export_func()
                if result:
                    exported_files.append(result)
                    logging.info(f"✓ {format_name} export successful")
                else:
                    logging.warning(f"✗ {format_name} export failed")
            except Exception as e:
                logging.error(f"✗ {format_name} export failed: {e}")
        
        # Create export summary
        summary_path = os.path.join(output_dir, f"{base_name}_export_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Model Export Summary\n")
            f.write(f"==================\n\n")
            f.write(f"Export Date: {datetime.now().isoformat()}\n")
            f.write(f"Model Type: {self.model_type}\n")
            f.write(f"Tiny Model: {self.tiny}\n")
            f.write(f"Input Size: {self.input_size}\n")
            f.write(f"Score Threshold: {self.score_threshold}\n\n")
            f.write(f"Exported Files:\n")
            for file_path in exported_files:
                if os.path.exists(file_path):
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        f.write(f"  - {file_path} ({size:,} bytes)\n")
                    else:
                        f.write(f"  - {file_path} (Directory)\n")
        
        exported_files.append(summary_path)
        return exported_files

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Export YOLO model for crowd detection")
    parser.add_argument('--model', type=str, default='yolov4', choices=['yolov3', 'yolov4'],
                       help='Model type (default: yolov4)')
    parser.add_argument('--tiny', action='store_true', help='Use tiny version of the model')
    parser.add_argument('--input-size', type=int, default=416, help='Input size (default: 416)')
    parser.add_argument('--weights', type=str, default='./data/yolov4.weights',
                       help='Path to weights file')
    parser.add_argument('--score-threshold', type=float, default=0.2,
                       help='Score threshold for filtering (default: 0.2)')
    parser.add_argument('--output-dir', type=str, default='./exported_models',
                       help='Output directory for exported models')
    parser.add_argument('--format', type=str, choices=['all', 'savedmodel', 'h5', 'tflite', 'weights'],
                       default='all', help='Export format (default: all)')
    
    args = parser.parse_args()
    
    print("YOLO Model Exporter for Crowd Detection")
    print("="*40)
    
    # Create exporter
    exporter = ModelExporter(
        model_type=args.model,
        tiny=args.tiny,
        input_size=args.input_size,
        weights_path=args.weights,
        score_threshold=args.score_threshold
    )
    
    # Create model
    if not exporter.create_model():
        print("Failed to create model. Exiting.")
        return 1
    
    print(f"Model created successfully!")
    print(f"Model summary:")
    exporter.model.summary()
    
    # Export model
    if args.format == 'all':
        exported_files = exporter.export_all(args.output_dir)
        print(f"\nExported {len(exported_files)} files to {args.output_dir}:")
        for file_path in exported_files:
            print(f"  - {file_path}")
    else:
        # Export specific format
        os.makedirs(args.output_dir, exist_ok=True)
        base_name = f"{args.model}{'_tiny' if args.tiny else ''}_{args.input_size}"
        
        if args.format == 'savedmodel':
            result = exporter.export_savedmodel(os.path.join(args.output_dir, f"{base_name}_savedmodel"))
        elif args.format == 'h5':
            result = exporter.export_h5(os.path.join(args.output_dir, f"{base_name}.h5"))
        elif args.format == 'tflite':
            result = exporter.export_tflite(os.path.join(args.output_dir, f"{base_name}.tflite"))
        elif args.format == 'weights':
            result = exporter.export_weights(os.path.join(args.output_dir, f"{base_name}_weights.h5"))
        
        if result:
            print(f"\nModel exported successfully to: {result}")
        else:
            print(f"\nFailed to export model in {args.format} format")
            return 1
    
    print("\n✓ Export completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
