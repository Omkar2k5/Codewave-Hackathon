#!/usr/bin/env python3
"""
Simple model export script that works with TensorFlow 2.19+
This script creates a basic export without the complex decode functions.
"""

import os
import sys
import tensorflow as tf
import numpy as np
from datetime import datetime
import json

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.yolov4 import YOLO
import core.utils as utils
from core.config import cfg

def create_simple_model(model_type='yolov4', tiny=False, input_size=416, weights_path='./data/yolov4.weights'):
    """Create a simple YOLO model for export"""
    
    print(f"Creating {model_type}{'_tiny' if tiny else ''} model...")
    
    # Load configuration
    class SimpleFlags:
        def __init__(self):
            self.tiny = tiny
            self.model = model_type
    
    flags = SimpleFlags()
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(flags)
    
    print(f"Model configuration:")
    print(f"  - Model type: {model_type}")
    print(f"  - Tiny: {tiny}")
    print(f"  - Input size: {input_size}")
    print(f"  - Number of classes: {NUM_CLASS}")
    
    # Create input layer
    input_layer = tf.keras.layers.Input([input_size, input_size, 3], name='input_image')
    
    # Create feature maps using YOLO backbone
    feature_maps = YOLO(input_layer, NUM_CLASS, model_type, tiny)
    
    # Create a simple model that outputs raw feature maps
    model = tf.keras.Model(input_layer, feature_maps, name=f'{model_type}{"_tiny" if tiny else ""}')
    
    # Load weights if they exist
    if os.path.exists(weights_path):
        try:
            utils.load_weights(model, weights_path, model_type, tiny)
            print(f"✓ Loaded weights from {weights_path}")
        except Exception as e:
            print(f"✗ Failed to load weights: {e}")
            print("Model will be exported without pre-trained weights")
    else:
        print(f"✗ Weights file {weights_path} not found")
        print("Model will be exported without pre-trained weights")
    
    return model, NUM_CLASS

def export_model_formats(model, model_name, output_dir, num_classes):
    """Export model in multiple formats"""
    
    os.makedirs(output_dir, exist_ok=True)
    exported_files = []
    
    print(f"\nExporting model to {output_dir}...")
    
    # 1. SavedModel format
    try:
        savedmodel_path = os.path.join(output_dir, f"{model_name}_savedmodel")
        model.save(savedmodel_path, save_format='tf')
        exported_files.append(savedmodel_path)
        print(f"✓ SavedModel exported to {savedmodel_path}")
    except Exception as e:
        print(f"✗ SavedModel export failed: {e}")
    
    # 2. HDF5 format
    try:
        h5_path = os.path.join(output_dir, f"{model_name}.h5")
        model.save(h5_path, save_format='h5')
        exported_files.append(h5_path)
        print(f"✓ HDF5 model exported to {h5_path}")
    except Exception as e:
        print(f"✗ HDF5 export failed: {e}")
    
    # 3. Weights only
    try:
        weights_path = os.path.join(output_dir, f"{model_name}_weights.h5")
        model.save_weights(weights_path)
        exported_files.append(weights_path)
        print(f"✓ Weights exported to {weights_path}")
    except Exception as e:
        print(f"✗ Weights export failed: {e}")
    
    # 4. Model architecture as JSON
    try:
        json_path = os.path.join(output_dir, f"{model_name}_architecture.json")
        with open(json_path, 'w') as f:
            f.write(model.to_json())
        exported_files.append(json_path)
        print(f"✓ Architecture exported to {json_path}")
    except Exception as e:
        print(f"✗ Architecture export failed: {e}")
    
    # 5. Model metadata
    try:
        metadata = {
            'model_name': model_name,
            'num_classes': num_classes,
            'input_shape': model.input_shape,
            'output_shapes': [output.shape.as_list() for output in model.outputs],
            'total_params': model.count_params(),
            'export_timestamp': datetime.now().isoformat(),
            'tensorflow_version': tf.__version__
        }
        
        metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        exported_files.append(metadata_path)
        print(f"✓ Metadata exported to {metadata_path}")
    except Exception as e:
        print(f"✗ Metadata export failed: {e}")
    
    # 6. Export summary
    try:
        summary_path = os.path.join(output_dir, f"{model_name}_export_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Model Export Summary\n")
            f.write(f"==================\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Export Date: {datetime.now().isoformat()}\n")
            f.write(f"TensorFlow Version: {tf.__version__}\n")
            f.write(f"Total Parameters: {model.count_params():,}\n")
            f.write(f"Input Shape: {model.input_shape}\n")
            f.write(f"Output Shapes: {[output.shape.as_list() for output in model.outputs]}\n\n")
            f.write(f"Exported Files:\n")
            for file_path in exported_files:
                if os.path.exists(file_path):
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        f.write(f"  - {os.path.basename(file_path)} ({size:,} bytes)\n")
                    else:
                        f.write(f"  - {os.path.basename(file_path)} (Directory)\n")
            
            # Add model summary
            f.write(f"\nModel Summary:\n")
            f.write("-" * 50 + "\n")
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        exported_files.append(summary_path)
        print(f"✓ Export summary saved to {summary_path}")
    except Exception as e:
        print(f"✗ Export summary failed: {e}")
    
    return exported_files

def main():
    """Main export function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple YOLO model export")
    parser.add_argument('--model', type=str, default='yolov4', choices=['yolov3', 'yolov4'],
                       help='Model type (default: yolov4)')
    parser.add_argument('--tiny', action='store_true', help='Use tiny version')
    parser.add_argument('--input-size', type=int, default=416, help='Input size (default: 416)')
    parser.add_argument('--weights', type=str, default='./data/yolov4.weights',
                       help='Path to weights file')
    parser.add_argument('--output-dir', type=str, default='./exported_models',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("Simple YOLO Model Exporter")
    print("=" * 30)
    
    # Adjust weights path for tiny models
    if args.tiny and args.weights == './data/yolov4.weights':
        args.weights = f'./data/{args.model}-tiny.weights'
    
    # Create model
    try:
        model, num_classes = create_simple_model(
            model_type=args.model,
            tiny=args.tiny,
            input_size=args.input_size,
            weights_path=args.weights
        )
        
        print(f"\n✓ Model created successfully!")
        print(f"Model summary:")
        model.summary()
        
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return 1
    
    # Export model
    model_name = f"{args.model}{'_tiny' if args.tiny else ''}_{args.input_size}"
    output_dir = os.path.join(args.output_dir, model_name)
    
    try:
        exported_files = export_model_formats(model, model_name, output_dir, num_classes)
        
        print(f"\n✓ Export completed successfully!")
        print(f"Exported {len(exported_files)} files to {output_dir}")
        print(f"\nExported files:")
        for file_path in exported_files:
            print(f"  - {file_path}")
            
        return 0
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
