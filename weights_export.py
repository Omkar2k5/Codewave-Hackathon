#!/usr/bin/env python3
"""
Weights-only export script for YOLO models.
This script extracts and saves just the weights from the .weights file
in a format that can be used with TensorFlow/Keras.
"""

import os
import sys
import numpy as np
import json
from datetime import datetime

def read_darknet_weights(weights_file):
    """Read weights from Darknet .weights file"""
    
    print(f"Reading weights from {weights_file}...")
    
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Weights file {weights_file} not found")
    
    with open(weights_file, 'rb') as f:
        # Read header
        major, minor, revision, seen, _ = np.fromfile(f, dtype=np.int32, count=5)
        
        print(f"Weights file info:")
        print(f"  - Version: {major}.{minor}.{revision}")
        print(f"  - Images seen during training: {seen}")
        
        # Read all remaining weights
        weights_data = np.fromfile(f, dtype=np.float32)
        
        print(f"  - Total weights: {len(weights_data):,}")
        print(f"  - File size: {os.path.getsize(weights_file):,} bytes")
    
    return {
        'header': {
            'major': int(major),
            'minor': int(minor), 
            'revision': int(revision),
            'seen': int(seen)
        },
        'weights': weights_data
    }

def save_weights_numpy(weights_data, output_path):
    """Save weights as numpy array"""
    np.save(output_path, weights_data['weights'])
    print(f"[OK] Weights saved as numpy array to {output_path}")
    return output_path

def save_weights_json(weights_data, output_path):
    """Save weights metadata as JSON"""
    metadata = {
        'header': weights_data['header'],
        'weights_shape': weights_data['weights'].shape,
        'weights_dtype': str(weights_data['weights'].dtype),
        'total_parameters': int(len(weights_data['weights'])),
        'export_timestamp': datetime.now().isoformat()
    }

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Weights metadata saved to {output_path}")
    return output_path

def create_conversion_script(model_name, output_dir):
    """Create a script to help convert weights back to TensorFlow format"""
    
    script_content = f'''#!/usr/bin/env python3
"""
Weight conversion script for {model_name}
This script helps convert the exported weights back to TensorFlow format.
"""

import numpy as np
import json

def load_weights():
    """Load the exported weights"""
    weights = np.load('{model_name}_weights.npy')
    
    with open('{model_name}_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded weights:")
    print(f"  - Shape: {{weights.shape}}")
    print(f"  - Total parameters: {{len(weights):,}}")
    print(f"  - Original version: {{metadata['header']['major']}}.{{metadata['header']['minor']}}.{{metadata['header']['revision']}}")
    
    return weights, metadata

def get_weight_statistics():
    """Get statistics about the weights"""
    weights, metadata = load_weights()
    
    print(f"\\nWeight statistics:")
    print(f"  - Min: {{weights.min():.6f}}")
    print(f"  - Max: {{weights.max():.6f}}")
    print(f"  - Mean: {{weights.mean():.6f}}")
    print(f"  - Std: {{weights.std():.6f}}")
    print(f"  - Non-zero: {{np.count_nonzero(weights):,}} / {{len(weights):,}} ({{np.count_nonzero(weights)/len(weights)*100:.1f}}%)")

if __name__ == '__main__':
    get_weight_statistics()
'''
    
    script_path = os.path.join(output_dir, f"{model_name}_weight_loader.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"[OK] Weight conversion script saved to {script_path}")
    return script_path

def export_weights(weights_file, output_dir, model_name=None):
    """Export weights in multiple formats"""
    
    # Determine model name from weights file if not provided
    if model_name is None:
        model_name = os.path.splitext(os.path.basename(weights_file))[0]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Exporting weights for {model_name}...")
    
    # Read weights
    try:
        weights_data = read_darknet_weights(weights_file)
    except Exception as e:
        print(f"✗ Failed to read weights: {e}")
        return []
    
    exported_files = []
    
    # Export as numpy array
    try:
        numpy_path = os.path.join(output_dir, f"{model_name}_weights.npy")
        save_weights_numpy(weights_data, numpy_path)
        exported_files.append(numpy_path)
    except Exception as e:
        print(f"[ERROR] Failed to save numpy weights: {e}")

    # Export metadata
    try:
        json_path = os.path.join(output_dir, f"{model_name}_metadata.json")
        save_weights_json(weights_data, json_path)
        exported_files.append(json_path)
    except Exception as e:
        print(f"[ERROR] Failed to save metadata: {e}")

    # Create conversion script
    try:
        script_path = create_conversion_script(model_name, output_dir)
        exported_files.append(script_path)
    except Exception as e:
        print(f"[ERROR] Failed to create conversion script: {e}")

    # Create summary
    try:
        summary_path = os.path.join(output_dir, f"{model_name}_export_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Weights Export Summary\n")
            f.write(f"====================\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Source: {weights_file}\n")
            f.write(f"Export Date: {datetime.now().isoformat()}\n")
            f.write(f"Darknet Version: {weights_data['header']['major']}.{weights_data['header']['minor']}.{weights_data['header']['revision']}\n")
            f.write(f"Images Seen: {weights_data['header']['seen']:,}\n")
            f.write(f"Total Parameters: {len(weights_data['weights']):,}\n")
            f.write(f"Weight Range: [{weights_data['weights'].min():.6f}, {weights_data['weights'].max():.6f}]\n\n")
            f.write(f"Exported Files:\n")
            for file_path in exported_files:
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    f.write(f"  - {os.path.basename(file_path)} ({size:,} bytes)\n")

        exported_files.append(summary_path)
        print(f"[OK] Export summary saved to {summary_path}")
    except Exception as e:
        print(f"[ERROR] Failed to create summary: {e}")
    
    return exported_files

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export YOLO weights")
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to .weights file')
    parser.add_argument('--output-dir', type=str, default='./exported_weights',
                       help='Output directory (default: ./exported_weights)')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Model name (default: derived from weights filename)')
    
    args = parser.parse_args()
    
    print("YOLO Weights Exporter")
    print("=" * 25)
    
    try:
        exported_files = export_weights(args.weights, args.output_dir, args.model_name)
        
        if exported_files:
            print(f"\n[OK] Export completed successfully!")
            print(f"Exported {len(exported_files)} files to {args.output_dir}")
            print(f"\nExported files:")
            for file_path in exported_files:
                print(f"  - {file_path}")

            print(f"\nTo use the weights:")
            print(f"1. Load with numpy: weights = np.load('{os.path.basename(exported_files[0])}')")
            print(f"2. Run the conversion script: python {os.path.basename([f for f in exported_files if f.endswith('_weight_loader.py')][0])}")

            return 0
        else:
            print(f"\n[ERROR] No files were exported")
            return 1

    except Exception as e:
        print(f"\n[ERROR] Export failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
