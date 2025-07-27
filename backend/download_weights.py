#!/usr/bin/env python3
"""
Download YOLOv4 weights for the crowd detection model.
This script downloads the pre-trained YOLOv4 weights from the official repository.
"""

import os
import sys
import urllib.request
import urllib.error
from tqdm import tqdm
import hashlib

# URLs for different YOLO model weights
WEIGHT_URLS = {
    'yolov4': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights',
    'yolov4-tiny': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights',
    'yolov3': 'https://pjreddie.com/media/files/yolov3.weights',
    'yolov3-tiny': 'https://pjreddie.com/media/files/yolov3-tiny.weights'
}

# Expected file sizes (approximate, in bytes)
EXPECTED_SIZES = {
    'yolov4': 257717640,  # ~246 MB
    'yolov4-tiny': 23266596,  # ~22 MB
    'yolov3': 248007048,  # ~236 MB
    'yolov3-tiny': 35434956  # ~34 MB
}

def download_with_progress(url, filename):
    """Download file with progress bar"""
    try:
        # Get file size
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('Content-Length', 0))
        
        # Download with progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {filename}") as pbar:
            def update_progress(block_num, block_size, total_size):
                pbar.update(block_size)
            
            urllib.request.urlretrieve(url, filename, reporthook=update_progress)
        
        return True
    except urllib.error.URLError as e:
        print(f"Error downloading {filename}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error downloading {filename}: {e}")
        return False

def verify_download(filename, expected_size=None):
    """Verify downloaded file"""
    if not os.path.exists(filename):
        return False
    
    file_size = os.path.getsize(filename)
    
    if expected_size and abs(file_size - expected_size) > 1024:  # Allow 1KB difference
        print(f"Warning: {filename} size ({file_size}) differs from expected ({expected_size})")
        return False
    
    print(f"✓ {filename} downloaded successfully ({file_size} bytes)")
    return True

def download_weights(model_type='yolov4', data_dir='./data'):
    """Download weights for specified model type"""
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    if model_type not in WEIGHT_URLS:
        print(f"Error: Unknown model type '{model_type}'")
        print(f"Available models: {list(WEIGHT_URLS.keys())}")
        return False
    
    url = WEIGHT_URLS[model_type]
    filename = os.path.join(data_dir, f"{model_type}.weights")
    
    # Check if file already exists
    if os.path.exists(filename):
        if verify_download(filename, EXPECTED_SIZES.get(model_type)):
            print(f"✓ {filename} already exists and appears valid")
            return True
        else:
            print(f"Existing {filename} appears corrupted, re-downloading...")
            os.remove(filename)
    
    print(f"Downloading {model_type} weights from {url}")
    
    # Download the file
    success = download_with_progress(url, filename)
    
    if success:
        # Verify the download
        if verify_download(filename, EXPECTED_SIZES.get(model_type)):
            print(f"✓ Successfully downloaded {model_type} weights to {filename}")
            return True
        else:
            print(f"✗ Download verification failed for {filename}")
            return False
    else:
        print(f"✗ Failed to download {model_type} weights")
        return False

def download_all_weights(data_dir='./data'):
    """Download all available model weights"""
    print("Downloading all YOLO model weights...")
    
    results = {}
    for model_type in WEIGHT_URLS.keys():
        print(f"\n--- Downloading {model_type} ---")
        results[model_type] = download_weights(model_type, data_dir)
    
    print("\n" + "="*50)
    print("Download Summary:")
    for model_type, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {model_type}: {status}")
    
    return results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download YOLO model weights")
    parser.add_argument('--model', type=str, default='yolov4', 
                       choices=list(WEIGHT_URLS.keys()) + ['all'],
                       help='Model type to download (default: yolov4)')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory to save weights (default: ./data)')
    
    args = parser.parse_args()
    
    print("YOLO Weights Downloader")
    print("="*30)
    
    if args.model == 'all':
        results = download_all_weights(args.data_dir)
        success = all(results.values())
    else:
        success = download_weights(args.model, args.data_dir)
    
    if success:
        print(f"\n✓ All downloads completed successfully!")
        print(f"Weights saved to: {os.path.abspath(args.data_dir)}")
        return 0
    else:
        print(f"\n✗ Some downloads failed. Please check your internet connection and try again.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
