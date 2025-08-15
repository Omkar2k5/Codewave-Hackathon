#!/usr/bin/env python3
"""
Test script for web exported crowd detection model
This script tests the exported ONNX model to ensure it works correctly for web deployment
"""

import cv2
import numpy as np
import os
import sys

def test_web_export_model():
    """Test the web exported ONNX model"""
    try:
        import onnxruntime as ort
        print("✅ ONNX Runtime available")
    except ImportError:
        print("❌ ONNX Runtime not available. Install with: pip install onnxruntime")
        return False
    
    # Check if model exists
    model_path = "web_export_models/crowd_detection_model.onnx"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    print(f"📁 Loading web export model: {model_path}")
    
    try:
        # Load ONNX model
        session = ort.InferenceSession(model_path)
        print("✅ Web export ONNX model loaded successfully")
        
        # Get model input/output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"📊 Model Info:")
        print(f"   Input: {input_info.name} - {input_info.shape} - {input_info.type}")
        print(f"   Output: {output_info.name} - {output_info.shape} - {output_info.type}")
        
        # Create test input (640x640 RGB image)
        test_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
        print(f"🔄 Testing with input shape: {test_input.shape}")
        
        # Run inference
        outputs = session.run(None, {input_info.name: test_input})
        print(f"✅ Inference successful! Output shape: {outputs[0].shape}")
        
        # Test with real image if available
        test_real_image(session)
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_real_image(session):
    """Test with a real image if available"""
    try:
        # Try to get a frame from test video
        test_video = "data/video/test_1.mp4"
        if os.path.exists(test_video):
            cap = cv2.VideoCapture(test_video)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print(f"📹 Testing with real frame from: {test_video}")
                
                # Preprocess frame
                resized = cv2.resize(frame, (640, 640))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                normalized = rgb.astype(np.float32) / 255.0
                input_tensor = np.transpose(normalized, (2, 0, 1))
                input_tensor = np.expand_dims(input_tensor, axis=0)
                
                # Run inference
                input_name = session.get_inputs()[0].name
                outputs = session.run(None, {input_name: input_tensor})
                
                print(f"✅ Real image inference successful!")
                print(f"📊 Output shape: {outputs[0].shape}")
                
                # Count high-confidence detections
                detections = outputs[0][0]  # Remove batch dimension
                high_conf_count = 0
                for detection in detections:
                    if len(detection) >= 5 and detection[4] > 0.5:
                        high_conf_count += 1
                
                print(f"🎯 High-confidence detections (>0.5): {high_conf_count}")
                return True
        
        print("⚠️ No test video found, skipping real image test")
        return True
        
    except Exception as e:
        print(f"❌ Real image test failed: {e}")
        return False

def test_web_integration_files():
    """Test if all web integration files are present"""
    print("\n🔍 Checking web integration files...")
    
    required_files = [
        "web_export_models/crowd_detection_model.onnx",
        "web_export_models/model_metadata.json",
        "web_export_models/preprocessing_utils.py",
        "web_export_models/python_inference_example.py",
        "web_export_models/web_integration_example.html",
        "web_export_models/README.md"
    ]
    
    all_present = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   ✅ {file_path} ({size:,} bytes)")
        else:
            print(f"   ❌ {file_path} (Missing)")
            all_present = False
    
    return all_present

def main():
    """Main test function"""
    print("🧪 Testing Web Export Crowd Detection Model")
    print("=" * 60)
    
    # Test 1: Check if all files are present
    print("\n🔬 Test 1: File Presence Check")
    files_test = test_web_integration_files()
    
    # Test 2: Model loading and inference
    print("\n🔬 Test 2: Model Loading and Inference")
    model_test = test_web_export_model()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 40)
    print(f"✅ File Presence Check: {'PASSED' if files_test else 'FAILED'}")
    print(f"✅ Model Inference Test: {'PASSED' if model_test else 'FAILED'}")
    
    if files_test and model_test:
        print("\n🎉 All tests passed! Web export model is ready for deployment.")
        print("\n💡 Next steps for web integration:")
        print("   1. Copy web_export_models/ folder to your web application")
        print("   2. Install onnxruntime-web: npm install onnxruntime-web")
        print("   3. Use the provided HTML/JavaScript example")
        print("   4. Test with your live camera feed")
        print("   5. Customize the detection visualization")
        
        print(f"\n📁 Key files for web deployment:")
        print(f"   🎯 crowd_detection_model.onnx (main model)")
        print(f"   📄 web_integration_example.html (example code)")
        print(f"   📚 README.md (documentation)")
        print(f"   🐍 python_inference_example.py (Python reference)")
        
        return 0
    else:
        print("\n❌ Some tests failed. Please check the export process.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
