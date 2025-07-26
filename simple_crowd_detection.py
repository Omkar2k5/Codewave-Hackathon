#!/usr/bin/env python3
"""
Simple crowd detection script that works with existing setup
Uses OpenCV for basic object detection and tracking
"""

import cv2
import numpy as np
import os
import argparse
from datetime import datetime

def load_yolo_model():
    """Load YOLO model using OpenCV DNN"""
    
    # Check if we have the weights file
    weights_path = "data/yolov4.weights"
    config_path = "data/yolov4.cfg"
    
    # If config doesn't exist, create a basic one
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        print("Trying alternative approach...")
        return None
    
    try:
        # Load YOLO
        net = cv2.dnn.readNet(weights_path, config_path)
        
        # Get output layer names
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        return net, output_layers
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

def detect_objects_opencv(frame, net, output_layers, confidence_threshold=0.5):
    """Detect objects using OpenCV DNN"""
    
    height, width, channels = frame.shape
    
    # Prepare image for detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    # Extract information from outputs
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, confidences, class_ids

def simple_crowd_detection(video_path, output_path):
    """Simple crowd detection using basic computer vision"""
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Load COCO class names
    classes = []
    try:
        with open("data/classes/coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
    except:
        # Default classes if file not found
        classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck"]
    
    # Try to load YOLO model
    yolo_model = load_yolo_model()
    
    frame_count = 0
    person_count_history = []
    
    # Background subtractor for motion detection
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Progress indicator
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
        
        # Create a copy for processing
        processed_frame = frame.copy()
        
        # Method 1: Try YOLO detection if available
        person_count = 0
        if yolo_model:
            net, output_layers = yolo_model
            try:
                boxes, confidences, class_ids = detect_objects_opencv(frame, net, output_layers)
                
                # Apply non-maximum suppression
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                
                if len(indexes) > 0:
                    for i in indexes.flatten():
                        if class_ids[i] == 0:  # Person class
                            person_count += 1
                            x, y, w, h = boxes[i]
                            
                            # Draw bounding box
                            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(processed_frame, f"Person {confidences[i]:.2f}", 
                                      (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print(f"YOLO detection error: {e}")
        
        # Method 2: Background subtraction for motion detection (fallback)
        if person_count == 0:
            fgMask = backSub.apply(frame)
            
            # Find contours
            contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area (assuming people)
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 5000:  # Reasonable size for a person
                    person_count += 1
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(processed_frame, "Motion", (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Track person count
        person_count_history.append(person_count)
        
        # Calculate crowd density
        if len(person_count_history) > 30:  # Use last 30 frames
            avg_count = np.mean(person_count_history[-30:])
            crowd_level = "Low" if avg_count < 5 else "Medium" if avg_count < 15 else "High"
        else:
            avg_count = person_count
            crowd_level = "Low"
        
        # Add information overlay
        cv2.putText(processed_frame, f"People Detected: {person_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(processed_frame, f"Crowd Level: {crowd_level}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(processed_frame, f"Frame: {frame_count}/{total_frames}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(processed_frame, timestamp, 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Write frame
        out.write(processed_frame)
    
    # Cleanup
    cap.release()
    out.release()
    
    # Print summary
    if person_count_history:
        avg_people = np.mean(person_count_history)
        max_people = max(person_count_history)
        print(f"\nProcessing complete!")
        print(f"Average people detected: {avg_people:.1f}")
        print(f"Maximum people detected: {max_people}")
        print(f"Output saved to: {output_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Simple Crowd Detection')
    parser.add_argument('--video', default='data/video/input.mp4', help='Input video path')
    parser.add_argument('--output', default='outputs/crowd_detection_output.mp4', help='Output video path')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run detection
    success = simple_crowd_detection(args.video, args.output)
    
    if success:
        print("Crowd detection completed successfully!")
    else:
        print("Crowd detection failed!")

if __name__ == "__main__":
    main()
