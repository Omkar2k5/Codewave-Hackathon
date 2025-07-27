#!/usr/bin/env python3
"""
YOLO-based crowd detection script that mimics the original op_1.mp4 style
Uses OpenCV DNN module with YOLOv4 for proper object detection and tracking
"""

import cv2
import numpy as np
import os
import argparse
from datetime import datetime
import random

class YOLOCrowdDetector:
    def __init__(self, weights_path, config_path, classes_path):
        self.weights_path = weights_path
        self.config_path = config_path
        self.classes_path = classes_path
        self.net = None
        self.output_layers = None
        self.classes = []
        self.colors = []
        self.load_model()
        
    def load_model(self):
        """Load YOLO model and classes"""
        try:
            # Load YOLO
            self.net = cv2.dnn.readNet(self.weights_path, self.config_path)
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Load class names
            with open(self.classes_path, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Generate colors for each class
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
            
            print(f"✅ YOLO model loaded successfully!")
            print(f"   - Classes: {len(self.classes)}")
            print(f"   - Output layers: {len(self.output_layers)}")
            
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            raise
    
    def detect_objects(self, frame, confidence_threshold=0.3, nms_threshold=0.4):
        """Detect objects in frame using YOLO"""
        height, width, channels = frame.shape
        
        # Prepare image for detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
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
        
        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        # Debug information
        if len(boxes) > 0:
            print(f"Debug: Found {len(boxes)} detections before NMS, {len(indexes) if len(indexes) > 0 else 0} after NMS")

        return boxes, confidences, class_ids, indexes
    
    def draw_detections(self, frame, boxes, confidences, class_ids, indexes):
        """Draw detection boxes and labels on frame"""
        person_count = 0
        
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = self.colors[class_ids[i]]
                
                # Only count and draw persons (class_id = 0 for COCO dataset)
                if class_ids[i] == 0:  # Person class
                    person_count += 1
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw label with confidence
                    label_text = f"{label}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                                (x + label_size[0], y), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label_text, (x, y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Draw person ID
                    cv2.putText(frame, f"ID:{person_count}", (x, y + h + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
        
        return person_count
    
    def process_video(self, video_path, output_path):
        """Process video with YOLO crowd detection"""
        print(f"🎬 Processing video: {video_path}")
        print(f"💾 Output will be saved to: {output_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        person_count_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"🔄 Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
            
            # Detect objects
            boxes, confidences, class_ids, indexes = self.detect_objects(frame)
            
            # Draw detections
            person_count = self.draw_detections(frame, boxes, confidences, class_ids, indexes)
            
            # Track person count
            person_count_history.append(person_count)
            
            # Calculate crowd statistics
            if len(person_count_history) > 30:  # Use last 30 frames for average
                avg_count = np.mean(person_count_history[-30:])
            else:
                avg_count = person_count
            
            # Determine crowd level
            if avg_count < 3:
                crowd_level = "LOW"
                crowd_color = (0, 255, 0)  # Green
            elif avg_count < 8:
                crowd_level = "MEDIUM"
                crowd_color = (0, 255, 255)  # Yellow
            else:
                crowd_level = "HIGH"
                crowd_color = (0, 0, 255)  # Red
            
            # Add information overlay (similar to op_1.mp4 style)
            overlay_height = 120
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, overlay_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Add text information
            cv2.putText(frame, f"CROWD DETECTION SYSTEM", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"People Detected: {person_count}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Crowd Level: {crowd_level}", 
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, crowd_color, 2)
            cv2.putText(frame, f"Avg Count: {avg_count:.1f}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add frame counter and timestamp
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                       (width - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, f"Time: {timestamp}", 
                       (width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Print summary
        if person_count_history:
            avg_people = np.mean(person_count_history)
            max_people = max(person_count_history)
            print(f"\n✅ Processing complete!")
            print(f"📊 Statistics:")
            print(f"   - Average people detected: {avg_people:.1f}")
            print(f"   - Maximum people detected: {max_people}")
            print(f"   - Total frames processed: {frame_count}")
            print(f"💾 Output saved to: {output_path}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='YOLO Crowd Detection')
    parser.add_argument('--video', default='data/video/input.mp4', help='Input video path')
    parser.add_argument('--output', default='outputs/crowd_detection_yolo.mp4', help='Output video path')
    parser.add_argument('--weights', default='data/yolov4.weights', help='YOLO weights path')
    parser.add_argument('--config', default='data/yolov4.cfg', help='YOLO config path')
    parser.add_argument('--classes', default='data/classes/coco.names', help='Classes file path')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        # Create detector
        detector = YOLOCrowdDetector(args.weights, args.config, args.classes)
        
        # Process video
        success = detector.process_video(args.video, args.output)
        
        if success:
            print("🎉 Crowd detection completed successfully!")
        else:
            print("❌ Crowd detection failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
