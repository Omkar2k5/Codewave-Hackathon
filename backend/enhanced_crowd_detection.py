#!/usr/bin/env python3
"""
Enhanced crowd detection system with improved sensitivity and crowd grouping
Handles overlapping detections and groups them into crowd clusters
"""

import cv2
import numpy as np
import os
import argparse
from datetime import datetime
from sklearn.cluster import DBSCAN
import math

class EnhancedCrowdDetector:
    def __init__(self):
        # Multiple HOG detectors with different parameters for better sensitivity
        self.hog1 = cv2.HOGDescriptor()
        self.hog1.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        self.hog2 = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
        self.hog2.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Background subtractor
        self.backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=500, varThreshold=50)
        
        # Cascade classifier for additional detection
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        except:
            self.face_cascade = None
            self.body_cascade = None
        
    def detect_people_multi_hog(self, frame):
        """Enhanced people detection using multiple HOG configurations"""
        all_boxes = []
        all_weights = []
        
        # Method 1: Standard HOG with multiple scales
        for scale in [0.8, 1.0, 1.2]:
            if scale != 1.0:
                scaled_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            else:
                scaled_frame = frame
            
            # HOG detection with lower threshold for better sensitivity
            boxes1, weights1 = self.hog1.detectMultiScale(
                scaled_frame,
                winStride=(4, 4),  # Smaller stride for better detection
                padding=(16, 16),
                scale=1.02  # Smaller scale step
            )
            
            # Scale boxes back if needed
            if scale != 1.0:
                boxes1 = boxes1 / scale
                
            if len(boxes1) > 0:
                all_boxes.extend(boxes1.astype(int))
                all_weights.extend(weights1)
        
        # Method 2: Alternative HOG configuration
        boxes2, weights2 = self.hog2.detectMultiScale(
            frame,
            winStride=(6, 6),
            padding=(24, 24),
            scale=1.03
        )
        
        if len(boxes2) > 0:
            all_boxes.extend(boxes2.astype(int))
            all_weights.extend(weights2)
        
        return all_boxes, all_weights
    
    def detect_faces_and_bodies(self, frame):
        """Additional detection using cascade classifiers"""
        additional_boxes = []
        
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Face detection
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
            )
            
            # Convert face boxes to person boxes (estimate body from face)
            for (x, y, w, h) in faces:
                # Estimate body box from face
                body_w = int(w * 2.5)
                body_h = int(h * 6)
                body_x = max(0, x - int(w * 0.75))
                body_y = y
                
                # Ensure box is within frame
                body_x = max(0, min(body_x, frame.shape[1] - body_w))
                body_y = max(0, min(body_y, frame.shape[0] - body_h))
                body_w = min(body_w, frame.shape[1] - body_x)
                body_h = min(body_h, frame.shape[0] - body_y)
                
                additional_boxes.append([body_x, body_y, body_w, body_h, 'Face'])
            
            # Full body detection
            if self.body_cascade is not None:
                bodies = self.body_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 60)
                )
                
                for (x, y, w, h) in bodies:
                    additional_boxes.append([x, y, w, h, 'Body'])
        
        return additional_boxes
    
    def detect_motion_enhanced(self, frame):
        """Enhanced motion detection with better filtering"""
        fgMask = self.backSub.apply(frame)
        
        # Enhanced morphological operations
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel1)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel2)
        
        # Find contours
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 400 < area < 15000:  # Wider area range for crowds
                x, y, w, h = cv2.boundingRect(contour)
                
                # More flexible aspect ratio for crowded scenes
                aspect_ratio = h / w if w > 0 else 0
                if 0.8 < aspect_ratio < 5.0:
                    motion_boxes.append([x, y, w, h, 'Motion'])
        
        return motion_boxes
    
    def group_into_crowds(self, all_detections):
        """Group overlapping detections into crowd clusters using DBSCAN"""
        if len(all_detections) == 0:
            return []
        
        # Extract centers of all detections
        centers = []
        for detection in all_detections:
            x, y, w, h = detection[:4]
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append([center_x, center_y])
        
        centers = np.array(centers)
        
        # Use DBSCAN clustering to group nearby detections
        # eps = maximum distance between points in same cluster
        # min_samples = minimum points to form a cluster
        clustering = DBSCAN(eps=80, min_samples=1).fit(centers)
        labels = clustering.labels_
        
        # Group detections by cluster
        crowds = {}
        for i, label in enumerate(labels):
            if label not in crowds:
                crowds[label] = []
            crowds[label].append(all_detections[i])
        
        # Create crowd bounding boxes
        crowd_boxes = []
        for crowd_id, detections in crowds.items():
            if len(detections) == 1:
                # Single person
                detection = detections[0]
                crowd_boxes.append({
                    'box': detection[:4],
                    'type': 'individual',
                    'count': 1,
                    'detections': detections,
                    'methods': [detection[4]] if len(detection) > 4 else ['Unknown']
                })
            else:
                # Multiple overlapping detections = crowd
                # Calculate bounding box that encompasses all detections
                min_x = min([d[0] for d in detections])
                min_y = min([d[1] for d in detections])
                max_x = max([d[0] + d[2] for d in detections])
                max_y = max([d[1] + d[3] for d in detections])
                
                crowd_box = [min_x, min_y, max_x - min_x, max_y - min_y]
                
                # Estimate number of people in crowd based on area and detection count
                crowd_area = (max_x - min_x) * (max_y - min_y)
                estimated_people = max(len(detections), int(crowd_area / 3000))  # Rough estimate
                
                methods = list(set([d[4] for d in detections if len(d) > 4]))
                
                crowd_boxes.append({
                    'box': crowd_box,
                    'type': 'crowd',
                    'count': estimated_people,
                    'detections': detections,
                    'methods': methods
                })
        
        return crowd_boxes
    
    def merge_all_detections(self, hog_boxes, additional_boxes, motion_boxes):
        """Merge all detection methods"""
        all_detections = []
        
        # Add HOG detections
        for i, box in enumerate(hog_boxes):
            x, y, w, h = box
            all_detections.append([x, y, w, h, 'HOG'])
        
        # Add additional detections (face/body)
        all_detections.extend(additional_boxes)
        
        # Add motion detections
        all_detections.extend(motion_boxes)
        
        return all_detections
    
    def draw_crowd_detections(self, frame, crowd_boxes):
        """Draw crowd-aware detections"""
        total_people = 0
        
        for i, crowd_info in enumerate(crowd_boxes):
            x, y, w, h = crowd_info['box']
            crowd_type = crowd_info['type']
            people_count = crowd_info['count']
            methods = crowd_info['methods']
            
            total_people += people_count
            
            # Choose color based on crowd type
            if crowd_type == 'crowd':
                color = (0, 0, 255)  # Red for crowds
                label = f"Crowd: {people_count} people"
                thickness = 3
            else:
                color = (0, 255, 0)  # Green for individuals
                label = f"Person ({', '.join(methods)})"
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 15), 
                         (x + label_size[0] + 10, y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # For crowds, show individual detection boxes in lighter color
            if crowd_type == 'crowd':
                light_color = tuple([int(c * 0.5) for c in color])
                for detection in crowd_info['detections']:
                    dx, dy, dw, dh = detection[:4]
                    cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), light_color, 1)
        
        return total_people
    
    def process_video(self, video_path, output_path):
        """Process video with enhanced crowd detection"""
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
            
            # Multi-method detection
            hog_boxes, hog_weights = self.detect_people_multi_hog(frame)
            additional_boxes = self.detect_faces_and_bodies(frame)
            motion_boxes = self.detect_motion_enhanced(frame)
            
            # Merge all detections
            all_detections = self.merge_all_detections(hog_boxes, additional_boxes, motion_boxes)
            
            # Group into crowds
            crowd_boxes = self.group_into_crowds(all_detections)
            
            # Draw crowd-aware detections
            total_people = self.draw_crowd_detections(frame, crowd_boxes)
            
            # Track person count
            person_count_history.append(total_people)
            
            # Calculate crowd statistics
            if len(person_count_history) > 30:
                avg_count = np.mean(person_count_history[-30:])
            else:
                avg_count = total_people
            
            # Determine crowd level
            if avg_count < 5:
                crowd_level = "LOW"
                crowd_color = (0, 255, 0)  # Green
            elif avg_count < 15:
                crowd_level = "MEDIUM"
                crowd_color = (0, 255, 255)  # Yellow
            else:
                crowd_level = "HIGH"
                crowd_color = (0, 0, 255)  # Red
            
            # Add information overlay
            overlay_height = 140
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, overlay_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Add text information following the workflow
            cv2.putText(frame, f"YOLO + TRACKING + DBSCAN CROWD DETECTION",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"People Detected: {total_people}",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Crowds Formed: {len([c for c in crowd_boxes if c['type'] == 'crowd'])}",
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Crowd Level: {crowd_level}",
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, crowd_color, 2)
            cv2.putText(frame, f"Avg: {avg_count:.1f} people",
                       (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add frame info
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                       (width - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, f"Time: {timestamp}", 
                       (width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Pipeline: Detection→Tracking→Clustering",
                       (width - 320, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Write frame
            out.write(frame)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Print summary
        if person_count_history:
            avg_people = np.mean(person_count_history)
            max_people = max(person_count_history)
            print(f"\n✅ Enhanced processing complete!")
            print(f"📊 Statistics:")
            print(f"   - Average people detected: {avg_people:.1f}")
            print(f"   - Maximum people detected: {max_people}")
            print(f"   - Total frames processed: {frame_count}")
            print(f"💾 Output saved to: {output_path}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Enhanced Crowd Detection with Clustering')
    parser.add_argument('--video', default='data/video/input.mp4', help='Input video path')
    parser.add_argument('--output', default='outputs/crowd_detection_enhanced.mp4', help='Output video path')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        # Create detector
        detector = EnhancedCrowdDetector()
        
        # Process video
        success = detector.process_video(args.video, args.output)
        
        if success:
            print("🎉 Enhanced crowd detection completed successfully!")
        else:
            print("❌ Enhanced crowd detection failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
