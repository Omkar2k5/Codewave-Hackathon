#!/usr/bin/env python3
"""
OpenCV-based crowd detection that mimics op_1.mp4 style
Uses multiple detection methods for robust person detection
"""

import cv2
import numpy as np
import os
import argparse
from datetime import datetime

class OpenCVCrowdDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.tracker_id = 0
        self.tracks = {}
        
    def detect_people_hog(self, frame):
        """Detect people using HOG descriptor"""
        # Resize frame for better performance
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Detect people
        boxes, weights = self.hog.detectMultiScale(small_frame, winStride=(8,8), padding=(32,32), scale=1.05)
        
        # Scale boxes back to original size
        boxes = boxes / scale
        boxes = boxes.astype(int)
        
        return boxes, weights
    
    def detect_motion(self, frame):
        """Detect motion using background subtraction"""
        fgMask = self.backSub.apply(frame)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 800 < area < 8000:  # Filter by area (person-sized objects)
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by aspect ratio (people are taller than wide)
                aspect_ratio = h / w if w > 0 else 0
                if 1.2 < aspect_ratio < 4.0:
                    motion_boxes.append([x, y, w, h])
        
        return motion_boxes
    
    def merge_detections(self, hog_boxes, motion_boxes, frame_shape):
        """Merge HOG and motion detections"""
        all_boxes = []
        
        # Add HOG detections
        for box in hog_boxes:
            x, y, w, h = box
            all_boxes.append([x, y, w, h, 'HOG'])
        
        # Add motion detections
        for box in motion_boxes:
            x, y, w, h = box
            all_boxes.append([x, y, w, h, 'Motion'])
        
        # Simple NMS to remove overlapping boxes
        final_boxes = []
        for i, box1 in enumerate(all_boxes):
            x1, y1, w1, h1, type1 = box1
            overlap = False
            
            for j, box2 in enumerate(all_boxes):
                if i != j:
                    x2, y2, w2, h2, type2 = box2
                    
                    # Calculate overlap
                    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    overlap_area = overlap_x * overlap_y
                    
                    area1 = w1 * h1
                    area2 = w2 * h2
                    
                    if overlap_area > 0.3 * min(area1, area2):
                        # Prefer HOG detections over motion
                        if type1 == 'Motion' and type2 == 'HOG':
                            overlap = True
                            break
            
            if not overlap:
                final_boxes.append([x1, y1, w1, h1, type1])
        
        return final_boxes
    
    def draw_detections(self, frame, boxes):
        """Draw detection boxes with tracking-style visualization"""
        person_count = len(boxes)
        
        for i, box in enumerate(boxes):
            x, y, w, h, detection_type = box
            
            # Choose color based on detection type
            if detection_type == 'HOG':
                color = (0, 255, 0)  # Green for HOG
                label = f"Person {i+1} (HOG)"
            else:
                color = (255, 0, 0)  # Blue for Motion
                label = f"Person {i+1} (Motion)"
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 3, color, -1)
            
            # Draw ID number
            cv2.putText(frame, f"#{i+1}", (x + w - 30, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return person_count
    
    def process_video(self, video_path, output_path):
        """Process video with crowd detection"""
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
            
            # Detect people using HOG
            hog_boxes, hog_weights = self.detect_people_hog(frame)
            
            # Detect motion
            motion_boxes = self.detect_motion(frame)
            
            # Merge detections
            final_boxes = self.merge_detections(hog_boxes, motion_boxes, frame.shape)
            
            # Draw detections
            person_count = self.draw_detections(frame, final_boxes)
            
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
            cv2.putText(frame, f"REAL-TIME CROWD DETECTION", 
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
            
            # Add detection method info
            cv2.putText(frame, f"Methods: HOG + Motion", 
                       (width - 200, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
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
    parser = argparse.ArgumentParser(description='OpenCV Crowd Detection')
    parser.add_argument('--video', default='data/video/input.mp4', help='Input video path')
    parser.add_argument('--output', default='outputs/crowd_detection_opencv.mp4', help='Output video path')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        # Create detector
        detector = OpenCVCrowdDetector()
        
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
