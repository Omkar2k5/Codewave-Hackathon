#!/usr/bin/env python3
"""
Simplified but proper crowd detection following the workflow:
Input: Video → YOLO Detection → Simple Tracking → DBSCAN Clustering → Output
Uses OpenCV DNN to avoid TensorFlow issues
"""

import cv2
import numpy as np
import os
import argparse
from datetime import datetime
from sklearn.cluster import DBSCAN
import math

class SimpleTracker:
    """Simple tracking implementation to replace DeepSORT"""
    def __init__(self, max_disappeared=10, max_distance=100):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid):
        """Register a new object"""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, object_id):
        """Deregister an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, rects):
        """Update tracker with new detections"""
        if len(rects) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.get_objects()
        
        # Initialize centroids array
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        
        # Compute centroids
        for (i, (x, y, w, h)) in enumerate(rects):
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids[i] = (cx, cy)
        
        # If no existing objects, register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Get existing object centroids
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())
            
            # Compute distance matrix
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            # Keep track of used row and column indices
            used_rows = set()
            used_cols = set()
            
            # Update existing objects
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if D[row, col] <= self.max_distance:
                    object_id = object_ids[row]
                    self.objects[object_id] = input_centroids[col]
                    self.disappeared[object_id] = 0
                    
                    used_rows.add(row)
                    used_cols.add(col)
            
            # Handle unmatched detections and objects
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            
            if D.shape[0] >= D.shape[1]:
                # More objects than detections
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # More detections than objects
                for col in unused_cols:
                    self.register(input_centroids[col])
        
        return self.get_objects()
    
    def get_objects(self):
        """Get current tracked objects"""
        return self.objects

class YOLOTrackingCrowdDetector:
    def __init__(self, weights_path, config_path, classes_path):
        self.weights_path = weights_path
        self.config_path = config_path
        self.classes_path = classes_path
        
        # Load YOLO
        self.net = cv2.dnn.readNet(weights_path, config_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Load classes
        with open(classes_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Initialize tracker
        self.tracker = SimpleTracker(max_disappeared=30, max_distance=80)
        
        print("✅ YOLO + Simple Tracking + DBSCAN system initialized!")
    
    def detect_people_yolo(self, frame, confidence_threshold=0.3, nms_threshold=0.4):
        """Step 1: YOLO Detection - Find people in frame"""
        height, width = frame.shape[:2]
        
        # Prepare input
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Only detect persons (class_id = 0)
                if class_id == 0 and confidence > confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply NMS
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        final_boxes = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                final_boxes.append(boxes[i])

        # Debug: Print detection info occasionally
        if len(boxes) > 0:
            print(f"Debug: Found {len(boxes)} raw detections, {len(final_boxes)} after NMS")

        return final_boxes
    
    def track_people(self, boxes):
        """Step 2: Simple Tracking - Track each person maintaining identity"""
        tracked_objects = self.tracker.update(boxes)
        
        # Convert to format with bounding boxes
        tracked_with_boxes = []
        for track_id, centroid in tracked_objects.items():
            # Find closest box to this centroid
            if boxes:
                distances = []
                for box in boxes:
                    box_center = [box[0] + box[2]//2, box[1] + box[3]//2]
                    dist = math.sqrt((centroid[0] - box_center[0])**2 + (centroid[1] - box_center[1])**2)
                    distances.append(dist)
                
                closest_box_idx = np.argmin(distances)
                if distances[closest_box_idx] < 100:  # Within reasonable distance
                    box = boxes[closest_box_idx]
                    tracked_with_boxes.append({
                        'track_id': track_id,
                        'bbox': box,
                        'center': centroid
                    })
        
        return tracked_with_boxes
    
    def cluster_crowds_dbscan(self, tracked_objects):
        """Step 3: DBSCAN Clustering - Group individuals into crowds"""
        if len(tracked_objects) == 0:
            return []
        elif len(tracked_objects) == 1:
            return [{'type': 'individual', 'objects': tracked_objects, 'center': tracked_objects[0]['center'], 'count': 1}]
        
        # Extract centers for clustering
        centers = np.array([obj['center'] for obj in tracked_objects])
        
        # DBSCAN clustering
        # eps: maximum distance between two samples to be in same cluster (pixels)
        # min_samples: minimum number of samples in a cluster
        clustering = DBSCAN(eps=120, min_samples=2).fit(centers)
        labels = clustering.labels_
        
        # Group objects by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(tracked_objects[i])
        
        # Create crowd groups
        crowd_groups = []
        for cluster_id, objects in clusters.items():
            if cluster_id == -1:
                # Noise points (individuals)
                for obj in objects:
                    crowd_groups.append({
                        'type': 'individual',
                        'objects': [obj],
                        'center': obj['center'],
                        'count': 1
                    })
            else:
                # Crowd cluster
                crowd_center = np.mean([obj['center'] for obj in objects], axis=0)
                crowd_groups.append({
                    'type': 'crowd',
                    'objects': objects,
                    'center': crowd_center,
                    'count': len(objects)
                })
        
        return crowd_groups
    
    def draw_annotations(self, frame, crowd_groups):
        """Step 4: Draw annotations showing detected crowds"""
        total_people = 0
        crowd_count = 0
        
        for group in crowd_groups:
            total_people += group['count']
            
            if group['type'] == 'crowd':
                crowd_count += 1
                # Calculate crowd bounding box
                min_x = min([obj['bbox'][0] for obj in group['objects']])
                min_y = min([obj['bbox'][1] for obj in group['objects']])
                max_x = max([obj['bbox'][0] + obj['bbox'][2] for obj in group['objects']])
                max_y = max([obj['bbox'][1] + obj['bbox'][3] for obj in group['objects']])
                
                # Draw crowd box in red
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 3)
                cv2.putText(frame, f"Crowd: {group['count']} people", 
                           (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw individual tracks within crowd (lighter red)
                for obj in group['objects']:
                    x, y, w, h = obj['bbox']
                    track_id = obj['track_id']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 255), 2)
                    cv2.putText(frame, f"ID:{track_id}", 
                               (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
                    # Draw center point
                    cv2.circle(frame, tuple(obj['center']), 3, (0, 0, 255), -1)
            else:
                # Individual person
                if len(group['objects']) > 0:
                    obj = group['objects'][0]
                    x, y, w, h = obj['bbox']
                    track_id = obj['track_id']

                    # Draw individual box in green
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person ID:{track_id}",
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Draw center point
                    cv2.circle(frame, tuple(obj['center']), 3, (0, 255, 0), -1)
        
        return total_people, crowd_count
    
    def process_video(self, video_path, output_path):
        """Main processing pipeline following the specified workflow"""
        print(f"🎬 Processing video: {video_path}")
        print(f"💾 Output will be saved to: {output_path}")
        print("📋 Workflow: Video → YOLO Detection → Simple Tracking → DBSCAN Clustering → Output")
        
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
        people_history = []
        crowd_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"🔄 Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
            
            # Step 1: YOLO Detection
            boxes = self.detect_people_yolo(frame, confidence_threshold=0.4)
            
            # Step 2: Simple Tracking
            tracked_objects = self.track_people(boxes)
            
            # Step 3: DBSCAN Clustering
            crowd_groups = self.cluster_crowds_dbscan(tracked_objects)
            
            # Step 4: Draw Annotations
            total_people, crowd_count = self.draw_annotations(frame, crowd_groups)
            
            # Track statistics
            people_history.append(total_people)
            crowd_history.append(crowd_count)
            
            # Calculate averages
            if len(people_history) > 30:
                avg_people = np.mean(people_history[-30:])
                avg_crowds = np.mean(crowd_history[-30:])
            else:
                avg_people = total_people
                avg_crowds = crowd_count
            
            # Determine crowd level
            if avg_people < 5:
                crowd_level = "LOW"
                crowd_color = (0, 255, 0)
            elif avg_people < 15:
                crowd_level = "MEDIUM"
                crowd_color = (0, 255, 255)
            else:
                crowd_level = "HIGH"
                crowd_color = (0, 0, 255)
            
            # Add information overlay
            overlay_height = 140
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, overlay_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Add text information
            cv2.putText(frame, f"YOLO + TRACKING + DBSCAN CROWD DETECTION", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"People Detected: {total_people}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Crowds Formed: {crowd_count}", 
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Crowd Level: {crowd_level}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, crowd_color, 2)
            cv2.putText(frame, f"Avg: {avg_people:.1f} people, {avg_crowds:.1f} crowds", 
                       (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add workflow info
            cv2.putText(frame, f"Pipeline: YOLO→Tracking→DBSCAN", 
                       (width - 280, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                       (width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, f"Time: {timestamp}", 
                       (width - 150, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Print summary
        if people_history:
            avg_people = np.mean(people_history)
            max_people = max(people_history)
            avg_crowds = np.mean(crowd_history)
            max_crowds = max(crowd_history)
            
            print(f"\n✅ YOLO + Tracking + DBSCAN processing complete!")
            print(f"📊 Statistics:")
            print(f"   - Average people detected: {avg_people:.1f}")
            print(f"   - Maximum people detected: {max_people}")
            print(f"   - Average crowds formed: {avg_crowds:.1f}")
            print(f"   - Maximum crowds formed: {max_crowds}")
            print(f"   - Total frames processed: {frame_count}")
            print(f"💾 Output saved to: {output_path}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='YOLO + Tracking + DBSCAN Crowd Detection')
    parser.add_argument('--video', default='data/video/input.mp4', help='Input video path')
    parser.add_argument('--output', default='outputs/crowd_detection_proper.mp4', help='Output video path')
    parser.add_argument('--weights', default='data/yolov4.weights', help='YOLO weights path')
    parser.add_argument('--config', default='data/yolov4.cfg', help='YOLO config path')
    parser.add_argument('--classes', default='data/classes/coco.names', help='Classes file path')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        # Create detector
        detector = YOLOTrackingCrowdDetector(args.weights, args.config, args.classes)
        
        # Process video
        success = detector.process_video(args.video, args.output)
        
        if success:
            print("🎉 YOLO + Tracking + DBSCAN crowd detection completed successfully!")
        else:
            print("❌ Crowd detection failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
