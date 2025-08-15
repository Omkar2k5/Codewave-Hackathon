#!/usr/bin/env python3
"""
Live Camera Crowd Detection with CV2 Window
==========================================

This script opens a CV2 window, uses your laptop's camera for live video,
and overlays the crowd detection model to show real-time detection.

Features:
- Live camera feed in CV2 window
- Real-time person detection with bounding boxes
- Live statistics and performance metrics
- Interactive controls
- Visual feedback showing model processing

Author: Crowd Detection System
Date: 2025-01-15
"""

import cv2
import numpy as np
import time
from datetime import datetime
import os
from sklearn.cluster import DBSCAN
from collections import deque, defaultdict
import math

# YOLOv8 imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLOv8 available for live detection")
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ YOLOv8 not available. Install with: pip install ultralytics")

class LiveCameraDetection:
    """
    Live camera crowd detection with CV2 window display
    Optimized for smooth video streaming with CUDA support
    """

    def __init__(self, model_path='yolov8m.pt', camera_id=0):
        self.model_path = model_path
        self.camera_id = camera_id
        self.model = None

        # Detection parameters
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45

        # Performance optimization settings
        self.use_cuda = False
        self.detection_skip_frames = 2  # Process every 3rd frame for speed
        self.frame_counter = 0
        self.last_groups = []  # Store last detection results for skipped frames
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.processing_time = 0
        
        # Statistics
        self.frame_count = 0
        self.total_people_detected = 0
        self.max_people_in_frame = 0
        self.detection_history = []

        # Enhanced tracking system with temporal filtering (from advanced_crowd_detection.py)
        self.next_id = 0
        self.tracks = {}
        self.disappeared = {}
        self.max_disappeared = 15
        self.max_distance = 80

        # Temporal filtering for smooth bounding boxes
        self.bbox_history = defaultdict(lambda: deque(maxlen=3))
        self.confidence_history = defaultdict(lambda: deque(maxlen=2))

        # Group tracking and merging
        self.group_rectangles = {}
        self.proximity_threshold = 80  # Distance threshold for merging rectangles
        self.movement_trails = defaultdict(lambda: deque(maxlen=30))  # Track movement for 30 frames
        
        # Colors for visualization
        self.colors = {
            'high_conf': (0, 255, 0),      # Green for high confidence
            'medium_conf': (0, 255, 255),  # Yellow for medium confidence  
            'low_conf': (0, 165, 255),     # Orange for low confidence
            'text': (255, 255, 255),       # White for text
            'background': (0, 0, 0),       # Black for backgrounds
            'border': (255, 255, 255)      # White for borders
        }
        
        print(f"🎥 Initialized Live Camera Detection")
        print(f"📁 Model: {model_path}")
        print(f"📹 Camera ID: {camera_id}")
    
    def load_model(self):
        """Load the YOLOv8 model with CUDA optimization"""
        if not YOLO_AVAILABLE:
            print("❌ YOLOv8 not available")
            return False

        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False

            self.model = YOLO(self.model_path)

            # Check for CUDA availability and configure
            try:
                import torch
                if torch.cuda.is_available():
                    self.use_cuda = True
                    # Move model to GPU
                    self.model.to('cuda')
                    print(f"✅ Model loaded with CUDA acceleration: {self.model_path}")
                    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
                else:
                    print(f"✅ Model loaded (CPU mode): {self.model_path}")
                    print("⚠️ CUDA not available, using CPU")
            except ImportError:
                print(f"✅ Model loaded (CPU mode): {self.model_path}")
                print("⚠️ PyTorch not available for CUDA detection")

            return True

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def detect_people(self, frame):
        """Detect people in the current frame with performance optimization"""
        start_time = time.time()

        # Frame skipping for performance optimization
        self.frame_counter += 1
        if self.frame_counter % (self.detection_skip_frames + 1) != 0:
            # Return previous detections for skipped frames
            self.processing_time = time.time() - start_time
            return getattr(self, 'last_detections', [])

        try:
            # Configure inference parameters for speed
            device = 'cuda' if self.use_cuda else 'cpu'

            # Run YOLOv8 detection with optimized settings
            results = self.model(
                frame,
                verbose=False,
                conf=self.confidence_threshold,
                device=device,
                half=self.use_cuda,  # Use FP16 for CUDA
                imgsz=640  # Fixed input size for consistency
            )

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class ID and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])

                        # Only detect persons (class 0 in COCO dataset)
                        if class_id == 0 and confidence > self.confidence_threshold:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                            # Format for tracking: [x, y, w, h, method, confidence]
                            detections.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1), 'YOLOv8', confidence])

            # Store for frame skipping
            self.last_detections = detections

            # Calculate processing time
            self.processing_time = time.time() - start_time

            return detections

        except Exception as e:
            print(f"❌ Detection error: {e}")
            self.processing_time = time.time() - start_time
            return []

    def apply_temporal_filtering(self, track_id, bbox, confidence):
        """Apply temporal filtering for smooth bounding boxes"""
        # Add to history
        self.bbox_history[track_id].append(bbox)
        self.confidence_history[track_id].append(confidence)

        # Calculate smoothed values
        if len(self.bbox_history[track_id]) > 1:
            # Average the bounding boxes
            bboxes = list(self.bbox_history[track_id])
            smoothed_bbox = [
                int(np.mean([b[0] for b in bboxes])),  # x
                int(np.mean([b[1] for b in bboxes])),  # y
                int(np.mean([b[2] for b in bboxes])),  # w
                int(np.mean([b[3] for b in bboxes]))   # h
            ]
        else:
            smoothed_bbox = bbox

        # Average confidence
        if len(self.confidence_history[track_id]) > 1:
            smoothed_confidence = np.mean(list(self.confidence_history[track_id]))
        else:
            smoothed_confidence = confidence

        return smoothed_bbox, smoothed_confidence

    def update_tracking(self, detections):
        """Update tracking system with new detections"""
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    if track_id in self.tracks:
                        del self.tracks[track_id]
                    del self.disappeared[track_id]
            return []

        # Extract centers from detections
        detection_centers = []
        for det in detections:
            x, y, w, h = det[:4]
            center = [x + w//2, y + h//2]
            detection_centers.append(center)

        # If no existing tracks, create new ones
        if len(self.tracks) == 0:
            for i, det in enumerate(detections):
                x, y, w, h = det[:4]
                self.tracks[self.next_id] = {
                    'center': detection_centers[i],
                    'bbox': [x, y, w, h],
                    'method': det[4],
                    'confidence': det[5],
                    'age': 0
                }
                self.disappeared[self.next_id] = 0
                self.next_id += 1
        else:
            # Match detections to existing tracks
            track_centers = [track['center'] for track in self.tracks.values()]
            track_ids = list(self.tracks.keys())

            if len(track_centers) > 0 and len(detection_centers) > 0:
                # Calculate distance matrix
                D = np.linalg.norm(
                    np.array(track_centers)[:, np.newaxis] - np.array(detection_centers), axis=2
                )

                # Hungarian algorithm (simplified)
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                used_rows = set()
                used_cols = set()

                # Update matched tracks
                for row, col in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue

                    if D[row, col] <= self.max_distance:
                        track_id = track_ids[row]
                        det = detections[col]
                        x, y, w, h = det[:4]

                        # Apply temporal filtering for smooth bounding boxes
                        smoothed_bbox, smoothed_confidence = self.apply_temporal_filtering(
                            track_id, [x, y, w, h], det[5]
                        )

                        # Update smoothed center
                        smoothed_center = [smoothed_bbox[0] + smoothed_bbox[2]//2,
                                         smoothed_bbox[1] + smoothed_bbox[3]//2]

                        self.tracks[track_id].update({
                            'center': smoothed_center,
                            'bbox': smoothed_bbox,
                            'method': det[4],
                            'confidence': smoothed_confidence,
                            'age': self.tracks[track_id]['age'] + 1
                        })
                        self.disappeared[track_id] = 0

                        used_rows.add(row)
                        used_cols.add(col)

                # Handle unmatched tracks
                for row in range(len(track_ids)):
                    if row not in used_rows:
                        track_id = track_ids[row]
                        self.disappeared[track_id] += 1
                        if self.disappeared[track_id] > self.max_disappeared:
                            del self.tracks[track_id]
                            del self.disappeared[track_id]

                # Create new tracks for unmatched detections
                for col in range(len(detections)):
                    if col not in used_cols:
                        det = detections[col]
                        x, y, w, h = det[:4]
                        self.tracks[self.next_id] = {
                            'center': detection_centers[col],
                            'bbox': [x, y, w, h],
                            'method': det[4],
                            'confidence': det[5],
                            'age': 0
                        }
                        self.disappeared[self.next_id] = 0
                        self.next_id += 1

        # Skip movement trails for performance (not displayed)

        # Return tracks as list of tuples for grouping
        return [(track_id, track) for track_id, track in self.tracks.items()]

    def should_merge_boxes(self, box1, box2):
        """Check if two boxes should be merged based on proximity and overlap"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate centers
        center1 = [(x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2]
        center2 = [(x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2]

        # Calculate distance between centers
        distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

        # Check if boxes overlap or are close enough
        overlap_x = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        overlap_y = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        overlap_area = overlap_x * overlap_y

        # Merge if overlapping or within proximity threshold
        return overlap_area > 0 or distance < self.proximity_threshold

    def merge_nearby_rectangles(self, tracks):
        """Merge ALL overlapping and nearby rectangles into unified groups"""
        if len(tracks) == 0:
            return []

        # Extract bounding boxes and track info
        boxes = []
        track_info = []
        for track_id, track in tracks:
            x, y, w, h = track['bbox']
            boxes.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2]
            track_info.append((track_id, track))

        # Use Union-Find (Disjoint Set) algorithm for proper grouping
        parent = list(range(len(boxes)))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Find all pairs that should be merged
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if self.should_merge_boxes(boxes[i], boxes[j]):
                    union(i, j)

        # Group boxes by their root parent
        groups_dict = {}
        for i in range(len(boxes)):
            root = find(i)
            if root not in groups_dict:
                groups_dict[root] = []
            groups_dict[root].append(i)

        # Create final groups
        groups = []
        for group_indices in groups_dict.values():
            # Calculate merged bounding box
            min_x1 = min(boxes[i][0] for i in group_indices)
            min_y1 = min(boxes[i][1] for i in group_indices)
            max_x2 = max(boxes[i][2] for i in group_indices)
            max_y2 = max(boxes[i][3] for i in group_indices)

            # Create group info
            group_tracks = [track_info[idx] for idx in group_indices]
            groups.append({
                'bbox': [min_x1, min_y1, max_x2 - min_x1, max_y2 - min_y1],  # [x, y, w, h]
                'tracks': group_tracks,
                'count': len(group_tracks),
                'is_group': len(group_tracks) > 1,
                'individual_boxes': [boxes[i] for i in group_indices]
            })

        return groups

    def get_confidence_color(self, confidence):
        """Get color based on confidence level"""
        if confidence >= 0.8:
            return self.colors['high_conf']
        elif confidence >= 0.6:
            return self.colors['medium_conf']
        else:
            return self.colors['low_conf']
    
    def draw_merged_groups(self, frame, groups):
        """Draw merged group rectangles with crowd detection"""
        for group in groups:
            x, y, w, h = group['bbox']
            count = group['count']
            is_group = group['is_group']

            # Choose color based on group size
            if is_group:
                if count == 2:
                    color = (0, 255, 255)  # Cyan for pairs
                    label = f"PAIR ({count})"
                elif count <= 4:
                    color = (0, 165, 255)  # Orange for small groups
                    label = f"SMALL GROUP ({count})"
                elif count <= 8:
                    color = (0, 100, 255)  # Dark orange for medium groups
                    label = f"MEDIUM GROUP ({count})"
                else:
                    color = (0, 0, 255)  # Red for large groups/crowds
                    label = f"CROWD ({count})"
                thickness = 4
            else:
                # Individual person
                track_id, track = group['tracks'][0]
                method = track['method']
                confidence = track['confidence']

                if method == 'YOLOv8':
                    color = (255, 0, 255)  # Magenta for YOLOv8
                    thickness = 3
                    label = f"PERSON ({confidence:.2f})"
                else:
                    color = (0, 255, 0)  # Green for other methods
                    thickness = 2
                    label = f"PERSON ({confidence:.2f})"

            # Draw main bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

            # Draw individual boxes within groups (lighter color)
            if is_group and len(group['individual_boxes']) > 1:
                light_color = tuple(int(c * 0.6) for c in color)
                for box in group['individual_boxes']:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), light_color, 1)

            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Label background
            cv2.rectangle(frame, (x, y - label_size[1] - 10),
                         (x + label_size[0] + 10, y), color, -1)

            # Label text
            cv2.putText(frame, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)

            # Draw count indicator for groups
            if is_group:
                # Draw circle with count
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(frame, (center_x, center_y), 20, color, -1)
                cv2.putText(frame, str(count), (center_x - 8, center_y + 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 2)

        return frame

    # Movement trails removed for cleaner display and better performance
    
    def calculate_crowd_metrics(self, groups, frame_shape):
        """Calculate crowd density and metrics from groups"""
        height, width = frame_shape[:2]

        # Count total people and groups
        total_people = sum(group['count'] for group in groups)
        num_groups = len([g for g in groups if g['is_group']])
        num_individuals = len([g for g in groups if not g['is_group']])
        largest_group = max([g['count'] for g in groups], default=0)

        # Calculate density per area
        frame_area = width * height
        people_density = (total_people / frame_area) * 10000  # Per 10k pixels

        # Determine density level and color based on total people and grouping
        if total_people == 0:
            density_level = "Empty"
            density_color = (128, 128, 128)  # Gray
        elif total_people <= 2:
            density_level = "Very Low"
            density_color = (0, 255, 0)  # Green
        elif total_people <= 5:
            density_level = "Low"
            density_color = (0, 255, 128)  # Light green
        elif total_people <= 10:
            density_level = "Medium"
            density_color = (0, 255, 255)  # Yellow
        elif total_people <= 20:
            density_level = "High"
            density_color = (0, 165, 255)  # Orange
        else:
            density_level = "Very High"
            density_color = (0, 0, 255)  # Red

        # Adjust for grouping behavior
        if num_groups > 0:
            if largest_group >= 8:
                density_level = "Crowded"
                density_color = (0, 0, 255)  # Red
            elif largest_group >= 5:
                density_level = "Grouped"
                density_color = (0, 100, 255)  # Dark orange

        # Calculate confidence statistics
        all_confidences = []
        for group in groups:
            for track_id, track in group['tracks']:
                all_confidences.append(track['confidence'])

        if all_confidences:
            avg_confidence = np.mean(all_confidences)
            max_confidence = np.max(all_confidences)
            min_confidence = np.min(all_confidences)
        else:
            avg_confidence = max_confidence = min_confidence = 0.0

        return {
            'total_people': total_people,
            'num_groups': num_groups,
            'num_individuals': num_individuals,
            'largest_group': largest_group,
            'density_level': density_level,
            'density_color': density_color,
            'people_density': people_density,
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'min_confidence': min_confidence
        }
    
    def draw_live_statistics(self, frame, metrics):
        """Draw live statistics overlay on the frame"""
        height, width = frame.shape[:2]
        
        # Main statistics panel (top-right)
        panel_width = 320
        panel_height = 180
        panel_x = width - panel_width - 10
        panel_y = 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     self.colors['border'], 2)
        
        # Title
        cv2.putText(frame, "LIVE CROWD DETECTION", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Statistics
        y_offset = panel_y + 50
        line_height = 18
        
        stats = [
            f"Frame: {self.frame_count}",
            f"People: {metrics['total_people']}",
            f"Groups: {metrics['num_groups']}",
            f"Individuals: {metrics['num_individuals']}",
            f"Largest Group: {metrics['largest_group']}",
            f"Density: {metrics['density_level']}",
            f"FPS: {self.current_fps:.1f}",
            f"Process: {self.processing_time*1000:.0f}ms",
            f"Device: {'CUDA' if self.use_cuda else 'CPU'}"
        ]
        
        for i, stat in enumerate(stats):
            color = self.colors['text']
            if "Density:" in stat:
                color = metrics['density_color']
            elif "FPS:" in stat and self.current_fps < 10:
                color = (0, 165, 255)  # Orange for low FPS
            
            cv2.putText(frame, stat, (panel_x + 10, y_offset + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def draw_controls_info(self, frame):
        """Draw control information at the bottom"""
        height, width = frame.shape[:2]
        
        controls = [
            "Controls: 'q'=Quit | 's'=Screenshot | 'r'=Record | SPACE=Pause"
        ]
        
        for i, control in enumerate(controls):
            y_pos = height - 20 + i * 20
            cv2.putText(frame, control, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return frame
    
    def update_statistics(self, groups):
        """Update detection statistics from groups"""
        self.frame_count += 1
        total_people = sum(group['count'] for group in groups)
        self.total_people_detected += total_people
        self.max_people_in_frame = max(self.max_people_in_frame, total_people)

        # Keep detection history for analysis
        self.detection_history.append(total_people)
        if len(self.detection_history) > 100:  # Keep last 100 frames
            self.detection_history.pop(0)
    
    def update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def save_screenshot(self, frame):
        """Save screenshot with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/live_camera_screenshot_{timestamp}.jpg"
        
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
        return filename

    def run_live_detection(self):
        """Main function to run live camera detection"""
        print("🚀 Starting Live Camera Detection...")

        # Load model
        if not self.load_model():
            print("❌ Failed to load model. Exiting.")
            return False

        # Initialize camera
        print(f"📹 Opening camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print(f"❌ Failed to open camera {self.camera_id}")
            print("💡 Try different camera IDs: 0, 1, 2...")
            return False

        # Set camera properties for optimal performance and smooth streaming
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPEG for better performance

        # Get actual camera properties
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

        print(f"✅ Camera opened successfully")
        print(f"📊 Camera resolution: {actual_width}x{actual_height} @ {actual_fps} FPS")
        print(f"🎯 Confidence threshold: {self.confidence_threshold}")
        print(f"🚀 Performance mode: {'CUDA Accelerated' if self.use_cuda else 'CPU'}")
        print(f"⚡ Frame skip optimization: Every {self.detection_skip_frames + 1} frames")

        # Create window
        window_name = 'Live Camera Crowd Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        print("\n🎮 Controls:")
        print("   'q' or ESC - Quit")
        print("   's' - Save screenshot")
        print("   'r' - Start/stop recording")
        print("   SPACE - Pause/resume")
        print("   '+' - Increase confidence")
        print("   '-' - Decrease confidence")

        print("\n🔴 Live detection starting... Press 'q' to quit")

        # Recording variables
        recording = False
        video_writer = None
        paused = False

        try:
            while True:
                if not paused:
                    # Capture frame
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Failed to capture frame")
                        break

                    # Step 1: Detect people in the frame
                    detections = self.detect_people(frame)

                    # Step 2: Update tracking system
                    tracks = self.update_tracking(detections)

                    # Step 3: Merge nearby rectangles into groups
                    groups = self.merge_nearby_rectangles(tracks)

                    # Step 4: Calculate crowd metrics from groups
                    metrics = self.calculate_crowd_metrics(groups, frame.shape)

                    # Step 5: Update statistics
                    self.update_statistics(groups)
                    self.update_fps()

                    # Step 6: Draw visualizations (optimized for performance)
                    # Draw merged group rectangles (main visualization)
                    frame = self.draw_merged_groups(frame, groups)

                    # Draw statistics and controls
                    frame = self.draw_live_statistics(frame, metrics)
                    frame = self.draw_controls_info(frame)

                    # Add recording indicator
                    if recording:
                        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                        cv2.putText(frame, "REC", (50, 35),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Add pause indicator
                    if paused:
                        cv2.putText(frame, "PAUSED", (frame.shape[1]//2 - 50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

                    # Record frame if recording
                    if recording and video_writer:
                        video_writer.write(frame)

                # Display frame
                cv2.imshow(window_name, frame)

                # Handle key presses with minimal wait for smoother video
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("👋 Quitting live detection...")
                    break

                elif key == ord('s'):
                    screenshot_path = self.save_screenshot(frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")

                elif key == ord('r'):
                    if recording:
                        # Stop recording
                        if video_writer:
                            video_writer.release()
                            video_writer = None
                        recording = False
                        print("⏹️ Recording stopped")
                    else:
                        # Start recording
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"outputs/live_camera_recording_{timestamp}.mp4"
                        os.makedirs("outputs", exist_ok=True)

                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(filename, fourcc, 20.0,
                                                     (frame.shape[1], frame.shape[0]))
                        recording = True
                        print(f"🔴 Recording started: {filename}")

                elif key == ord(' '):  # SPACE
                    paused = not paused
                    print(f"⏸️ {'Paused' if paused else 'Resumed'}")

                elif key == ord('+') or key == ord('='):
                    self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                    print(f"🎯 Confidence threshold: {self.confidence_threshold:.2f}")

                elif key == ord('-'):
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                    print(f"🎯 Confidence threshold: {self.confidence_threshold:.2f}")

        except KeyboardInterrupt:
            print("\n⚠️ Detection interrupted by user")

        except Exception as e:
            print(f"❌ Error during detection: {e}")

        finally:
            # Cleanup
            if recording and video_writer:
                video_writer.release()
                print("⏹️ Recording stopped and saved")

            cap.release()
            cv2.destroyAllWindows()

            # Print final statistics
            print("\n📊 Final Detection Statistics:")
            print(f"   - Total frames processed: {self.frame_count}")
            print(f"   - Total people detected: {self.total_people_detected}")
            print(f"   - Max people in single frame: {self.max_people_in_frame}")
            if self.detection_history:
                avg_people = sum(self.detection_history) / len(self.detection_history)
                print(f"   - Average people per frame: {avg_people:.1f}")
            print(f"   - Final FPS: {self.current_fps:.1f}")
            print(f"   - Average processing time: {self.processing_time*1000:.1f}ms")

        return True


def main():
    """Main function to run the live camera detection"""
    print("🎯 Live Camera Crowd Detection")
    print("=" * 50)

    # Check if model exists
    model_path = 'yolov8m.pt'
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("💡 Please ensure yolov8m.pt is in the current directory")
        return 1

    # Create detector instance
    detector = LiveCameraDetection(model_path=model_path, camera_id=0)

    # Run live detection
    success = detector.run_live_detection()

    if success:
        print("✅ Live detection completed successfully!")
        return 0
    else:
        print("❌ Live detection failed!")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
