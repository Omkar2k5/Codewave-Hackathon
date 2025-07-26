#!/usr/bin/env python3
"""
Advanced Crowd Detection System with Enhanced Features:
- Overlay bounding boxes on detected people
- Visualize heatmaps showing crowd density over time
- Dynamic zone labels (Congested/Normal/Clear)
- Temporal tracking for movement patterns
- Real-time statistics panel
- 768x576 resolution at 7 FPS (matching op_4.mp4 aesthetic)
"""

import cv2
import numpy as np
import os
import argparse
from datetime import datetime
from sklearn.cluster import DBSCAN
from collections import deque, defaultdict
import math
import colorsys

# YOLOv8 imports for high-accuracy detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLOv8 (ultralytics) available - High accuracy detection enabled")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLOv8 not available, falling back to HOG detection")
    print("💡 Install with: pip install ultralytics")

class AdvancedCrowdDetector:
    def __init__(self):
        # Initialize YOLOv8 model (highest accuracy detector)
        self.yolo_model = None
        self.use_yolo = False

        if YOLO_AVAILABLE:
            try:
                print("🔄 Loading YOLOv8 model (high accuracy)...")
                # Use YOLOv8 medium model for better accuracy (can use yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
                self.yolo_model = YOLO('yolov8m.pt')  # Medium model for good balance of speed/accuracy
                self.use_yolo = True
                print("✅ YOLOv8 medium model loaded successfully!")
                print("🎯 High-accuracy COCO pretrained detection enabled")
            except Exception as e:
                print(f"⚠️ Failed to load YOLOv8: {e}")
                print("📦 Falling back to traditional methods")
                self.yolo_model = None
                self.use_yolo = False

        # Initialize traditional detection components as fallback/supplement
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=500, varThreshold=30)
        
        # Enhanced tracking system with temporal filtering
        self.next_id = 0
        self.tracks = {}
        self.disappeared = {}
        self.max_disappeared = 15
        self.max_distance = 80

        # Temporal filtering for smooth bounding boxes
        self.bbox_history = defaultdict(lambda: deque(maxlen=5))  # Store last 5 bounding boxes
        self.confidence_history = defaultdict(lambda: deque(maxlen=3))  # Store confidence history

        # Optimized confidence thresholds
        self.yolo_confidence_threshold = 0.4  # Optimized for YOLOv8 to reduce false positives
        self.hog_confidence_threshold = 0.6   # Higher threshold for HOG to reduce noise
        
        # Group tracking and merging
        self.group_rectangles = {}  # Store merged rectangles for groups
        self.proximity_threshold = 80  # More aggressive distance threshold for merging rectangles
        self.movement_trails = defaultdict(lambda: deque(maxlen=30))  # Track movement for 30 frames

        # Statistics tracking
        self.frame_stats = deque(maxlen=100)
        self.zone_history = deque(maxlen=50)
        
        print("✅ Advanced Crowd Detection System initialized!")
        if self.use_yolo:
            print("🎯 Features: YOLOv8 High-Accuracy Detection | Temporal Filtering | Group Merging | Movement Tracking")
        else:
            print("📊 Features: HOG Detection | Temporal Filtering | Group Merging | Movement Tracking | Real-time Stats")
    
    def detect_people_enhanced(self, frame):
        """Enhanced people detection with YOLOv8 high-accuracy detection"""
        detections = []

        # Method 1: YOLOv8 Detection (Primary - Highest Accuracy)
        if self.use_yolo and self.yolo_model is not None:
            try:
                results = self.yolo_model(frame, verbose=False)

                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get class ID and confidence
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])

                            # Only detect persons (class 0 in COCO dataset)
                            if class_id == 0 and confidence > self.yolo_confidence_threshold:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                                detections.append([x, y, w, h, 'YOLOv8', confidence])
            except Exception as e:
                print(f"YOLOv8 detection error: {e}")

        # Method 2: HOG Detection (Fallback/Supplement)
        if not self.use_yolo or len(detections) == 0:
            try:
                boxes, weights = self.hog.detectMultiScale(
                    frame, winStride=(8, 8), padding=(16, 16), scale=1.05
                )
                for i, (x, y, w, h) in enumerate(boxes):
                    if weights[i] > self.hog_confidence_threshold:  # Optimized confidence threshold
                        detections.append([x, y, w, h, 'HOG', weights[i]])
            except:
                pass
        
        # Method 2: Background Subtraction
        fgMask = self.backSub.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 800 < area < 8000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                if 1.2 < aspect_ratio < 4.0:
                    confidence = min(area / 3000, 1.0)
                    detections.append([x, y, w, h, 'Motion', confidence])
        
        return detections

    def apply_temporal_filtering(self, track_id, bbox, confidence):
        """Apply temporal filtering for smooth bounding boxes"""
        # Store current bbox and confidence
        self.bbox_history[track_id].append(bbox)
        self.confidence_history[track_id].append(confidence)

        # Calculate smoothed bounding box using weighted average
        if len(self.bbox_history[track_id]) > 1:
            weights = np.linspace(0.1, 1.0, len(self.bbox_history[track_id]))
            weights = weights / weights.sum()

            smoothed_bbox = np.zeros(4)
            for i, (bbox_hist, weight) in enumerate(zip(self.bbox_history[track_id], weights)):
                smoothed_bbox += np.array(bbox_hist) * weight

            smoothed_bbox = smoothed_bbox.astype(int).tolist()
        else:
            smoothed_bbox = bbox

        # Calculate smoothed confidence
        smoothed_confidence = np.mean(list(self.confidence_history[track_id]))

        return smoothed_bbox, smoothed_confidence

    def validate_detection_quality(self, bbox, confidence, method):
        """Validate detection quality to reduce false positives"""
        x, y, w, h = bbox

        # Basic size validation
        if w < 20 or h < 40:  # Too small to be a person
            return False

        if w > 300 or h > 500:  # Too large, likely false positive
            return False

        # Aspect ratio validation
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < 1.2 or aspect_ratio > 4.0:  # Not person-like aspect ratio
            return False

        # Method-specific confidence validation
        if method == 'YOLOv8' and confidence < self.yolo_confidence_threshold:
            return False
        elif method == 'HOG' and confidence < self.hog_confidence_threshold:
            return False

        return True
    
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
                
                # Hungarian-like assignment (simplified)
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
        
        # Update movement trails
        for track_id, track in self.tracks.items():
            self.movement_trails[track_id].append(track['center'])
        
        # Clean up old trails
        active_ids = set(self.tracks.keys())
        for trail_id in list(self.movement_trails.keys()):
            if trail_id not in active_ids:
                del self.movement_trails[trail_id]
        
        return list(self.tracks.items())
    
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

        # Create final groups with merged bounding boxes
        groups = []
        for group_indices in groups_dict.values():
            # Calculate unified bounding box for this group
            min_x1 = min(boxes[i][0] for i in group_indices)
            min_y1 = min(boxes[i][1] for i in group_indices)
            max_x2 = max(boxes[i][2] for i in group_indices)
            max_y2 = max(boxes[i][3] for i in group_indices)

            # Add some padding to ensure complete coverage
            padding = 10
            min_x1 = max(0, min_x1 - padding)
            min_y1 = max(0, min_y1 - padding)
            max_x2 = max_x2 + padding
            max_y2 = max_y2 + padding

            # Create group info
            group_tracks = [track_info[idx] for idx in group_indices]
            groups.append({
                'bbox': [min_x1, min_y1, max_x2 - min_x1, max_y2 - min_y1],  # [x, y, w, h]
                'tracks': group_tracks,
                'count': len(group_tracks),
                'is_group': len(group_tracks) > 1,
                'individual_boxes': [boxes[i] for i in group_indices]  # Store original boxes for reference
            })

        return groups

    def should_merge_boxes(self, box1, box2):
        """Check if two boxes should be merged - more aggressive overlap detection"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Check for any overlap (including touching edges)
        overlap = not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)

        if overlap:
            return True

        # Calculate centers
        center1_x = (x1_1 + x2_1) / 2
        center1_y = (y1_1 + y2_1) / 2
        center2_x = (x1_2 + x2_2) / 2
        center2_y = (y1_2 + y2_2) / 2

        # Calculate distance between centers
        distance = math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)

        # More aggressive proximity threshold
        return distance <= self.proximity_threshold

    def calculate_overlap_area(self, box1, box2):
        """Calculate the area of overlap between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0  # No overlap

        return (x2_i - x1_i) * (y2_i - y1_i)
    
    def calculate_overall_density(self, groups, frame_shape):
        """Calculate overall crowd density without zone subdivision"""
        height, width = frame_shape[:2]
        total_people = sum(group['count'] for group in groups)

        # Calculate density metrics
        frame_area = width * height
        people_per_area = (total_people / frame_area) * 10000  # Per 10k pixels

        # Determine overall density level
        if total_people == 0:
            density_level = "Empty"
            density_color = (128, 128, 128)  # Gray
        elif people_per_area < 1.0:
            density_level = "Low"
            density_color = (0, 255, 0)  # Green
        elif people_per_area < 3.0:
            density_level = "Medium"
            density_color = (0, 255, 255)  # Yellow
        elif people_per_area < 6.0:
            density_level = "High"
            density_color = (0, 165, 255)  # Orange
        else:
            density_level = "Very High"
            density_color = (0, 0, 255)  # Red

        return {
            'total_people': total_people,
            'density_level': density_level,
            'density_color': density_color,
            'people_per_area': people_per_area
        }
    
    def draw_merged_groups(self, frame, groups):
        """Draw ONLY merged group rectangles - no overlapping boxes"""
        for group_idx, group in enumerate(groups):
            x, y, w, h = group['bbox']
            count = group['count']
            is_group = group['is_group']

            # Choose color based on group size and detection method
            if is_group:
                if count == 2:
                    color = (0, 255, 255)  # Cyan for pairs
                elif count <= 4:
                    color = (0, 165, 255)  # Orange for small groups
                elif count <= 8:
                    color = (0, 100, 255)  # Dark orange for medium groups
                else:
                    color = (0, 0, 255)  # Red for large groups
                thickness = 4
                label = f"GROUP-{group_idx+1}: {count} people"
            else:
                track_id, track = group['tracks'][0]
                method = track['method']

                # Color based on detection method
                if method == 'YOLOv8':
                    color = (255, 0, 255)  # Magenta for YOLOv8 (high accuracy)
                    thickness = 3
                    label = f"YOLO-{track_id} (High Accuracy)"
                else:
                    color = (0, 255, 0)  # Green for traditional methods
                    thickness = 2
                    label = f"PERSON-{track_id} ({method})"

            # Draw SINGLE unified bounding box for the entire group
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

            # Draw label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 15),
                         (x + label_size[0] + 15, y), color, -1)
            cv2.putText(frame, label, (x + 7, y - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Draw group center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 8, color, -1)
            cv2.circle(frame, (center_x, center_y), 8, (255, 255, 255), 2)

            # For groups, show individual detection points INSIDE the unified box
            if is_group:
                for i, (track_id, track) in enumerate(group['tracks']):
                    track_center = track['center']
                    # Use different colors for individual points within group
                    point_color = (255, 255, 255)  # White for visibility
                    cv2.circle(frame, tuple(map(int, track_center)), 4, point_color, -1)
                    cv2.circle(frame, tuple(map(int, track_center)), 4, (0, 0, 0), 1)

                    # Small ID label for each person in group
                    cv2.putText(frame, str(track_id),
                               (int(track_center[0]) - 8, int(track_center[1]) - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                    cv2.putText(frame, str(track_id),
                               (int(track_center[0]) - 8, int(track_center[1]) - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Add count indicator in corner
            count_text = str(count)
            cv2.circle(frame, (x + w - 15, y + 15), 12, color, -1)
            cv2.putText(frame, count_text, (x + w - 20, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def draw_zone_labels(self, frame, zones):
        """Draw zone status labels"""
        for zone_name, zone_data in zones.items():
            x1, y1, x2, y2 = zone_data['bounds']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            status = zone_data['status']
            color = zone_data['color']
            
            # Draw zone boundary
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            
            # Draw status label with people count
            label = f"{zone_name}: {status} ({zone_data['people_count']})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]

            # Background for text
            cv2.rectangle(frame,
                         (center_x - label_size[0]//2 - 5, center_y - 10),
                         (center_x + label_size[0]//2 + 5, center_y + 5),
                         (0, 0, 0), -1)

            # Text
            cv2.putText(frame, label,
                       (center_x - label_size[0]//2, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def draw_movement_trails(self, frame, groups):
        """Draw movement trails for tracked objects"""
        for group in groups:
            for track_id, track in group['tracks']:
                if track_id in self.movement_trails:
                    trail = list(self.movement_trails[track_id])
                    if len(trail) > 1:
                        # Draw trail with fading effect
                        for i in range(1, len(trail)):
                            alpha = i / len(trail)
                            thickness = max(1, int(3 * alpha))

                            # Color based on track age
                            hue = (track_id * 137.5) % 360  # Golden ratio for color distribution
                            rgb = colorsys.hsv_to_rgb(hue/360, 0.8, 0.9)
                            color = tuple(int(c * 255) for c in rgb)

                            cv2.line(frame, tuple(map(int, trail[i-1])), tuple(map(int, trail[i])),
                                    color, thickness)
    

    
    def draw_statistics_panel(self, frame, groups, density_info, frame_count, total_frames):
        """Draw comprehensive statistics panel"""
        height, width = frame.shape[:2]
        panel_height = 120

        # Create semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Current statistics
        total_people = sum(group['count'] for group in groups)
        total_groups = len([g for g in groups if g['is_group']])
        individual_people = len([g for g in groups if not g['is_group']])
        density_level = density_info['density_level']
        density_color = density_info['density_color']
        people_per_area = density_info['people_per_area']
        
        # Update frame stats
        self.frame_stats.append({
            'people': total_people,
            'groups': total_groups,
            'individuals': individual_people,
            'density_level': density_level,
            'people_per_area': people_per_area
        })
        
        # Calculate averages
        if len(self.frame_stats) > 0:
            avg_people = np.mean([s['people'] for s in self.frame_stats])
            avg_density = np.mean([s['people_per_area'] for s in self.frame_stats])
        else:
            avg_people = total_people
            avg_density = people_per_area
        
        # Draw statistics
        y_offset = 20
        
        # Title with detection method
        if hasattr(self, 'use_yolo') and self.use_yolo:
            title = "YOLOv8 HIGH-ACCURACY CROWD DETECTION"
        else:
            title = "ADVANCED CROWD DETECTION - GROUP MERGING"
        cv2.putText(frame, title,
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Current stats
        y_offset += 25
        cv2.putText(frame, f"People: {total_people} | Groups: {total_groups} | Individuals: {individual_people}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_offset += 20
        cv2.putText(frame, f"Density: {density_level} | {people_per_area:.2f} people/10k px",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, density_color, 2)
        
        # Progress and time
        y_offset += 20
        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
        cv2.putText(frame, f"Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Timestamp and features
        y_offset += 20
        timestamp = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, 'use_yolo') and self.use_yolo:
            features_text = f"Time: {timestamp} | YOLOv8 + Temporal Filtering + Group Merging"
        else:
            features_text = f"Time: {timestamp} | Temporal Filtering + Group Merging + Tracking"
        cv2.putText(frame, features_text,
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Right side - Detection method info
        right_x = width - 280
        cv2.putText(frame, "DETECTION METHODS",
                   (right_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_pos = 40
        if hasattr(self, 'use_yolo') and self.use_yolo:
            cv2.putText(frame, "YOLOv8: Magenta (High Accuracy)",
                       (right_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            y_pos += 20

        cv2.putText(frame, "HOG: Green (Traditional)",
                   (right_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y_pos += 20

        cv2.putText(frame, "Groups: Cyan/Orange/Red",
                   (right_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    def process_video(self, video_path, output_path):
        """Process video with advanced crowd detection features"""
        print(f"🎬 Processing video: {video_path}")
        print(f"💾 Output: {output_path} (768x576 @ 7 FPS)")
        print("🔥 Features: Group Merging | Zone Analysis | Movement Tracking | Statistics")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return False
        
        # Get original video properties
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Input: {original_width}x{original_height}, {original_fps} FPS, {total_frames} frames")
        
        # Target specifications (matching op_4.mp4)
        target_width, target_height = 768, 576
        target_fps = 7
        
        # Calculate frame skip ratio
        frame_skip = max(1, original_fps // target_fps)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (target_width, target_height))
        
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames to achieve target FPS
            if frame_count % frame_skip != 0:
                continue
            
            processed_count += 1
            
            # Resize frame to target resolution
            frame = cv2.resize(frame, (target_width, target_height))
            
            # Progress indicator
            if processed_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"🔄 Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
            
            # Step 1: Detect people
            detections = self.detect_people_enhanced(frame)

            # Step 2: Update tracking
            tracks = self.update_tracking(detections)

            # Step 3: Merge nearby rectangles into groups
            groups = self.merge_nearby_rectangles(tracks)

            # Step 4: Calculate overall density
            density_info = self.calculate_overall_density(groups, frame.shape)

            # Step 5: Draw all visualizations
            # Draw movement trails
            self.draw_movement_trails(frame, groups)

            # Draw merged group rectangles
            self.draw_merged_groups(frame, groups)

            # Draw statistics panel
            self.draw_statistics_panel(frame, groups, density_info, processed_count, total_frames // frame_skip)
            
            # Write frame
            out.write(frame)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Final statistics
        if len(self.frame_stats) > 0:
            avg_people = np.mean([s['people'] for s in self.frame_stats])
            max_people = max([s['people'] for s in self.frame_stats])
            avg_groups = np.mean([s['groups'] for s in self.frame_stats])
            max_groups = max([s['groups'] for s in self.frame_stats])
            avg_density = np.mean([s['people_per_area'] for s in self.frame_stats])

            print(f"\n✅ Advanced processing complete!")
            print(f"📊 Final Statistics:")
            print(f"   - Frames processed: {processed_count}")
            print(f"   - Average people detected: {avg_people:.1f}")
            print(f"   - Maximum people detected: {max_people}")
            print(f"   - Average groups formed: {avg_groups:.1f}")
            print(f"   - Maximum groups formed: {max_groups}")
            print(f"   - Average density: {avg_density:.2f} people/10k pixels")
            print(f"   - Output resolution: {target_width}x{target_height} @ {target_fps} FPS")
            print(f"   - Temporal filtering: Enabled for smooth bounding boxes")
            print(f"   - Optimized thresholds: YOLOv8={self.yolo_confidence_threshold}, HOG={self.hog_confidence_threshold}")
            print(f"💾 Output saved to: {output_path}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Advanced Crowd Detection with Enhanced Features')
    parser.add_argument('--video', default='data/video/input.mp4', help='Input video path')
    parser.add_argument('--output', default='outputs/advanced_crowd_detection.mp4', help='Output video path')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        # Create detector
        detector = AdvancedCrowdDetector()
        
        # Process video
        success = detector.process_video(args.video, args.output)
        
        if success:
            print("🎉 Advanced crowd detection completed successfully!")
            print("🔥 Features delivered: Group Merging ✓ | Zone Analysis ✓ | Movement Tracking ✓ | Statistics ✓")
        else:
            print("❌ Advanced crowd detection failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
