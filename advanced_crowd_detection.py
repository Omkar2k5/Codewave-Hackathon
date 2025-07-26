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

class AdvancedCrowdDetector:
    def __init__(self):
        # Initialize detection components
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=500, varThreshold=30)
        
        # Tracking system
        self.next_id = 0
        self.tracks = {}
        self.disappeared = {}
        self.max_disappeared = 15
        self.max_distance = 80
        
        # Heatmap and temporal data
        self.heatmap_history = deque(maxlen=150)  # Store last 150 frames for heatmap
        self.density_zones = {}
        self.movement_trails = defaultdict(lambda: deque(maxlen=30))  # Track movement for 30 frames
        
        # Statistics tracking
        self.frame_stats = deque(maxlen=100)
        self.zone_history = deque(maxlen=50)
        
        print("✅ Advanced Crowd Detection System initialized!")
        print("📊 Features: Heatmaps | Zone Analysis | Movement Tracking | Real-time Stats")
    
    def detect_people_enhanced(self, frame):
        """Enhanced people detection with multiple methods"""
        detections = []
        
        # Method 1: HOG Detection
        try:
            boxes, weights = self.hog.detectMultiScale(
                frame, winStride=(8, 8), padding=(16, 16), scale=1.05
            )
            for i, (x, y, w, h) in enumerate(boxes):
                if weights[i] > 0.5:  # Confidence threshold
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
                        
                        self.tracks[track_id].update({
                            'center': detection_centers[col],
                            'bbox': [x, y, w, h],
                            'method': det[4],
                            'confidence': det[5],
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
    
    def update_heatmap(self, tracks, frame_shape):
        """Update density heatmap based on current detections"""
        height, width = frame_shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Add current detections to heatmap
        for track_id, track in tracks:
            center = track['center']
            x, y = int(center[0]), int(center[1])
            
            # Create gaussian blob around detection
            for dy in range(-30, 31):
                for dx in range(-30, 31):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        distance = math.sqrt(dx*dx + dy*dy)
                        if distance <= 30:
                            intensity = math.exp(-(distance**2) / (2 * 15**2))
                            heatmap[ny, nx] += intensity
        
        self.heatmap_history.append(heatmap)
        
        # Create accumulated heatmap
        if len(self.heatmap_history) > 0:
            accumulated = np.sum(self.heatmap_history, axis=0)
            # Normalize
            if accumulated.max() > 0:
                accumulated = accumulated / accumulated.max()
            return accumulated
        
        return heatmap
    
    def analyze_zones(self, heatmap, frame_shape):
        """Analyze different zones and classify density levels"""
        height, width = frame_shape[:2]
        zones = {}
        
        # Define zones (divide frame into 3x3 grid)
        zone_height = height // 3
        zone_width = width // 3
        
        zone_labels = [
            ['Top-Left', 'Top-Center', 'Top-Right'],
            ['Mid-Left', 'Center', 'Mid-Right'],
            ['Bottom-Left', 'Bottom-Center', 'Bottom-Right']
        ]
        
        for i in range(3):
            for j in range(3):
                y1 = i * zone_height
                y2 = (i + 1) * zone_height if i < 2 else height
                x1 = j * zone_width
                x2 = (j + 1) * zone_width if j < 2 else width
                
                zone_heatmap = heatmap[y1:y2, x1:x2]
                density = np.mean(zone_heatmap)
                
                # Classify density
                if density < 0.1:
                    status = "Clear"
                    color = (0, 255, 0)  # Green
                elif density < 0.3:
                    status = "Normal"
                    color = (0, 255, 255)  # Yellow
                else:
                    status = "Congested"
                    color = (0, 0, 255)  # Red
                
                zones[zone_labels[i][j]] = {
                    'density': density,
                    'status': status,
                    'color': color,
                    'bounds': (x1, y1, x2, y2)
                }
        
        return zones
    
    def draw_heatmap_overlay(self, frame, heatmap):
        """Draw heatmap overlay on frame"""
        if heatmap.max() > 0:
            # Convert heatmap to color
            heatmap_normalized = (heatmap * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
            
            # Create overlay
            overlay = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
            return overlay
        return frame
    
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
            
            # Draw status label
            label = f"{zone_name}: {status}"
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
    
    def draw_movement_trails(self, frame, tracks):
        """Draw movement trails for tracked objects"""
        for track_id, track in tracks:
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
    
    def draw_detections_and_tracking(self, frame, tracks):
        """Draw bounding boxes and tracking information"""
        for track_id, track in tracks:
            x, y, w, h = track['bbox']
            center = track['center']
            method = track['method']
            confidence = track['confidence']
            age = track['age']
            
            # Color based on method
            if method == 'HOG':
                color = (0, 255, 0)  # Green
            else:
                color = (255, 0, 0)  # Blue
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            cv2.circle(frame, tuple(map(int, center)), 4, color, -1)
            
            # Draw ID and info
            label = f"ID:{track_id} ({method})"
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw confidence and age
            info = f"C:{confidence:.2f} A:{age}"
            cv2.putText(frame, info, (x, y + h + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def draw_statistics_panel(self, frame, tracks, zones, frame_count, total_frames):
        """Draw comprehensive statistics panel"""
        height, width = frame.shape[:2]
        panel_height = 120
        
        # Create semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Current statistics
        total_people = len(tracks)
        congested_zones = len([z for z in zones.values() if z['status'] == 'Congested'])
        normal_zones = len([z for z in zones.values() if z['status'] == 'Normal'])
        clear_zones = len([z for z in zones.values() if z['status'] == 'Clear'])
        
        # Update frame stats
        self.frame_stats.append({
            'people': total_people,
            'congested': congested_zones,
            'normal': normal_zones,
            'clear': clear_zones
        })
        
        # Calculate averages
        if len(self.frame_stats) > 0:
            avg_people = np.mean([s['people'] for s in self.frame_stats])
            avg_congested = np.mean([s['congested'] for s in self.frame_stats])
        else:
            avg_people = total_people
            avg_congested = congested_zones
        
        # Draw statistics
        y_offset = 20
        
        # Title
        cv2.putText(frame, "ADVANCED CROWD DETECTION SYSTEM", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Current stats
        y_offset += 25
        cv2.putText(frame, f"People Detected: {total_people} | Avg: {avg_people:.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(frame, f"Zones - Congested: {congested_zones} | Normal: {normal_zones} | Clear: {clear_zones}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Progress and time
        y_offset += 20
        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
        cv2.putText(frame, f"Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Timestamp and features
        y_offset += 20
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {timestamp} | Features: Heatmap + Zones + Tracking", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Right side - Zone summary
        right_x = width - 300
        cv2.putText(frame, "ZONE STATUS SUMMARY", 
                   (right_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos = 40
        for zone_name, zone_data in zones.items():
            status = zone_data['status']
            color = zone_data['color']
            density = zone_data['density']
            
            text = f"{zone_name}: {status} ({density:.2f})"
            cv2.putText(frame, text, (right_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y_pos += 15
    
    def process_video(self, video_path, output_path):
        """Process video with advanced crowd detection features"""
        print(f"🎬 Processing video: {video_path}")
        print(f"💾 Output: {output_path} (768x576 @ 7 FPS)")
        print("🔥 Features: Heatmaps | Zone Analysis | Movement Tracking | Statistics")
        
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
            
            # Step 3: Update heatmap
            heatmap = self.update_heatmap(tracks, frame.shape)
            
            # Step 4: Analyze zones
            zones = self.analyze_zones(heatmap, frame.shape)
            
            # Step 5: Draw all visualizations
            # Draw heatmap overlay
            frame = self.draw_heatmap_overlay(frame, heatmap)
            
            # Draw movement trails
            self.draw_movement_trails(frame, tracks)
            
            # Draw detections and tracking
            self.draw_detections_and_tracking(frame, tracks)
            
            # Draw zone labels
            self.draw_zone_labels(frame, zones)
            
            # Draw statistics panel
            self.draw_statistics_panel(frame, tracks, zones, processed_count, total_frames // frame_skip)
            
            # Write frame
            out.write(frame)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Final statistics
        if len(self.frame_stats) > 0:
            avg_people = np.mean([s['people'] for s in self.frame_stats])
            max_people = max([s['people'] for s in self.frame_stats])
            avg_congested = np.mean([s['congested'] for s in self.frame_stats])
            
            print(f"\n✅ Advanced processing complete!")
            print(f"📊 Final Statistics:")
            print(f"   - Frames processed: {processed_count}")
            print(f"   - Average people detected: {avg_people:.1f}")
            print(f"   - Maximum people detected: {max_people}")
            print(f"   - Average congested zones: {avg_congested:.1f}")
            print(f"   - Output resolution: {target_width}x{target_height} @ {target_fps} FPS")
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
            print("🔥 Features delivered: Heatmaps ✓ | Zone Analysis ✓ | Movement Tracking ✓ | Statistics ✓")
        else:
            print("❌ Advanced crowd detection failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
