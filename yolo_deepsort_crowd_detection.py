#!/usr/bin/env python3
"""
Proper crowd detection following the specified workflow:
Input: Video stream → YOLOv4 Detection → DeepSORT Tracking → DBSCAN Clustering → Output
"""

import cv2
import numpy as np
import os
import argparse
from datetime import datetime
from sklearn.cluster import DBSCAN
import tensorflow as tf
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg

class YOLODeepSORTCrowdDetector:
    def __init__(self, weights_path, model_name='yolov4', input_size=416):
        self.weights_path = weights_path
        self.model_name = model_name
        self.input_size = input_size
        self.iou_threshold = 0.45
        self.score_threshold = 0.50
        
        # Initialize YOLOv4 model
        self.yolo_model = None
        self.load_yolo_model()
        
        # Initialize DeepSORT
        self.max_cosine_distance = 0.4
        self.nn_budget = None
        self.nms_max_overlap = 1.0
        
        # Initialize deep sort
        model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(metric)
        
        # Load class names
        self.class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        
        print("✅ YOLOv4 + DeepSORT + DBSCAN system initialized successfully!")
    
    def load_yolo_model(self):
        """Load YOLOv4 model from weights"""
        try:
            # Check if we have a saved model
            saved_model_path = f"checkpoints/{self.model_name}-{self.input_size}"
            
            if os.path.exists(saved_model_path):
                print(f"Loading saved model from {saved_model_path}")
                self.yolo_model = tf.saved_model.load(saved_model_path, tags=[tag_constants.SERVING])
                self.infer = self.yolo_model.signatures['serving_default']
            else:
                print("Creating YOLOv4 model from weights...")
                # Create model from weights
                STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config_yolo(self.model_name)
                
                input_layer = tf.keras.layers.Input([self.input_size, self.input_size, 3])
                feature_maps = utils.YOLOv4(input_layer, NUM_CLASS, self.model_name)
                
                bbox_tensors = []
                prob_tensors = []
                
                for i, fm in enumerate(feature_maps):
                    if i == 0:
                        output_tensors = utils.decode(fm, self.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                    elif i == 1:
                        output_tensors = utils.decode(fm, self.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                    else:
                        output_tensors = utils.decode(fm, self.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                    bbox_tensors.append(output_tensors[0])
                    prob_tensors.append(output_tensors[1])
                
                pred_bbox = tf.concat(bbox_tensors, axis=1)
                pred_prob = tf.concat(prob_tensors, axis=1)
                
                self.yolo_model = tf.keras.Model(input_layer, [pred_bbox, pred_prob])
                utils.load_weights(self.yolo_model, self.weights_path, self.model_name)
                
                @tf.function
                def infer_func(x):
                    return self.yolo_model(x)
                
                self.infer = infer_func
                
                print("✅ YOLOv4 model created successfully!")
                
        except Exception as e:
            print(f"❌ Error loading YOLOv4 model: {e}")
            print("Falling back to OpenCV DNN...")
            self.use_opencv_fallback()
    
    def use_opencv_fallback(self):
        """Fallback to OpenCV DNN if TensorFlow fails"""
        try:
            config_path = "data/yolov4.cfg"
            self.net = cv2.dnn.readNet(self.weights_path, config_path)
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            self.use_opencv = True
            print("✅ OpenCV DNN fallback loaded successfully!")
        except Exception as e:
            print(f"❌ OpenCV fallback also failed: {e}")
            raise
    
    def detect_people_yolo(self, frame):
        """Step 1: YOLOv4 Detection - Find people in frame"""
        try:
            if hasattr(self, 'use_opencv') and self.use_opencv:
                return self.detect_opencv(frame)
            else:
                return self.detect_tensorflow(frame)
        except Exception as e:
            print(f"Detection error: {e}")
            return [], [], []
    
    def detect_opencv(self, frame):
        """OpenCV DNN detection"""
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
                if class_id == 0 and confidence > self.score_threshold:
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
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.score_threshold, self.iou_threshold)
        
        final_boxes = []
        final_scores = []
        final_classes = []
        
        if len(indexes) > 0:
            for i in indexes.flatten():
                final_boxes.append(boxes[i])
                final_scores.append(confidences[i])
                final_classes.append(class_ids[i])
        
        return final_boxes, final_scores, final_classes
    
    def detect_tensorflow(self, frame):
        """TensorFlow detection"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = tf.image.resize(frame_rgb, (self.input_size, self.input_size))
        frame_resized = frame_resized / 255.0
        frame_resized = tf.expand_dims(frame_resized, 0)
        
        pred_bbox = self.infer(frame_resized)
        
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold
        )
        
        # Convert to numpy and filter for persons only
        boxes = boxes.numpy()[0]
        scores = scores.numpy()[0]
        classes = classes.numpy()[0]
        valid_detections = valid_detections.numpy()[0]
        
        # Filter for person class (0) and valid detections
        person_boxes = []
        person_scores = []
        person_classes = []
        
        for i in range(int(valid_detections)):
            if classes[i] == 0:  # Person class
                person_boxes.append(boxes[i])
                person_scores.append(scores[i])
                person_classes.append(classes[i])
        
        return person_boxes, person_scores, person_classes
    
    def track_people_deepsort(self, frame, boxes, scores, classes):
        """Step 2: DeepSORT Tracking - Track each person maintaining identity"""
        height, width = frame.shape[:2]
        
        # Convert boxes to detection format
        detections = []
        
        for box, score, cls in zip(boxes, scores, classes):
            if hasattr(self, 'use_opencv') and self.use_opencv:
                # OpenCV format: [x, y, w, h]
                x, y, w, h = box
                bbox = [x, y, w, h]
            else:
                # TensorFlow format: [y1, x1, y2, x2] normalized
                y1, x1, y2, x2 = box
                x = int(x1 * width)
                y = int(y1 * height)
                w = int((x2 - x1) * width)
                h = int((y2 - y1) * height)
                bbox = [x, y, w, h]
            
            detections.append(Detection(bbox, score, cls))
        
        # Encode detections for DeepSORT
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Extract features for DeepSORT
        features = self.encoder(frame, boxs)
        detections = [Detection(boxs[i], scores[i], classes[i], features[i]) for i in range(len(detections))]
        
        # Update tracker
        self.tracker.predict()
        self.tracker.update(detections)
        
        # Get tracked objects
        tracked_objects = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            bbox = track.to_tlbr()
            class_name = track.get_class()
            track_id = track.track_id
            
            tracked_objects.append({
                'bbox': bbox,
                'track_id': track_id,
                'class': class_name,
                'center': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            })
        
        return tracked_objects
    
    def cluster_crowds_dbscan(self, tracked_objects):
        """Step 3: DBSCAN Clustering - Group individuals into crowds"""
        if len(tracked_objects) < 2:
            # Not enough people for clustering
            return [{'type': 'individual', 'objects': tracked_objects, 'center': None}]
        
        # Extract centers for clustering
        centers = np.array([obj['center'] for obj in tracked_objects])
        
        # DBSCAN clustering
        # eps: maximum distance between two samples to be in same cluster
        # min_samples: minimum number of samples in a cluster
        clustering = DBSCAN(eps=100, min_samples=2).fit(centers)
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
                # Calculate crowd center
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
                # Draw crowd bounding box
                min_x = min([obj['bbox'][0] for obj in group['objects']])
                min_y = min([obj['bbox'][1] for obj in group['objects']])
                max_x = max([obj['bbox'][2] for obj in group['objects']])
                max_y = max([obj['bbox'][3] for obj in group['objects']])
                
                # Draw crowd box in red
                cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 0, 255), 3)
                cv2.putText(frame, f"Crowd: {group['count']} people", 
                           (int(min_x), int(min_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw individual tracks within crowd
                for obj in group['objects']:
                    bbox = obj['bbox']
                    track_id = obj['track_id']
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)
                    cv2.putText(frame, f"ID:{track_id}", 
                               (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            else:
                # Individual person
                obj = group['objects'][0]
                bbox = obj['bbox']
                track_id = obj['track_id']
                
                # Draw individual box in green
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"Person ID:{track_id}", 
                           (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return total_people, crowd_count
    
    def process_video(self, video_path, output_path):
        """Main processing pipeline following the specified workflow"""
        print(f"🎬 Processing video: {video_path}")
        print(f"💾 Output will be saved to: {output_path}")
        print("📋 Workflow: Video → YOLOv4 Detection → DeepSORT Tracking → DBSCAN Clustering → Output")
        
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
            
            # Step 1: YOLOv4 Detection
            boxes, scores, classes = self.detect_people_yolo(frame)
            
            # Step 2: DeepSORT Tracking
            tracked_objects = self.track_people_deepsort(frame, boxes, scores, classes)
            
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
            
            # Add information overlay
            overlay_height = 140
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, overlay_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Add text information
            cv2.putText(frame, f"YOLOv4 + DeepSORT + DBSCAN CROWD DETECTION", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"People Detected: {total_people}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Crowds Formed: {crowd_count}", 
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Avg People: {avg_people:.1f} | Avg Crowds: {avg_crowds:.1f}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add workflow info
            cv2.putText(frame, f"Pipeline: YOLO→DeepSORT→DBSCAN", 
                       (width - 300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, f"Time: {timestamp}", 
                       (width - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
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
            
            print(f"\n✅ YOLOv4 + DeepSORT + DBSCAN processing complete!")
            print(f"📊 Statistics:")
            print(f"   - Average people detected: {avg_people:.1f}")
            print(f"   - Maximum people detected: {max_people}")
            print(f"   - Average crowds formed: {avg_crowds:.1f}")
            print(f"   - Maximum crowds formed: {max_crowds}")
            print(f"   - Total frames processed: {frame_count}")
            print(f"💾 Output saved to: {output_path}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='YOLOv4 + DeepSORT + DBSCAN Crowd Detection')
    parser.add_argument('--video', default='data/video/input.mp4', help='Input video path')
    parser.add_argument('--output', default='outputs/crowd_detection_yolo_deepsort.mp4', help='Output video path')
    parser.add_argument('--weights', default='data/yolov4.weights', help='YOLOv4 weights path')
    parser.add_argument('--model', default='yolov4', help='Model name')
    parser.add_argument('--size', type=int, default=416, help='Input size')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        # Create detector
        detector = YOLODeepSORTCrowdDetector(args.weights, args.model, args.size)
        
        # Process video
        success = detector.process_video(args.video, args.output)
        
        if success:
            print("🎉 YOLOv4 + DeepSORT + DBSCAN crowd detection completed successfully!")
        else:
            print("❌ Crowd detection failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
