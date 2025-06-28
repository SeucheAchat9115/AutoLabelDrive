import torch
import cv2
import numpy as np
from PIL import Image
import streamlit as st

class ObjectDetector:
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.confidence_threshold = 0.5
        
        # Autonomous driving relevant classes from COCO dataset
        self.autonomous_driving_classes = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            9: 'traffic light',
            11: 'stop sign'
        }
    
    def load_model(self, model_name='yolov5s'):
        """Load YOLOv5 model"""
        try:
            # Load model from torch hub
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            st.success(f"Model {model_name} loaded successfully on {self.device}")
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            raise e
    
    def set_confidence(self, confidence):
        """Set confidence threshold"""
        self.confidence_threshold = confidence
        if self.model:
            self.model.conf = confidence
    
    def detect_objects(self, image):
        """Run object detection on image"""
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        try:
            # Run inference
            results = self.model(image)
            
            # Extract detections
            detections = []
            
            # Get predictions
            pred = results.pred[0]  # predictions for first image
            
            if len(pred) > 0:
                # Convert to numpy
                pred_np = pred.cpu().numpy()
                
                for detection in pred_np:
                    x1, y1, x2, y2, conf, cls = detection
                    
                    # Filter by confidence
                    if conf < self.confidence_threshold:
                        continue
                    
                    # Get class name
                    class_id = int(cls)
                    
                    # Only include autonomous driving relevant classes
                    if class_id in self.autonomous_driving_classes:
                        class_name = self.autonomous_driving_classes[class_id]
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class': class_name,
                            'class_id': class_id
                        })
            
            return detections
            
        except Exception as e:
            st.error(f"Error during object detection: {str(e)}")
            return []
    
    def detect_batch(self, images):
        """Run object detection on batch of images"""
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        try:
            # Run inference on batch
            results = self.model(images)
            
            batch_detections = []
            
            for i, result in enumerate(results.pred):
                detections = []
                
                if len(result) > 0:
                    # Convert to numpy
                    pred_np = result.cpu().numpy()
                    
                    for detection in pred_np:
                        x1, y1, x2, y2, conf, cls = detection
                        
                        # Filter by confidence
                        if conf < self.confidence_threshold:
                            continue
                        
                        # Get class name
                        class_id = int(cls)
                        
                        # Only include autonomous driving relevant classes
                        if class_id in self.autonomous_driving_classes:
                            class_name = self.autonomous_driving_classes[class_id]
                            
                            detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(conf),
                                'class': class_name,
                                'class_id': class_id
                            })
                
                batch_detections.append(detections)
            
            return batch_detections
            
        except Exception as e:
            st.error(f"Error during batch object detection: {str(e)}")
            return []
    
    def get_model_info(self):
        """Get information about loaded model"""
        if self.model is None:
            return None
        
        return {
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'classes': self.autonomous_driving_classes
        }
    
    def preprocess_image(self, image_path):
        """Preprocess image for detection"""
        try:
            # Read image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image_rgb
            
        except Exception as e:
            st.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def postprocess_detections(self, detections, image_shape):
        """Post-process detections (e.g., NMS, filtering)"""
        # Basic filtering already done in detect_objects
        # Additional post-processing can be added here
        
        # Filter out detections that are too small or too large
        filtered_detections = []
        height, width = image_shape[:2]
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            
            # Calculate box dimensions
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            
            # Filter by relative size (avoid tiny detections)
            if box_area > (width * height * 0.001):  # At least 0.1% of image
                filtered_detections.append(det)
        
        return filtered_detections
