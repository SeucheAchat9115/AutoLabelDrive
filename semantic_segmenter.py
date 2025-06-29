import torch
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

class SemanticSegmenter:
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.transform = None
        
        # Autonomous driving relevant classes from COCO/Cityscapes
        self.autonomous_driving_classes = {
            0: 'background',
            1: 'person',
            2: 'bicycle',
            3: 'car',
            4: 'motorcycle',
            5: 'bus',
            6: 'truck',
            7: 'traffic light',
            8: 'traffic sign',
            9: 'road',
            10: 'sidewalk',
            11: 'building',
            12: 'sky',
            13: 'vegetation'
        }
        
        # Color map for visualization
        self.color_map = {
            0: [0, 0, 0],         # background - black
            1: [220, 20, 60],     # person - crimson
            2: [255, 0, 0],       # bicycle - red
            3: [0, 0, 142],       # car - dark blue
            4: [0, 0, 230],       # motorcycle - blue
            5: [106, 0, 228],     # bus - violet
            6: [0, 60, 100],      # truck - dark cyan
            7: [250, 170, 30],    # traffic light - orange
            8: [220, 220, 0],     # traffic sign - yellow
            9: [128, 64, 128],    # road - purple
            10: [244, 35, 232],   # sidewalk - magenta
            11: [70, 70, 70],     # building - dark gray
            12: [70, 130, 180],   # sky - steel blue
            13: [107, 142, 35]    # vegetation - olive
        }
    
    def load_model(self, model_name='deeplabv3_resnet50'):
        """Load semantic segmentation model"""
        try:
            if model_name == 'deeplabv3_resnet50':
                # Load pre-trained DeepLabV3 model
                weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
                self.model = deeplabv3_resnet50(weights=weights)
                self.model.to(self.device)
                self.model.eval()
                
                # Set up preprocessing transform
                preprocess = weights.transforms()
                self.transform = preprocess
                
                st.success(f"Semantic segmentation model {model_name} loaded successfully on {self.device}")
            else:
                raise ValueError(f"Unsupported model: {model_name}")
                
        except Exception as e:
            st.error(f"Error loading semantic segmentation model: {str(e)}")
            raise e
    
    def segment_image(self, image):
        """Run semantic segmentation on image"""
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        try:
            # Convert OpenCV BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume it's already RGB from YOLOv5 preprocessing
                pil_image = Image.fromarray(image)
            else:
                raise ValueError("Invalid image format")
            
            # Preprocess image
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)['out']
                prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            return prediction
            
        except Exception as e:
            st.error(f"Error during semantic segmentation: {str(e)}")
            return None
    
    def visualize_segmentation(self, image, segmentation_mask, alpha=0.6):
        """Overlay segmentation mask on original image"""
        try:
            # Create colored mask
            colored_mask = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8)
            
            for class_id, color in self.color_map.items():
                colored_mask[segmentation_mask == class_id] = color
            
            # Resize mask to match image size if needed
            if colored_mask.shape[:2] != image.shape[:2]:
                colored_mask = cv2.resize(colored_mask, (image.shape[1], image.shape[0]))
            
            # Blend with original image
            result = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
            
            return result
            
        except Exception as e:
            st.error(f"Error visualizing segmentation: {str(e)}")
            return image
    
    def get_segmentation_stats(self, segmentation_mask):
        """Get statistics about segmentation results"""
        try:
            stats = {}
            unique_classes, counts = np.unique(segmentation_mask, return_counts=True)
            total_pixels = segmentation_mask.size
            
            for class_id, count in zip(unique_classes, counts):
                class_name = self.autonomous_driving_classes.get(class_id, f'class_{class_id}')
                percentage = (count / total_pixels) * 100
                stats[class_name] = {
                    'pixel_count': int(count),
                    'percentage': round(percentage, 2)
                }
            
            return stats
            
        except Exception as e:
            st.error(f"Error computing segmentation stats: {str(e)}")
            return {}
    
    def extract_segments_by_class(self, image, segmentation_mask, target_classes=None):
        """Extract specific segments from the image"""
        if target_classes is None:
            target_classes = ['car', 'truck', 'bus', 'person', 'bicycle', 'motorcycle']
        
        segments = {}
        
        try:
            for class_name in target_classes:
                # Find class ID
                class_id = None
                for cid, cname in self.autonomous_driving_classes.items():
                    if cname == class_name:
                        class_id = cid
                        break
                
                if class_id is not None:
                    # Create mask for this class
                    class_mask = (segmentation_mask == class_id)
                    
                    if np.any(class_mask):
                        # Apply mask to image
                        segmented_image = image.copy()
                        segmented_image[~class_mask] = 0  # Set non-class pixels to black
                        segments[class_name] = {
                            'image': segmented_image,
                            'mask': class_mask,
                            'pixel_count': np.sum(class_mask)
                        }
            
            return segments
            
        except Exception as e:
            st.error(f"Error extracting segments: {str(e)}")
            return {}
    
    def get_model_info(self):
        """Get information about loaded model"""
        if self.model is None:
            return None
        
        return {
            'device': self.device,
            'classes': self.autonomous_driving_classes,
            'model_type': 'DeepLabV3 ResNet50'
        }
    
    def combine_with_detection(self, image, detections, segmentation_mask):
        """Combine object detection bounding boxes with semantic segmentation"""
        try:
            # Start with segmentation visualization
            result = self.visualize_segmentation(image, segmentation_mask, alpha=0.3)
            
            # Add detection bounding boxes on top
            for det in detections:
                bbox = det['bbox']
                class_name = det['class']
                confidence = det['confidence']
                
                # Draw bounding box
                cv2.rectangle(
                    result,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (255, 255, 255),  # White border
                    2
                )
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(
                    result,
                    (int(bbox[0]), int(bbox[1]) - label_size[1] - 10),
                    (int(bbox[0]) + label_size[0], int(bbox[1])),
                    (255, 255, 255),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    result,
                    label,
                    (int(bbox[0]), int(bbox[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )
            
            return result
            
        except Exception as e:
            st.error(f"Error combining detection and segmentation: {str(e)}")
            return image