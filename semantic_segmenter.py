import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Try to import torch and torchvision
TORCH_AVAILABLE = False
TORCHVISION_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    try:
        import torchvision.transforms as transforms
        from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
        TORCHVISION_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass

class SemanticSegmenter:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.transform = None
        
        # Autonomous driving relevant classes
        self.class_names = {
            0: 'background',
            1: 'person',
            2: 'bicycle', 
            3: 'car',
            4: 'motorcycle',
            5: 'bus',
            6: 'truck',
            7: 'traffic_light',
            8: 'traffic_sign',
            9: 'road',
            10: 'sidewalk',
            11: 'building',
            12: 'sky',
            13: 'vegetation'
        }
        
        # Color mapping for visualization
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
        if not TORCH_AVAILABLE or not TORCHVISION_AVAILABLE:
            st.info("Using color-based segmentation (advanced models unavailable)")
            self.model_loaded = True
            return True
            
        try:
            if model_name == 'deeplabv3_resnet50':
                # Load pre-trained DeepLabV3 model
                weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
                self.model = deeplabv3_resnet50(weights=weights)
                self.model.to(self.device)
                self.model.eval()
                
                # Set up preprocessing transform
                self.transform = weights.transforms()
                
                st.success(f"Semantic segmentation model {model_name} loaded successfully on {self.device}")
                self.model_loaded = True
                return True
            else:
                raise ValueError(f"Unsupported model: {model_name}")
                
        except Exception as e:
            st.info(f"Using color-based segmentation (DeepLabV3 unavailable)")
            self.model_loaded = True
            return True
    
    def segment_image(self, image):
        """Run semantic segmentation on image"""
        if not self.model_loaded:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Use simplified segmentation if advanced model unavailable
        if not TORCH_AVAILABLE or not TORCHVISION_AVAILABLE or self.model is None:
            return self._simple_segmentation(image)
        
        try:
            # Convert OpenCV BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
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
            # Fallback to simple segmentation
            return self._simple_segmentation(image)
    
    def _simple_segmentation(self, image):
        """Simple color-based segmentation fallback"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h, w = img_array.shape[:2]
        
        # Create segmentation mask based on color ranges
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Sky detection (blue regions in upper part)
        sky_mask = (hsv[:, :, 1] < 100) & (hsv[:, :, 2] > 100)
        sky_mask[:h//2, :] = sky_mask[:h//2, :] | ((hsv[:h//2, :, 0] > 100) & (hsv[:h//2, :, 0] < 130))
        mask[sky_mask] = 12  # sky class
        
        # Road detection (dark regions in lower part)
        road_mask = (hsv[h//2:, :, 2] < 80) & (hsv[h//2:, :, 1] < 100)
        mask[h//2:, :][road_mask] = 9  # road class
        
        # Green vegetation detection
        green_mask = (hsv[:, :, 0] > 40) & (hsv[:, :, 0] < 80) & (hsv[:, :, 1] > 50)
        mask[green_mask] = 13  # vegetation class
        
        # Building detection (gray regions)
        gray_mask = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 50) & (hsv[:, :, 2] < 200)
        mask[gray_mask] = 11  # building class
        
        return mask
    
    def visualize_segmentation(self, image, segmentation_mask, alpha=0.6):
        """Overlay segmentation mask on original image"""
        try:
            # Create colored mask
            colored_mask = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8)
            
            for class_id, color in self.color_map.items():
                colored_mask[segmentation_mask == class_id] = color
            
            # Overlay on original image
            if len(image.shape) == 3:
                overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
            else:
                # Convert grayscale to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                overlay = cv2.addWeighted(image_rgb, 1-alpha, colored_mask, alpha, 0)
            
            return overlay
            
        except Exception as e:
            st.error(f"Error in segmentation visualization: {str(e)}")
            return image
    
    def get_segmentation_stats(self, segmentation_mask):
        """Get statistics about segmentation results"""
        try:
            unique_classes, counts = np.unique(segmentation_mask, return_counts=True)
            total_pixels = segmentation_mask.size
            
            stats = {}
            for class_id, count in zip(unique_classes, counts):
                if class_id in self.class_names:
                    class_name = self.class_names[class_id]
                    percentage = (count / total_pixels) * 100
                    stats[class_name] = {
                        'pixel_count': int(count),
                        'percentage': float(percentage)
                    }
            
            return stats
            
        except Exception as e:
            st.error(f"Error getting segmentation stats: {str(e)}")
            return {}
    
    def extract_segments_by_class(self, image, segmentation_mask, target_classes=None):
        """Extract specific segments from the image"""
        try:
            if target_classes is None:
                target_classes = [3, 9]  # car, road by default
            
            extracted = {}
            for class_id in target_classes:
                if class_id in self.class_names:
                    class_mask = (segmentation_mask == class_id)
                    masked_image = image.copy()
                    masked_image[~class_mask] = 0
                    extracted[self.class_names[class_id]] = masked_image
            
            return extracted
            
        except Exception as e:
            st.error(f"Error extracting segments: {str(e)}")
            return {}
    
    def get_model_info(self):
        """Get information about loaded model"""
        if self.model_loaded:
            if TORCH_AVAILABLE and TORCHVISION_AVAILABLE and self.model is not None:
                return f"DeepLabV3 ResNet50 on {self.device}"
            else:
                return "Color-based segmentation (fallback)"
        else:
            return "No model loaded"
    
    def combine_with_detection(self, image, detections, segmentation_mask):
        """Combine object detection bounding boxes with semantic segmentation"""
        try:
            # Start with segmentation overlay
            segmented_image = self.visualize_segmentation(image, segmentation_mask, alpha=0.4)
            
            # Add bounding boxes
            for detection in detections:
                bbox = detection['bbox']
                class_name = detection['class']
                confidence = detection['confidence']
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(segmented_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(segmented_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return segmented_image
            
        except Exception as e:
            st.error(f"Error combining detection and segmentation: {str(e)}")
            return image