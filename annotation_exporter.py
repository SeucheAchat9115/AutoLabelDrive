import os
import json
import csv
import shutil
from datetime import datetime
from pathlib import Path
import streamlit as st
from project_utils import create_temp_dir

class AnnotationExporter:
    def __init__(self):
        # Create exports directory in the repository root
        self.export_dir = os.path.join(os.getcwd(), 'exports')
        os.makedirs(self.export_dir, exist_ok=True)
    
    def export_yolo(self, annotations, include_images=True):
        """Export annotations in YOLO format"""
        try:
            # Create export directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            yolo_dir = os.path.join(self.export_dir, f'yolo_export_{timestamp}')
            os.makedirs(yolo_dir, exist_ok=True)
            
            # Create subdirectories
            labels_dir = os.path.join(yolo_dir, 'labels')
            os.makedirs(labels_dir, exist_ok=True)
            
            if include_images:
                images_dir = os.path.join(yolo_dir, 'images')
                os.makedirs(images_dir, exist_ok=True)
            
            # YOLO class mapping
            class_mapping = {
                'person': 0,
                'bicycle': 1,
                'car': 2,
                'motorcycle': 3,
                'bus': 4,
                'truck': 5,
                'traffic light': 6,
                'stop sign': 7
            }
            
            # Write classes file
            classes_file = os.path.join(yolo_dir, 'classes.txt')
            with open(classes_file, 'w') as f:
                for class_name in sorted(class_mapping.keys()):
                    f.write(f"{class_name}\n")
            
            # Process each frame
            for frame_idx, annotation in annotations.items():
                frame_path = annotation['frame_path']
                detections = annotation['detections']
                frame_shape = annotation['frame_shape']
                
                # Get frame filename without extension
                frame_name = Path(frame_path).stem
                
                # Copy image if requested
                if include_images:
                    dst_image_path = os.path.join(images_dir, f"{frame_name}.jpg")
                    shutil.copy2(frame_path, dst_image_path)
                
                # Create YOLO label file
                label_file = os.path.join(labels_dir, f"{frame_name}.txt")
                
                with open(label_file, 'w') as f:
                    for det in detections:
                        # Convert bbox to YOLO format (normalized xywh)
                        bbox = det['bbox']
                        x1, y1, x2, y2 = bbox
                        
                        # Normalize coordinates
                        img_width, img_height = frame_shape[1], frame_shape[0]
                        x_center = (x1 + x2) / 2 / img_width
                        y_center = (y1 + y2) / 2 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        # Get class ID
                        class_name = det['class']
                        class_id = class_mapping.get(class_name, 0)
                        
                        # Write YOLO format line
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            # Create dataset YAML file
            yaml_file = os.path.join(yolo_dir, 'dataset.yaml')
            with open(yaml_file, 'w') as f:
                f.write(f"path: {yolo_dir}\n")
                f.write("train: images\n")
                f.write("val: images\n")
                f.write("\n")
                f.write(f"nc: {len(class_mapping)}\n")
                f.write("names:\n")
                for class_name in sorted(class_mapping.keys()):
                    f.write(f"  {class_mapping[class_name]}: {class_name}\n")
            
            return yolo_dir
            
        except Exception as e:
            st.error(f"Error exporting YOLO format: {str(e)}")
            raise e
    
    def export_coco(self, annotations, include_images=True):
        """Export annotations in COCO JSON format"""
        try:
            # Create export directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            coco_dir = os.path.join(self.export_dir, f'coco_export_{timestamp}')
            os.makedirs(coco_dir, exist_ok=True)
            
            if include_images:
                images_dir = os.path.join(coco_dir, 'images')
                os.makedirs(images_dir, exist_ok=True)
            
            # COCO format structure
            coco_data = {
                "info": {
                    "description": "Auto-labeled dataset for autonomous driving",
                    "version": "1.0",
                    "year": datetime.now().year,
                    "contributor": "Auto-Labeling Tool",
                    "date_created": datetime.now().isoformat()
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": []
            }
            
            # Define categories
            categories = [
                {"id": 1, "name": "person", "supercategory": "person"},
                {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
                {"id": 3, "name": "car", "supercategory": "vehicle"},
                {"id": 4, "name": "motorcycle", "supercategory": "vehicle"},
                {"id": 5, "name": "bus", "supercategory": "vehicle"},
                {"id": 6, "name": "truck", "supercategory": "vehicle"},
                {"id": 7, "name": "traffic light", "supercategory": "traffic"},
                {"id": 8, "name": "stop sign", "supercategory": "traffic"}
            ]
            
            coco_data["categories"] = categories
            
            # Create category name to ID mapping
            cat_name_to_id = {cat["name"]: cat["id"] for cat in categories}
            
            annotation_id = 1
            
            # Process each frame
            for frame_idx, annotation in annotations.items():
                frame_path = annotation['frame_path']
                detections = annotation['detections']
                frame_shape = annotation['frame_shape']
                
                # Get frame info
                frame_name = Path(frame_path).name
                img_height, img_width = frame_shape[:2]
                
                # Copy image if requested
                if include_images:
                    dst_image_path = os.path.join(images_dir, frame_name)
                    shutil.copy2(frame_path, dst_image_path)
                
                # Add image info
                image_info = {
                    "id": frame_idx + 1,
                    "width": img_width,
                    "height": img_height,
                    "file_name": frame_name
                }
                coco_data["images"].append(image_info)
                
                # Add annotations for this image
                for det in detections:
                    bbox = det['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    # COCO bbox format: [x, y, width, height]
                    coco_bbox = [x1, y1, x2 - x1, y2 - y1]
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Get category ID
                    class_name = det['class']
                    category_id = cat_name_to_id.get(class_name, 1)
                    
                    annotation_info = {
                        "id": annotation_id,
                        "image_id": frame_idx + 1,
                        "category_id": category_id,
                        "bbox": coco_bbox,
                        "area": area,
                        "iscrowd": 0,
                        "confidence": det['confidence']
                    }
                    
                    coco_data["annotations"].append(annotation_info)
                    annotation_id += 1
            
            # Save COCO JSON file
            json_file = os.path.join(coco_dir, 'annotations.json')
            with open(json_file, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            return coco_dir
            
        except Exception as e:
            st.error(f"Error exporting COCO format: {str(e)}")
            raise e
    
    def export_csv(self, annotations):
        """Export annotations in CSV format"""
        try:
            # Create export directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = os.path.join(self.export_dir, f'annotations_{timestamp}.csv')
            
            # Prepare CSV data
            csv_data = []
            
            for frame_idx, annotation in annotations.items():
                frame_path = annotation['frame_path']
                detections = annotation['detections']
                frame_shape = annotation['frame_shape']
                
                frame_name = Path(frame_path).name
                img_height, img_width = frame_shape[:2]
                
                for det in detections:
                    bbox = det['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    csv_row = {
                        'frame_index': frame_idx,
                        'frame_name': frame_name,
                        'frame_width': img_width,
                        'frame_height': img_height,
                        'class': det['class'],
                        'confidence': det['confidence'],
                        'bbox_x1': x1,
                        'bbox_y1': y1,
                        'bbox_x2': x2,
                        'bbox_y2': y2,
                        'bbox_width': x2 - x1,
                        'bbox_height': y2 - y1,
                        'bbox_center_x': (x1 + x2) / 2,
                        'bbox_center_y': (y1 + y2) / 2
                    }
                    
                    csv_data.append(csv_row)
            
            # Write CSV file
            if csv_data:
                fieldnames = csv_data[0].keys()
                
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_data)
            
            return csv_file
            
        except Exception as e:
            st.error(f"Error exporting CSV format: {str(e)}")
            raise e
    
    def get_export_summary(self, annotations):
        """Get summary statistics for export"""
        total_frames = len(annotations)
        total_detections = sum(len(ann['detections']) for ann in annotations.values())
        
        # Count by class
        class_counts = {}
        for ann in annotations.values():
            for det in ann['detections']:
                class_name = det['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'total_frames': total_frames,
            'total_detections': total_detections,
            'class_counts': class_counts
        }
