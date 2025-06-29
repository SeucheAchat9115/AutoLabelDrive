import streamlit as st
import os
import tempfile
import json
from pathlib import Path
import pandas as pd
from PIL import Image
import cv2
import numpy as np

from video_processor import VideoProcessor
from object_detector import ObjectDetector
from annotation_exporter import AnnotationExporter
from semantic_segmenter import SemanticSegmenter
from project_utils import create_temp_dir, cleanup_temp_files

# Configure page
st.set_page_config(
    page_title="Auto-Labeling Tool for Autonomous Driving",
    page_icon="🚗",
    layout="wide"
)

# Initialize session state
if 'video_processor' not in st.session_state:
    st.session_state.video_processor = VideoProcessor()
if 'object_detector' not in st.session_state:
    st.session_state.object_detector = ObjectDetector()
if 'semantic_segmenter' not in st.session_state:
    st.session_state.semantic_segmenter = SemanticSegmenter()
if 'annotation_exporter' not in st.session_state:
    st.session_state.annotation_exporter = AnnotationExporter()
if 'current_video_path' not in st.session_state:
    st.session_state.current_video_path = None
if 'frames' not in st.session_state:
    st.session_state.frames = []
if 'annotations' not in st.session_state:
    st.session_state.annotations = {}
if 'segmentations' not in st.session_state:
    st.session_state.segmentations = {}
if 'current_frame_index' not in st.session_state:
    st.session_state.current_frame_index = 0

def main():
    st.title("🚗 Auto-Labeling Tool for Autonomous Driving")
    st.markdown("Automatically detect and annotate objects in YouTube videos with object detection and semantic segmentation for autonomous driving applications.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Select Detection Model",
            ["YOLOv5s", "YOLOv5m", "YOLOv5l"],
            help="Larger models are more accurate but slower"
        )
        
        # Segmentation toggle
        enable_segmentation = st.checkbox(
            "Enable Semantic Segmentation",
            value=False,
            help="Add pixel-level semantic segmentation (slower but more detailed)"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum confidence for detections"
        )
        
        # Frame sampling rate
        frame_rate = st.slider(
            "Frame Sampling Rate",
            min_value=1,
            max_value=30,
            value=5,
            help="Process every Nth frame"
        )
        
        # Max frames setting
        max_frames = st.number_input(
            "Max Frames to Process",
            min_value=10,
            max_value=2000,
            value=100,
            step=10,
            help="Maximum number of frames to extract and process"
        )
        
        # Update detector settings
        st.session_state.object_detector.set_confidence(confidence_threshold)
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["📹 Video Input", "🔍 Review Annotations", "📁 Export"])
    
    with tab1:
        video_input_tab(model_type, frame_rate, max_frames, enable_segmentation)
    
    with tab2:
        review_annotations_tab()
    
    with tab3:
        export_annotations_tab()

def video_input_tab(model_type, frame_rate, max_frames, enable_segmentation):
    st.header("Video Input")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["YouTube URL", "Upload Video File"],
        horizontal=True
    )
    
    if input_method == "YouTube URL":
        youtube_url = st.text_input(
            "Enter YouTube URL:",
            value="https://youtu.be/YBS8rkP4yCg?si=-2Hr5QkTSAF-5knT",
            placeholder="https://www.youtube.com/watch?v=..."
        )
        
        if st.button("Download and Process", type="primary"):
            if youtube_url:
                process_youtube_video(youtube_url, model_type, frame_rate, max_frames, enable_segmentation)
            else:
                st.error("Please enter a valid YouTube URL")
    
    else:
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to process"
        )
        
        if uploaded_file is not None and st.button("Process Video", type="primary"):
            process_uploaded_video(uploaded_file, model_type, frame_rate, max_frames, enable_segmentation)

def process_youtube_video(url, model_type, frame_rate, max_frames, enable_segmentation):
    """Process YouTube video for object detection"""
    with st.spinner("Downloading video from YouTube..."):
        try:
            # Download video
            video_path = st.session_state.video_processor.download_youtube_video(url)
            if video_path:
                st.success(f"Video downloaded successfully!")
                process_video_frames(video_path, model_type, frame_rate, max_frames, enable_segmentation)
            else:
                st.error("Failed to download video from YouTube")
        except Exception as e:
            st.error(f"Error downloading video: {str(e)}")

def process_uploaded_video(uploaded_file, model_type, frame_rate, max_frames, enable_segmentation):
    """Process uploaded video file"""
    with st.spinner("Processing uploaded video..."):
        try:
            # Save uploaded file temporarily
            temp_dir = create_temp_dir()
            video_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            st.success("Video uploaded successfully!")
            process_video_frames(video_path, model_type, frame_rate, max_frames, enable_segmentation)
        except Exception as e:
            st.error(f"Error processing uploaded video: {str(e)}")

def process_video_frames(video_path, model_type, frame_rate, max_frames, enable_segmentation):
    """Extract frames and run object detection"""
    st.session_state.current_video_path = video_path
    
    with st.spinner("Extracting frames from video..."):
        try:
            frames = st.session_state.video_processor.extract_frames(
                video_path, 
                frame_rate=frame_rate,
                max_frames=max_frames
            )
            
            if not frames:
                st.error("No frames could be extracted from the video")
                return
            
            st.session_state.frames = frames
            st.success(f"Extracted {len(frames)} frames")
            
        except Exception as e:
            st.error(f"Error extracting frames: {str(e)}")
            return
    
    # Load models if not already loaded
    with st.spinner(f"Loading {model_type} model..."):
        try:
            st.session_state.object_detector.load_model(model_type.lower())
            st.success("Object detection model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading detection model: {str(e)}")
            return
    
    # Load segmentation model if enabled
    if enable_segmentation:
        with st.spinner("Loading semantic segmentation model..."):
            try:
                st.session_state.semantic_segmenter.load_model('deeplabv3_resnet50')
                st.success("Semantic segmentation model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading segmentation model: {str(e)}")
                # Continue without segmentation
                enable_segmentation = False
    
    # Run object detection and segmentation on frames
    processing_text = "Running object detection"
    if enable_segmentation:
        processing_text += " and semantic segmentation"
    processing_text += " on frames..."
    
    with st.spinner(processing_text):
        try:
            progress_bar = st.progress(0)
            annotations = {}
            segmentations = {}
            
            for i, frame_path in enumerate(frames):
                # Load frame
                frame = cv2.imread(frame_path)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run detection
                detections = st.session_state.object_detector.detect_objects(frame_rgb)
                
                # Run segmentation if enabled
                segmentation_mask = None
                if enable_segmentation:
                    try:
                        segmentation_mask = st.session_state.semantic_segmenter.segment_image(frame_rgb)
                    except Exception as e:
                        st.warning(f"Segmentation failed for frame {i}: {str(e)}")
                
                annotations[i] = {
                    'frame_path': frame_path,
                    'detections': detections,
                    'frame_shape': frame_rgb.shape
                }
                
                if segmentation_mask is not None:
                    segmentations[i] = {
                        'frame_path': frame_path,
                        'segmentation_mask': segmentation_mask,
                        'frame_shape': frame_rgb.shape
                    }
                
                # Update progress
                progress_bar.progress((i + 1) / len(frames))
            
            st.session_state.annotations = annotations
            st.session_state.segmentations = segmentations
            
            # Show summary
            total_detections = sum(len(ann['detections']) for ann in annotations.values())
            summary_text = f"Object detection completed on {len(frames)} frames! Total detections: {total_detections}"
            
            if enable_segmentation and segmentations:
                summary_text += f"\nSemantic segmentation completed on {len(segmentations)} frames!"
            
            st.success(summary_text)
            
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")

def review_annotations_tab():
    st.header("Review Annotations")
    
    if not st.session_state.annotations:
        st.info("No annotations available. Please process a video first.")
        return
    
    # Display mode selection
    has_segmentation = bool(st.session_state.segmentations)
    
    if has_segmentation:
        display_mode = st.radio(
            "Display Mode:",
            ["Object Detection Only", "Semantic Segmentation Only", "Combined View"],
            horizontal=True
        )
    else:
        display_mode = st.radio(
            "Display Mode:",
            ["Object Detection Only"],
            horizontal=True,
            help="Enable semantic segmentation in the sidebar to unlock additional display modes"
        )
        display_mode = "Object Detection Only"  # Force to detection only
    
    # Frame navigation
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
    
    with col1:
        if st.button("⬅️ Previous"):
            if st.session_state.current_frame_index > 0:
                st.session_state.current_frame_index -= 1
                st.rerun()
    
    with col2:
        if st.button("➡️ Next"):
            if st.session_state.current_frame_index < len(st.session_state.annotations) - 1:
                st.session_state.current_frame_index += 1
                st.rerun()
    
    with col3:
        frame_num = st.number_input(
            "Frame",
            min_value=0,
            max_value=len(st.session_state.annotations) - 1,
            value=st.session_state.current_frame_index,
            key="frame_selector"
        )
        if frame_num != st.session_state.current_frame_index:
            st.session_state.current_frame_index = frame_num
            st.rerun()
    
    with col4:
        st.write(f"Frame {st.session_state.current_frame_index + 1} of {len(st.session_state.annotations)}")
    
    # Display current frame with annotations
    if st.session_state.current_frame_index in st.session_state.annotations:
        annotation = st.session_state.annotations[st.session_state.current_frame_index]
        
        # Load and display frame
        frame = cv2.imread(annotation['frame_path'])
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get segmentation data if available
        segmentation_data = st.session_state.segmentations.get(st.session_state.current_frame_index)
        
        # Generate display based on mode
        if display_mode == "Object Detection Only":
            annotated_frame = draw_bounding_boxes(frame_rgb, annotation['detections'])
        elif display_mode == "Semantic Segmentation Only" and segmentation_data:
            annotated_frame = st.session_state.semantic_segmenter.visualize_segmentation(
                frame_rgb, segmentation_data['segmentation_mask']
            )
        elif display_mode == "Combined View" and segmentation_data:
            annotated_frame = st.session_state.semantic_segmenter.combine_with_detection(
                frame_rgb, annotation['detections'], segmentation_data['segmentation_mask']
            )
        else:
            # Fallback to detection only
            annotated_frame = draw_bounding_boxes(frame_rgb, annotation['detections'])
        
        st.image(annotated_frame, caption=f"Frame {st.session_state.current_frame_index + 1}", use_container_width=True)
        
        # Show analysis details based on display mode
        col1, col2 = st.columns(2)
        
        with col1:
            # Show detection details
            if annotation['detections']:
                st.subheader("Object Detections:")
                detection_data = []
                for det in annotation['detections']:
                    detection_data.append({
                        'Class': det['class'],
                        'Confidence': f"{det['confidence']:.2f}",
                        'Bounding Box': f"({det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, {det['bbox'][2]:.0f}, {det['bbox'][3]:.0f})"
                    })
                
                st.dataframe(pd.DataFrame(detection_data), use_container_width=True)
            else:
                st.info("No detections found in this frame.")
        
        with col2:
            # Show segmentation statistics if available
            if segmentation_data and display_mode != "Object Detection Only":
                st.subheader("Segmentation Statistics:")
                seg_stats = st.session_state.semantic_segmenter.get_segmentation_stats(
                    segmentation_data['segmentation_mask']
                )
                
                if seg_stats:
                    stats_data = []
                    for class_name, stats in seg_stats.items():
                        if stats['percentage'] > 0.1:  # Only show classes with >0.1% coverage
                            stats_data.append({
                                'Class': class_name,
                                'Coverage': f"{stats['percentage']:.1f}%",
                                'Pixels': stats['pixel_count']
                            })
                    
                    if stats_data:
                        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
                    else:
                        st.info("No significant segments found.")
                else:
                    st.info("No segmentation data available.")
            elif display_mode == "Object Detection Only":
                st.info("Enable segmentation to see pixel-level analysis.")

def draw_bounding_boxes(image, detections):
    """Draw bounding boxes on image"""
    annotated = image.copy()
    
    # Color map for different classes
    color_map = {
        'car': (255, 0, 0),      # Red
        'truck': (255, 165, 0),  # Orange
        'bus': (255, 255, 0),    # Yellow
        'person': (0, 255, 0),   # Green
        'bicycle': (0, 0, 255),  # Blue
        'motorcycle': (128, 0, 128), # Purple
        'traffic light': (255, 192, 203), # Pink
        'stop sign': (165, 42, 42)  # Brown
    }
    
    for det in detections:
        bbox = det['bbox']
        class_name = det['class']
        confidence = det['confidence']
        
        # Get color for this class
        color = color_map.get(class_name, (128, 128, 128))  # Default gray
        
        # Draw bounding box
        cv2.rectangle(
            annotated,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color,
            2
        )
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw label background
        cv2.rectangle(
            annotated,
            (int(bbox[0]), int(bbox[1]) - label_size[1] - 10),
            (int(bbox[0]) + label_size[0], int(bbox[1])),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated,
            label,
            (int(bbox[0]), int(bbox[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return annotated

def export_annotations_tab():
    st.header("Export Annotations")
    
    if not st.session_state.annotations:
        st.info("No annotations available. Please process a video first.")
        return
    
    # Export format selection
    export_format = st.selectbox(
        "Select export format:",
        ["YOLO", "COCO JSON", "CSV"]
    )
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        include_images = st.checkbox("Include image files", value=True)
    
    with col2:
        min_confidence = st.slider(
            "Minimum confidence for export",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1
        )
    
    if st.button("Export Annotations", type="primary"):
        try:
            with st.spinner("Exporting annotations..."):
                # Filter annotations by confidence
                filtered_annotations = filter_annotations_by_confidence(
                    st.session_state.annotations, 
                    min_confidence
                )
                
                if export_format == "YOLO":
                    export_path = st.session_state.annotation_exporter.export_yolo(
                        filtered_annotations, 
                        include_images=include_images
                    )
                elif export_format == "COCO JSON":
                    export_path = st.session_state.annotation_exporter.export_coco(
                        filtered_annotations,
                        include_images=include_images
                    )
                else:  # CSV
                    export_path = st.session_state.annotation_exporter.export_csv(
                        filtered_annotations
                    )
                
                st.success(f"Annotations exported successfully!")
                st.info(f"Export location: {export_path}")
                
                # Provide download link if possible
                if os.path.exists(export_path) and export_format == "CSV":
                    with open(export_path, 'r') as f:
                        csv_data = f.read()
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="annotations.csv",
                        mime="text/csv"
                    )
                
        except Exception as e:
            st.error(f"Error exporting annotations: {str(e)}")
    
    # Show export statistics
    if st.session_state.annotations:
        st.subheader("Export Statistics")
        
        total_frames = len(st.session_state.annotations)
        total_detections = sum(len(ann['detections']) for ann in st.session_state.annotations.values())
        
        # Count detections by class
        class_counts = {}
        for ann in st.session_state.annotations.values():
            for det in ann['detections']:
                if det['confidence'] >= min_confidence:
                    class_name = det['class']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Frames", total_frames)
            st.metric("Total Detections", total_detections)
        
        with col2:
            if class_counts:
                st.write("**Detections by Class:**")
                for class_name, count in sorted(class_counts.items()):
                    st.write(f"- {class_name}: {count}")

def filter_annotations_by_confidence(annotations, min_confidence):
    """Filter annotations by minimum confidence threshold"""
    filtered = {}
    
    for frame_idx, annotation in annotations.items():
        filtered_detections = [
            det for det in annotation['detections'] 
            if det['confidence'] >= min_confidence
        ]
        
        filtered[frame_idx] = {
            'frame_path': annotation['frame_path'],
            'detections': filtered_detections,
            'frame_shape': annotation['frame_shape']
        }
    
    return filtered

if __name__ == "__main__":
    main()
