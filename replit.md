# Auto-Labeling Tool for Autonomous Driving

## Overview

This is a Streamlit-based web application designed for automatically detecting and annotating objects in YouTube videos for autonomous driving applications. The tool leverages YOLOv5 models to identify relevant objects like vehicles, pedestrians, traffic signs, and other road elements, making it useful for creating training datasets for autonomous driving systems.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

- **Frontend**: Streamlit web interface providing an intuitive user experience
- **Video Processing**: YouTube video downloading and frame extraction capabilities
- **Object Detection**: YOLOv5-based computer vision for identifying autonomous driving-relevant objects
- **Annotation Export**: Multiple export formats for training data compatibility
- **Utilities**: Common helper functions for file management and validation

## Key Components

### 1. Video Processor (`video_processor.py`)
- **Purpose**: Downloads YouTube videos and extracts frames for processing
- **Key Features**:
  - YouTube video downloading via yt-dlp
  - Configurable video quality selection
  - Frame extraction at specified intervals
  - Temporary file management

### 2. Object Detector (`object_detector.py`)
- **Purpose**: Performs computer vision object detection using YOLOv5
- **Key Features**:
  - Multiple YOLOv5 model variants (s, m, l)
  - GPU acceleration when available
  - Focuses on autonomous driving-relevant classes (vehicles, pedestrians, traffic signs)
  - Configurable confidence thresholds

### 3. Annotation Exporter (`annotation_exporter.py`)
- **Purpose**: Exports detected annotations in various formats
- **Key Features**:
  - YOLO format export for training compatibility
  - Optional image inclusion with annotations
  - Structured directory organization
  - Class mapping for autonomous driving objects

### 4. Main Application (`app.py`)
- **Purpose**: Streamlit web interface orchestrating all components
- **Key Features**:
  - Wide layout configuration optimized for image viewing
  - Session state management for persistent data
  - Sidebar configuration options
  - Model selection interface

### 5. Utilities (`utils.py`)
- **Purpose**: Common helper functions and file management
- **Key Features**:
  - Temporary directory creation and cleanup
  - File validation for images and videos
  - Size formatting and duration calculations

## Data Flow

1. **Input**: User provides YouTube URL or uploads video file
2. **Download/Upload**: Video is downloaded from YouTube or uploaded to temporary storage
3. **Frame Extraction**: Video is processed to extract frames at specified intervals
4. **Object Detection**: Each frame is analyzed using YOLOv5 to detect relevant objects
5. **Annotation**: Detected objects are stored with bounding box coordinates and class labels
6. **Export**: Annotations are exported in YOLO format with optional image inclusion
7. **Cleanup**: Temporary files are cleaned up after processing

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **OpenCV (cv2)**: Computer vision and video processing
- **PyTorch**: Deep learning framework for YOLOv5
- **yt-dlp**: YouTube video downloading
- **PIL (Pillow)**: Image processing
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation

### Model Dependencies
- **YOLOv5**: Pre-trained object detection models from Ultralytics
- **COCO Dataset Classes**: Using standard COCO class mappings

## Deployment Strategy

The application is designed for deployment on Replit with the following considerations:

- **Environment**: Python-based Streamlit application
- **Resource Requirements**: GPU support preferred but CPU fallback available
- **File Management**: Uses temporary directories for processing with automatic cleanup
- **External Access**: Requires internet access for YouTube video downloading and model loading

### Deployment Requirements
- Python 3.7+
- Sufficient storage for temporary video and frame files
- Internet connectivity for model downloads and YouTube access
- Optional: GPU acceleration for faster inference

## Changelog
- June 28, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.