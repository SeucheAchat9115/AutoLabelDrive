# Auto-Labeling Tool for Autonomous Driving

A Streamlit-based web application for automatically detecting and annotating objects in YouTube videos using YOLOv5. Designed specifically for creating training datasets for autonomous driving systems.

## Features

- **YouTube Video Processing**: Download and process videos directly from YouTube URLs
- **Video File Upload**: Support for MP4, AVI, MOV, and MKV formats
- **Object Detection**: YOLOv5-based detection focusing on autonomous driving relevant objects:
  - Vehicles (cars, trucks, buses, motorcycles)
  - Pedestrians
  - Bicycles
  - Traffic signs and lights
- **Semantic Segmentation**: DeepLabV3-based pixel-level segmentation for detailed scene understanding:
  - Road surface detection
  - Sidewalk and building identification
  - Vegetation and sky segmentation
  - Combined detection + segmentation visualization
- **Interactive Review**: Browse through detected frames with bounding box visualizations
- **Multiple Export Formats**:
  - YOLO format for training
  - COCO JSON format
  - CSV format for analysis
- **Configurable Processing**: Adjust frame sampling rate, confidence thresholds, and maximum frame limits

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd auto-labeling-tool
   ```

2. **Install dependencies**:
   ```bash
   pip install streamlit torch torchvision opencv-python yt-dlp pandas pillow seaborn numpy
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py --server.port 5000
   ```

## Usage

### 1. Video Input
- **YouTube**: Enter a YouTube URL and click "Download and Process"
- **File Upload**: Upload a video file and click "Process Video"

### 2. Configuration (Sidebar)
- **Model Selection**: Choose between YOLOv5s, YOLOv5m, or YOLOv5l
- **Semantic Segmentation**: Enable pixel-level segmentation for detailed scene analysis
- **Confidence Threshold**: Set minimum confidence for detections (0.1-1.0)
- **Frame Sampling Rate**: Process every Nth frame (1-30)
- **Max Frames**: Limit processing to specified number of frames (10-2000)

### 3. Review Annotations
- Navigate through processed frames
- Choose display mode: Object Detection Only, Semantic Segmentation Only, or Combined View
- View bounding boxes with class labels and confidence scores
- See pixel-level segmentation statistics and coverage percentages
- Analyze both object-level and scene-level information

### 4. Export Annotations
- Choose export format (YOLO, COCO JSON, or CSV)
- Set minimum confidence threshold for export
- Option to include image files
- Files saved to `exports/` directory

## Project Structure

```
auto-labeling-tool/
├── app.py                 # Main Streamlit application
├── video_processor.py     # YouTube download and frame extraction
├── object_detector.py     # YOLOv5 object detection
├── annotation_exporter.py # Export functionality
├── project_utils.py       # Utility functions
├── exports/              # Output directory for annotations
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── README.md
├── .gitignore
└── requirements.txt
```

## Technical Details

### Object Detection
- Uses YOLOv5 models from Ultralytics
- GPU acceleration when available, CPU fallback
- Filters detections to autonomous driving relevant classes
- Configurable confidence thresholds and post-processing

### Export Formats

**YOLO Format**:
- Normalized bounding box coordinates (x_center, y_center, width, height)
- Class mappings in `classes.txt`
- Dataset YAML configuration file
- Optional image copies

**COCO JSON Format**:
- Standard COCO dataset structure
- Includes image metadata and category definitions
- Bounding boxes in (x, y, width, height) format

**CSV Format**:
- Tabular data with frame and detection information
- Includes confidence scores and bounding box coordinates
- Easy to import into data analysis tools

### Performance Optimization
- Configurable frame sampling to reduce processing time
- Maximum frame limits to prevent memory issues
- Temporary file cleanup
- Progress tracking for long-running operations

## System Requirements

- Python 3.7+
- Internet connection (for YouTube downloads and model loading)
- Recommended: GPU support for faster inference
- Sufficient storage for video files and extracted frames

## Troubleshooting

### Common Issues

1. **Port already in use**: Kill existing Streamlit processes or use a different port
2. **YouTube download fails**: Check internet connection and video accessibility
3. **Model loading errors**: Ensure PyTorch is properly installed
4. **Memory issues**: Reduce max frames or frame sampling rate

### Performance Tips

- Use smaller YOLOv5 models (s vs l) for faster processing
- Increase frame sampling rate to process fewer frames
- Set reasonable max frame limits for testing
- Close browser tabs when not needed to save memory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for object detection
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube video downloading
- [Streamlit](https://streamlit.io/) for the web interface