import os
import tempfile
import shutil
import streamlit as st
from pathlib import Path

def create_temp_dir(suffix=''):
    """Create a temporary directory"""
    temp_dir = tempfile.mkdtemp(suffix=f'_{suffix}' if suffix else '')
    return temp_dir

def cleanup_temp_files(temp_dir):
    """Clean up temporary files"""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        st.warning(f"Could not clean up temporary files: {str(e)}")

def ensure_dir_exists(directory):
    """Ensure directory exists, create if not"""
    os.makedirs(directory, exist_ok=True)
    return directory

def get_file_size(file_path):
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    except:
        return 0

def validate_image_file(file_path):
    """Validate if file is a valid image"""
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    file_ext = Path(file_path).suffix.lower()
    return file_ext in valid_extensions

def validate_video_file(file_path):
    """Validate if file is a valid video"""
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    file_ext = Path(file_path).suffix.lower()
    return file_ext in valid_extensions

def format_duration(seconds):
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def safe_filename(filename):
    """Make filename safe for filesystem"""
    import re
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    return filename

def get_youtube_video_id(url):
    """Extract video ID from YouTube URL"""
    import re
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def validate_youtube_url(url):
    """Validate if URL is a valid YouTube URL"""
    youtube_patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'https?://(?:www\.)?youtu\.be/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/embed/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/v/[\w-]+'
    ]
    
    import re
    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return True
    
    return False

def bytes_to_human_readable(size_bytes):
    """Convert bytes to human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def create_progress_callback(progress_bar):
    """Create a progress callback for long-running operations"""
    def callback(current, total):
        if total > 0:
            progress = current / total
            progress_bar.progress(progress)
    
    return callback

class VideoMetadata:
    """Class to store video metadata"""
    def __init__(self, width=0, height=0, fps=0, duration=0, total_frames=0):
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.total_frames = total_frames
    
    def __str__(self):
        return f"Video: {self.width}x{self.height}, {self.fps:.1f}fps, {format_duration(self.duration)}, {self.total_frames} frames"
    
    def to_dict(self):
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'duration': self.duration,
            'total_frames': self.total_frames
        }

def check_disk_space(path, required_space_mb):
    """Check if there's enough disk space available"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_mb = free / (1024 * 1024)
        return free_mb >= required_space_mb
    except:
        return True  # Assume sufficient space if check fails

def estimate_processing_time(total_frames, fps=30):
    """Estimate processing time based on frame count"""
    # Rough estimate: 0.1 seconds per frame for detection
    estimated_seconds = total_frames * 0.1
    return format_duration(estimated_seconds)
