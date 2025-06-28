import os
import cv2
import yt_dlp
import tempfile
from pathlib import Path
import streamlit as st
from project_utils import create_temp_dir

class VideoProcessor:
    def __init__(self):
        self.temp_dir = create_temp_dir()
    
    def download_youtube_video(self, url, quality='720p'):
        """Download video from YouTube URL"""
        try:
            # Configure yt-dlp options
            ydl_opts = {
                'format': f'best[height<={quality[:-1]}]/best',
                'outtmpl': os.path.join(self.temp_dir, '%(title)s.%(ext)s'),
                'restrictfilenames': True,
                'noplaylist': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first to get the filename
                info = ydl.extract_info(url, download=False)
                filename = ydl.prepare_filename(info)
                
                # Download the video
                ydl.download([url])
                
                # Return the path to downloaded file
                if os.path.exists(filename):
                    return filename
                else:
                    # Try to find the downloaded file
                    for file in os.listdir(self.temp_dir):
                        if file.startswith(info.get('title', '')[:20]):
                            return os.path.join(self.temp_dir, file)
                    
                    return None
                    
        except Exception as e:
            st.error(f"Error downloading video: {str(e)}")
            return None
    
    def extract_frames(self, video_path, frame_rate=5, max_frames=1000):
        """Extract frames from video at specified frame rate"""
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            st.info(f"Video info: FPS={fps:.1f}, Duration={duration:.1f}s, Total frames={total_frames}")
            
            frames_dir = os.path.join(self.temp_dir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            
            frame_paths = []
            frame_count = 0
            extracted_count = 0
            
            # Calculate frame interval
            frame_interval = max(1, int(fps / frame_rate))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at specified interval
                if frame_count % frame_interval == 0:
                    frame_filename = f"frame_{extracted_count:06d}.jpg"
                    frame_path = os.path.join(frames_dir, frame_filename)
                    
                    # Save frame
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    extracted_count += 1
                    
                    # Limit number of frames to prevent memory issues
                    if extracted_count >= max_frames:
                        st.warning(f"Reached maximum frame limit ({max_frames}). Processing first {max_frames} frames only.")
                        break
                
                frame_count += 1
            
            cap.release()
            
            if not frame_paths:
                raise ValueError("No frames could be extracted from the video")
            
            return frame_paths
            
        except Exception as e:
            st.error(f"Error extracting frames: {str(e)}")
            return []
    
    def get_video_info(self, video_path):
        """Get video information"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': total_frames,
                'duration': duration
            }
            
        except Exception as e:
            st.error(f"Error getting video info: {str(e)}")
            return None
    
    def validate_video_file(self, video_path):
        """Validate if the file is a valid video"""
        try:
            cap = cv2.VideoCapture(video_path)
            is_valid = cap.isOpened()
            cap.release()
            return is_valid
        except:
            return False
