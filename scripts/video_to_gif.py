import cv2
import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional
import argparse
from pathlib import Path


class VideoToGifConverter:
    """
    A class for converting video segments to high-quality GIF animations.
    """
    
    def __init__(self, fps: int = 10, resize_factor: float = 1.0):
        """
        Initialize the converter.
        
        Args:
            fps: Frames per second for the GIF (default: 10)
            resize_factor: Factor to resize the video (default: 1.0 = no resize)
        """
        self.fps = fps
        self.resize_factor = resize_factor
    
    def time_to_seconds(self, time_str: str) -> float:
        """
        Convert time string (HH:MM:SS or MM:SS) to seconds.
        
        Args:
            time_str: Time string in format HH:MM:SS or MM:SS
            
        Returns:
            Time in seconds
        """
        parts = time_str.split(':')
        if len(parts) == 2:
            # MM:SS format
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:
            # HH:MM:SS format
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            raise ValueError(f"Invalid time format: {time_str}. Use HH:MM:SS or MM:SS")
    
    def extract_frames(self, video_path: str, start_time: str, end_time: str) -> list:
        """
        Extract frames from video between start_time and end_time.
        
        Args:
            video_path: Path to the video file
            start_time: Start time in format HH:MM:SS or MM:SS
            end_time: End time in format HH:MM:SS or MM:SS
            
        Returns:
            List of PIL Image objects
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Convert times to seconds
        start_seconds = self.time_to_seconds(start_time)
        end_seconds = self.time_to_seconds(end_time)
        
        # Convert to frame numbers
        start_frame = int(start_seconds * fps_video)
        end_frame = int(end_seconds * fps_video)
        
        # Ensure frame numbers are within bounds
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        
        frames = []
        frame_interval = max(1, int(fps_video / self.fps))  # Sample frames to match desired FPS
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if self.resize_factor != 1.0:
                height, width = frame_rgb.shape[:2]
                new_width = int(width * self.resize_factor)
                new_height = int(height * self.resize_factor)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        
        cap.release()
        return frames
    
    def create_gif(self, video_path: str, start_time: str, end_time: str, 
                   output_path: str, optimize: bool = True) -> str:
        """
        Create GIF from video segment.
        
        Args:
            video_path: Path to the video file
            start_time: Start time in format HH:MM:SS or MM:SS
            end_time: End time in format HH:MM:SS or MM:SS
            output_path: Path for the output GIF file
            optimize: Whether to optimize the GIF (default: True)
            
        Returns:
            Path to the created GIF file
        """
        print(f"Extracting frames from {video_path}")
        print(f"Time range: {start_time} - {end_time}")
        
        frames = self.extract_frames(video_path, start_time, end_time)
        
        if not frames:
            raise ValueError("No frames extracted from the video segment")
        
        print(f"Extracted {len(frames)} frames")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save as GIF
        print(f"Saving GIF to {output_path}")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=optimize,
            duration=int(1000 / self.fps),  # Duration in milliseconds
            loop=0
        )
        
        print(f"GIF created successfully: {output_path}")
        return output_path


def main():
    """
    Main function for command line usage.
    """
    parser = argparse.ArgumentParser(description="Convert video segment to GIF animation")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("start_time", help="Start time (HH:MM:SS or MM:SS)")
    parser.add_argument("end_time", help="End time (HH:MM:SS or MM:SS)")
    parser.add_argument("output_path", help="Path for the output GIF file")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for GIF (default: 10)")
    parser.add_argument("--resize", type=float, default=1.0, help="Resize factor (default: 1.0)")
    parser.add_argument("--no-optimize", action="store_true", help="Disable GIF optimization")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    # Create converter
    converter = VideoToGifConverter(fps=args.fps, resize_factor=args.resize)
    
    try:
        # Create GIF
        converter.create_gif(
            video_path=args.video_path,
            start_time=args.start_time,
            end_time=args.end_time,
            output_path=args.output_path,
            optimize=not args.no_optimize
        )
        print("GIF creation completed successfully!")
        
    except Exception as e:
        print(f"Error creating GIF: {e}")


if __name__ == "__main__":
    main() 