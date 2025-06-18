import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def create_gif_from_video(video_path, output_gif_path, start_time=4, duration=3, fps=10, resize_factor=0.5):
    """
    Create a GIF animation from a video segment.
    
    Args:
        video_path: Path to input video
        output_gif_path: Path to output GIF file
        start_time: Start time in seconds
        duration: Duration in seconds
        fps: Frames per second for the GIF
        resize_factor: Factor to resize frames (0.5 = half size)
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {total_frames} frames, {video_fps:.1f} fps, {width}x{height}")
    
    # Calculate frame positions
    start_frame = int(start_time * video_fps)
    end_frame = int((start_time + duration) * video_fps)
    frames_to_extract = end_frame - start_frame
    
    # Calculate new dimensions
    new_width = int(width * resize_factor)
    new_height = int(height * resize_factor)
    
    print(f"Extracting frames {start_frame} to {end_frame} ({frames_to_extract} frames)")
    print(f"Resizing to {new_width}x{new_height}")
    
    # Set video position to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Extract frames
    frames = []
    frame_interval = max(1, int(video_fps / fps))  # Skip frames to match desired FPS
    
    for i in tqdm(range(frames_to_extract), desc="Extracting frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only take every nth frame to match desired FPS
        if i % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_resized)
            frames.append(pil_image)
    
    cap.release()
    
    if not frames:
        print("Error: No frames extracted!")
        return
    
    print(f"Extracted {len(frames)} frames for GIF")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
    
    # Save as GIF
    print(f"Saving GIF to {output_gif_path}")
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),  # Duration in milliseconds
        loop=0,  # 0 means loop forever
        optimize=True,
        quality=85
    )
    
    print(f"GIF created successfully!")
    print(f"Size: {new_width}x{new_height}")
    print(f"Frames: {len(frames)}")
    print(f"Duration: {len(frames) / fps:.1f} seconds")
    print(f"File: {output_gif_path}")


def main():
    # Configuration
    video_path = r'C:\Users\illya\Documents\volleyball_analitics\volleystat\results\video\segments\segment_01_start_000409.mp4'
    output_gif_path = r'C:\Users\illya\Documents\volleyball_analitics\volleystat\results\video\segments\volleyball_animation.gif'
    
    # Create GIF from 00:00:04 to 00:00:07 (3 seconds)
    create_gif_from_video(
        video_path=video_path,
        output_gif_path=output_gif_path,
        start_time=4,      # Start at 4 seconds
        duration=3,        # 3 seconds duration
        fps=15,            # 15 FPS for smooth animation
        resize_factor=0.6  # Resize to 60% for smaller file size
    )


if __name__ == "__main__":
    main() 