#!/usr/bin/env python3
"""
Interactive YOLO Dataset Creator with Enhanced Masks
Uses Hough Circles to create precise volleyball ball masks
Supports resuming work from existing dataset
"""

import json
import cv2  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import base64
from io import BytesIO
from PIL import Image
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os


def rgb2gray(image: np.ndarray) -> np.ndarray:
    """Converts image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return image / 255.0 if image.max() > 1 else image


class InteractiveYOLODatasetCreator:
    """Interactive YOLO dataset creator with enhanced masks."""
    
    def __init__(self, output_dir: str, target_samples: int = 800):
        """Initialize dataset creator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_samples = target_samples
        
        # Create YOLO dataset structure
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
        
        # Optimal Hough Circles parameters
        self.hough_params = {
            'dp': 1,
            'minDist': 28,
            'param1': 100,
            'param2': 20,
            'fixed_radius': 15
        }
        
        # Statistics counters
        self.processed_count = 0
        self.accepted_count = 0
        self.rejected_count = 0
        self.rejected_no_circle = 0  # Rejected due to no circle found
        
        # Current processing data
        self.current_image = None
        self.current_circle = None
        self.current_filename = None
        self.current_bbox = None
        self.user_choice = None
        
        # Set of processed files for resuming work
        self.processed_files: Set[str] = set()
        
        # Load existing progress
        self.load_existing_progress()
        
    def load_existing_progress(self):
        """Load information about existing progress."""
        existing_images = list((self.output_dir / "images").glob("*.jpg"))
        existing_labels = list((self.output_dir / "labels").glob("*.txt"))
        
        # Check correspondence between images and labels
        valid_samples = []
        for img_path in existing_images:
            label_path = self.output_dir / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                valid_samples.append(img_path.stem)
                self.processed_files.add(img_path.stem)
        
        self.accepted_count = len(valid_samples)
        
        print(f"Found existing dataset:")
        print(f"   Ready for YOLO: {self.accepted_count}/{self.target_samples}")
        
        # Progress bar
        progress_percent = (self.accepted_count / self.target_samples) * 100
        progress_bar_length = 50
        filled_length = int(progress_bar_length * self.accepted_count // self.target_samples)
        bar = '█' * filled_length + '-' * (progress_bar_length - filled_length)
        print(f"   Progress: |{bar}| {progress_percent:.1f}%")
        
        remaining = max(0, self.target_samples - self.accepted_count)
        print(f"   Remaining until YOLO ready: {remaining}")
        
        if self.accepted_count >= self.target_samples:
            print(f"YOLO DATASET READY! Reached target of {self.target_samples} samples")
            return True
        
        return False
    
    def is_file_already_processed(self, filename: str) -> bool:
        """Check if file has already been processed."""
        base_name = Path(filename).stem
        return base_name in self.processed_files
    
    def is_dataset_complete(self) -> bool:
        """Check if dataset is complete."""
        return self.accepted_count >= self.target_samples
    
    def load_label_studio_data(self, json_path: Path) -> List[Dict]:
        """Load data from Label Studio JSON file."""
        print(f"Loading annotations from: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter only items with volleyball ball annotations
        annotated_data = []
        for item in data:
            if item.get('annotations') and len(item['annotations']) > 0:
                for annotation in item['annotations']:
                    if annotation.get('result'):
                        for result in annotation['result']:
                            if (result.get('value', {}).get('rectanglelabels') and 
                                'Volleyball Ball' in result['value']['rectanglelabels']):
                                annotated_data.append(item)
                                break
        
        print(f"Found {len(annotated_data)} items with ball annotations")
        return annotated_data
    
    def decode_base64_image(self, base64_string: str) -> np.ndarray:
        """Decode base64 image to numpy array."""
        try:
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            image_bytes = base64.b64decode(base64_string)
            pil_image = Image.open(BytesIO(image_bytes))
            image_array = np.array(pil_image)
            
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            return image_array
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None
    
    def extract_bbox_from_label_studio(self, item: Dict) -> Tuple[str, Tuple[int, int, int, int], np.ndarray]:
        """Extract bounding box coordinates and image from Label Studio."""
        filename = item.get('file_upload', 'unknown')
        
        # Get image from data
        image = None
        if 'data' in item and 'image' in item['data']:
            image_data = item['data']['image']
            image = self.decode_base64_image(image_data)
        
        for annotation in item['annotations']:
            if annotation.get('result'):
                for result in annotation['result']:
                    if (result.get('value', {}).get('rectanglelabels') and 
                        'Volleyball Ball' in result['value']['rectanglelabels']):
                        
                        value = result['value']
                        original_width = result['original_width']
                        original_height = result['original_height']
                        
                        # Convert percentage coordinates to absolute
                        x_percent = value['x']
                        y_percent = value['y']
                        width_percent = value['width']
                        height_percent = value['height']
                        
                        x1 = int((x_percent / 100.0) * original_width)
                        y1 = int((y_percent / 100.0) * original_height)
                        x2 = int(((x_percent + width_percent) / 100.0) * original_width)
                        y2 = int(((y_percent + height_percent) / 100.0) * original_height)
                        
                        return filename, (x1, y1, x2, y2), image
        
        return None, None, None
    
    def create_ball_crop_with_margin(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                                   margin_pixels: int = 4) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Create ball crop with additional pixels on both axes.
        
        Args:
            image: Original image
            bbox: Ball bounding box (x1, y1, x2, y2)
            margin_pixels: Additional pixels on both axes
            
        Returns:
            Cropped image and new bbox coordinates relative to crop
        """
        x1, y1, x2, y2 = bbox
        
        # Add margin_pixels on all sides
        crop_x1 = max(0, x1 - margin_pixels)
        crop_y1 = max(0, y1 - margin_pixels)
        crop_x2 = min(image.shape[1], x2 + margin_pixels)
        crop_y2 = min(image.shape[0], y2 + margin_pixels)
        
        # Create crop
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Bbox coordinates relative to crop
        new_bbox = (x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1)
        
        return crop, new_bbox
    
    def detect_circle_with_hough(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int]]:
        """
        Apply Hough Circles with optimal parameters.
        
        Args:
            image: Cropped image
            bbox: Bounding box relative to crop
            
        Returns:
            Best found circle (x, y, r) or None
        """
        # Convert to grayscale
        gray = rgb2gray(image)
        
        # Create mask only for bbox area with margin
        mask = np.zeros(gray.shape, dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        
        margin = 3
        mask_x1 = max(0, x1 - margin)
        mask_y1 = max(0, y1 - margin)
        mask_x2 = min(gray.shape[1], x2 + margin)
        mask_y2 = min(gray.shape[0], y2 + margin)
        
        mask[mask_y1:mask_y2, mask_x1:mask_x2] = 255
        
        # Apply mask
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Convert to uint8 for HoughCircles
        masked_gray_uint8 = (masked_gray * 255).astype(np.uint8)
        
        try:
            # Apply Hough Circle Transform with optimal parameters
            circles = cv2.HoughCircles(
                masked_gray_uint8,
                cv2.HOUGH_GRADIENT,
                dp=self.hough_params['dp'],
                minDist=self.hough_params['minDist'],
                param1=self.hough_params['param1'],
                param2=self.hough_params['param2'],
                minRadius=self.hough_params['fixed_radius'] - 2,
                maxRadius=self.hough_params['fixed_radius'] + 2
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # Return first found circle
                if len(circles) > 0:
                    x, y, r = circles[0]
                    # Check if circle center is within bbox area
                    if (mask_x1 <= x <= mask_x2 and mask_y1 <= y <= mask_y2):
                        return (x, y, r)
            
            return None
            
        except Exception as e:
            print(f"HoughCircles error: {e}")
            return None
    
    def create_circle_mask(self, image_shape: Tuple[int, int], circle: Tuple[int, int, int]) -> np.ndarray:
        """Create circle mask."""
        mask = np.zeros(image_shape, dtype=np.uint8)
        x, y, r = circle
        cv2.circle(mask, (x, y), r, 255, -1)
        return mask
    
    def show_interactive_preview(self, image: np.ndarray, crop: np.ndarray, circle: Optional[Tuple[int, int, int]], 
                               bbox: Tuple[int, int, int, int], crop_bbox: Tuple[int, int, int, int], filename: str) -> bool:
        """
        Show interactive window for decision making.
        
        Returns:
            True if user accepted, False if rejected
        """
        self.user_choice = None
        
        # Create window
        root = tk.Tk()
        progress_percent = (self.accepted_count / self.target_samples) * 100
        root.title(f"YOLO Dataset Creator - {filename} | Ready: {self.accepted_count}/{self.target_samples} ({progress_percent:.1f}%)")
        root.geometry("1200x800")
        
        # Focus window for keyboard events
        root.focus_set()
        
        # Create matplotlib figure with 4 panels
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Progress bar for title
        progress_bar_length = 30
        filled_length = int(progress_bar_length * self.accepted_count // self.target_samples)
        progress_bar = '█' * filled_length + '░' * (progress_bar_length - filled_length)
        
        fig.suptitle(f'YOLO Dataset: {filename} | {progress_bar} {self.accepted_count}/{self.target_samples} ({progress_percent:.1f}%)', 
                    fontsize=14, fontweight='bold')
        
        # 1. Crop with original bbox
        crop_with_bbox = crop.copy()
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
        cv2.rectangle(crop_with_bbox, (crop_x1, crop_y1), (crop_x2, crop_y2), (128, 128, 128), 2)
        
        axes[0, 0].imshow(crop_with_bbox[..., ::-1])  # BGR to RGB
        axes[0, 0].set_title('Crop with original BBox')
        axes[0, 0].axis('off')
        
        # 2. Crop with found circle
        crop_with_circle = crop.copy()
        if circle:
            x, y, r = circle
            cv2.circle(crop_with_circle, (x, y), r, (0, 255, 0), 1)  # Line thickness 1 pixel
            cv2.circle(crop_with_circle, (x, y), 1, (0, 0, 255), 2)  # Center slightly smaller
            
        axes[0, 1].imshow(crop_with_circle[..., ::-1])  # BGR to RGB
        axes[0, 1].set_title('Crop with found circle' if circle else 'Crop - circle NOT found')
        axes[0, 1].axis('off')
        
        # 3. Original image with new circle (if found)
        if circle:
            # Convert circle coordinates back to original image
            original_circle = self.convert_crop_circle_to_original_coords(circle, bbox, margin_pixels=4)
            
            original_with_new_circle = image.copy()
            orig_x, orig_y, orig_r = original_circle
            cv2.circle(original_with_new_circle, (orig_x, orig_y), orig_r, (0, 255, 0), 1)
            cv2.circle(original_with_new_circle, (orig_x, orig_y), 1, (0, 0, 255), 2)
            
            axes[1, 0].imshow(original_with_new_circle[..., ::-1])  # BGR to RGB
            axes[1, 0].set_title(f'Original with new circle\nCenter: ({orig_x}, {orig_y}), R: {orig_r}')
        else:
            axes[1, 0].text(0.5, 0.5, 'Original\nCircle not found', 
                           transform=axes[1, 0].transAxes, ha='center', va='center',
                           fontsize=16, color='red')
            axes[1, 0].set_title('Original image')
        axes[1, 0].axis('off')
        
        # 4. Circle mask on original image
        if circle:
            original_circle = self.convert_crop_circle_to_original_coords(circle, bbox, margin_pixels=4)
            mask = self.create_circle_mask(image.shape[:2], original_circle)
            axes[1, 1].imshow(mask, cmap='gray')
            axes[1, 1].set_title('Mask on original image')
        else:
            axes[1, 1].text(0.5, 0.5, 'Mask\nnot created', 
                           transform=axes[1, 1].transAxes, ha='center', va='center',
                           fontsize=16, color='red')
            axes[1, 1].set_title('Circle mask')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Embed matplotlib in tkinter
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Create buttons
        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        def accept():
            self.user_choice = True
            root.quit()
            root.destroy()
        
        def reject():
            self.user_choice = False
            root.quit()
            root.destroy()
        
        def stop_processing():
            self.user_choice = 'stop'
            root.quit()
            root.destroy()
        
        def on_key_press(event):
            """Handle key press events."""
            key = event.keysym.lower()
            if key == 'y' or key == 'return' or key == 'space':
                # Y, Enter or Space - accept
                accept()
            elif key == 'n' or key == 'escape':
                # N or Escape - reject
                reject()
            elif key == 'q':
                # Q - stop processing
                stop_processing()
        
        # Bind key handler
        root.bind('<Key>', on_key_press)
        
        # Buttons
        tk.Button(button_frame, text="ACCEPT (Y/Enter/Space)", 
                 command=accept, bg='lightgreen', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="REJECT (N/Esc)", 
                 command=reject, bg='lightcoral', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="STOP (Q)", 
                 command=stop_processing, bg='orange', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # YOLO dataset progress info
        remaining = max(0, self.target_samples - self.accepted_count)
        progress_info = tk.Label(button_frame, 
                               text=f"YOLO Readiness: {self.accepted_count}/{self.target_samples} | Remaining: {remaining}", 
                               font=('Arial', 11, 'bold'), fg='darkgreen')
        progress_info.pack(side=tk.LEFT, padx=10)
        
        # Keyboard shortcuts info
        keyboard_info = tk.Label(button_frame, 
                               text="Keys: Y/Enter/Space=Accept | N/Esc=Reject | Q=Stop", 
                               font=('Arial', 10, 'bold'), fg='blue')
        keyboard_info.pack(side=tk.LEFT, padx=10)
        
        # Parameters info
        info_text = f"Parameters: dp={self.hough_params['dp']}, minDist={self.hough_params['minDist']}, "
        info_text += f"param1={self.hough_params['param1']}, param2={self.hough_params['param2']}, r={self.hough_params['fixed_radius']}"
        
        tk.Label(button_frame, text=info_text, font=('Arial', 8)).pack(side=tk.RIGHT, padx=5)
        
        # Start window
        root.mainloop()
        
        plt.close(fig)
        return self.user_choice if self.user_choice is not None else False
    
    def convert_crop_circle_to_original_coords(self, crop_circle: Tuple[int, int, int], 
                                             original_bbox: Tuple[int, int, int, int], 
                                             margin_pixels: int = 4) -> Tuple[int, int, int]:
        """
        Convert circle coordinates from crop back to original image.
        
        Args:
            crop_circle: (x, y, r) circle coordinates on crop
            original_bbox: (x1, y1, x2, y2) original bbox
            margin_pixels: Additional pixels used for crop
            
        Returns:
            (x, y, r) circle coordinates on original image
        """
        crop_x, crop_y, crop_r = crop_circle
        orig_x1, orig_y1, orig_x2, orig_y2 = original_bbox
        
        # Crop coordinates relative to original image
        crop_start_x = max(0, orig_x1 - margin_pixels)
        crop_start_y = max(0, orig_y1 - margin_pixels)
        
        # Convert circle coordinates back to original image
        original_x = crop_x + crop_start_x
        original_y = crop_y + crop_start_y
        original_r = crop_r  # Radius remains the same
        
        return (original_x, original_y, original_r)
    
    def convert_to_yolo_format(self, circle: Tuple[int, int, int], image_shape: Tuple[int, int]) -> str:
        """
        Convert circle to YOLO format.
        
        Args:
            circle: (x, y, r) circle coordinates on original image
            image_shape: (height, width) original image dimensions
            
        Returns:
            String in YOLO format: "class_id center_x center_y width height"
        """
        x, y, r = circle
        height, width = image_shape
        
        # Bbox coordinates around circle
        bbox_x1 = max(0, x - r)
        bbox_y1 = max(0, y - r)
        bbox_x2 = min(width, x + r)
        bbox_y2 = min(height, y + r)
        
        # Center and dimensions in normalized coordinates
        center_x = (bbox_x1 + bbox_x2) / 2 / width
        center_y = (bbox_y1 + bbox_y2) / 2 / height
        bbox_width = (bbox_x2 - bbox_x1) / width
        bbox_height = (bbox_y2 - bbox_y1) / height
        
        # YOLO format: class_id center_x center_y width height
        return f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}"
    
    def save_yolo_sample(self, original_image: np.ndarray, crop_circle: Tuple[int, int, int], 
                       original_bbox: Tuple[int, int, int, int], filename: str):
        """
        Save sample in YOLO format with original image and correct coordinates.
        
        Args:
            original_image: Original full image
            crop_circle: Circle coordinates found on crop
            original_bbox: Original bbox used for crop
            filename: File name
        """
        base_name = Path(filename).stem
        
        # Generate unique name if file already exists
        counter = 1
        original_base_name = base_name
        while base_name in self.processed_files:
            base_name = f"{original_base_name}_{counter}"
            counter += 1
        
        # Convert circle coordinates from crop back to original image
        original_circle = self.convert_crop_circle_to_original_coords(
            crop_circle, original_bbox, margin_pixels=4
        )
        
        # Save original image
        image_path = self.output_dir / "images" / f"{base_name}.jpg"
        cv2.imwrite(str(image_path), original_image)
        
        # Save label with coordinates relative to original image
        label_path = self.output_dir / "labels" / f"{base_name}.txt"
        yolo_annotation = self.convert_to_yolo_format(original_circle, original_image.shape[:2])
        
        with open(label_path, 'w') as f:
            f.write(yolo_annotation)
        
        # Add to processed files
        self.processed_files.add(base_name)
        
        # Output information about saved coordinates
        orig_x, orig_y, orig_r = original_circle
        print(f"Saved: {image_path.name} + {label_path.name}")
        print(f"   Original circle coordinates: center=({orig_x}, {orig_y}), radius={orig_r}")
    
    def process_chunk_file(self, chunk_path: Path):
        """Process one chunk file."""
        print(f"\nProcessing chunk file: {chunk_path.name}")
        
        # Check if dataset is not complete
        if self.is_dataset_complete():
            print(f"DATASET COMPLETE! Reached {self.target_samples} samples")
            return 'complete'
        
        # Load data
        data = self.load_label_studio_data(chunk_path)
        
        if len(data) == 0:
            print("No data with annotations found")
            return 'empty'
        
        # Process each element
        for i, item in enumerate(data):
            # Check if dataset is not complete
            if self.is_dataset_complete():
                print(f"DATASET COMPLETE! Reached {self.target_samples} samples")
                return 'complete'
            
            print(f"\nProcessing image {i+1}/{len(data)}")
            
            # Extract information
            filename, bbox, image = self.extract_bbox_from_label_studio(item)
            
            if bbox is None or image is None:
                print(f"Failed to extract bbox or image for {filename}")
                continue
            
            # Check if file already processed
            if self.is_file_already_processed(filename):
                print(f"File {filename} already processed, skipping")
                continue
            
            print(f"   File: {filename}")
            print(f"   BBox: {bbox}")
            
            # Create crop with additional pixels
            crop, crop_bbox = self.create_ball_crop_with_margin(image, bbox, margin_pixels=4)
            
            # Apply Hough Circles
            circle = self.detect_circle_with_hough(crop, crop_bbox)
            
            if circle:
                print(f"   Found circle: center=({circle[0]}, {circle[1]}), radius={circle[2]}")
            else:
                print(f"   Circle not found - skipping image")
                self.rejected_no_circle += 1
                continue
            
            # Show interactive window only if circle found
            self.processed_count += 1
            
            user_choice = self.show_interactive_preview(image, crop, circle, bbox, crop_bbox, filename)
            
            if user_choice == 'stop':
                print(f"Processing stopped by user")
                return 'stopped'
            elif user_choice and circle:
                # Save in YOLO format with original image and correct coordinates
                self.save_yolo_sample(image, circle, bbox, filename)
                self.accepted_count += 1
                print(f"   ACCEPTED and saved")
            else:
                self.rejected_count += 1
                print(f"   REJECTED by user")
            
            # Output current statistics with progress bar
            remaining = max(0, self.target_samples - self.accepted_count)
            progress_percent = (self.accepted_count / self.target_samples) * 100
            
            # Mini progress bar for console
            mini_bar_length = 20
            mini_filled = int(mini_bar_length * self.accepted_count // self.target_samples)
            mini_bar = '█' * mini_filled + '░' * (mini_bar_length - mini_filled)
            
            print(f"   YOLO Progress: |{mini_bar}| {self.accepted_count}/{self.target_samples} ({progress_percent:.1f}%)")
            print(f"   Remaining: {remaining} | Rejected: {self.rejected_count} | No circles: {self.rejected_no_circle}")
        
        return 'completed_chunk'
    
    def process_all_chunks(self, chunks_dir: Path):
        """Process all chunk files in directory."""
        chunk_files = list(chunks_dir.glob("volleyball_ball_detection_chunk_*.json"))
        chunk_files.sort()
        
        print(f"Found {len(chunk_files)} chunk files")
        
        if self.is_dataset_complete():
            print(f"DATASET ALREADY COMPLETE! We have {self.accepted_count} samples")
            return
        
        for chunk_file in chunk_files:
            result = self.process_chunk_file(chunk_file)
            
            if result == 'complete':
                break
            elif result == 'stopped':
                break
            elif result == 'empty':
                continue
            
            # Ask whether to continue only if dataset not complete
            if not self.is_dataset_complete() and self.processed_count > 0:
                remaining = max(0, self.target_samples - self.accepted_count)
                progress_percent = (self.accepted_count / self.target_samples) * 100
                
                # Progress bar for dialog
                dialog_bar_length = 30
                dialog_filled = int(dialog_bar_length * self.accepted_count // self.target_samples)
                dialog_bar = '█' * dialog_filled + '░' * (dialog_bar_length - dialog_filled)
                
                continue_processing = messagebox.askyesno(
                    "Continue creating YOLO dataset?", 
                    f"YOLO Readiness: {self.accepted_count}/{self.target_samples} ({progress_percent:.1f}%)\n"
                    f"Progress: |{dialog_bar}|\n\n"
                    f"Remaining until ready: {remaining}\n"
                    f"Rejected by user: {self.rejected_count}\n"
                    f"Skipped without circles: {self.rejected_no_circle}\n\n"
                    f"Continue processing next chunk file?"
                )
                
                if not continue_processing:
                    break
        
        self.print_final_statistics()
    
    def print_final_statistics(self):
        """Print final statistics."""
        print(f"\n" + "="*70)
        print(f"FINAL YOLO DATASET STATISTICS")
        print(f"="*70)
        
        # Main progress bar
        progress_percent = (self.accepted_count / self.target_samples) * 100
        final_bar_length = 50
        final_filled = int(final_bar_length * self.accepted_count // self.target_samples)
        final_bar = '█' * final_filled + '░' * (final_bar_length - final_filled)
        
        print(f"YOLO READINESS:")
        print(f"   |{final_bar}| {self.accepted_count}/{self.target_samples} ({progress_percent:.1f}%)")
        print()
        
        total_processed = self.processed_count + self.rejected_no_circle
        print(f"Total reviewed: {total_processed} images")
        print(f"   Shown for review: {self.processed_count} images (with circles)")
        print(f"   Skipped without circles: {self.rejected_no_circle} images")
        print(f"Accepted for YOLO: {self.accepted_count}/{self.target_samples} samples")
        print(f"Rejected by user: {self.rejected_count} samples")
        
        if self.processed_count > 0:
            success_rate = (self.accepted_count / self.processed_count) * 100
            print(f"Approval success rate: {success_rate:.1f}%")
        
        remaining = max(0, self.target_samples - self.accepted_count)
        if remaining > 0:
            print(f"Remaining until YOLO ready: {remaining} samples")
            print(f"Status: YOLO dataset NOT READY for training")
        else:
            print(f"YOLO DATASET READY FOR TRAINING!")
        
        print(f"\nYOLO dataset saved in: {self.output_dir}")
        print(f"   Images: {self.output_dir / 'images'}")
        print(f"   Labels: {self.output_dir / 'labels'}")
        
        # Create data.yaml for YOLO
        self.create_yolo_config()
    
    def create_yolo_config(self):
        """Create YOLO configuration file data.yaml."""
        config_content = f"""# YOLO Dataset Configuration
# Volleyball Ball Detection Dataset
# Generated by Interactive YOLO Dataset Creator

path: {self.output_dir.absolute()}  # dataset root dir
train: images  # train images (relative to 'path')
val: images    # val images (relative to 'path') 
test: images   # test images (optional, relative to 'path')

# Classes
nc: 1  # number of classes
names: ['volleyball_ball']  # class names

# Dataset info
total_samples: {self.accepted_count}
target_samples: {self.target_samples}
completed: {'yes' if self.is_dataset_complete() else 'no'}
"""
        
        config_path = self.output_dir / "data.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"Created YOLO configuration: {config_path}")


def main():
    """Main function."""
    print("Interactive YOLO Dataset Creator")
    print("Uses Hough Circles for enhanced masks")
    print("Supports resuming work from existing dataset")
    print("Goal: create 800 quality samples for YOLO training")
    
    # Paths
    chunks_dir = Path(r"C:\Users\illya\Documents\volleyball_analitics\volleystat\data\label_studio_chunks_with_images")
    output_dir = Path(r"C:\Users\illya\Documents\volleyball_analitics\volleystat\data\yolo_dataset_improved")
    
    if not chunks_dir.exists():
        print(f"Chunks directory not found: {chunks_dir}")
        return
    
    # Create dataset creator with target of 800 samples
    creator = InteractiveYOLODatasetCreator(str(output_dir), target_samples=800)
    
    # Check if dataset already complete
    if creator.is_dataset_complete():
        print(f"Dataset already complete! We have {creator.accepted_count} samples")
        creator.create_yolo_config()
        return
    
    # Process all chunks
    creator.process_all_chunks(chunks_dir)


if __name__ == "__main__":
    main() 