#!/usr/bin/env python3
"""
Dataset Curator - FastAPI app for curating volleyball ball dataset
"""

import json
import shutil
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from datetime import datetime
import base64

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


class CurationDecision(BaseModel):
    """Model for curation decision."""
    action: str  # 'accept' or 'skip'
    image_id: str


class DatasetCurator:
    """Dataset curator for volleyball ball images."""
    
    def __init__(
        self, 
        input_data_dir: str = "../data/result_detection_yolo",
        output_dataset_dir: str = "../data/curated_dataset",
        target_count: int = 1000
    ):
        """Initialize dataset curator."""
        self.input_data_dir = Path(input_data_dir)
        self.output_dataset_dir = Path(output_dataset_dir)
        self.target_count = target_count
        
        # Create output directories
        self.output_dataset_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dataset_dir / "images").mkdir(exist_ok=True)
        (self.output_dataset_dir / "labels").mkdir(exist_ok=True)
        
        # Load available images
        self.available_images = self._discover_images()
        self.current_index = 0
        
        # Load existing dataset info
        self.dataset_info_file = self.output_dataset_dir / "dataset_info.json"
        self.accepted_images = self._load_dataset_info()
        
        print(f"🖼️  Found {len(self.available_images)} images for curation")
        print(f"📁 Output dataset directory: {self.output_dataset_dir}")
        print(f"✅ Already accepted: {len(self.accepted_images)} images")
        print(f"🎯 Target: {self.target_count} images")
    
    def _discover_images(self) -> List[Dict[str, Any]]:
        """Discover all available images from detection results."""
        images = []
        
        if not self.input_data_dir.exists():
            print(f"⚠️  Input directory not found: {self.input_data_dir}")
            return images
        
        # Find all video processing directories
        video_dirs = [d for d in self.input_data_dir.iterdir() if d.is_dir() and d.name.startswith("video_")]
        
        for video_dir in video_dirs:
            # Load annotations
            annotations_file = video_dir / "annotations.json"
            if not annotations_file.exists():
                continue
            
            try:
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                
                images_dir = video_dir / "images"
                if not images_dir.exists():
                    continue
                
                # Process each annotation
                for annotation in annotations:
                    if not annotation.get('annotations'):
                        continue
                    
                    # Get image filename from annotation
                    file_upload = annotation.get('file_upload', '')
                    if not file_upload:
                        continue
                    
                    image_path = images_dir / file_upload
                    if not image_path.exists():
                        continue
                    
                    # Extract bounding boxes from annotation
                    bboxes = []
                    for ann in annotation['annotations']:
                        for result in ann.get('result', []):
                            if result.get('type') == 'rectanglelabels':
                                value = result.get('value', {})
                                bbox = {
                                    'x': value.get('x', 0),
                                    'y': value.get('y', 0),
                                    'width': value.get('width', 0),
                                    'height': value.get('height', 0),
                                    'label': value.get('rectanglelabels', ['Ball'])[0]
                                }
                                bboxes.append(bbox)
                    
                    if bboxes:
                        image_info = {
                            'id': str(uuid.uuid4()),
                            'image_path': str(image_path),
                            'video_dir': str(video_dir),
                            'filename': file_upload,
                            'bboxes': bboxes,
                            'annotation': annotation
                        }
                        images.append(image_info)
            
            except Exception as e:
                print(f"⚠️  Error processing {video_dir}: {e}")
                continue
        
        return images
    
    def _load_dataset_info(self) -> List[str]:
        """Load information about already accepted images."""
        if not self.dataset_info_file.exists():
            return []
        
        try:
            with open(self.dataset_info_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('accepted_images', [])
        except:
            return []
    
    def _save_dataset_info(self) -> None:
        """Save dataset information."""
        data = {
            'accepted_images': self.accepted_images,
            'target_count': self.target_count,
            'created_at': datetime.now().isoformat(),
            'total_accepted': len(self.accepted_images)
        }
        
        with open(self.dataset_info_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Create YOLO data.yaml file if we have enough images
        if len(self.accepted_images) >= 100:  # Create config when we have at least 100 images
            self._create_yolo_config()
    
    def get_current_image(self) -> Optional[Dict[str, Any]]:
        """Get current image for curation."""
        if self.current_index >= len(self.available_images):
            return None
        
        # Skip already accepted images
        skipped_count = 0
        while (self.current_index < len(self.available_images) and 
               self.available_images[self.current_index]['id'] in self.accepted_images):
            self.current_index += 1
            skipped_count += 1
        
        if skipped_count > 0:
            print(f"🔄 Skipped {skipped_count} already processed images")
        
        if self.current_index >= len(self.available_images):
            return None
        
        # Check if we've reached the target
        if len(self.accepted_images) >= self.target_count:
            print(f"🎉 Target reached! {len(self.accepted_images)} images curated")
            return None
        
        return self.available_images[self.current_index]
    
    def process_decision(self, image_id: str, action: str) -> Dict[str, Any]:
        """Process curation decision."""
        # Find image by ID
        image_info = None
        for img in self.available_images:
            if img['id'] == image_id:
                image_info = img
                break
        
        if not image_info:
            raise HTTPException(status_code=404, detail="Image not found")
        
        if action == 'accept':
            if image_id not in self.accepted_images:
                # Copy image and create YOLO label
                self._copy_image_to_dataset(image_info)
                self.accepted_images.append(image_id)
                self._save_dataset_info()
                print(f"✅ Accepted image: {image_info['filename']} ({len(self.accepted_images)}/{self.target_count})")
            else:
                print(f"⚠️  Image already accepted: {image_info['filename']}")
        
        # Move to next image
        self.current_index += 1
        
        return {
            'action': action,
            'image_id': image_id,
            'accepted_count': len(self.accepted_images),
            'target_count': self.target_count,
            'progress': len(self.accepted_images) / self.target_count * 100,
            'completed': len(self.accepted_images) >= self.target_count
        }
    
    def _copy_image_to_dataset(self, image_info: Dict[str, Any]) -> None:
        """Copy image to dataset and create YOLO label."""
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_id = image_info['id'][:8]
        filename_base = f"{timestamp}_{image_id}"
        
        # Copy image
        source_path = Path(image_info['image_path'])
        target_image_path = self.output_dataset_dir / "images" / f"{filename_base}.jpg"
        shutil.copy2(source_path, target_image_path)
        
        # Create YOLO label
        target_label_path = self.output_dataset_dir / "labels" / f"{filename_base}.txt"
        self._create_yolo_label(image_info, target_label_path)
    
    def _create_yolo_label(self, image_info: Dict[str, Any], label_path: Path) -> None:
        """Create YOLO format label file."""
        # Load image to get dimensions
        image = cv2.imread(image_info['image_path'])
        if image is None:
            return
        
        height, width = image.shape[:2]
        
        # Convert bboxes to YOLO format
        with open(label_path, 'w') as f:
            for bbox in image_info['bboxes']:
                # Convert percentage to absolute coordinates
                x_abs = (bbox['x'] / 100) * width
                y_abs = (bbox['y'] / 100) * height
                w_abs = (bbox['width'] / 100) * width
                h_abs = (bbox['height'] / 100) * height
                
                # Convert to YOLO format (center_x, center_y, width, height) normalized
                center_x = (x_abs + w_abs / 2) / width
                center_y = (y_abs + h_abs / 2) / height
                norm_width = w_abs / width
                norm_height = h_abs / height
                
                # Write to file: class_id center_x center_y width height
                f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    def _create_yolo_config(self) -> None:
        """Create YOLO dataset configuration file."""
        config_content = f"""# Volleyball Ball Dataset
# Created by Dataset Curator on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

path: {self.output_dataset_dir.absolute()}  # dataset root dir
train: images  # train images (relative to 'path')
val: images    # val images (relative to 'path') 
test:          # test images (optional)

# Classes
nc: 1  # number of classes
names: ['Volleyball Ball']  # class names

# Training info
total_images: {len(self.accepted_images)}
target_count: {self.target_count}
created_at: {datetime.now().isoformat()}
"""
        
        config_file = self.output_dataset_dir / "data.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"📝 Created YOLO config: {config_file}")
    
    def get_image_with_visualization(self, image_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get image with bounding box visualization."""
        if not image_info:
            return {'error': 'No image available'}
        
        # Load image
        image_path = image_info['image_path']
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'Could not load image: {image_path}'}
        
        # Create visualization
        vis_image = image.copy()
        height, width = image.shape[:2]
        
        # Draw bounding boxes
        for bbox in image_info['bboxes']:
            # Convert percentage to absolute coordinates
            x = int((bbox['x'] / 100) * width)
            y = int((bbox['y'] / 100) * height)
            w = int((bbox['width'] / 100) * width)
            h = int((bbox['height'] / 100) * height)
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = bbox.get('label', 'Ball')
            cv2.putText(vis_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add image info
        info_text = f"Image: {image_info['filename']}"
        cv2.putText(vis_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        bbox_count = len(image_info['bboxes'])
        bbox_text = f"Bboxes: {bbox_count}"
        cv2.putText(vis_image, bbox_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Resize if too large
        max_size = 800
        if width > max_size or height > max_size:
            scale = min(max_size / width, max_size / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            vis_image = cv2.resize(vis_image, (new_width, new_height))
        
        # Convert to base64
        image_base64 = self._image_to_base64(vis_image)
        
        return {
            'image_id': image_info['id'],
            'filename': image_info['filename'],
            'image_base64': image_base64,
            'bbox_count': bbox_count,
            'image_size': f"{width}x{height}"
        }
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string."""
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{image_base64}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get curation statistics."""
        return {
            'total_available': len(self.available_images),
            'accepted_count': len(self.accepted_images),
            'target_count': self.target_count,
            'progress_percent': (len(self.accepted_images) / self.target_count * 100) if self.target_count > 0 else 0,
            'remaining': max(0, self.target_count - len(self.accepted_images)),
            'completed': len(self.accepted_images) >= self.target_count
        }


# Initialize curator
curator = DatasetCurator(
    input_data_dir="../data/result_detection_yolo",
    output_dataset_dir="../data/curated_dataset",
    target_count=1000
)

# Create FastAPI app
app = FastAPI(title="Dataset Curator", description="Volleyball Ball Dataset Curation Tool")

@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """Serve the main curation page."""
    try:
        with open("templates/curator.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Fallback simple HTML
        return """
        <html><head><title>Dataset Curator</title></head>
        <body><h1>Dataset Curator</h1>
        <p>HTML template not found. Please create templates/curator.html</p>
        </body></html>
        """

@app.get("/api/stats")
async def get_stats():
    """Get curation statistics."""
    return curator.get_stats()

@app.get("/api/current-image")
async def get_current_image():
    """Get current image for curation."""
    current_image = curator.get_current_image()
    
    # Check if we've completed the target
    accepted_count = len(curator.accepted_images)
    if accepted_count >= curator.target_count:
        print(f"✅ Curation completed! {accepted_count}/{curator.target_count} images")
        return {"completed": True, "message": "Curation completed!"}
    elif not current_image:
        print(f"📋 No more images available. Curated: {accepted_count}/{curator.target_count}")
        return {"completed": True, "message": "No more images available"}
    
    # Get image with visualization
    visualization_data = curator.get_image_with_visualization(current_image)
    
    visualization_data['current_progress'] = f"{len(curator.accepted_images)}/{curator.target_count}"
    
    return visualization_data

@app.post("/api/decide")
async def make_decision(decision: CurationDecision):
    """Process curation decision."""
    try:
        result = curator.process_decision(decision.image_id, decision.action)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Dataset Curator...")
    print("📱 Open http://localhost:8000 in your browser")
    print("🎮 Use keyboard: 'A' to Accept, 'S' to Skip")
    print("⚠️  Keep this terminal window open!")
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 