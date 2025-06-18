"""
Asset loader utility for the volleyball analytics project.
"""

import os
from pathlib import Path

class AssetLoader:
    def __init__(self, base_path=None):
        self.base_path = base_path or Path(__file__).parent.parent
    
    def get_model_path(self, model_name):
        model_paths = {
            'yolo': self.base_path / 'models' / 'pretrained' / 'yolov8n.pt',
            'face_detector': self.base_path / 'models' / 'pretrained' / 'opencv_face_detector.caffemodel',
            'face_detector_config': self.base_path / 'models' / 'configs' / 'opencv_face_detector.prototxt',
        }
        return str(model_paths.get(model_name, ''))
    
    def get_asset_path(self, asset_name):
        asset_paths = {
            'test_image': self.base_path / 'assets' / 'images' / 'test_frame_yolo_faces.jpg',
        }
        return str(asset_paths.get(asset_name, ''))
    
    def validate_paths(self):
        required_assets = [
            self.get_model_path('yolo'),
            self.get_model_path('face_detector'),
            self.get_model_path('face_detector_config'),
        ]
        
        missing = []
        for path in required_assets:
            if not os.path.exists(path):
                missing.append(path)
        
        if missing:
            print("Warning: Missing assets:")
            for path in missing:
                print(f"  - {path}")
            return False
        return True

asset_loader = AssetLoader()
