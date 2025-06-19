#!/usr/bin/env python3
"""
Embed images as base64 in Label Studio JSON files.
This allows importing JSON files directly without separate image uploads.
"""

import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any
import argparse
from PIL import Image
import io


class ImageEmbedder:
    """Embed images as base64 in Label Studio JSON files."""
    
    def __init__(self, json_dir: str, images_dir: str, output_dir: str):
        """
        Initialize image embedder.
        
        Args:
            json_dir: Directory containing JSON files
            images_dir: Directory containing images
            output_dir: Directory to save embedded JSON files
        """
        self.json_dir = Path(json_dir)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'files_processed': 0,
            'images_embedded': 0,
            'missing_images': [],
            'errors': []
        }
    
    def find_image_file(self, image_filename: str) -> Path:
        """
        Find image file recursively in images directory.
        
        Args:
            image_filename: Name of the image file to find
            
        Returns:
            Path to the image file if found, None otherwise
        """
        # Search recursively for the image file
        for ext in ['.jpg', '.jpeg', '.png']:
            search_pattern = f"*{image_filename}{ext}" if '.' not in image_filename else f"*{image_filename}"
            for img_path in self.images_dir.rglob(search_pattern):
                if img_path.name == image_filename or img_path.stem == image_filename:
                    return img_path
        
        return None
    
    def encode_image_to_base64(self, image_path: Path) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string with data URL prefix
        """
        try:
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                
            # Determine MIME type based on file extension
            ext = image_path.suffix.lower()
            if ext in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            elif ext == '.png':
                mime_type = 'image/png'
            else:
                mime_type = 'image/jpeg'  # Default
            
            # Encode to base64
            base64_data = base64.b64encode(img_data).decode('utf-8')
            
            # Return data URL
            return f"data:{mime_type};base64,{base64_data}"
            
        except Exception as e:
            raise Exception(f"Error encoding image {image_path}: {str(e)}")
    
    def embed_images_in_json(self, json_file: Path) -> bool:
        """
        Embed images as base64 in a JSON file.
        
        Args:
            json_file: Path to JSON file to process
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"🔄 Processing: {json_file.name}")
            
            # Read the JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if it's an array of tasks
            if isinstance(data, list):
                tasks = data
            else:
                print(f"   ⚠️ Unexpected JSON format, skipping...")
                return False
            
            # Process each task
            embedded_count = 0
            for task in tasks:
                # Get image filename from file_upload or data.image
                image_filename = task.get('file_upload', '')
                if not image_filename:
                    # Try to extract from data.image path
                    image_path = task.get('data', {}).get('image', '')
                    if image_path:
                        image_filename = Path(image_path).name
                
                if image_filename:
                    # Find the actual image file
                    image_file = self.find_image_file(image_filename)
                    
                    if image_file and image_file.exists():
                        # Encode image to base64
                        base64_data = self.encode_image_to_base64(image_file)
                        
                        # Update the task data
                        task['data']['image'] = base64_data
                        embedded_count += 1
                        print(f"   ✅ Embedded: {image_filename}")
                    else:
                        self.stats['missing_images'].append(image_filename)
                        print(f"   ❌ Missing image: {image_filename}")
            
            # Save the updated JSON file
            output_file = self.output_dir / json_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.stats['images_embedded'] += embedded_count
            self.stats['files_processed'] += 1
            
            print(f"   💾 Saved: {embedded_count} images embedded")
            return True
            
        except Exception as e:
            error_msg = f"Error processing {json_file}: {str(e)}"
            print(f"   ❌ {error_msg}")
            self.stats['errors'].append(error_msg)
            return False
    
    def process_all_files(self):
        """Process all JSON files in the directory."""
        print("🖼️ Embedding images in Label Studio JSON files...")
        print(f"📁 JSON directory: {self.json_dir}")
        print(f"📁 Images directory: {self.images_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all JSON files
        json_files = [f for f in self.json_dir.glob("*.json") 
                     if f.name not in ['dataset_summary.json', 'format_fix_summary.json', 'chunk_info.json']]
        self.stats['total_files'] = len(json_files)
        
        print(f"📊 Found {self.stats['total_files']} JSON files")
        
        # Process each file
        for json_file in json_files:
            self.embed_images_in_json(json_file)
        
        # Create summary report
        self.create_summary_report()
        
        print(f"\n✅ Image embedding completed!")
        print(f"📊 Final Statistics:")
        print(f"   Total files processed: {self.stats['total_files']}")
        print(f"   Files processed: {self.stats['files_processed']}")
        print(f"   Images embedded: {self.stats['images_embedded']}")
        print(f"   Missing images: {len(self.stats['missing_images'])}")
        print(f"   Output directory: {self.output_dir}")
        
        if self.stats['missing_images']:
            print(f"\n⚠️ Missing images:")
            for img in self.stats['missing_images'][:10]:  # Show first 10
                print(f"   - {img}")
            if len(self.stats['missing_images']) > 10:
                print(f"   ... and {len(self.stats['missing_images']) - 10} more")
        
        if self.stats['errors']:
            print(f"\n❌ Errors encountered: {len(self.stats['errors'])}")
    
    def create_summary_report(self):
        """Create a summary report."""
        summary_data = {
            "embedding_info": {
                "total_files": self.stats['total_files'],
                "files_processed": self.stats['files_processed'],
                "images_embedded": self.stats['images_embedded'],
                "missing_images_count": len(self.stats['missing_images']),
                "format": "Label Studio JSON with embedded images"
            },
            "embedded_files": []
        }
        
        # List all embedded JSON files
        for json_file in sorted(self.output_dir.glob("*.json")):
            file_size = json_file.stat().st_size
            summary_data["embedded_files"].append({
                "json_file": json_file.name,
                "size_mb": round(file_size / (1024 * 1024), 2)
            })
        
        summary_path = self.output_dir / "embedding_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Embedding summary saved to: {summary_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Embed images as base64 in Label Studio JSON files")
    parser.add_argument("--json-dir", "-j", 
                       default="volleystat/data/label_studio_chunks_fixed_correct",
                       help="Directory containing JSON files")
    parser.add_argument("--images-dir", "-i", 
                       default="volleystat/data/label_studio_chunks_json_fixed",
                       help="Directory containing images")
    parser.add_argument("--output", "-o", 
                       default="volleystat/data/label_studio_chunks_with_images",
                       help="Output directory for embedded JSON files")
    
    args = parser.parse_args()
    
    # Create embedder
    embedder = ImageEmbedder(
        json_dir=args.json_dir,
        images_dir=args.images_dir,
        output_dir=args.output
    )
    
    # Process all files
    embedder.process_all_files()
    
    print("\n✅ Image embedding completed!")


if __name__ == "__main__":
    main() 