# Volleyball Dataset Curator

![Dataset Curation](../assets/images/curator_interface.png)

The Dataset Curator is a web-based tool for efficiently curating volleyball ball detection datasets. It provides an intuitive interface for accepting or rejecting ball detection results with keyboard shortcuts and real-time visualization.

## Features

### Web-Based Interface
- **Browser-based UI** - Works in any modern browser
- **Keyboard shortcuts** - 'A' to Accept, 'S' to Skip for rapid curation
- **Real-time preview** - See detection results instantly
- **Progress tracking** - Visual progress bar and statistics

### Smart Curation
- **Bounding box visualization** - Shows detected ball regions
- **Confidence scoring** - Displays detection confidence levels
- **Quality filtering** - Automatic filtering of low-quality detections
- **Batch processing** - Handle thousands of images efficiently

### Export Capabilities
- **Label Studio format** - Export for further annotation
- **YOLO format** - Ready for training object detection models
- **Statistics export** - Detailed curation metrics
- **Progress persistence** - Resume curation sessions

## Installation

### 1. Install Dependencies
```bash
cd volleystat/curator
pip install -r requirements.txt
```

### 2. Prepare Dataset
Place your detection results in the data directory:
```
volleystat/
├── data/
│   ├── detected_balls/          # Images with ball detections
│   ├── detection_results/       # Detection JSON files
│   └── curated_dataset/        # Output curated images
```

## Usage

### Quick Start
```bash
cd volleystat/curator

# Check dataset status
python check_dataset.py

# Start curator (Windows)
start_curator.bat

# Start curator (PowerShell)
.\start_curator.ps1

# Start curator (Python)
python run_curator.py
```

### Web Interface
1. **Open browser** - Navigate to http://localhost:8000
2. **Review images** - Ball detections are highlighted with bounding boxes
3. **Make decisions** - Use keyboard shortcuts:
   - **'A' key** - Accept image (good detection)
   - **'S' key** - Skip image (poor detection)
4. **Monitor progress** - Watch progress bar and statistics
5. **Export results** - Curated dataset saved automatically

### Configuration

Edit `curator.py` for custom settings:

```python
class DatasetCurator:
    def __init__(self):
        # Dataset paths
        self.data_dir = Path("../data/detected_balls")
        self.output_dir = Path("../data/curated_dataset")
        
        # Curation settings
        self.target_count = 1000          # Target number of curated images
        self.min_confidence = 0.3         # Minimum detection confidence
        self.min_bbox_size = 20          # Minimum bounding box size
        
        # Server settings
        self.host = "localhost"
        self.port = 8000
```

## Workflow

### 1. Data Preparation
```bash
# Run ball detection on videos
python ../scripts/detection/detect_ball_on_video.py

# Check detection results
python check_dataset.py
```

### 2. Curation Process
```bash
# Start web curator
python run_curator.py

# Open http://localhost:8000
# Use 'A' to accept, 'S' to skip
# Monitor progress in terminal
```

### 3. Export Results
```bash
# Export to YOLO format
python export_to_yolo.py

# Export to Label Studio
python export_to_labelstudio.py

# View statistics
python get_curation_stats.py
```

## Technical Details

### Architecture
- **FastAPI backend** - Handles image serving and decision processing
- **Jinja2 templates** - Renders web interface
- **SQLite database** - Stores curation decisions (optional)
- **Path management** - Handles cross-platform file paths

### File Structure
```
curator/
├── README.md              # This documentation
├── curator.py             # Main curator application
├── run_curator.py         # Application launcher
├── check_dataset.py       # Dataset validation
├── test_curator.py        # Unit tests
├── requirements.txt       # Python dependencies
├── start_curator.bat      # Windows launcher
├── start_curator.ps1      # PowerShell launcher
└── templates/
    └── curator.html       # Web interface template
```

### Data Flow
1. **Input**: Ball detection results (images + bounding boxes)
2. **Processing**: Web interface displays images with overlays
3. **Decision**: User accepts/skips using keyboard shortcuts
4. **Output**: Curated dataset in specified format
5. **Export**: Multiple export formats available

## Advanced Usage

### Custom Detection Sources
```python
# Add custom detection source
def load_custom_detections(self, source_path):
    """Load detections from custom format."""
    # Implementation for custom format
    pass
```

### Batch Curation
```python
# Configure for large datasets
curator = DatasetCurator()
curator.target_count = 5000
curator.batch_size = 100
curator.auto_save_interval = 50
```

### Quality Filters
```python
# Add quality filtering
def apply_quality_filters(self, detections):
    """Filter detections by quality metrics."""
    filtered = []
    for det in detections:
        if (det['confidence'] > 0.5 and 
            det['bbox_area'] > 400 and
            det['aspect_ratio'] < 3.0):
            filtered.append(det)
    return filtered
```

## Integration

### With YOLO Training
```bash
# Export curated dataset
python export_to_yolo.py

# Train YOLO model
cd ../scripts/training
python train_yolo_curated.py --dataset ../curator/exports/yolo_dataset
```

### With Label Studio
```bash
# Export for annotation refinement
python export_to_labelstudio.py

# Import refined annotations
python import_from_labelstudio.py
```

## Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Change port in curator.py
self.port = 8001
```

**2. Images Not Loading**
```bash
# Check image paths
python check_dataset.py

# Verify permissions
ls -la ../data/detected_balls/
```

**3. Slow Performance**
```bash
# Reduce image size
self.display_size = (800, 600)

# Enable caching
self.enable_image_cache = True
```

### Performance Optimization

**For Large Datasets (>10k images):**
- Enable image caching
- Use image thumbnails for preview
- Implement pagination
- Add database indexing

**For Network Deployment:**
- Configure authentication
- Use HTTPS
- Add CORS settings
- Implement session management

## API Reference

### Core Methods
```python
# Get current image for curation
current_image = curator.get_current_image()

# Process user decision
result = curator.process_decision(image_id, action)

# Get curation statistics
stats = curator.get_stats()

# Export curated dataset
curator.export_dataset(format='yolo')
```

### REST Endpoints
- `GET /` - Web interface
- `GET /stats` - Curation statistics  
- `GET /current` - Current image data
- `POST /decision` - Process curation decision
- `GET /export/{format}` - Export dataset

## Contributing

### Adding New Features
1. Fork the repository
2. Create feature branch
3. Add tests in `test_curator.py`
4. Update documentation
5. Submit pull request

### Testing
```bash
# Run unit tests
python test_curator.py

# Test with sample data
python create_test_dataset.py
python run_curator.py
```

## License
MIT License - See main project README for details. 