# Frame Extraction Module for Cross-Attention CNN Research

This module is part of Phase 2.2 of the Cross-Attention CNN Research project. It extracts frames from video data for further processing in face detection and optical flow computation steps.

## Features

- Extract frames from video files with configurable sampling rate
- Parallel processing for improved performance
- Support for extracting frames based on video directories or annotation files
- Frame resizing option
- Comprehensive logging and error handling
- Progress tracking with tqdm

## Usage

You can use the frame extraction module in two ways:

### 1. Using the extract_frames.py utility script

```bash
python scripts/extract_frames.py --video-dir /path/to/videos --output-dir /path/to/save/frames
```

### 2. Importing the module in your Python code

```python
from scripts.preprocessing.frame_extractor import process_video_directory

# Extract frames from a directory of videos
stats = process_video_directory(
    video_dir="/path/to/videos",
    output_base_dir="/path/to/save/frames",
    sampling_rate=1,
    max_frames=None,  # Extract all frames
    frame_size=(224, 224),  # Resize frames to 224x224
    num_workers=4,  # Use 4 parallel workers
)
```

## Command-Line Arguments

### Required Arguments

One of the following input sources is required:
- `--video-dir`: Directory containing video files
- `--annotation-file`: Path to annotation file (requires `--video-base-dir`)

Output directory:
- `--output-dir`: Directory to save extracted frames

### Optional Arguments

Frame extraction options:
- `--sampling-rate`: Extract every nth frame (default: 1)
- `--max-frames`: Maximum number of frames to extract per video
- `--resize`: Resize frames to WxH (e.g., 224x224)

Processing options:
- `--workers`: Number of parallel workers (default: 4)
- `--video-base-dir`: Base directory containing video files (required with `--annotation-file`)

Logging options:
- `--log-file`: Path to log file
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Examples

### Extract frames from a directory of videos

```bash
python scripts/extract_frames.py --video-dir data/raw/train-1 --output-dir data/processed/frames --sampling-rate 5 --max-frames 100 --resize 224x224
```

This will extract every 5th frame from each video in the directory, up to a maximum of 100 frames per video, and resize them to 224x224 pixels.

### Extract frames based on annotation file

```bash
python scripts/extract_frames.py --annotation-file data/raw/train-annotation/annotation_training.pkl --video-base-dir data/raw/train-1 --output-dir data/processed/frames --sampling-rate 5
```

This will extract every 5th frame from the videos specified in the annotation file.

## Next Steps in Phase 2.2

After frame extraction, the following steps remain to be implemented:

1. Face detection and alignment system
2. Optical flow computation module
3. Data augmentation pipeline
4. Preprocessing failure handling system
