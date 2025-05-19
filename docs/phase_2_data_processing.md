# Phase 2: Data Processing Pipeline

## Overview

Phase 2 of the Cross-Attention CNN Research project focused on implementing a robust data processing pipeline for video data. This phase includes frame extraction, face detection and alignment, and optical flow computation. These processed outputs serve as inputs to the neural network models developed in Phase 3.

## Components

### 1. Frame Extraction Module

The frame extraction module extracts frames from video data for further processing in face detection and optical flow computation steps.

#### Features

- Extract frames from video files with configurable sampling rate
- Parallel processing for improved performance
- Support for extracting frames based on video directories or annotation files
- Frame resizing option
- Comprehensive logging and error handling
- Progress tracking with tqdm

#### Usage

**Using the extract_frames.py utility script:**

```bash
python scripts/extract_frames.py --video-dir /path/to/videos --output-dir /path/to/save/frames
```

**Importing the module in Python code:**

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

#### Command-Line Arguments

**Required Arguments:**
- `--video-dir` or `--annotation-file` (with `--video-base-dir`): Input source
- `--output-dir`: Directory to save extracted frames

**Optional Arguments:**
- `--sampling-rate`: Extract every nth frame (default: 1)
- `--max-frames`: Maximum number of frames to extract per video
- `--resize`: Resize frames to WxH (e.g., 224x224)
- `--workers`: Number of parallel workers (default: 4)
- `--log-file`: Path to log file
- `--log-level`: Set logging level

#### Examples

**Extract frames from a directory of videos:**

```bash
python scripts/extract_frames.py --video-dir data/raw/train-1 --output-dir data/processed/frames --sampling-rate 5 --max-frames 100 --resize 224x224
```

**Extract frames based on annotation file:**

```bash
python scripts/extract_frames.py --annotation-file data/raw/train-annotation/annotation_training.pkl --video-base-dir data/raw/train-1 --output-dir data/processed/frames --sampling-rate 5
```

### 2. Face Detection and Alignment System

The face detection and alignment system detects faces in video frames, aligns them based on eye positions, and prepares them for feature extraction.

#### System Components

1. **Core Face Detection Module**: `scripts/preprocessing/face_detector.py`
2. **Command-line Interface**: `scripts/process_faces.py`
3. **Analysis Tool**: `scripts/analyze_face_detection.py`

#### Features

**Face Detection:**
- Uses MTCNN for state-of-the-art face detection
- Configurable confidence threshold to filter out low-quality detections
- Option to detect multiple faces or focus on the largest face
- Parallel processing for efficiency

**Face Alignment:**
- Aligns faces based on eye positions
- Standardizes face orientation for consistent input to neural networks
- Adds appropriate margins around faces
- Resizes to a consistent resolution

**Logging and Visualization:**
- Comprehensive logging of detection and alignment process
- Visualization of success rates and processing statistics
- Analysis of face detection results across the dataset

#### Usage

**Step 1: Extract Frames from Videos**

```bash
python scripts/extract_frames.py --video-dir "data/raw/train-1/training80_01" --output-dir "data/processed/frames" --sampling-rate 5 --max-frames 100 --resize 224x224
```

**Step 2: Process Frames to Detect and Align Faces**

```bash
python scripts/process_faces.py --input-dir "data/processed/frames" --output-dir "data/faces" --confidence 0.9 --min-size 80 --target-size 224x224 --visualize
```

**Step 3: Analyze Face Detection Results**

```bash
python scripts/analyze_face_detection.py --face-dir "data/faces" --output-dir "results/visualizations"
```

#### Best Practices

1. **Confidence Threshold**: Adjust the confidence threshold based on your dataset. Lower values will detect more faces but may include false positives.
2. **Processing Parameters**: Tune the minimum face size and target size based on your specific requirements.
3. **Multi-processing**: Adjust the number of workers based on your system's capabilities for optimal performance.
4. **Batch Processing**: For large datasets, consider processing in batches or using the recursive option.

### 3. Optical Flow Computation

The optical flow computation module calculates motion vectors between consecutive frames, which serve as input for the dynamic feature extractor in Phase 3.

#### Features

- Computes optical flow using Farneback or TVL1 methods
- Supports batch processing of video sequences
- Visualization options for flow fields
- Integration with the frame extraction pipeline
- Configurable parameters for flow computation
- Parallel processing for efficiency

#### Usage

**Computing Optical Flow:**

```bash
python scripts/compute_optical_flow.py --frame-dir "data/processed/frames" --output-dir "data/optical_flow" --method farneback --visualize
```

**Processing a Batch of Videos:**

```bash
python scripts/process_all_optical_flow.py --video-dir "data/raw/train-1" --output-dir "data/optical_flow" --sampling-rate 2 --method tvl1
```

#### Flow Visualization

The optical flow visualization tools in `scripts/analyze_optical_flow.py` provide:

- Color-coded flow field visualization
- Motion vector visualization
- Flow magnitude heatmaps
- Temporal flow pattern analysis

### 4. Data Pipeline Integration

The complete data processing pipeline integrates all components to prepare data for the model training phase:

1. Video data is processed to extract frames at a specified sampling rate
2. Extracted frames are processed for face detection and alignment
3. Consecutive frames are used to compute optical flow
4. Processed faces and optical flow data are organized for efficient batch loading
5. Quality control measures identify and handle failed processing cases

#### Pipeline Execution

The integrated pipeline can be executed with:

```bash
python scripts/prepare_dataset.py --video-dir "data/raw" --output-dir "data/processed" --sampling-rate 5 --face-confidence 0.9 --flow-method farneback --workers 8
```

## Milestone Completion

| Component | Status | Documentation |
|-----------|--------|---------------|
| Frame Extraction | Completed | `docs/frame_extraction.md` |
| Face Detection & Alignment | Completed | `docs/face_detection_completed.md` |
| Optical Flow Computation | Completed | `docs/optical_flow.md` |
| Data Pipeline Integration | Completed | `docs/data_pipeline_logging.md` |

## Performance Metrics

The data processing pipeline achieves the following performance metrics:

- **Frame Extraction**: 25-30 frames per second (depending on resolution)
- **Face Detection**: 10-15 faces per second (MTCNN, single-process)
- **Face Detection (Parallel)**: 30-40 faces per second (with 4 workers)
- **Optical Flow**: 5-8 frame pairs per second (Farneback method)
- **Optical Flow (TVL1)**: 3-5 frame pairs per second (higher quality)
- **End-to-End Processing**: ~2-3 videos per minute (with 8 workers)

## Next Steps

With the data processing pipeline completed in Phase 2, the project moves to Phase 3: Model Architecture Development, focusing on:

1. Static feature extractor implementation (2D CNN)
2. Dynamic feature extractor implementation (3D CNN for optical flow)
3. Cross-attention mechanism for feature fusion
4. Implementation of the prediction head for personality traits

---
*Last updated: 2025-05-20*
