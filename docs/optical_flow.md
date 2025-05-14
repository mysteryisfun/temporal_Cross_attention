# Optical Flow Computation Documentation

## Overview

The optical flow computation module is a key part of the Cross-Attention CNN Research project's data preprocessing pipeline. Optical flow quantifies the motion between consecutive frames, providing valuable temporal information for understanding facial expressions and movements.

## Implementation

The module is implemented in two main components:

1. **OpticalFlowComputer Class** (`scripts/preprocessing/optical_flow_computer.py`):
   - Core class for computing optical flow between consecutive frames
   - Supports multiple optical flow algorithms: Farneback and TVL1
   - Provides utilities for visualization and analysis

2. **Command-line Interface** (`scripts/compute_optical_flow.py`):
   - User-friendly interface for processing face directories
   - Configurable parameters for optical flow computation
   - Options for visualization and analysis

## Supported Optical Flow Algorithms

### Farneback Algorithm

The Farneback algorithm is a dense optical flow estimation method that computes the flow vector for every pixel in the frame. It is based on polynomial expansion of the image signals.

Key parameters:
- `pyr_scale`: Scale factor for pyramid structures (default: 0.5)
- `levels`: Number of pyramid layers (default: 3)
- `winsize`: Window size for averaging (default: 15)
- `iterations`: Number of iterations (default: 3)
- `poly_n`: Size of pixel neighborhood for polynomial approximation (default: 5)
- `poly_sigma`: Standard deviation of Gaussian for smoothing derivatives (default: 1.2)

### TVL1 Algorithm

The TVL1 (Total Variation L1) algorithm is a more robust optical flow algorithm that handles occlusions better and produces sharper flow boundaries. It is based on a variational approach.

> **Note:** The TVL1 algorithm requires the OpenCV contrib modules to be installed. If you encounter an error like `module 'cv2' has no attribute 'optflow'`, you'll need to reinstall OpenCV with the extra modules.
>
> ```bash
> pip uninstall opencv-python
> pip install opencv-contrib-python
> ```

Key parameters:
- `tau`: Time step for optimization (default: 0.25)
- `lambda_`: Weight for data term (default: 0.15)
- `theta`: Weight for smoothness term (default: 0.3)
- `nscales`: Number of scales for multi-scale processing (default: 5)
- `warps`: Number of warps per scale (default: 5)
- `epsilon`: Stopping criterion (default: 0.01)
- `iterations`: Maximum iterations (default: 300)

## Output Format

The optical flow module produces two types of outputs for each pair of consecutive frames:

1. **RGB Visualization** (.jpg):
   - Color-coded representation of optical flow
   - Hue represents flow direction
   - Intensity represents flow magnitude

2. **Raw Flow Data** (.npy):
   - NumPy arrays containing the raw (u, v) flow vectors
   - Can be used for further processing and analysis

## Directory Structure

The optical flow results are organized as follows:

```
data/
  optical_flow/
    video_id_1/
      frame_000000_flow_rgb.jpg
      frame_000000_flow_raw.npy
      frame_000010_flow_rgb.jpg
      frame_000010_flow_raw.npy
      ...
    video_id_2/
      ...
    flow_samples.png          # Visualization of sample flows
    flow_magnitude_analysis.png # Analysis of flow magnitudes
    flow_magnitude_analysis.json # Flow statistics in JSON format
```

## Usage Examples

### Basic Usage

```bash
python scripts/compute_optical_flow.py --faces-dir data/faces --output-dir data/optical_flow
```

### Using TVL1 Algorithm

```bash
python scripts/compute_optical_flow.py --faces-dir data/faces --output-dir data/optical_flow --method tvl1
```

### Advanced Configuration

```bash
python scripts/compute_optical_flow.py --faces-dir data/faces --output-dir data/optical_flow \
  --method farneback --pyr-scale 0.5 --levels 4 --winsize 20 --iterations 5 \
  --poly-n 7 --poly-sigma 1.5 --visualize --analyze
```

## Performance Considerations

- Processing time scales with the number and resolution of frames
- The TVL1 algorithm is generally more accurate but slower than Farneback
- The module supports parallel processing to improve performance

## Further Improvements

- Integration with deep learning-based optical flow methods
- Adaptive parameter selection based on video characteristics
- Enhanced visualization and analysis tools
