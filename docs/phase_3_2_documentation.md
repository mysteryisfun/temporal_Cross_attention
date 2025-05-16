# Phase 3.2: Dynamic Feature Extraction from Optical Flow Documentation

## Overview

Phase 3.2 implements a dynamic feature extraction system that processes optical flow data using a 3D CNN architecture. This system extracts temporal patterns from motion information captured in optical flow sequences. The implementation is based on the I3D (Inflated 3D ConvNet) architecture, which is well-suited for processing temporal information in video data.

This update includes two significant enhancements:
1. **Temporal Feature Visualization**: Advanced visualization techniques for analyzing how features evolve over time
2. **Model Serialization**: A comprehensive system for saving and loading models with metadata and versioning

## Components

The dynamic feature extraction system consists of the following components:

1. **Dynamic Feature Extractor**: A 3D CNN model based on the I3D architecture that extracts features from optical flow sequences.
2. **Optical Flow Sequence Loader**: Utility for loading, preprocessing, and batching optical flow sequences for the model.
3. **Visualization Tools**: Utilities for visualizing optical flow sequences, feature activations, and temporal patterns.
4. **Model Serialization Tools**: Utilities for saving and loading models with metadata and versioning.
5. **Evaluation Metrics**: Tools for measuring the consistency and discriminability of the extracted features.

## Model Architecture

The dynamic feature extractor uses an I3D architecture with the following characteristics:

- **Input**: Optical flow sequences with shape [batch_size, frames, height, width, channels].
- **Backbone**: 3D convolutions with Inception-style blocks.
- **Temporal Processing**: 3D convolutions and pooling operations to capture motion patterns over time.
- **Output**: Feature vectors that represent the temporal dynamics of the input optical flow sequences.

The I3D architecture includes:
- Initial 3D convolution and pooling layers
- Multiple 3D Inception blocks
- Global average pooling
- Final dense layers for feature extraction

## Usage

### 1. Preprocessing Optical Flow Data

Before using the dynamic feature extractor, optical flow must be computed from video frames. The `optical_flow_computer.py` script is used for this purpose:

```bash
python scripts/preprocessing/optical_flow_computer.py --input_dir /path/to/videos --output_dir data/optical_flow --method farneback
```

### 2. Implementing and Testing the Dynamic Feature Extractor

The `implement_dynamic_feature_extractor.py` script is used to initialize, train, and test the dynamic feature extractor:

```bash
# Implement and save the model
python scripts/implement_dynamic_feature_extractor.py --data_dir data/optical_flow --output_dir results/dynamic_feature_extractor --mode implement

# Test the model on optical flow sequences
python scripts/implement_dynamic_feature_extractor.py --data_dir data/optical_flow --output_dir results/dynamic_feature_extractor --mode test

# Evaluate the model's performance
python scripts/implement_dynamic_feature_extractor.py --data_dir data/optical_flow --output_dir results/dynamic_feature_extractor --mode evaluate
```

### 3. Testing with Optical Flow Sequences

The `test_dynamic_feature_extractor.py` script provides more detailed testing and visualization of the model's performance:

```bash
python scripts/test_dynamic_feature_extractor.py --data_dir data/optical_flow --output_dir results/dynamic_feature_extractor/test --num_samples 5
```

### 4. Optical Flow Sequence Loader

The `optical_flow_sequence_loader.py` script provides utilities for loading and preprocessing optical flow sequences:

```bash
# Load and preprocess existing optical flow data
python scripts/optical_flow_sequence_loader.py --mode load --input_dir data/optical_flow --output_dir data/optical_flow/processed

# Compute optical flow from a video file
python scripts/optical_flow_sequence_loader.py --mode compute --video_path videos/sample.mp4 --output_dir data/optical_flow/processed
```

## API Reference

### DynamicFeatureExtractor

The core class for extracting features from optical flow sequences.

```python
from models.dynamic_feature_extractor.feature_extractor import DynamicFeatureExtractor

# Initialize the extractor
extractor = DynamicFeatureExtractor(config_path="config/model_config/dynamic_model_config.yaml")

# Load preprocessed flow sequence
flow_sequence = np.load("data/optical_flow/sequence.npy")

# Preprocess flow sequence
preprocessed = extractor.preprocess_flow_sequence(flow_sequence)

# Extract features
features = extractor.extract_features(preprocessed)
```

### OpticalFlowSequenceLoader

Utility for loading and preprocessing optical flow sequences.

```python
from scripts.optical_flow_sequence_loader import OpticalFlowSequenceLoader

# Initialize loader
loader = OpticalFlowSequenceLoader(flow_dir="data/optical_flow", sequence_length=16)

# Load a batch of sequences
flow_batch = loader.create_sequence_batch(flow_files, batch_size=8)
```

### FeatureVisualization (Enhanced)

Utilities for visualizing the model's intermediate activations and features with advanced temporal visualization.

```python
from models.dynamic_feature_extractor.visualization import FeatureVisualization

# Initialize visualizer for a specific layer
visualizer = FeatureVisualization(model=extractor.model, layer_name="inception_3a_concat")

# Get activations
activations = visualizer.get_intermediate_activations(preprocessed)

# Basic visualization
visualizer.visualize_layer_activations(activations, frame_idx=0)

# Enhanced temporal visualizations
visualizer.visualize_temporal_feature_evolution(
    activations=activations,
    feature_indices=[0, 10, 20],
    save_path="results/visualizations/temporal_evolution.png"
)

visualizer.create_feature_activation_heatmap(
    activations=activations,
    feature_idx=10,
    save_path="results/visualizations/feature_heatmap.png"
)

visualizer.visualize_3d_feature_activation(
    activations=activations,
    feature_idx=10,
    save_path="results/visualizations/feature_3d.png"
)

visualizer.create_feature_evolution_animation(
    activations=activations,
    feature_idx=10,
    save_path="results/visualizations/feature_evolution.gif"
)

visualizer.create_comparative_temporal_view(
    activations=activations,
    feature_indices=[0, 10, 20],
    flow_sequence=preprocessed,
    save_path="results/visualizations/comparative_view.png"
)
```

### ModelSerializer

Tools for saving, loading, and managing model versions with metadata.

```python
from models.dynamic_feature_extractor.serialization import ModelSerializer

# Save model with metadata
extractor = DynamicFeatureExtractor()
metadata = {
    "author": "Research Team",
    "description": "Dynamic feature extractor for optical flow sequences",
    "training_dataset": "Custom dataset"
}

# Using the extractor's save method
save_path = extractor.save_model_with_serializer(
    base_dir="results/model_checkpoints",
    metadata=metadata
)

# Or using ModelSerializer directly
serializer = ModelSerializer(model=extractor.model, model_name="dynamic_feature_extractor")
save_path = serializer.save_model(
    base_dir="results/model_checkpoints",
    metadata=metadata
)

# List available versions
versions = ModelSerializer.list_available_versions(
    base_dir="results/model_checkpoints", 
    model_name="dynamic_feature_extractor"
)

# Load the latest model version
model, metadata, version = ModelSerializer.load_latest_model(
    base_dir="results/model_checkpoints",
    model_name="dynamic_feature_extractor"
)

# Export model for inference
export_path = ModelSerializer.export_for_inference(
    model=model,
    export_dir="results/model_checkpoints/inference"
)
```

## Visualization Enhancements

### Temporal Feature Visualization

The temporal feature visualization component enhances our ability to understand and analyze how features evolve over time in dynamic feature extraction. This is particularly important for optical flow processing, where temporal patterns are key to understanding motion dynamics.

#### Key Visualization Features

1. **Temporal Evolution Visualization**
   - Displays how feature activations change across frames
   - Supports multiple features in a single visualization
   - Provides clear temporal patterns through line graphs

2. **Feature Activation Heatmaps**
   - Creates spatial heatmaps showing where features activate most strongly
   - Combines temporal and spatial information in a single view
   - Helps identify regions of interest across frames

3. **3D Feature Visualization**
   - Generates 3D visualizations of feature activations over time
   - Allows for intuitive understanding of feature behavior in time-space
   - Supports interactive rotation and exploration (when viewed in interactive mode)

4. **Feature Evolution Animation**
   - Creates animated GIFs showing how feature activations evolve over time
   - Provides dynamic visualization of temporal patterns
   - Particularly useful for presentations and reports

5. **Comparative Temporal View**
   - Compares multiple features side-by-side across time
   - Optionally includes the original optical flow for reference
   - Facilitates comparative analysis of different features

These visualization capabilities are implemented in the `FeatureVisualization` class within the `visualization.py` module, with advanced methods for visualizing temporal patterns.

### Model Serialization

The model serialization component provides a robust system for saving and loading models with metadata, version management, and exporting for inference. This is essential for maintaining model lineage, tracking experiments, and deploying models in production.

#### Key Serialization Features

1. **Model Saving with Metadata**
   - Saves model with comprehensive metadata (author, version, description, etc.)
   - Preserves model architecture, weights, and optimizer state
   - Supports both TensorFlow SavedModel and HDF5 formats

2. **Version Management**
   - Automatically generates version numbers based on timestamps
   - Allows for explicit version specification
   - Provides methods for listing and retrieving available versions

3. **Model Loading**
   - Loads models with associated metadata
   - Supports loading the latest version of a model
   - Handles custom objects and layers

4. **Export for Inference**
   - Exports models in TensorFlow SavedModel format for deployment
   - Optimizes models for inference
   - Supports custom signature definitions

The serialization functionality is implemented in the `ModelSerializer` class within the `serialization.py` module, with the `DynamicFeatureExtractor` class extended with a `save_model_with_serializer` method for advanced model saving capabilities.

## Integration with Existing Pipeline

The dynamic feature extractor integrates with the existing optical flow computation pipeline:

1. Optical flow is computed from videos using `OpticalFlowComputer` in `scripts/preprocessing/optical_flow_computer.py`.
2. The optical flow sequences are loaded and preprocessed using `OpticalFlowSequenceLoader`.
3. The preprocessed sequences are fed into the `DynamicFeatureExtractor` model for feature extraction.
4. The extracted features can be evaluated using `FeatureExtractorEvaluator` and visualized using the visualization tools.

## Conclusion

The dynamic feature extraction system implemented in Phase 3.2 provides a robust foundation for extracting temporal motion features from optical flow sequences. These features capture important dynamic patterns in videos and can be used for a variety of downstream tasks in the Cross-Attention CNN Research project.
