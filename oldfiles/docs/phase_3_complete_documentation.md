# Phase 3: Cross-Attention CNN Architecture Implementation

## Table of Contents
1. [Introduction](#introduction)
2. [Phase 3.1: Static Feature Extractor](#phase-31-static-feature-extractor)
   - [Overview](#31-overview)
   - [Implementation Components](#implementation-components)
   - [Model Architecture](#31-model-architecture)
   - [Results and Findings](#results-and-findings)
3. [Phase 3.2: Dynamic Feature Extractor](#phase-32-dynamic-feature-extractor)
   - [Overview](#32-overview)
   - [Implementation Components](#32-implementation-components)
   - [Model Architecture](#32-model-architecture)
   - [Visualization Enhancements](#visualization-enhancements)
   - [Integration with Existing Pipeline](#integration-with-existing-pipeline)
4. [Phase 3.3: Cross-Attention Mechanism](#phase-33-cross-attention-mechanism)
   - [Overview](#33-overview)
   - [Cross-Attention Architecture](#cross-attention-architecture)
   - [Feature Fusion Module](#feature-fusion-module)
   - [Personality Prediction Head](#personality-prediction-head)
   - [Complete Model Architecture](#complete-model-architecture)
5. [Model Visualization Tools](#model-visualization-tools)
   - [Feature Map Visualization](#feature-map-visualization)
   - [Attention Visualization](#attention-visualization)
   - [Model Architecture Comparison](#model-architecture-comparison)
6. [Integration Guide](#integration-guide)
   - [Components Overview](#components-overview)
   - [Integration Steps](#integration-steps)
   - [Potential Integration Issues](#potential-integration-issues)
7. [Next Steps](#next-steps)

## Introduction

Phase 3 of the Cross-Attention CNN Research project focused on implementing the core architecture for personality trait prediction from video data. This phase was divided into three sub-phases:

1. **Phase 3.1**: Implementation of the static feature extractor (2D CNN for face images)
2. **Phase 3.2**: Implementation of the dynamic feature extractor (3D CNN for optical flow)
3. **Phase 3.3**: Implementation of the cross-attention mechanism, feature fusion, and prediction head

This comprehensive documentation merges the documentation created for each sub-phase to provide a complete reference for the Phase 3 implementation.

## Phase 3.1: Static Feature Extractor

### 3.1 Overview

Phase 3.1 focused on implementing a 2D CNN architecture using TensorFlow with ResNet-50 as the pre-trained model backbone for static feature extraction. This component extracts spatial features from individual face frames, which will later be combined with dynamic features through the cross-attention mechanism.

### Implementation Components

#### 1. Static Feature Extractor

The static feature extractor is implemented in `models/static_feature_extractor/feature_extractor.py`. It uses TensorFlow's ResNet-50 model as the backbone architecture and provides the following functionality:

- Loading and configuring a pre-trained ResNet-50 model
- Controlling which layers are frozen during training (first 4 layers are frozen by default)
- Extracting features from face images into a 2048-dimensional feature vector
- Preprocessing images according to the requirements of the pre-trained model
- Saving and loading model weights

The implementation follows the configuration specified in `config/model_config/model_config.yaml`, which defines parameters such as the base model, whether pre-training is used, which layers to freeze, and the output feature dimensions.

#### 2. Feature Visualization

The feature visualization module in `models/static_feature_extractor/visualization.py` provides tools for understanding and visualizing the features extracted by the CNN, including:

- Intermediate layer activation visualization
- Class activation map (CAM) generation
- Filter visualization through feature maximization
- Overlay of attention maps on original images

#### 3. Model Evaluation

The evaluation module in `models/static_feature_extractor/evaluation.py` provides methods for assessing the quality and discriminability of the extracted features, with metrics including:

- Within-subject feature consistency (average cosine similarity of 0.961)
- Between-subject discriminability (average cosine distance of 0.310)
- PCA visualization of feature space
- Layer activation analysis

#### 4. Testing Scripts

The project includes comprehensive testing scripts:
- `scripts/test_feature_extractor.py`: Tests the feature extractor on sample face images
- `scripts/visualize_single_image.py`: Visualizes CNN feature activations on individual images
- `scripts/implement_static_feature_extractor.py`: Main implementation script with various execution modes

### 3.1 Model Architecture

The static feature extractor is based on ResNet-50, which is a deep convolutional neural network with 50 layers organized in residual blocks. Key architectural features include:

1. **Pre-trained ResNet-50 base**: Uses weights pre-trained on ImageNet
2. **Layer freezing**: First four convolutional layers are frozen
3. **Global average pooling**: Applied to reduce spatial dimensions
4. **Feature output**: 2048-dimensional feature vector

## Phase 3.2: Dynamic Feature Extractor

### 3.2 Overview

Phase 3.2 implemented a dynamic feature extraction system that processes optical flow data using a 3D CNN architecture. This system extracts temporal patterns from motion information captured in optical flow sequences, based on the I3D (Inflated 3D ConvNet) architecture.

### 3.2 Implementation Components

#### 1. Dynamic Feature Extractor

The `DynamicFeatureExtractor` class in `models/dynamic_feature_extractor/feature_extractor.py` implements:
- 3D CNN model based on I3D architecture
- Feature extraction from optical flow sequences
- Preprocessing of optical flow data
- Model saving and loading with versioning

#### 2. Optical Flow Sequence Loader

The `OpticalFlowSequenceLoader` class in `scripts/optical_flow_sequence_loader.py` provides:
- Loading, preprocessing, and batching optical flow sequences
- Sequence length adjustment
- Integration with optical flow computation

#### 3. Visualization Tools

Enhanced visualization tools in `models/dynamic_feature_extractor/visualization.py` include:
- Layer activation visualization for 3D data
- Temporal feature evolution visualization
- Feature activation heatmaps
- 3D feature visualization
- Feature evolution animation
- Comparative temporal views

#### 4. Model Serialization

The `ModelSerializer` class in `models/dynamic_feature_extractor/serialization.py` provides:
- Model saving with metadata
- Version management
- Convenient loading interface
- Export for inference

#### 5. Evaluation Metrics

Evaluation tools in `models/dynamic_feature_extractor/evaluation.py` include:
- Feature consistency evaluation
- Feature discriminability analysis
- Feature dimension distribution analysis

### 3.2 Model Architecture

The dynamic feature extractor uses an I3D architecture with:
- 3D convolutional layers for spatial-temporal feature extraction
- Inception blocks for multi-scale feature learning
- 3D pooling operations for temporal dimensionality reduction
- Dropout for regularization
- Dense layers for final feature output (1024-dimensional feature vector)

The model processes optical flow sequences with shape `[batch_size, frames, height, width, 2]` where the last dimension represents the u and v components of optical flow.

### Visualization Enhancements

#### Temporal Feature Visualization

The temporal feature visualization component enhances our ability to understand and analyze how features evolve over time in dynamic feature extraction. Key visualization features include:

1. **Temporal Evolution Visualization**
   - Displays how feature activations change across frames
   - Supports multiple features in a single visualization

2. **Feature Activation Heatmaps**
   - Creates spatial heatmaps showing where features activate most strongly
   - Combines temporal and spatial information in a single view

3. **3D Feature Visualization**
   - Generates 3D visualizations of feature activations over time
   - Allows for intuitive understanding of feature behavior in time-space

4. **Feature Evolution Animation**
   - Creates animated GIFs showing how feature activations evolve over time
   - Provides dynamic visualization of temporal patterns

5. **Comparative Temporal View**
   - Compares multiple features side-by-side across time
   - Optionally includes the original optical flow for reference

### Integration with Existing Pipeline

The dynamic feature extractor integrates with the existing optical flow computation pipeline:

1. Optical flow is computed from videos using `OpticalFlowComputer`
2. The optical flow sequences are loaded and preprocessed using `OpticalFlowSequenceLoader`
3. The preprocessed sequences are fed into the `DynamicFeatureExtractor` for feature extraction
4. The extracted features can be evaluated and visualized using the provided tools

## Phase 3.3: Cross-Attention Mechanism

### 3.3 Overview

Phase 3.3 focused on developing the cross-attention mechanism to fuse static facial features with dynamic motion features for personality trait prediction. This phase also implemented the feature fusion module and the prediction head for the final personality trait outputs.

### Cross-Attention Architecture

The cross-attention mechanism is implemented in `src/models/cross_attention.py` as the `CrossAttention` class. Key features include:

- **Multi-head attention** for feature interaction between modalities
- **Projection layers** to handle different input dimensions
- **Residual connections** and **layer normalization** for training stability

The architecture processes:
- **Input:**
  - Static features: [batch_size, static_dim]
  - Dynamic features: [batch_size, dynamic_dim]
- **Projection:** Dense layers to a common fusion_dim
- **Multi-Head Attention:** Computes attention between static and dynamic features
- **Residual Connection:** Adds static features to attention output
- **Layer Normalization:** Stabilizes output
- **Output:** [batch_size, fusion_dim]

### Feature Fusion Module

The `FeatureFusionModule` (implemented in `src/models/personality_model.py`) takes the outputs from the cross-attention mechanism and refines them through:

- **Batch normalization** to stabilize the feature representations
- **Dropout regularization** (default rate: 0.3) to prevent overfitting
- Maintaining the fusion dimension throughout (default: 128)

Architecture:
```
Cross-Attention Output → Batch Normalization → Dropout → Processed Fusion Features
```

### Personality Prediction Head

The `PersonalityPredictionHead` (implemented in `src/models/personality_model.py`) transforms the fused features into personality trait predictions:

- **Hidden layer** with ReLU activation provides non-linearity (default size: 64)
- **Output layer** with 5 units and linear activation for OCEAN trait score regression
- No activation is applied to outputs as these are regression values

Architecture:
```
Fusion Features → Dense Layer (64, ReLU) → Output Layer (5, Linear) → OCEAN Predictions
```

### Complete Model Architecture

The `CompletePersonalityModel` class integrates all components into a single, end-to-end trainable model:

```
     Static Features     Dynamic Features
            │                   │
            ▼                   ▼
      ┌─────────────────────────────┐
      │      Cross-Attention        │
      └─────────────────────────────┘
                    │
                    ▼
      ┌─────────────────────────────┐
      │     Feature Fusion Module   │
      └─────────────────────────────┘
                    │
                    ▼
      ┌─────────────────────────────┐
      │   Personality Prediction    │
      │             Head            │
      └─────────────────────────────┘
                    │
                    ▼
      ┌─────────────────────────────┐
      │         O  C  E  A  N       │
      └─────────────────────────────┘
```

Total model parameters: 371,717 (trainable: 371,461)

## Model Visualization Tools

The project includes several tools for model visualization and analysis.

### Feature Map Visualization

The feature map visualization tool (`scripts/visualize_feature_maps.py`) extracts and displays activations from intermediate layers of neural networks, providing insights into:
- What patterns each filter detects
- How representations evolve through the network
- Which features are most activated for different inputs

### Attention Visualization

Attention visualization tools (`scripts/visualize_attention.py`) provide:
- Functions to extract attention weights from the model
- Visualization of attention maps for interpretability
- Support for saving visualizations to disk
- Attention score analysis for feature relationships

### Model Architecture Comparison

The model architecture comparison tool (`scripts/compare_architectures.py`) provides:
- Parameter efficiency analysis
- Layer-by-layer comparison
- Model complexity assessment
- Visual architecture graph comparison

## Integration Guide

### Components Overview

The personality prediction system consists of three main components:

1. **Cross-Attention Mechanism** (`src/models/cross_attention.py`)
   - Enables information exchange between static and dynamic features
   - Implemented with multi-head attention, handling different input dimensions

2. **Feature Fusion Module** (`src/models/personality_model.py`)
   - Normalizes and regularizes the fused features
   - Prepares fusion outputs for prediction

3. **Personality Prediction Head** (`src/models/personality_model.py`)
   - Transforms fused features into OCEAN trait predictions
   - Uses a simple feed-forward architecture

All components are integrated in the `CompletePersonalityModel` class.

### Integration Steps

1. **Initialize the Model**

```python
model = CompletePersonalityModel(
    static_dim=512,     # Dimension of static features (from face CNN)
    dynamic_dim=256,    # Dimension of dynamic features (from optical flow CNN)
    fusion_dim=128,     # Dimension for fusion representation
    num_heads=4,        # Number of attention heads
    dropout_rate=0.3    # Dropout rate for regularization
)
```

2. **Feature Extraction Pipeline**

Before passing data to the model, ensure you have:
- Static features extracted from faces using the static feature extractor
- Dynamic features extracted from optical flow using the dynamic feature extractor

```python
# Get static features from face images
static_features = static_feature_extractor(face_images)

# Get dynamic features from optical flow sequences
dynamic_features = dynamic_feature_extractor(optical_flow_sequences)
```

3. **Make Predictions**

```python
# Generate predictions for personality traits
predictions = model((static_features, dynamic_features), training=True)
```

The output shape is `[batch_size, 5]` representing scores for the OCEAN traits:
- predictions[:, 0]: Openness
- predictions[:, 1]: Conscientiousness
- predictions[:, 2]: Extraversion
- predictions[:, 3]: Agreeableness
- predictions[:, 4]: Neuroticism

4. **Training Setup**

```python
# Loss function (MSE for regression)
loss_fn = tf.keras.losses.MeanSquaredError()

# Optimizer (Adam with learning rate scheduling recommended)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile the model
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=['mae', 'mse']  # Mean Absolute Error and Mean Squared Error
)
```

5. **Visualization During Training/Evaluation**

```python
from scripts.visualize_attention import get_attention_weights, plot_attention_map

# Extract and visualize attention weights
attention_weights = get_attention_weights(
    model.cross_attention,
    static_features,
    dynamic_features
)
plot_attention_map(attention_weights, save_path="results/visualizations/attention_map.png")
```

### Potential Integration Issues

#### Dimension Mismatch
If your feature extractors output dimensions different from the expected ones:
1. Adjust the model initialization parameters, or
2. Add projection layers to match dimensions before passing to the model

#### Training Stability
- Start with a low learning rate (0.0001-0.001)
- Consider gradient clipping
- Use early stopping to prevent overfitting

#### Memory Constraints
- For large batch sizes, reduce the fusion dimension or number of attention heads
- Use mixed precision training for GPU efficiency

## Next Steps

With Phase 3 complete, the project is ready to move to Phase 4: Training System Development, which includes:
1. Implementing the data loader and batch generator
2. Creating the training loop with validation
3. Building a learning rate scheduler
4. Developing early stopping mechanism
5. Implementing model checkpoint system

---
*Last updated: 2025-05-20*
