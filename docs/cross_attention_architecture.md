# Cross-Attention Architecture Documentation

## Overview
The Cross-Attention mechanism fuses static (face) and dynamic (motion) features using multi-head attention, residual connections, and normalization. This enables the model to learn important interactions between modalities for personality trait prediction.

## Architecture
- **Input:**
  - Static features: [batch_size, static_dim]
  - Dynamic features: [batch_size, dynamic_dim]
- **Projection:** Dense layers to a common fusion_dim
- **Multi-Head Attention:** Computes attention between static and dynamic features
- **Residual Connection:** Adds static features to attention output
- **Layer Normalization:** Stabilizes output
- **Output:** [batch_size, fusion_dim]

## Implementation
- Implemented in `src/models/cross_attention.py` as `CrossAttention` (TensorFlow/Keras)
- Tested with random data in `scripts/test_cross_attention.py` (output shape and correctness verified)

## Next Step: Attention Visualization Tools
- Visualize attention weights from the MultiHeadAttention layer
- Provide tools to:
  - Extract and plot attention maps for a given input batch
  - Save attention visualizations for analysis
  - Integrate with model evaluation for interpretability

---
*Last updated: 2025-05-20*


# Model Visualization and Comparison Tools

This document provides details about the visualization and comparison tools implemented for the Cross-Attention CNN model architecture.

## Table of Contents
- [Feature Map Visualization](#feature-map-visualization)
- [Model Architecture Comparison](#model-architecture-comparison)

## Feature Map Visualization

### Purpose
The feature map visualization tool extracts and displays activations from intermediate layers of neural networks, providing insights into:
- What patterns each filter detects
- How representations evolve through the network
- Which features are most activated for different inputs

### Implementation
- Located in: `scripts/visualize_feature_maps.py`
- Key Functions:
  - `visualize_feature_maps(model, layer_name, input_data, save_dir, max_maps)`: Extracts and plots feature maps from a specific layer

### Usage
```powershell
if (Test-Path .\env\Scripts\activate.ps1) {
    .\env\Scripts\activate.ps1
    python .\scripts\visualize_feature_maps.py
}
```

### Extension Possibilities
- Batch visualization across multiple inputs
- Animation of feature evolution during training
- Correlation analysis between feature activations and model predictions

## Model Architecture Comparison

### Purpose
The model architecture comparison tool provides quantitative and qualitative comparisons between different model architectures, enabling:
- Parameter efficiency analysis
- Layer-by-layer comparison
- Model complexity assessment

### Implementation
- Located in: `scripts/compare_architectures.py`
- Key Functions:
  - `get_model_summary(model)`: Returns a string representation of model structure
  - `get_param_count(model)`: Returns the total parameter count
  - `compare_models(model_a, model_b, name_a, name_b, save_path)`: Compares two models and saves results

### Usage
```powershell
if (Test-Path .\env\Scripts\activate.ps1) {
    .\env\Scripts\activate.ps1
    python .\scripts\compare_architectures.py
}
```

### Extension Possibilities
- Performance benchmarking (inference time, memory usage)
- Visual architecture graph comparison
- FLOPS calculation for computational efficiency analysis

## Integration with Cross-Attention Mechanism

These tools are particularly useful for analyzing the cross-attention mechanism:

1. **Feature Map Visualization**: Reveals what patterns the attention heads focus on when processing static and dynamic features.

2. **Architecture Comparison**: Helps evaluate the parameter and computational efficiency of different attention configurations (number of heads, fusion dimension).

---
*Last updated: 2025-05-20*
