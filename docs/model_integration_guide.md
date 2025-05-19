# Integration Guide: Cross-Attention Personality Model

This document provides guidance for integrating the cross-attention-based personality prediction model into the training pipeline for Phase 4.

## Components Overview

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

## Integration Steps

### 1. Import Requirements

```python
from src.models.personality_model import CompletePersonalityModel
```

### 2. Initialize the Model

```python
model = CompletePersonalityModel(
    static_dim=512,     # Dimension of static features (from face CNN)
    dynamic_dim=256,    # Dimension of dynamic features (from optical flow CNN)
    fusion_dim=128,     # Dimension for fusion representation
    num_heads=4,        # Number of attention heads
    dropout_rate=0.3    # Dropout rate for regularization
)
```

### 3. Feature Extraction Pipeline

Before passing data to the model, ensure you have:
- Static features extracted from faces using the static feature extractor
- Dynamic features extracted from optical flow using the dynamic feature extractor

Example:
```python
# Get static features from face images
static_features = static_feature_extractor(face_images)

# Get dynamic features from optical flow sequences
dynamic_features = dynamic_feature_extractor(optical_flow_sequences)
```

### 4. Make Predictions

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

### 5. Training Setup

For training, use the following:

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

### 6. Visualization During Training/Evaluation

To visualize attention weights during training or evaluation:

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

## Potential Integration Issues

### Dimension Mismatch
If your feature extractors output dimensions different from the expected ones (512 for static, 256 for dynamic):
1. Adjust the model initialization parameters, or
2. Add projection layers to match dimensions before passing to the model

### Training Stability
- Start with a low learning rate (0.0001-0.001)
- Consider gradient clipping (`optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)`)
- Use early stopping to prevent overfitting

### Memory Constraints
- For large batch sizes, reduce the fusion dimension or number of attention heads
- Use mixed precision training for GPU efficiency

## Testing the Integration

Before starting full training, validate the integration with:

```python
# Run the test script to ensure components work together
if (Test-Path .\env\Scripts\activate.ps1) {
    .\env\Scripts\activate.ps1
    python .\scripts\test_personality_model.py
}
```

---
*Last updated: 2025-05-20*
