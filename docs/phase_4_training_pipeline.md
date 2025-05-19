# Phase 4: Training Pipeline and Metrics

## Phase Overview

Phase 4 of the Cross-Attention CNN Research project focuses on developing a comprehensive training pipeline with advanced metrics and visualizations for personality trait prediction. This phase is divided into:

- **Phase 4.1**: Basic Training Pipeline
- **Phase 4.2**: Loss Functions and Metrics
- **Phase 4.3**: Training System Logging

## Phase 4.1: Basic Training Pipeline

Phase 4.1 established the foundational training pipeline that:

- Loads pre-extracted static and dynamic features along with OCEAN labels
- Implements data loading and batch generation
- Provides a training loop with validation
- Includes learning rate scheduling, early stopping, and model checkpointing
- Freezes feature extractors while training cross-attention and dense layers

## Phase 4.2: Loss Functions and Metrics

Phase 4.2 enhances the training pipeline with:

- **R-squared metric**: Implementation of the coefficient of determination (R²) as a custom TensorFlow metric
- **Per-trait metrics**: Tracking of MAE, MSE, and R² for each OCEAN personality trait individually
- **Comprehensive visualizations**: Tools for analyzing training progress and model performance
- **Integration with experiment tracking**: TensorBoard and W&B support for detailed monitoring

## Usage Instructions

### Training the Model

```powershell
# Activate the environment
.\env\Scripts\activate.ps1

# Train the model using sample data
python scripts/train_personality_model.py --static_features data_sample/static_features.npy --dynamic_features data_sample/dynamic_features.npy --labels data_sample/labels.npy --val_split 0.2 --batch_size 16 --epochs 50 --output results/personality_model_trained.h5
```

### Visualizing Training Metrics

```powershell
# Visualize training history
python scripts/visualize_training_metrics.py --history results/training_history.npy --output results/training_visualizations
```

### Analyzing Model Predictions

```powershell
# Analyze model predictions in detail
python scripts/analyze_model_predictions.py --model results/personality_model_trained.h5 --static_features data_sample/static_features.npy --dynamic_features data_sample/dynamic_features.npy --labels data_sample/labels.npy --output results/prediction_analysis
```

### Viewing TensorBoard Visualizations

```powershell
# Launch TensorBoard to view training metrics
tensorboard --logdir results/tensorboard_logs
```

## Key Files

```
utils/
  metrics.py                    # R-squared and per-trait metric implementations

scripts/
  train_personality_model.py    # Main training pipeline
  visualize_training_metrics.py # Training visualization tools
  analyze_model_predictions.py  # Detailed prediction analysis

docs/
  phase_4.2_loss_functions_and_metrics.md # Documentation for Phase 4.2
```

## Directory Structure

After training, the following directory structure will be created:

```
results/
  personality_model_trained.h5  # Trained model weights
  checkpoint_weights.h5         # Best model weights from training
  training_history.npy          # Saved training metrics
  tensorboard_logs/             # TensorBoard log files
  training_visualizations/      # Generated visualizations
    loss.png                    # Loss curves
    mae.png                     # MAE metric plot
    mse.png                     # MSE metric plot
    r_squared.png               # R² metric plot
    per_trait_r_squared.png     # R² by personality trait
    r2_comparison.png           # Comparison of R² across traits
    metric_correlation.png      # Correlation between metrics
  prediction_analysis/          # Model prediction analysis
    metrics.json                # Detailed evaluation metrics
    openness_scatter.png        # Predicted vs true for Openness
    conscientiousness_scatter.png # Predicted vs true for Conscientiousness
    ...
```

## Next Steps

- **Phase 4.3**: Implement detailed training system logging and visualization
- **Future Work**: Explore additional loss functions and multi-task learning approaches
