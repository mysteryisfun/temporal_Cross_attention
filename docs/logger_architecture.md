# Logger System Architecture Document

## Overview

This document describes the comprehensive logging system designed for the Cross-Attention CNN Research project. The logging framework is responsible for tracking experiments, metrics, visualizations, and system diagnostics throughout the research process.

## Key Components

### 1. Base Logger

The base logging system utilizes Python's built-in logging module extended with custom functionality for research experiments. It provides:

- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Multiple output handlers (console, file)
- Structured logging format with timestamps and module info

### 2. Experiment Tracking

The experiment tracking system integrates with:

- **TensorBoard**: For metric visualization, model graphs, and image logging
- **Weights & Biases (W&B)**: For experiment comparison, artifact tracking, and collaboration
- **Custom database**: For experiment metadata and results storage

### 3. Metrics Logging

The metrics logging component handles:

- Training metrics (loss, accuracy, learning rate)
- Validation metrics (per-epoch evaluation)
- Test metrics (final model evaluation)
- Custom research metrics (per-trait performance)

### 4. Visualization Logging

Visualization capabilities include:

- Attention map visualization
- Feature map logging
- Training/validation curves
- Confusion matrices
- Prediction vs. ground truth comparisons

### 5. System Logging

System monitoring includes:

- GPU utilization tracking
- Memory usage monitoring
- Computation time logging
- I/O operation tracking

## Configuration Schema

The logging system is configured through YAML files with the following structure:

```yaml
# General logging settings
general:
  log_level: "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  log_to_file: true
  log_file_path: "../logs/research.log"
  
# Experiment tracking
experiment_tracking:
  # Weights & Biases configuration
  wandb:
    enabled: true
    project_name: "personality-trait-prediction"
    entity: null  # Username or team name
    log_freq: 10  # Log every N batches
    
  # TensorBoard configuration
  tensorboard:
    enabled: true
    log_dir: "../logs/tensorboard"
    
  # Custom experiment database
  custom_db:
    enabled: true
    db_path: "../logs/experiment_tracker.csv"
```

## Implementation

The logger implementation follows a modular approach with the following classes:

1. `ResearchLogger`: Core logging functionality with experiment tracking
2. `MetricsLogger`: Specialized logger for performance metrics
3. `VisualizationLogger`: Handles visualization logging
4. `SystemMonitor`: Tracks hardware resource usage

## Usage Examples

### Basic Logger Initialization

```python
from utils.logger import ResearchLogger

# Initialize with config file
logger = ResearchLogger(config_path="config/logging_config.yaml", 
                        experiment_name="cross_attention_experiment_1")
```

### Metrics Logging

```python
# Log training metrics
metrics = {
    "loss": 0.342,
    "accuracy": 0.89,
    "learning_rate": 0.001
}
logger.log_metrics(metrics, step=epoch)
```

### Model Architecture Logging

```python
# Log model architecture
logger.log_model_architecture(model)
```

### Visualization Logging

```python
# Log attention map
logger.log_attention_map(
    name="facial_attention", 
    attention_map=attention_weights,
    original_image=input_image,
    step=epoch
)
```

## Integration Points

The logging system integrates with:

1. **Training Pipeline**: For metrics and visualization logging
2. **Data Pipeline**: For dataset statistics and preprocessing metrics
3. **Evaluation System**: For performance metrics and error analysis
4. **Model Architecture**: For parameter counts and architecture visualization

## Error Handling

The logging system includes robust error handling to:

- Gracefully handle logging failures
- Prevent experiment crashes due to logging issues
- Provide fallback mechanisms when external services are unavailable
- Archive logs for debugging purposes

## Future Extensions

Planned enhancements include:

1. Distributed logging for multi-GPU training
2. Automated report generation from logged metrics
3. Real-time alerts for experiment issues
4. Comparative visualization across multiple experiments
5. Integration with additional experiment tracking platforms
