# Data Pipeline Logging System

This document describes the data pipeline logging system implemented for the Cross-Attention CNN Research project, which tracks dataset statistics, preprocessing metrics, and generates visualizations for the preprocessed data.

## Overview

The data pipeline logging system consists of three main components:

1. **Dataset Statistics Logging**: Collects and logs comprehensive statistics about the dataset, including personality trait distributions, face detection metrics, and optical flow quality metrics.

2. **Preprocessing Metrics Tracking**: Tracks metrics for each step of the preprocessing pipeline, including frame extraction, face detection, and optical flow computation.

3. **Visualization Tools**: Generates visualizations of the preprocessed data, including face detection results, optical flow quality, and side-by-side comparisons.

All components are integrated into a central `DataPipelineLogger` class that extends the project's existing `ResearchLogger` system. The system is designed to handle NumPy data types and provides comprehensive logging and visualization capabilities.

### Key Features

- Comprehensive dataset statistics collection
- Detailed preprocessing metrics tracking
- Rich visualization tools for qualitative assessment
- Integration with TensorBoard and Weights & Biases
- Automatic handling of NumPy data types for JSON serialization
- Test script for verifying functionality with sample data
- Extensive documentation and usage examplesThe system is designed to handle NumPy data types and provides comprehensive logging and visualization capabilities.

### Key Features

- Comprehensive dataset statistics collection
- Detailed preprocessing metrics tracking
- Rich visualization tools for qualitative assessment
- Integration with TensorBoard and Weights & Biases
- Automatic handling of NumPy data types for JSON serialization
- Test script for verifying functionality with sample data
- Extensive documentation and usage examples

## Components

### 1. Dataset Statistics Logging

The dataset statistics logging component collects and logs the following statistics:

- **Personality Trait Distributions**: Mean, standard deviation, min, max, and median for each personality trait (extraversion, neuroticism, agreeableness, conscientiousness, openness)
- **Face Detection Statistics**: Total faces detected, detection rate, per-video face counts
- **Optical Flow Statistics**: Total flows computed, flow magnitude statistics, quality metrics

Run using the `collect_dataset_statistics.py` script:

```bash
python scripts/collect_dataset_statistics.py --annotation-file "data/raw/annotation_validation.pkl" --faces-dir "data/faces" --flow-dir "data/optical_flow" --output-dir "results"
```

### 2. Preprocessing Metrics Tracking

The preprocessing metrics tracking component logs the following metrics for each preprocessing step:

- **Frame Extraction Metrics**: Total videos processed, success rate, frames per video, processing time
- **Face Detection Metrics**: Detection rate, confidence scores, processing time
- **Optical Flow Metrics**: Computation rate, flow quality, processing time

Run using the `track_preprocessing_metrics.py` script:

```bash
python scripts/track_preprocessing_metrics.py --frames-dir "data/processed" --faces-dir "data/faces" --flow-dir "data/optical_flow" --raw-dir "data/raw" --output-dir "results"
```

### 3. Visualization Tools

The visualization tools generate the following visualizations:

- **Face Detection Samples**: Grid of detected faces from selected videos
- **Optical Flow Samples**: Grid of optical flow visualizations
- **Side-by-Side Comparisons**: Faces and their corresponding optical flow
- **Preprocessing Pipeline Status**: Bar charts showing the status of each preprocessing step

Run using the `visualize_data_pipeline.py` script:

```bash
python scripts/visualize_data_pipeline.py --faces-dir "data/faces" --flow-dir "data/optical_flow" --metrics-dir "results/preprocessing_metrics" --output-dir "results/visualizations/data_pipeline"
```

## Integrated Analysis

All components can be run together using the `analyze_data_pipeline.py` script:

```bash
python scripts/analyze_data_pipeline.py --data-dir "data" --annotation-file "data/raw/annotation_validation.pkl" --output-dir "results" --log-config "config/logging_config.yaml"
```

This script will:
1. Collect dataset statistics
2. Track preprocessing metrics
3. Generate visualizations
4. Save all results to the specified output directory

## Output Structure

The data pipeline logging system generates the following outputs:

```
results/
  ├── dataset_statistics.json           # Summary of dataset statistics
  ├── preprocessing_summary.json        # Summary of preprocessing metrics
  ├── dataset_statistics/               # Detailed dataset statistics files
  │   └── test_dataset_statistics_20250515_221832.json  # Example statistics file with timestamp
  ├── preprocessing_metrics/            # Detailed preprocessing metrics files
  │   ├── frame_extraction_metrics_20250515_221832.csv  # Frame extraction metrics
  │   ├── face_detection_metrics_20250515_221832.csv    # Face detection metrics
  │   └── optical_flow_metrics_20250515_221832.csv      # Optical flow metrics
  └── visualizations/
      └── data_pipeline/                # Data pipeline visualizations
          ├── face_detection_results_20250515_221832.png  # Face detection results visualization
          ├── optical_flow_quality_20250515_221832.png    # Optical flow quality visualization
          ├── dataset_samples_20250515_221832.png         # Dataset samples visualization
          ├── side_by_side_*.png        # Side-by-side comparisons
          └── side_by_side_montage.png  # Montage of all comparisons
```

## Integration with Experiment Tracking

The data pipeline logging system integrates with the project's experiment tracking systems:

- **TensorBoard**: All metrics are logged to TensorBoard for visualization
- **Weights & Biases (W&B)**: Metrics and visualizations are logged to W&B if available
- **Custom CSV Logs**: Metrics are saved as CSV files for further analysis

## Using the DataPipelineLogger in Custom Scripts

The `DataPipelineLogger` class can be imported and used in custom scripts:

```python
from utils.data_pipeline_logger import DataPipelineLogger

# Initialize logger
logger = DataPipelineLogger(
    config_path="config/logging_config.yaml",
    experiment_name="custom_analysis",
    output_dir="results"
)

# Log dataset statistics
logger.log_dataset_statistics("custom_dataset", statistics_dict)

# Log preprocessing metrics
logger.log_preprocessing_metrics("custom_process", metrics_dict)

# Create visualizations
logger.visualize_face_detection_results(results_dict)
logger.visualize_optical_flow_quality(flow_stats_dict)
logger.visualize_dataset_samples(face_paths, flow_paths)

# Save summary
logger.save_preprocessing_summary()
```

### Handling Numpy Data Types

The system automatically handles NumPy data types when serializing to JSON using the `convert_numpy_types` utility function:

```python
from utils.json_utils import convert_numpy_types

# Convert numpy types to Python native types before serialization
processed_data = convert_numpy_types(numpy_data)
```

This function recursively converts:
- `numpy.integer` to Python `int`
- `numpy.floating` to Python `float`
- `numpy.ndarray` to Python `list`

## Testing the Data Pipeline Logging System

A test script is provided to verify all components of the data pipeline logging system using a small subset of the data:

```bash
python scripts/test_data_pipeline_logging.py --data-dir "data" --num-videos 5 --output-dir "results_test"
```

This script:
1. Creates a sample dataset with a specified number of videos
2. Runs all data pipeline logging components on the sample dataset
3. Saves all outputs to the specified directory

### Test Script Options:

- `--data-dir`: Path to the original data directory
- `--sample-dir`: Directory for sample dataset (default: "data_sample")
- `--output-dir`: Directory for test results (default: "results_test")
- `--num-videos`: Number of videos to include in sample (default: 5)
- `--no-create-sample`: Skip sample dataset creation (use existing)
- `--annotation-file`: Path to annotation file (optional)

The test script handles annotation file loading with multiple encoding options to ensure compatibility with different pickle file formats.

## Best Practices

1. **Regular Monitoring**: Run the data pipeline analysis regularly during preprocessing to track progress and identify issues.

2. **Incremental Logging**: Log metrics incrementally as each preprocessing step completes, rather than waiting until the end.

3. **Visualization Review**: Review visualizations to qualitatively assess the quality of preprocessing.

4. **Experiment Tracking**: Use the integration with experiment tracking systems to compare preprocessing results across different runs.

5. **Error Analysis**: Use the logged metrics to identify and debug preprocessing errors.

## Conclusion

The data pipeline logging system provides comprehensive tracking and visualization of the preprocessing pipeline, enabling effective monitoring, debugging, and documentation of the data preparation process for the Cross-Attention CNN Research project.

This system helps ensure data quality and reliability by:

1. Providing clear insights into dataset composition and characteristics
2. Tracking the performance of each preprocessing step
3. Generating visualizations for qualitative assessment
4. Maintaining comprehensive logs for reproducibility
5. Handling different data types including NumPy arrays

The system has been thoroughly tested with a sample subset of the data and is ready for integration into the main research workflow.

## Last Updated

May 15, 2025
