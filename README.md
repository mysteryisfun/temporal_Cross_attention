# Cross-Attention CNN for Personality Trait Prediction

This repository contains the implementation of a novel cross-attention convolutional neural network architecture for predicting personality traits from video data, using the ChaLearn First Impressions V2 dataset.

## Project Overview

This research project aims to develop and evaluate a dual-stream architecture that combines static facial features and dynamic temporal information using a cross-attention mechanism to predict the Big Five personality traits from short video clips.

### Architecture

The model consists of:
1. **Static Feature Extraction**: 2D CNN for processing facial frames
2. **Dynamic Feature Extraction**: 3D CNN for processing optical flow sequences
3. **Cross-Attention Mechanism**: Multi-head attention for feature fusion
4. **Prediction Head**: MLP for personality trait prediction

## Repository Structure

```
├── config/                    # Configuration files
│   ├── data_config/           # Dataset configuration
│   ├── model_config/          # Model architecture configuration
│   └── logging_config.yaml    # Logging configuration
├── data/                      # Data storage
│   ├── raw/                   # Raw dataset
│   ├── processed/             # Processed data
│   ├── faces/                 # Extracted face images
│   └── optical_flow/          # Computed optical flow
├── docs/                      # Documentation
├── logs/                      # Experiment logs
├── models/                    # Model implementations
│   ├── static_cnn/            # Static feature extraction
│   ├── dynamic_cnn/           # Dynamic feature extraction
│   ├── cross_attention/       # Cross-attention mechanism
│   └── fusion/                # Feature fusion components
├── notebooks/                 # Jupyter notebooks for experiments
├── scripts/                   # Utility scripts
│   ├── data_acquisition/      # Dataset download scripts
│   ├── preprocessing/         # Data preprocessing scripts
│   └── training/              # Training scripts
├── results/                   # Experimental results
│   ├── ablation_studies/      # Ablation study results
│   ├── visualizations/        # Attention visualizations
│   └── model_checkpoints/     # Saved model checkpoints
└── utils/                     # Utility functions
```

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 1.10+
- CUDA-capable GPU with 8GB+ VRAM

### Installation

1. Clone this repository:
```bash
git clone https://github.com/username/cross-attention-cnn.git
cd cross-attention-cnn
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate cross_attention_cnn
```

Or install with pip:
```bash
pip install -r requirements.txt
```

### Dataset

This project uses the ChaLearn First Impressions V2 dataset. To request access:
1. Visit [ChaLearn First Impressions V2](https://chalearnlap.cvc.uab.cat/dataset/24/description/)
2. Follow the instructions to request and download the dataset
3. Place the downloaded dataset in the `data/raw` directory

### Configuration

All experiment settings are stored in YAML configuration files in the `config` directory. Modify these files to adjust:
- Dataset parameters
- Model architecture
- Training hyperparameters
- Logging settings

### Running Experiments

1. Preprocess the dataset:
```bash
python scripts/preprocessing/preprocess_dataset.py
```

2. Train the model:
```bash
python scripts/training/train_model.py
```

3. Evaluate and visualize results:
```bash
python scripts/training/evaluate_model.py
```

## Documentation

For detailed information about the project:
- See `docs/project_scope.md` for project scope and requirements
- See `docs/development_roadmap.md` for the development timeline
- Check the `notebooks` directory for experimental analyses

## License

This project is research code provided for educational purposes. The ChaLearn First Impressions V2 dataset has its own license for research purposes only.

## Citation

If you use this code in your research, please cite:

```
@article{crossattentioncnn2023,
  title={Cross-Attention CNN for Personality Trait Prediction from Videos},
  author={Author, A.},
  journal={},
  year={2023}
}
```

### Latest Updates (Phase 4.3)

We've just completed Phase 4.3, which enhances the training pipeline with advanced training system logging:

- **Comprehensive Training Callbacks**: Custom Keras callbacks for detailed metric tracking, time monitoring, and learning rate tracking
- **Advanced Visualization Tools**: Scripts for generating publication-quality visualizations of training dynamics and model behavior
- **Attention Mechanism Visualization**: Tools for visualizing and analyzing the multi-head attention patterns
- **Training Time Analysis**: Automatic analysis of training performance with optimization recommendations

#### Quick Start with Phase 4.3 Components

```powershell
# Run the test script to verify all Phase 4.3 components
if (Test-Path .\env\Scripts\activate.ps1) {
    .\env\Scripts\activate.ps1
    .\test_phase_4.3.ps1
}

# Train a model with the enhanced logging system
if (Test-Path .\env\Scripts\activate.ps1) {
    .\env\Scripts\activate.ps1
    python scripts/train_personality_model.py --static_features data/static_features.npy --dynamic_features data/dynamic_features.npy --labels data/labels.npy --val_split 0.2 --batch_size 32 --epochs 50 --output results/personality_model_trained.h5
}

# Generate visualizations from training logs
if (Test-Path .\env\Scripts\activate.ps1) {
    .\env\Scripts\activate.ps1
    .\visualize_training_results.ps1
}
```
