"""
Configuration file for Temporal Cross-Attention Video Action Recognition
Contains paths, constants, and hyperparameters
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
VIDEOS_DIR = RAW_DATA_DIR / "videos"
LABELS_DIR = RAW_DATA_DIR / "labels"

# Source code paths
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = SRC_DIR / "models"
DATA_DIR = SRC_DIR / "data"
TRAINING_DIR = SRC_DIR / "training"
UTILS_DIR = SRC_DIR / "utils"

# Scripts and configs
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Experiments and results
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

# Dataset configuration
NUM_CLASSES = 174
VIDEO_EXT = ".webm"
FRAME_RATE = 12  # FPS
TARGET_FRAMES = 16  # Number of frames to sample per video

# Model dimensions
SEMANTIC_FEATURE_DIM = 1024  # DINOv3 ViT-S+/16
MOTION_FEATURE_DIM = 2048    # I3D features
CROSS_ATTENTION_DIM = 512    # Common embedding dimension
NUM_ATTENTION_HEADS = 8

# Training configuration
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
WARMUP_EPOCHS = 5

# Data loading
NUM_WORKERS = 4
PREFETCH_FACTOR = 2

# File paths
LABELS_JSON = LABELS_DIR / "labels.json"
TRAIN_JSON = LABELS_DIR / "train.json"
VALIDATION_JSON = LABELS_DIR / "validation.json"
TEST_JSON = LABELS_DIR / "test.json"

# Device configuration
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Model save paths
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pth"
LATEST_MODEL_PATH = CHECKPOINTS_DIR / "latest_model.pth"

# Feature cache paths
FEATURE_CACHE_DIR = DATA_ROOT / "features"
SEMANTIC_FEATURES_CACHE = FEATURE_CACHE_DIR / "semantic"
MOTION_FEATURES_CACHE = FEATURE_CACHE_DIR / "motion"
