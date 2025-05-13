"""
Test script to verify the logger functionality.
This script creates a logger instance and tests various logging functions.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the project root to sys.path to import the logger
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from utils.logger import ResearchLogger


def test_basic_logging():
    """Test basic logging functionality"""
    print("Testing basic logging...")
    
    # Create logger with configuration file
    config_path = os.path.join(project_root, "config", "logging_config.yaml")
    logger = ResearchLogger(config_path=config_path, experiment_name="logger_test")
    
    # Log some basic messages
    logger.logger.debug("This is a debug message")
    logger.logger.info("This is an info message")
    logger.logger.warning("This is a warning message")
    logger.logger.error("This is an error message")
    
    # Log system information
    logger.log_system_info()
    
    # Log metrics
    metrics = {
        "loss": 0.342,
        "accuracy": 0.89,
        "learning_rate": 0.001
    }
    logger.log_metrics(metrics, step=1)
      # Log hyperparameters
    hparams = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "epochs": 100
    }
    logger.log_hyperparameters(hparams, step=1)
      # Log dataset statistics
    dataset_info = {
        "train_size": 6000,
        "val_size": 2000,
        "test_size": 2000,
        "num_classes": 5,
        "video_length": 15
    }
    logger.log_dataset_stats(dataset_info)
    
    print("Attention map logging test...")
    # Test attention map logging with a dummy attention map
    try:
        attention_map = np.random.rand(224, 224)  # Random attention weights
        original_image = np.random.rand(224, 224, 3)  # Random RGB image
        logger.log_attention_map(
            name="test_attention_map", 
            attention_map=attention_map,
            original_image=original_image,
            step=1
        )
        print("Attention map logging successful")
    except Exception as e:
        print(f"Attention map logging failed: {str(e)}")
    # Close the logger
    logger.close()
    print("Basic logging test completed successfully")


if __name__ == "__main__":
    test_basic_logging()
