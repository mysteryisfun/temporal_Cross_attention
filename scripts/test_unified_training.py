"""
Test script for the unified training system (Phase 4.3)

This script demonstrates how to use the unified training system module
to train a personality model with advanced logging and visualization.

Usage:
    # If virtual environment exists
    if (Test-Path .\env\Scripts\activate.ps1) {
        .\env\Scripts\activate.ps1
        python scripts/test_unified_training.py --static_features data_sample/static_features.npy --dynamic_features data_sample/dynamic_features.npy --labels data_sample/labels.npy --output results/test_phase_4.3/unified_training/model.h5
    }
"""

import os
import argparse
import tensorflow as tf
import numpy as np
from src.training import TrainingSystem, train_personality_model
from src.models.personality_model import CompletePersonalityModel
from utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Test Unified Training System")
    parser.add_argument('--static_features', type=str, required=True, help='Path to static features .npy file')
    parser.add_argument('--dynamic_features', type=str, required=True, help='Path to dynamic features .npy file')
    parser.add_argument('--labels', type=str, required=True, help='Path to OCEAN labels .npy file')
    parser.add_argument('--output', type=str, default='results/test_phase_4.3/unified_training/model.h5', help='Output model file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (use small value for testing)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    return parser.parse_args()


def test_convenience_function(args):
    """
    Test the convenience function approach
    """
    print("Testing unified training system using convenience function...")
    print(f"Static features: {args.static_features}")
    print(f"Dynamic features: {args.dynamic_features}")
    print(f"Labels: {args.labels}")
    print(f"Output path: {args.output}")
    
    # Create base output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Train model using the convenience function
    model, history = train_personality_model(
        static_features_path=args.static_features,
        dynamic_features_path=args.dynamic_features,
        labels_path=args.labels,
        output_path=args.output,
        val_split=0.2,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    print(f"Model training completed. Weights saved to {args.output}")
    print(f"Training history has {len(history.history['loss'])} epochs")
    return model, history


def test_direct_system_usage(args):
    """
    Test direct usage of the TrainingSystem class
    """
    print("Testing unified training system using direct interface...")
    
    # Create a custom output directory
    output_dir = os.path.join(os.path.dirname(args.output), 'custom_training')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a logger
    logger = get_logger(experiment_name="test_direct_system")
    
    # Initialize the training system
    training_system = TrainingSystem(
        static_features_path=args.static_features,
        dynamic_features_path=args.dynamic_features,
        labels_path=args.labels,
        val_split=0.2,
        output_dir=output_dir,
        experiment_name="test_direct_system",
        logger=logger
    )
    
    # Create model with custom parameters
    static_dim = training_system.train_data[0].shape[1]
    dynamic_dim = training_system.train_data[1].shape[1]
    
    model = CompletePersonalityModel(
        static_dim=static_dim,
        dynamic_dim=dynamic_dim,
        fusion_dim=64,  # Custom fusion dimension
        num_heads=2,    # Fewer attention heads for faster training
        dropout_rate=0.5  # Higher dropout rate
    )
    
    # Set which layers to train (for demonstration, train all layers)
    for layer in model.layers:
        layer.trainable = True
    
    # Train model using the training system
    model, history = training_system.train_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Analyze results
    training_system.analyze_results(model, history, sample_size=5)
    
    print(f"Custom model training completed. Results saved to {output_dir}")
    return model, history


def main():
    """Main function to test the unified training system"""
    args = parse_args()
    
    # Test the convenience function approach (recommended for most use cases)
    model1, history1 = test_convenience_function(args)
    print("-" * 80)
    
    # Test direct usage of the TrainingSystem class (for advanced customization)
    model2, history2 = test_direct_system_usage(args)
    print("-" * 80)
    
    print("All tests completed successfully!")


if __name__ == "__main__":
    main()
