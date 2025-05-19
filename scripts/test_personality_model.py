"""
Test script for the feature fusion module and prediction head.

This script validates the functionality of the feature fusion module and personality prediction head,
ensuring they correctly process static and dynamic features and output personality trait predictions.

Usage:
    if (Test-Path .\env\Scripts\activate.ps1) {
        .\env\Scripts\activate.ps1
        python .\scripts\test_personality_model.py
    }
"""
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.models.personality_model import FeatureFusionModule, PersonalityPredictionHead, CompletePersonalityModel

def test_feature_fusion_module():
    """
    Test the FeatureFusionModule component.
    """
    # Create random fusion features
    batch_size = 8
    fusion_dim = 128
    fusion_features = tf.random.normal((batch_size, fusion_dim))
    
    # Initialize the fusion module
    fusion_module = FeatureFusionModule(fusion_dim=fusion_dim)
    
    # Process the features
    output = fusion_module(fusion_features, training=True)
    
    # Check output shape
    assert output.shape == (batch_size, fusion_dim), f"Expected shape {(batch_size, fusion_dim)}, got {output.shape}"
    print("Feature fusion module test passed.")
    
    return output

def test_prediction_head(fusion_features):
    """
    Test the PersonalityPredictionHead component.
    """
    # Initialize the prediction head
    prediction_head = PersonalityPredictionHead()
    
    # Generate predictions
    predictions = prediction_head(fusion_features)
    
    # Check output shape (batch_size, 5) for OCEAN traits
    assert predictions.shape == (fusion_features.shape[0], 5), f"Expected shape {(fusion_features.shape[0], 5)}, got {predictions.shape}"
    print("Personality prediction head test passed.")
    
    return predictions

def test_complete_model():
    """
    Test the complete model integrating cross-attention, fusion, and prediction.
    """
    # Create random input features
    batch_size = 8
    static_dim = 512
    dynamic_dim = 256
    static_features = tf.random.normal((batch_size, static_dim))
    dynamic_features = tf.random.normal((batch_size, dynamic_dim))
    
    # Initialize the complete model
    model = CompletePersonalityModel(static_dim, dynamic_dim)
    
    # Generate predictions
    predictions = model((static_features, dynamic_features), training=True)
    
    # Check output shape (batch_size, 5) for OCEAN traits
    assert predictions.shape == (batch_size, 5), f"Expected shape {(batch_size, 5)}, got {predictions.shape}"
    
    # Print model summary
    model.summary(print_fn=lambda x: print(x))
    
    print("Complete personality model test passed.")

def main():
    """
    Run all tests.
    """
    print("Testing feature fusion module...")
    fusion_features = test_feature_fusion_module()
    
    print("\nTesting prediction head...")
    test_prediction_head(fusion_features)
    
    print("\nTesting complete personality model...")
    test_complete_model()

if __name__ == "__main__":
    main()
