#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test_temporal_visualization_and_serialization.py
# Description: Test script for temporal feature visualization and model serialization

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import tensorflow as tf

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.logger import get_logger
from models.dynamic_feature_extractor.feature_extractor import DynamicFeatureExtractor
from models.dynamic_feature_extractor.visualization import FeatureVisualization
from models.dynamic_feature_extractor.serialization import ModelSerializer
from scripts.preprocessing.optical_flow_computer import OpticalFlowComputer

def test_temporal_visualization(video_path, output_dir, num_frames=16):
    """
    Test the temporal feature visualization capabilities with a video sample.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save test results.
        num_frames (int): Number of frames to extract from the video.
        
    Returns:
        bool: True if the test was successful, False otherwise.
    """
    # Initialize logger
    logger = get_logger(experiment_name="temporal_vis_test")
    logger.logger.info(f"Testing temporal visualization with video sample: {video_path}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    visualizations_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)
    
    try:
        # Step 1: Initialize the dynamic feature extractor
        logger.logger.info("Initializing dynamic feature extractor")
        extractor = DynamicFeatureExtractor()
        
        # Step 2: Compute optical flow from the video
        logger.logger.info("Computing optical flow")
        flow_computer = OpticalFlowComputer()
        frames = flow_computer.extract_frames_from_video(video_path, num_frames=num_frames)
        flow_sequence = flow_computer.compute_optical_flow_sequence(frames)
        
        # Save the flow sequence for reference
        optical_flow_dir = os.path.join(output_dir, "optical_flow")
        os.makedirs(optical_flow_dir, exist_ok=True)
        flow_path = os.path.join(optical_flow_dir, "sample_flow.npy")
        np.save(flow_path, flow_sequence)
        logger.logger.info(f"Saved flow sequence to {flow_path}")
        
        # Step 3: Preprocess and extract features
        logger.logger.info("Preprocessing flow sequence")
        preprocessed = extractor.preprocess_flow_sequence(flow_sequence)
        
        # Step 4: Get intermediate activations from various layers
        logger.logger.info("Extracting intermediate activations")
        
        # Create visualization object for first inception block
        inception_2a_vis = FeatureVisualization(
            model=extractor.model, 
            layer_name="inception_2a_concat"
        )
        
        # Create visualization object for middle inception block
        inception_3c_vis = FeatureVisualization(
            model=extractor.model, 
            layer_name="inception_3c_concat"
        )
        
        # Create visualization object for last inception block
        inception_4b_vis = FeatureVisualization(
            model=extractor.model, 
            layer_name="inception_4b_concat"
        )
        
        # Get activations
        start_time = time.time()
        inception_2a_act = inception_2a_vis.get_intermediate_activations(preprocessed)
        inception_3c_act = inception_3c_vis.get_intermediate_activations(preprocessed)
        inception_4b_act = inception_4b_vis.get_intermediate_activations(preprocessed)
        logger.logger.info(f"Activation extraction completed in {time.time() - start_time:.2f} seconds")
        
        # Log activation shapes
        logger.logger.info(f"Inception 2a activations shape: {inception_2a_act.shape}")
        logger.logger.info(f"Inception 3c activations shape: {inception_3c_act.shape}")
        logger.logger.info(f"Inception 4b activations shape: {inception_4b_act.shape}")
        
        # Step 5: Generate temporal visualizations
        logger.logger.info("Generating temporal visualizations")
        
        # 5.1: Temporal evolution line graph
        inception_3c_vis.visualize_temporal_feature_evolution(
            activations=inception_3c_act,
            feature_indices=[0, 10, 20, 30, 40],
            save_path=os.path.join(visualizations_dir, "temporal_evolution.png")
        )
        logger.logger.info("Created temporal evolution visualization")
        
        # 5.2: Feature activation heatmap
        inception_3c_vis.create_feature_activation_heatmap(
            activations=inception_3c_act,
            feature_idx=10,
            save_path=os.path.join(visualizations_dir, "feature_heatmap.png")
        )
        logger.logger.info("Created feature activation heatmap")
        
        # 5.3: 3D feature visualization
        inception_3c_vis.visualize_3d_feature_activation(
            activations=inception_3c_act,
            feature_idx=10,
            save_path=os.path.join(visualizations_dir, "feature_3d.png")
        )
        logger.logger.info("Created 3D feature visualization")
        
        # 5.4: Feature evolution animation
        inception_3c_vis.create_feature_evolution_animation(
            activations=inception_3c_act,
            feature_idx=10,
            save_path=os.path.join(visualizations_dir, "feature_evolution.gif")
        )
        logger.logger.info("Created feature evolution animation")
        
        # 5.5: Comparative temporal view
        inception_3c_vis.create_comparative_temporal_view(
            activations=inception_3c_act,
            feature_indices=[0, 10, 20],
            flow_sequence=preprocessed,
            save_path=os.path.join(visualizations_dir, "comparative_view.png")
        )
        logger.logger.info("Created comparative temporal view")
        
        return True
        
    except Exception as e:
        logger.logger.error(f"Error in temporal visualization test: {str(e)}")
        import traceback
        logger.logger.error(traceback.format_exc())
        return False


def test_model_serialization(output_dir):
    """
    Test the model serialization capabilities.
    
    Args:
        output_dir (str): Directory to save test results.
        
    Returns:
        bool: True if the test was successful, False otherwise.
    """
    # Initialize logger
    logger = get_logger(experiment_name="serialization_test")
    logger.logger.info("Testing model serialization")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        # Step 1: Initialize the dynamic feature extractor
        logger.logger.info("Initializing dynamic feature extractor")
        extractor = DynamicFeatureExtractor()
        
        # Step 2: Test basic model saving and loading
        logger.logger.info("Testing basic model saving")
        basic_model_path = os.path.join(models_dir, "basic_model")
        extractor.save_model(basic_model_path)
        
        # Create a new extractor and load the model
        logger.logger.info("Testing basic model loading")
        new_extractor = DynamicFeatureExtractor()
        load_success = new_extractor.load_model(basic_model_path)
        if load_success:
            logger.logger.info("Basic model loading successful")
        else:
            logger.logger.error("Basic model loading failed")
            return False
        
        # Step 3: Test ModelSerializer saving with versioning and metadata
        logger.logger.info("Testing ModelSerializer saving")
        
        # Add custom metadata
        metadata = {
            "author": "Research Team",
            "description": "Dynamic feature extractor for optical flow sequences",
            "training_dataset": "N/A - using pretrained weights",
            "usage_notes": "Extract features from optical flow for action recognition"
        }
        
        # Save the model with serializer
        serialized_path = extractor.save_model_with_serializer(
            base_dir=models_dir,
            metadata=metadata
        )
        
        if serialized_path:
            logger.logger.info(f"ModelSerializer saving successful: {serialized_path}")
        else:
            logger.logger.error("ModelSerializer saving failed")
            return False
        
        # Step 4: Test ModelSerializer model loading
        logger.logger.info("Testing ModelSerializer loading")
        
        # List available model versions
        versions = ModelSerializer.list_available_versions(models_dir, "dynamic_feature_extractor")
        logger.logger.info(f"Available versions: {versions}")
        
        # Load the latest model
        latest_model, latest_metadata, latest_version = ModelSerializer.load_latest_model(
            base_dir=models_dir,
            model_name="dynamic_feature_extractor"
        )
        
        if latest_model is not None:
            logger.logger.info(f"Loaded latest model version: {latest_version}")
            logger.logger.info(f"Model metadata: {latest_metadata.get('description')}")
        else:
            logger.logger.error("Failed to load latest model")
            return False
        
        # Step 5: Export model for inference
        logger.logger.info("Exporting model for inference")
        inference_dir = os.path.join(models_dir, "inference")
        export_path = ModelSerializer.export_for_inference(extractor.model, inference_dir)
        
        if export_path:
            logger.logger.info(f"Model exported for inference to {export_path}")
        else:
            logger.logger.error("Failed to export model for inference")
            return False
        
        return True
        
    except Exception as e:
        logger.logger.error(f"Error in model serialization test: {str(e)}")
        import traceback
        logger.logger.error(traceback.format_exc())
        return False


def main():
    """
    Main function to run tests for temporal visualization and model serialization.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test temporal visualization and model serialization"
    )
    parser.add_argument(
        "--video", 
        type=str, 
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/dynamic_feature_extractor/temporal_and_serialization_test",
        help="Directory to save test results"
    )
    parser.add_argument(
        "--num_frames", 
        type=int, 
        default=16,
        help="Number of frames to extract from the video"
    )
    parser.add_argument(
        "--skip_visualization", 
        action="store_true",
        help="Skip the temporal visualization test"
    )
    parser.add_argument(
        "--skip_serialization", 
        action="store_true",
        help="Skip the model serialization test"
    )
    
    args = parser.parse_args()
    
    # Run the tests
    if not args.skip_visualization:
        print("Running temporal visualization test...")
        vis_success = test_temporal_visualization(
            video_path=args.video,
            output_dir=args.output_dir,
            num_frames=args.num_frames
        )
        print(f"Temporal visualization test {'successful' if vis_success else 'failed'}")
    
    if not args.skip_serialization:
        print("Running model serialization test...")
        serial_success = test_model_serialization(
            output_dir=args.output_dir
        )
        print(f"Model serialization test {'successful' if serial_success else 'failed'}")
    
    print(f"All test results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
