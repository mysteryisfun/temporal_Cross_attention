#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: integrate_optical_flow_pipeline.py
# Description: Script to integrate dynamic feature extractor with optical flow pipeline

import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.logger import get_logger
from models.dynamic_feature_extractor.feature_extractor import DynamicFeatureExtractor
from models.dynamic_feature_extractor.visualization import visualize_optical_flow_sequence
from scripts.preprocessing.optical_flow_computer import OpticalFlowComputer
from scripts.optical_flow_sequence_loader import OpticalFlowSequenceLoader

def process_video_to_flow_features(video_path, output_dir, flow_method='farneback', 
                                  sequence_length=16, model_path=None):
    """
    Process a video from raw frames to optical flow features in one integrated pipeline.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save results.
        flow_method (str): Optical flow method ('farneback' or 'tvl1').
        sequence_length (int): Length of the flow sequences to process.
        model_path (str): Path to a pre-trained dynamic feature extractor model.
        
    Returns:
        dict: Processing results with paths to saved files.
    """
    # Initialize logger
    logger = get_logger(experiment_name="optical_flow_pipeline")
    logger.logger.info(f"Processing video: {video_path}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    flow_dir = os.path.join(output_dir, "optical_flow")
    features_dir = os.path.join(output_dir, "features")
    viz_dir = os.path.join(output_dir, "visualizations")
    
    os.makedirs(flow_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Initialize feature extractor
    logger.logger.info("Initializing dynamic feature extractor")
    if model_path and os.path.exists(model_path):
        extractor = DynamicFeatureExtractor()
        extractor.load_model(model_path)
        logger.logger.info(f"Loaded model from {model_path}")
    else:
        extractor = DynamicFeatureExtractor()
        logger.logger.info("Created new model (no pre-trained model found)")
    
    # Initialize optical flow sequence loader
    flow_loader = OpticalFlowSequenceLoader(
        sequence_length=sequence_length,
        resize_shape=(224, 224)
    )
    
    # Compute optical flow from video
    logger.logger.info("Computing optical flow")
    video_name = os.path.basename(video_path).split('.')[0]
    flow_path = os.path.join(flow_dir, f"{video_name}_flow_raw.npy")
    
    if os.path.exists(flow_path):
        logger.logger.info(f"Flow already exists, loading from {flow_path}")
        flow_sequence = np.load(flow_path)
    else:
        # Compute new flow
        flow_path = flow_loader.compute_and_save_flow(
            video_path=video_path,
            output_dir=flow_dir,
            method=flow_method,
            clip_length=sequence_length
        )
        
        if not flow_path:
            logger.logger.error("Failed to compute flow")
            return None
        
        flow_sequence = np.load(flow_path)
    
    logger.logger.info(f"Flow sequence shape: {flow_sequence.shape}")
    
    # Visualize optical flow
    flow_viz_path = os.path.join(viz_dir, f"{video_name}_flow_visualization.png")
    visualize_optical_flow_sequence(flow_sequence, save_path=flow_viz_path)
    logger.logger.info(f"Saved flow visualization to {flow_viz_path}")
    
    # Preprocess flow sequence for feature extraction
    logger.logger.info("Preprocessing flow sequence")
    if len(flow_sequence.shape) == 4:  # [frames, height, width, channels]
        flow_batch = np.expand_dims(flow_sequence, 0)
    else:
        flow_batch = flow_sequence
    
    preprocessed = extractor.preprocess_flow_sequence(flow_batch)
    
    # Extract features
    logger.logger.info("Extracting dynamic features")
    features = extractor.extract_features(preprocessed)
    logger.logger.info(f"Extracted features shape: {features.shape}")
    
    # Save features
    features_path = os.path.join(features_dir, f"{video_name}_dynamic_features.npy")
    np.save(features_path, features)
    logger.logger.info(f"Saved features to {features_path}")
    
    # Visualize feature distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(min(64, features.shape[1])), features[0][:64])
    plt.title("Dynamic Feature Vector (First 64 dimensions)")
    plt.xlabel("Feature Dimension")
    plt.ylabel("Feature Value")
    plt.grid(alpha=0.3)
    
    feature_viz_path = os.path.join(viz_dir, f"{video_name}_feature_distribution.png")
    plt.savefig(feature_viz_path)
    plt.close()
    logger.logger.info(f"Saved feature visualization to {feature_viz_path}")
    
    # Visualize feature activations
    from models.dynamic_feature_extractor.visualization import FeatureVisualization
    
    # Find the last convolutional layer
    layer_name = None
    for layer in reversed(extractor.model.layers):
        if isinstance(layer, tf.keras.layers.Conv3D):
            layer_name = layer.name
            break
    
    if layer_name:
        # Create a visualization object
        visualizer = FeatureVisualization(
            model=extractor.model,
            layer_name=layer_name
        )
        
        # Get activations
        activations = visualizer.get_intermediate_activations(preprocessed)
        
        # Visualize flow and activations
        activations_path = os.path.join(viz_dir, f"{video_name}_activations.png")
        visualizer.visualize_flow_and_activations(
            preprocessed,
            activations,
            save_path=activations_path
        )
        logger.logger.info(f"Saved activation visualization to {activations_path}")
        
        # Visualize temporal activation
        temporal_path = os.path.join(viz_dir, f"{video_name}_temporal_activation.png")
        visualizer.visualize_temporal_activation(
            activations,
            feature_idx=0,
            save_path=temporal_path
        )
        logger.logger.info(f"Saved temporal activation visualization to {temporal_path}")
    
    logger.logger.info("Completed processing pipeline")
    
    return {
        'video_path': video_path,
        'flow_path': flow_path,
        'features_path': features_path,
        'flow_viz_path': flow_viz_path,
        'feature_viz_path': feature_viz_path
    }

def process_directory(input_dir, output_dir, flow_method='farneback', 
                     sequence_length=16, model_path=None, limit=None):
    """
    Process all videos in a directory using the integrated pipeline.
    
    Args:
        input_dir (str): Directory containing input video files.
        output_dir (str): Directory to save results.
        flow_method (str): Optical flow method ('farneback' or 'tvl1').
        sequence_length (int): Length of the flow sequences to process.
        model_path (str): Path to a pre-trained dynamic feature extractor model.
        limit (int): Maximum number of videos to process.
        
    Returns:
        list: Processing results for all videos.
    """
    # Initialize logger
    logger = get_logger(experiment_name="batch_optical_flow_pipeline")
    logger.logger.info(f"Processing videos from {input_dir}")
    
    # Get all video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(list(Path(input_dir).glob(f"**/*{ext}")))
    
    logger.logger.info(f"Found {len(video_files)} video files")
    
    if limit:
        video_files = video_files[:limit]
        logger.logger.info(f"Processing {len(video_files)} videos (limited by user)")
    
    # Process each video
    results = []
    for video_path in tqdm(video_files, desc="Processing videos"):
        video_name = os.path.basename(video_path).split('.')[0]
        video_output_dir = os.path.join(output_dir, video_name)
        
        try:
            result = process_video_to_flow_features(
                video_path=str(video_path),
                output_dir=video_output_dir,
                flow_method=flow_method,
                sequence_length=sequence_length,
                model_path=model_path
            )
            
            if result:
                results.append(result)
        except Exception as e:
            logger.logger.error(f"Error processing {video_path}: {str(e)}")
    
    logger.logger.info(f"Processed {len(results)} videos successfully")
    
    return results

def main():
    """
    Main function to run the integrated optical flow and dynamic feature extraction pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Integrated optical flow and dynamic feature extraction pipeline")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input video file or directory of videos")
    parser.add_argument("--output_dir", type=str, default="results/dynamic_pipeline",
                       help="Directory to save results")
    parser.add_argument("--flow_method", type=str, choices=['farneback', 'tvl1'], default='farneback',
                       help="Optical flow computation method")
    parser.add_argument("--sequence_length", type=int, default=16,
                       help="Length of optical flow sequences")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to pre-trained dynamic feature extractor model")
    parser.add_argument("--limit", type=int, default=None,
                       help="Maximum number of videos to process (for directory input)")
    args = parser.parse_args()
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Process a single video
        process_video_to_flow_features(
            video_path=args.input,
            output_dir=args.output_dir,
            flow_method=args.flow_method,
            sequence_length=args.sequence_length,
            model_path=args.model_path
        )
    elif os.path.isdir(args.input):
        # Process a directory of videos
        process_directory(
            input_dir=args.input,
            output_dir=args.output_dir,
            flow_method=args.flow_method,
            sequence_length=args.sequence_length,
            model_path=args.model_path,
            limit=args.limit
        )
    else:
        print(f"Input path {args.input} is not a valid file or directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
