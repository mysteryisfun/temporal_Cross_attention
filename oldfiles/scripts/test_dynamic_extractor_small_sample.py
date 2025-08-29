#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test_dynamic_extractor_small_sample.py
# Description: Script to test dynamic feature extractor with a small sample

import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.logger import get_logger
from models.dynamic_feature_extractor.feature_extractor import DynamicFeatureExtractor
from scripts.preprocessing.optical_flow_computer import OpticalFlowComputer
from scripts.optical_flow_sequence_loader import OpticalFlowSequenceLoader

def test_with_video_sample(video_path, output_dir, num_frames=16):
    """
    Test the dynamic feature extractor with a small video sample.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save test results.
        num_frames (int): Number of frames to extract from the video.
    """
    # Initialize logger
    logger = get_logger(experiment_name="dynamic_extractor_test")
    logger.logger.info(f"Testing with video sample: {video_path}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    flow_dir = os.path.join(output_dir, "optical_flow")
    visualizations_dir = os.path.join(output_dir, "visualizations")
    features_dir = os.path.join(output_dir, "features")
    
    os.makedirs(flow_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    
    # Check if the video exists
    if not os.path.exists(video_path):
        logger.logger.error(f"Video file not found: {video_path}")
        return False
    
    # Initialize feature extractor
    logger.logger.info("Initializing dynamic feature extractor")
    
    try:
        config_path = os.path.join(project_root, "config", "model_config", "dynamic_model_config.yaml")
        extractor = DynamicFeatureExtractor(config_path=config_path)
        logger.logger.info("Feature extractor initialized successfully")
    except Exception as e:
        logger.logger.error(f"Error initializing feature extractor: {str(e)}")
        return False
    
    # Initialize optical flow computer
    logger.logger.info("Computing optical flow")
    flow_computer = OpticalFlowComputer(method='farneback')
    
    # Extract frames from the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.logger.error(f"Error opening video file: {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.logger.info(f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames")
    
    # Calculate frame interval to extract evenly distributed frames
    if frame_count <= num_frames:
        frame_interval = 1
    else:
        frame_interval = frame_count // num_frames
    
    # Extract frames
    frames = []
    frame_indices = []
    
    for i in range(min(num_frames, frame_count)):
        frame_idx = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frames.append(frame)
            frame_indices.append(frame_idx)
        else:
            logger.logger.warning(f"Failed to read frame at index {frame_idx}")
    
    cap.release()
    
    if len(frames) < 2:
        logger.logger.error("Not enough frames extracted for optical flow computation")
        return False
    
    logger.logger.info(f"Extracted {len(frames)} frames")
    
    # Compute optical flow
    flow_frames = []
    
    for i in range(len(frames) - 1):
        # Convert to grayscale
        prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
        
        # Compute flow
        flow = flow_computer.compute_flow(prev_gray, curr_gray)
        flow_frames.append(flow)
        
        # Visualize flow
        flow_rgb = flow_computer.flow_to_rgb(flow)
        flow_viz_path = os.path.join(visualizations_dir, f"flow_{i:03d}.png")
        cv2.imwrite(flow_viz_path, cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR))
    
    logger.logger.info(f"Computed {len(flow_frames)} flow frames")
    
    # Convert to numpy array
    flow_sequence = np.array(flow_frames)
    
    # Save flow sequence
    flow_path = os.path.join(flow_dir, "sample_flow.npy")
    np.save(flow_path, flow_sequence)
    logger.logger.info(f"Saved flow sequence to {flow_path}")
    
    # Preprocess flow sequence
    logger.logger.info("Preprocessing flow sequence")
    
    # Add one to match expected shape [frames, height, width, channels]
    if len(flow_sequence.shape) == 3:
        flow_sequence = flow_sequence.reshape(flow_sequence.shape[0], 
                                             flow_sequence.shape[1], 
                                             flow_sequence.shape[2], 2)
    
    # Resize to match the expected input shape
    target_size = (224, 224)
    resized_sequence = np.zeros((flow_sequence.shape[0], target_size[0], target_size[1], 2))
    
    for i in range(flow_sequence.shape[0]):
        for c in range(2):
            resized_sequence[i, :, :, c] = cv2.resize(flow_sequence[i, :, :, c], target_size)
    
    flow_sequence = resized_sequence
    
    # Preprocess using the extractor's method
    preprocessed = extractor.preprocess_flow_sequence(flow_sequence)
      # Extract features
    logger.logger.info("Extracting features")
    try:
        # Check if the preprocessed sequence matches the expected input shape
        expected_frames = extractor.config.get('input_shape', (16, 224, 224, 2))[0]
        actual_frames = preprocessed.shape[1]
        
        if actual_frames != expected_frames:
            logger.logger.warning(f"Flow sequence has {actual_frames} frames, but model expects {expected_frames}")
            # Pad or truncate to match expected frame count
            if actual_frames < expected_frames:
                logger.logger.info(f"Padding sequence from {actual_frames} to {expected_frames} frames")
                padding = np.zeros((1, expected_frames - actual_frames, 
                                    preprocessed.shape[2], 
                                    preprocessed.shape[3], 
                                    preprocessed.shape[4]))
                preprocessed = np.concatenate([preprocessed, padding], axis=1)
            else:
                logger.logger.info(f"Truncating sequence from {actual_frames} to {expected_frames} frames")
                preprocessed = preprocessed[:, :expected_frames]
        
        logger.logger.info(f"Preprocessed shape: {preprocessed.shape}")
        
        start_time = time.time()
        features = extractor.extract_features(preprocessed)
        end_time = time.time()
        
        logger.logger.info(f"Feature extraction completed in {end_time - start_time:.2f} seconds")
        logger.logger.info(f"Feature shape: {features.shape}")
        
        # Save features
        feature_path = os.path.join(features_dir, "dynamic_features.npy")
        np.save(feature_path, features)
        logger.logger.info(f"Saved features to {feature_path}")
        
        # Visualize features
        plt.figure(figsize=(12, 6))
        plt.bar(range(min(64, features.shape[1])), features[0][:64])
        plt.title("Dynamic Feature Vector (First 64 dimensions)")
        plt.xlabel("Feature Dimension")
        plt.ylabel("Feature Value")
        plt.grid(alpha=0.3)
        
        feature_viz_path = os.path.join(visualizations_dir, "feature_distribution.png")
        plt.savefig(feature_viz_path)
        plt.close()
        
        logger.logger.info(f"Created feature visualization at {feature_viz_path}")
        
        # Create visualization of flow sequence
        from models.dynamic_feature_extractor.visualization import visualize_optical_flow_sequence
        
        sequence_viz_path = os.path.join(visualizations_dir, "flow_sequence.png")
        visualize_optical_flow_sequence(flow_sequence, save_path=sequence_viz_path)
        logger.logger.info(f"Created flow sequence visualization at {sequence_viz_path}")
        
        return True
        
    except Exception as e:
        logger.logger.error(f"Error extracting features: {str(e)}")
        return False

def main():
    """
    Main function to run the small sample test.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test dynamic feature extractor with a small sample")
    parser.add_argument("--video", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--output_dir", type=str, default="results/dynamic_feature_extractor/small_test",
                       help="Directory to save test results")
    parser.add_argument("--num_frames", type=int, default=16,
                       help="Number of frames to extract from the video")
    args = parser.parse_args()
    
    # Run the test
    test_with_video_sample(
        video_path=args.video,
        output_dir=args.output_dir,
        num_frames=args.num_frames
    )

if __name__ == "__main__":
    main()
